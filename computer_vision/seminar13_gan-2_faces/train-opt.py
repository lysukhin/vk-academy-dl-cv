"""Script for running Optuna hyperparams optimization for GAN using FID as quality metric.

To store intermediate trials results Optuna uses local databases, e.g. MySQL.
To be able to use it, one should install the following requirements (Ubuntu):
```
sudo apt-get install mysql-server
sudo mysql_secure_installation
sudo apt-get install libmysqlclient-dev
pip install mysqlclient
```
Then MySQL should be setup for using without password for main user.

After running the script the dataframe with results can be viewed as follows:
```
import optuna

storage_name = "mysql://root@localhost/optuna"
study_name = # name used for running this script

study = optuna.load_study(study_name=study_name, storage=storage_name)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

df.head()
```
"""

import os
import pickle
from argparse import ArgumentParser
from subprocess import run

import numpy as np
import optuna
import torch
import tqdm
from optuna.trial import TrialState
from torch import optim

from seminar.data import get_data, ImageFolderNoLabels
from seminar.fid import GeneratorDataset, fid_transforms, compute_fid
from seminar.losses import LOSSES
from seminar.models import Generator, Discriminator, IMAGE_CHANNELS, weights_init, weights_init_ortho, add_spectral_norm
from seminar.utils import get_device, save_data_batch
from seminar.utils_todo import generate_noise


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--name", "-n", help="Experiment name.")
    parser.add_argument("--data", "-d", help="Path to data root.", default="./data/img_align_celeba")
    parser.add_argument("--loss", choices=list(LOSSES.keys()), help="Loss to use for GAN training.")
    parser.add_argument("--batch-size", "-b", type=int, default=512, help="Batch size.")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Num epochs.")
    parser.add_argument("--num-workers", "-j", type=int, default=4)
    parser.add_argument("--num-trials", type=int, default=64)
    parser.add_argument("--fid-sample-size", type=int, default=8192, help="Sample size for FID computing.")
    return parser.parse_args()


def train_epoch(generator, discriminator,
                generator_optimizer, discriminator_optimizer,
                dataloader, epoch, num_epochs,
                latent_dim, fixed_noise, loss_fn, n_critic,
                device):
    generator_loss_list, discriminator_loss_list = [], []

    for i, real_batch in enumerate(tqdm.tqdm(dataloader, desc=f"[#{epoch}]")):

        discriminator.train()
        generator.eval()

        for j in range(n_critic):
            discriminator.zero_grad()

            noise_vectors = generate_noise(dataloader.batch_size, latent_dim, device)
            fake_batch = generator(noise_vectors)

            discriminator_loss = loss_fn.discriminator(discriminator, generator, real_batch, fake_batch, device)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        generator.train()
        generator.zero_grad()

        noise_vectors = generate_noise(dataloader.batch_size, latent_dim, device)
        fake_batch = generator(noise_vectors)

        generator_loss = loss_fn.generator(discriminator, generator, real_batch, fake_batch, device)
        generator_loss.backward()
        generator_optimizer.step()

        # Save Losses for plotting later
        generator_loss_list.append(generator_loss.item())
        discriminator_loss_list.append(discriminator_loss.item())

    # Output training stats
    generator.eval()
    with torch.no_grad():
        generated_batch = generator(fixed_noise).detach().cpu()
    generator.train()

    epoch_results = {
        "generator_loss_list": generator_loss_list,
        "discriminator_loss_list": discriminator_loss_list,
        "generated_batch_list": [generated_batch]
    }

    return epoch_results


def main(args):
    device = get_device()

    cmd = f"mysql -u root -e 'CREATE DATABASE IF NOT EXISTS optuna;'"
    run(cmd, shell=True)

    dataset, dataloader = get_data(args.data, args.batch_size, args.num_workers)

    run_dirname = os.path.join("runs", args.name)
    os.makedirs(run_dirname, exist_ok=True)

    real_dataset_for_fid = ImageFolderNoLabels(args.data, fid_transforms)
    real_dataset_for_fid.image_filenames = real_dataset_for_fid.image_filenames[:args.fid_sample_size]

    def objective_fn(trial: optuna.Trial):

        trial_dirname = os.path.join(run_dirname, f"trial_{str(trial.number).zfill(2)}")
        os.makedirs(trial_dirname, exist_ok=True)
        results_filename = os.path.join(trial_dirname, "results.pkl")
        checkpoint_filename = os.path.join(trial_dirname, "checkpoint.pth.tar")
        images_dirname = os.path.join(trial_dirname, "images")
        os.makedirs(images_dirname, exist_ok=True)

        _latent_dim = trial.suggest_categorical("latent_dim", (64, 128))

        _gen_base_features = trial.suggest_categorical("gen_base_features", (16, 32, 64))
        _dis_base_features = trial.suggest_categorical("dis_base_features", (16, 32, 64))

        _gen_normalization = trial.suggest_categorical("gen_normalization", ("batch", "none"))
        _dis_normalization = trial.suggest_categorical("dis_normalization", ("batch", "none"))

        _init_ortho = False

        _gen_sn = False
        _dis_sn = trial.suggest_categorical("dis_sn", (True, False))

        _lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

        _beta1 = trial.suggest_categorical("beta1", (0.0, 0.5))
        _beta2 = trial.suggest_categorical("beta2", (0.9, 0.999))

        _n_critic = trial.suggest_int("n_critic", 1, 5)

        generator = Generator(input_channels=_latent_dim,
                              base_features=_gen_base_features,
                              normalization=_gen_normalization,
                              output_channels=IMAGE_CHANNELS)
        discriminator = Discriminator(input_channels=IMAGE_CHANNELS,
                                      base_features=_dis_base_features,
                                      normalization=_dis_normalization)

        init_fn = weights_init if not _init_ortho else weights_init_ortho
        generator.apply(init_fn)
        discriminator.apply(init_fn)

        if _gen_sn:
            generator.apply(add_spectral_norm)
        if _dis_sn:
            discriminator.apply(add_spectral_norm)

        generator.to(device)
        discriminator.to(device)

        generator_optimizer = optim.Adam(generator.parameters(), lr=_lr, betas=(_beta1, _beta2), amsgrad=True)
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=_lr, betas=(_beta1, _beta2), amsgrad=True)

        loss_fn = LOSSES[args.loss]()
        fixed_noise = generate_noise(args.batch_size, _latent_dim, device)

        fid = None
        results = {
            key: [] for key in [
                "generator_loss_list",
                "discriminator_loss_list",
                "generated_batch_list",
                "fid"
            ]
        }
        for epoch in range(args.epochs):
            epoch_results = train_epoch(generator, discriminator,
                                        generator_optimizer, discriminator_optimizer,
                                        dataloader, epoch, args.epochs,
                                        _latent_dim, fixed_noise, loss_fn, _n_critic,
                                        device)

            fake_dataset = GeneratorDataset(generator, _latent_dim, args.fid_sample_size, device)
            fid = compute_fid(real_dataset_for_fid, fake_dataset)

            with open(checkpoint_filename, "wb") as fp:
                torch.save({"generator": generator.state_dict(),
                            "discriminator": discriminator.state_dict()}, fp)

            if epoch % 5 == 0:
                image_filename = os.path.join(images_dirname, f"batch_ep={str(epoch).zfill(2)}.png")
                save_data_batch(epoch_results["generated_batch_list"][0], image_filename)

            discriminator_loss_mean = np.mean(epoch_results["discriminator_loss_list"])
            generator_loss_mean = np.mean(epoch_results["generator_loss_list"])
            print(f"[{epoch}/{args.epochs}]\t"
                  f"Loss_D: {discriminator_loss_mean.item():.4f}\tLoss_G: {generator_loss_mean.item():.4f}\t"
                  f"FID: {fid:.4f}")

            trial.report(fid, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return fid

    study = optuna.create_study(study_name=args.name,
                                storage=f"mysql://root@localhost/optuna",
                                pruner=optuna.pruners.HyperbandPruner(),
                                direction="minimize",
                                load_if_exists=True)

    study.optimize(objective_fn, n_trials=args.num_trials, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("\tNumber of finished trials: ", len(study.trials))
    print("\tNumber of pruned trials: ", len(pruned_trials))
    print("\tNumber of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("\tValue: ", trial.value)

    print("\tParams: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open(os.path.join(run_dirname, "study.pkl"), "wb") as fp:
        pickle.dump(study, fp)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = parse_arguments()
    main(args)
