import os
import pickle
from argparse import ArgumentParser

import torch
from torch import optim

from seminar.data import get_data
from seminar.losses import LOSSES
from seminar.models import Generator, Discriminator, LATENT_DIM, IMAGE_CHANNELS, DISCRIMINATOR_BASE_FEATURES, \
    GENERATOR_BASE_FEATURES, weights_init, weights_init_ortho, add_spectral_norm
from seminar.training import train_epoch
from seminar.utils import get_device, save_data_batch
from seminar.utils_todo import generate_noise


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--name", "-n", help="Experiment name.")
    parser.add_argument("--data", "-d", help="Path to data root.", default="./data/img_align_celeba/")
    parser.add_argument("--loss", choices=list(LOSSES.keys()), default="bce", help="Loss to use for GAN training.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--n-critic", "-nc", type=int, default=1, help="Number of D updates per 1 G update.")
    parser.add_argument("--gen-base-features", "-gbf", type=int, default=GENERATOR_BASE_FEATURES,
                        help="Generator base_features.")
    parser.add_argument("--dis-base-features", "-dbf", type=int, default=DISCRIMINATOR_BASE_FEATURES,
                        help="Discriminator base_features.")
    parser.add_argument("--gen-normalization", "-gn", choices=("batch", "none"), default="batch",
                        help="Generator normalization.")
    parser.add_argument("--dis-normalization", "-dn", choices=("batch", "none"), default="batch",
                        help="Discriminator normalization.")
    parser.add_argument("--gen-sn", "-gsn", action="store_true", help="Spectral normzalization for generator.")
    parser.add_argument("--dis-sn", "-dsn", action="store_true", help="Spectral normzalization for discriminator.")
    parser.add_argument("--init-ortho", action="store_true", help="Use orthogonal initialization.")
    parser.add_argument("--batch-size", "-b", type=int, default=512, help="Batch size.")
    parser.add_argument("--epochs", "-e", type=int, default=64, help="Num epochs.")
    parser.add_argument("--latent-dim", "-l", type=int, default=LATENT_DIM)
    parser.add_argument("--num-workers", "-j", type=int, default=8)
    return parser.parse_args()


def main(args):
    device = get_device()

    dataset, dataloader = get_data(args.data, args.batch_size, args.num_workers)

    generator = Generator(input_channels=args.latent_dim, base_features=args.gen_base_features,
                          normalization=args.gen_normalization, output_channels=IMAGE_CHANNELS)
    discriminator = Discriminator(input_channels=IMAGE_CHANNELS, base_features=args.dis_base_features,
                                  normalization=args.dis_normalization)

    print(generator)
    print(discriminator)

    init_fn = weights_init if not args.init_ortho else weights_init_ortho
    generator.apply(init_fn)
    discriminator.apply(init_fn)

    if args.gen_sn:
        generator.apply(add_spectral_norm)
    if args.dis_sn:
        discriminator.apply(add_spectral_norm)

    generator.to(device)
    discriminator.to(device)

    generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), amsgrad=True)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                         amsgrad=True)

    run_dirname = os.path.join("runs", args.name)
    os.makedirs(run_dirname)
    results_filename = os.path.join(run_dirname, "results.pkl")
    checkpoint_filename = os.path.join(run_dirname, "checkpoint.pth.tar")
    images_dirname = os.path.join(run_dirname, "images")
    os.makedirs(images_dirname)

    results = {
        key: [] for key in [
            "generator_loss_list",
            "discriminator_loss_list",
            "generated_batch_list"
        ]
    }

    loss_fn = LOSSES[args.loss]()
    fixed_noise = generate_noise(args.batch_size, args.latent_dim, device)

    for epoch in range(args.epochs):
        epoch_results = train_epoch(generator, discriminator,
                                    generator_optimizer, discriminator_optimizer,
                                    dataloader, epoch, args.epochs,
                                    args.latent_dim, fixed_noise, loss_fn, args.n_critic,
                                    device)
        for key in results:
            results[key].extend(epoch_results[key])

        with open(results_filename, "wb") as fp:
            pickle.dump(results, fp)

        with open(checkpoint_filename, "wb") as fp:
            torch.save({"generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict()}, fp)

        image_filename = os.path.join(images_dirname, f"batch_ep={str(epoch).zfill(2)}.png")
        save_data_batch(epoch_results["generated_batch_list"][0], image_filename)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
