import torch
import tqdm

from .utils_todo import generate_noise


def train_epoch(generator, discriminator,
                generator_optimizer, discriminator_optimizer,
                dataloader, epoch, num_epochs,
                latent_dim, fixed_noise, loss_fn, n_critic,
                device):
    generator.train()
    discriminator.train()
    generator_loss_list, discriminator_loss_list = [], []

    for i, real_batch in enumerate(tqdm.tqdm(dataloader, desc=f"[#{epoch}]")):

        # Train D (n_critic times)
        for j in range(n_critic):
            discriminator.zero_grad()

            noise_vectors = generate_noise(dataloader.batch_size, latent_dim, device)
            fake_batch = generator(noise_vectors)

            discriminator_loss = loss_fn.discriminator(discriminator, generator, real_batch, fake_batch, device)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        # Train G
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
    print(f"[{epoch}/{num_epochs}]\t"
          f"Loss_D: {discriminator_loss.item():.4f}\tLoss_G: {generator_loss.item():.4f}\t")
    
    # Generate fake images using fixed_noise 
    generator.eval()
    with torch.no_grad():
        generated_batch = generator(fixed_noise).detach().cpu()

    epoch_results = {
        "generator_loss_list": generator_loss_list,
        "discriminator_loss_list": discriminator_loss_list,
        "generated_batch_list": [generated_batch]
    }

    return epoch_results
