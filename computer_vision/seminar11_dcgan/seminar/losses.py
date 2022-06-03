import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

LABEL_REAL = 0.95
LABEL_FAKE = 0.05


class GANLoss(nn.Module):

    def generator(self, discriminator, generator, real_batch, fake_batch, device):
        raise NotImplementedError

    def discriminator(self, discriminator, generator, real_batch, fake_batch, device):
        raise NotImplementedError


class GANBCELoss(GANLoss):

    def generator(self, discriminator, generator, real_batch, fake_batch, device):
        prob_fake = discriminator(fake_batch.to(device))
        labels_fake = torch.ones_like(prob_fake) * LABEL_REAL
        generator_loss = binary_cross_entropy_with_logits(prob_fake, labels_fake)
        return generator_loss

    def discriminator(self, discriminator, generator, real_batch, fake_batch, device):
        prob_real = discriminator(real_batch.to(device))
        prob_fake = discriminator(fake_batch.detach().to(device))
        labels_real = torch.ones_like(prob_real) * LABEL_REAL
        labels_fake = torch.ones_like(prob_fake) * LABEL_FAKE
        discriminator_loss = (binary_cross_entropy_with_logits(prob_real, labels_real) +
                              binary_cross_entropy_with_logits(prob_fake, labels_fake)) / 2
        return discriminator_loss


class GANWLoss(GANLoss):

    def generator(self, discriminator, generator, real_batch, fake_batch, device):
        prob_fake = discriminator(fake_batch.to(device))
        generator_loss = - torch.mean(prob_fake)
        return generator_loss

    def discriminator(self, discriminator, generator, real_batch, fake_batch, device):
        prob_real = discriminator(real_batch.to(device))
        prob_fake = discriminator(fake_batch.detach().to(device))
        discriminator_loss = - torch.mean(prob_real) + torch.mean(prob_fake)
        return discriminator_loss


class GANHingeLoss(GANLoss):
    def generator(self, discriminator, generator, real_batch, fake_batch, device):
        prob_fake = discriminator(fake_batch.to(device))
        generator_loss = - torch.mean(prob_fake)
        return generator_loss

    def discriminator(self, discriminator, generator, real_batch, fake_batch, device):
        prob_real = discriminator(real_batch.to(device))
        prob_fake = discriminator(fake_batch.to(device).detach())
        discriminator_loss = torch.mean(torch.relu(1.0 - prob_real)) + torch.mean(torch.relu(1.0 + prob_fake))
        return discriminator_loss


LOSSES = {
    "bce": GANBCELoss,
    "wasserstein": GANWLoss,
    "hinge": GANHingeLoss
}
