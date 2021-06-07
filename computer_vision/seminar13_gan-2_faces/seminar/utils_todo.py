import torch


def generate_noise(batch_size, latent_dim, device=None):
    """Create batch_size normally distributed (0, I) vectors of length latent_dim.

    Args:
        - batch_size: Number of vectors to sample.
        - latent_dim: Dimension of latent space (= length of noise vector).
        - device: Device to store output on.

    Returns:
        Torch.Tensor of shape (batch_size, latent_dim, 1, 1).
    """
    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    return noise


def interpolate_noise(noise1, noise2, num=3):
    """Interpolate between two tensors of the same shape.

    Args:
        - noise1: First ('start') tensor (shaped b x latent_dim x *).
        - noise2: Second ('end') tensor (shaped b x latent_dim x *).
        - num: Number of points in interpolated output.

    Returns:
        List of num torch.Tensors (each shaped b x latent_dim x *).
    """
    step_size = 1 / (num - 1)
    weights = torch.tensor([i * step_size for i in range(num)]).to(noise1.device)
    interpolation_list = [torch.lerp(noise1, noise2, w) for w in weights]
    return interpolation_list
