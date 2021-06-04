import torch
from torch import nn
from torch.nn.utils import spectral_norm

LATENT_DIM = 128
IMAGE_CHANNELS = 3
GENERATOR_BASE_FEATURES = 64
DISCRIMINATOR_BASE_FEATURES = 64
NORMALIZATION = "batch"


class Generator(nn.Module):
    
    def __init__(self, input_channels=LATENT_DIM, base_features=GENERATOR_BASE_FEATURES, 
                 normalization=NORMALIZATION, output_channels=IMAGE_CHANNELS):
        """DCGAN Generator model class. 
        
        Args:
            - input_channels: Number of channels (depth) of input image.
            - base_features: Parameter to control width of convolutional layers.
            - normalization: Type of normalization layer to use ("batch" or None).
            - output_channels: Number or channels (depth) of output image.
        """
        super(Generator, self).__init__()
        # conv2d_t arithmetics: h_out = (h_in - 1) * stride + h_kernel - 2 * pad
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            
            nn.ConvTranspose2d(input_channels, base_features * 8, 4, 1, 0),
            nn.BatchNorm2d(base_features * 8) if normalization == "batch" else nn.Identity(),
            nn.ReLU(inplace=True),
            # state size: (base_features * 8) x 4 x 4

            nn.ConvTranspose2d(base_features * 8, base_features * 4, 4, 2, 1),
            nn.BatchNorm2d(base_features * 4) if normalization == "batch" else nn.Identity(),
            nn.ReLU(inplace=True),
            # state size: (base_features * 4) x 8 x 8

            nn.ConvTranspose2d(base_features * 4, base_features * 2, 4, 2, 1),
            nn.BatchNorm2d(base_features * 2) if normalization == "batch" else nn.Identity(),
            nn.ReLU(inplace=True),
            # state size: (base_features * 2) x 16 x 16
            
            nn.ConvTranspose2d(base_features * 2, base_features * 1, 4, 2, 1),
            nn.BatchNorm2d(base_features * 1) if normalization == "batch" else nn.Identity(),
            nn.ReLU(inplace=True),
            # state size: (base_features * 1) x 32 x 32

            nn.ConvTranspose2d(base_features * 1, output_channels, 4, 2, 1),
            nn.Tanh()
            # state size: output_channels x 64 x 64
        )

    def forward(self, inputs):
        return self.main(inputs)


class Discriminator(nn.Module):
        
    def __init__(self, input_channels=IMAGE_CHANNELS, base_features=DISCRIMINATOR_BASE_FEATURES, 
                 normalization=NORMALIZATION):
        """DCGAN Discriminator model class. 
        
        Args:
            - input_channels: Number of channels (depth) of input image.
            - base_features: Parameter to control width of convolutional layers.
            - normalization: Type of normalization layer to use ("batch" or None).
        """
        super(Discriminator, self).__init__()
        # conv2d arithmetics: h_out = (h_in + 2 * pad - h_kernel) / stride + 1
        self.main = nn.Sequential(
            # input size: input_channels x H x W
            
            nn.Conv2d(input_channels, base_features, 4, 2, 0),
            nn.BatchNorm2d(base_features) if normalization == "batch" else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: base_num_features x H/2 x W/2

            nn.Conv2d(base_features, base_features * 2, 4, 2, 0),
            nn.BatchNorm2d(base_features * 2) if normalization == "batch" else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (base_features * 2) x H/4 x W/4

            nn.Conv2d(base_features * 2, base_features * 4, 4, 2, 0),
            nn.BatchNorm2d(base_features * 4) if normalization == "batch" else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (base_features * 4) x H/8 x W/8
            
            nn.Conv2d(base_features * 4, base_features * 8, 4, 2, 0),
            nn.BatchNorm2d(base_features * 8) if normalization == "batch" else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (base_features * 8) x H/16 x W/16
            
            nn.AdaptiveAvgPool2d((1, 1)),
            # state size: (base_features * 8) x 1 x 1
            
            nn.Conv2d(base_features * 8, 1, 1, 1, 0),
            # state size: 1 x 1 x 1
            
            nn.Flatten()
        )

    def forward(self, inputs):
        return self.main(inputs)


def weights_init(m, scale=0.02):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        torch.nn.init.normal_(m.weight, 0.0, scale)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, scale)
        torch.nn.init.constant_(m.bias, 0)


def weights_init_ortho(m):
    raise NotImplementedError


def add_spectral_norm(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
        m.apply(spectral_norm)
