import torch
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics
from torchvision import transforms

from .utils_todo import generate_noise


class TensorToUInt8(torch.nn.Module):
    @staticmethod
    def forward(tensor):
        tensor = torch.clip(tensor, 0, 1)
        tensor = (tensor * 255).type(torch.uint8)
        return tensor


fid_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    TensorToUInt8(),
])


class GeneratorDataset(Dataset):
    def __init__(self, generator, latent_dim, target_len, device):
        self.generator = generator
        self.generator.to(device)
        self.latent_dim = latent_dim
        self.target_len = target_len
        self.device = device

    def __len__(self):
        return self.target_len

    def __getitem__(self, item):
        self.generator.eval()
        noise = generate_noise(1, self.latent_dim)
        with torch.no_grad():
            image = self.generator(noise.to(self.device)).squeeze(0).cpu()  # 1 x 64 x 64
        image = (image * 0.25) + 0.5
        image = TensorToUInt8.forward(image)
        # image = torch.clip(image, 0, 1)
        # image = (image * 255).type(torch.uint8)
        return image


def compute_fid(real_dataset, fake_dataset):
    metrics_dict = calculate_metrics(real_dataset, fake_dataset, cuda=True, fid=True, verbose=False)
    fid = metrics_dict["frechet_inception_distance"]
    return fid
