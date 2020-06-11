"""Helper tools for the seminar."""
import os
import time

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm


USE_CUDA = torch.cuda.is_available()


class ImageNetDataset(torch.utils.data.Dataset):
    """ImageNet dataset.

    Dataset folder must contain subfolders with names from "0" to "999".
    Each subfolder contains images corresponding to the label equal to folder name.
    """
    def __init__(self, root, num_samples=None, image_size=224, to_tensor=False, fp16=False):
        """Create dataset.

        Args:
            root: Dataset root path.
            image_size: Output image size along each dim.
            to_tensor: Prepare image for CNN input.
        """
        self._image_size = image_size
        self._to_tensor = to_tensor
        self._dtype = np.float16 if fp16 else np.float32
        self._paths = []
        self._labels = []
        for subfolder in os.listdir(root):
            try:
                label = int(subfolder)
            except ValueError:
                raise RuntimeError("Bad dataset folder format. Unknown label: {}".format(subfolder))
            if (label < 0) or (label > 999):
                raise RuntimeError("Bad label: {} (expected labels in the range [0, 999])".format(label))
            label_root = os.path.join(root, subfolder)
            for filename in os.listdir(label_root):
                path = os.path.join(label_root, filename)
                ext = os.path.splitext(path)[1].lower()
                if (not os.path.isfile(path)) or (ext != ".jpeg"):
                    raise RuntimeError("Bad image: {}".format(path))
                self._paths.append(path)
                self._labels.append(label)
        if num_samples is None:
            self._indices = np.arange(len(self._paths))
        else:
            np.random.seed(0)
            self._indices = np.random.permutation(len(self._paths))[:num_samples]
        self._mean = np.array([0.485, 0.456, 0.406], dtype=self._dtype)
        self._std = np.array([0.229, 0.224, 0.225], dtype=self._dtype)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        index = self._indices[index]
        image = cv2.imread(self._paths[index])
        image = cv2.resize(image, (self._image_size, self._image_size), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._to_tensor:
            image = image.astype(self._dtype) / 255
            image = (image - self._mean) / self._std
            image = torch.from_numpy(image.transpose((2, 0, 1)))
        return image, self._labels[index]


def model_info(model):
    num_parameters = 0
    for p in model.parameters():
        num_parameters += np.prod(p.shape)
    print(model)
    print()
    print("Number of parameters: {}".format(num_parameters))


class Estimator(pl.LightningModule):
    """Модуль, объединяющий модель, данные и оптимизацию."""
    def __init__(self, model, dataset, batch_size, num_workers):
        super().__init__()
        self._model = model
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._loss = torch.nn.CrossEntropyLoss()
        self._start_time = None
        self._num_samples = 0

    def forward(self, images):
        """Применение модели."""
        return self._model(images)
    
    @pl.data_loader
    def train_dataloader(self):
        """Создать загрузчик данных."""
        return torch.utils.data.DataLoader(self._dataset,
                                           batch_size=self._batch_size,
                                           num_workers=self._num_workers,
                                           pin_memory=USE_CUDA)
    
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """Подсчитать loss для порции данных."""
        if self._start_time is None:
            self._start_time = time.time()
        images, labels = batch
        result = self.forward(images)
        loss = self._loss(result, labels)
        self._num_samples += len(images)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        """После эпохи выводим оценку производительности."""
        end_time = time.time()
        print("Speed: {:.3f} ms per sample (total {} samples)".format(
            1000 * (end_time - self._start_time) / self._num_samples,
            self._num_samples))
        self._start_time = None
        self._num_samples = 0
        # Dummy implementation.
        return outputs[0]
    
    def configure_optimizers(self):
        """Создаем оптимизатор."""
        return torch.optim.Adam(self._model.parameters(), lr=0.001)


class report_time(object):
    def __init__(self, num_samples):
        self._start = None
        self._num_samples = num_samples

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end = time.time()
        print("Speed: {:.3f} ms per sample".format(1000 * (end - self._start) / self._num_samples))
        self._start = None


def eval_model(model, dataloader, simulate_train=False):
    with report_time(len(dataloader.dataset)):
        top1_correct = []
        top5_correct = []
        if simulate_train:
            model.train()
        else:
            model.eval()
        for images, labels in tqdm(dataloader):
            if USE_CUDA:
                images = images.cuda()
                labels = labels.cuda()
            if simulate_train:
                model_result = model(images)
                model_result.mean().backward()
            else:
                with torch.no_grad():
                    model_result = model(images)
            predictions = model_result.topk(5, 1).indices
            for label, topk in zip(labels, predictions):
                top1_correct.append(bool(label == topk[0]))
                top5_correct.append(bool(label in topk))
    print("Top 1 accuracy: {:.3f}".format(np.mean(top1_correct)))
    print("Top 5 accuracy: {:.3f}".format(np.mean(top5_correct)))
