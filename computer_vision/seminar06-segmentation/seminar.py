import random
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10

import cvmade


class Module(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer_fn, trainset, testset,
                 lr_scheduler_fn=None,
                 batch_size=16):
        super().__init__()
        self._model = model
        self._optimizer_fn = optimizer_fn
        self._lr_scheduler_fn = lr_scheduler_fn
        self._criterion = loss_fn()
        self._batch_size = batch_size
        self._trainset = trainset
        self._testset = testset
        
    def forward(self, input):
        return self._model(input)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        logits = self._model(x)
        loss = self._criterion(logits, y)
        self.logger.experiment.add_scalars("loss", 
                                           {"train": loss}, 
                                           global_step=self.global_step)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self._model(x)
        loss = self._criterion(logits, y)
        self.logger.experiment.add_scalars("loss", 
                                           {"val": loss}, 
                                           global_step=self.global_step)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self._trainset, batch_size=self._batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self._testset, batch_size=self._batch_size, shuffle=False, num_workers=4)

    def configure_optimizers(self):
        optimizer = self._optimizer_fn(self._model)
        scheduler = self._lr_scheduler_fn(optimizer)
        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer
    

def train_model(model, loss_fn, optimizer_fn, trainset, testset,
                lr_scheduler_fn=None,
                batch_size=4,
                eval_steps=250,
                num_epochs=1):
    model = Module(model, loss_fn, optimizer_fn, trainset, testset,
                   lr_scheduler_fn, batch_size)
    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()), max_epochs=num_epochs, val_check_interval=eval_steps)
    trainer.fit(model)


class CheckerError(Exception):
    pass


def check_ok():
    print("========")
    print("CHECK OK")

def check_conv1x1(layer_fn):
    layer = layer_fn(2, 5)
    if not isinstance(layer, torch.nn.Conv2d):
        raise CheckerError("conv1x1: неверные тип слоя")
    if layer.kernel_size != (1, 1):
        raise CheckerError("conv1x1: неверный размер свертки")
    if layer.in_channels != 2 or layer.out_channels != 5:
        raise CheckerError("conv1x1: неверное количество каналов")

def check_t_conv(layer_fn):
    layer = layer_fn(2, 5)
    if not isinstance(layer, torch.nn.ConvTranspose2d):
        raise CheckerError("transposed2x2: неверные тип слоя")
    if layer.kernel_size != (2, 2):
        raise CheckerError("transposed2x2: неверный размер свертки")
    if layer.stride != (2, 2):
        raise CheckerError("transposed2x2: неверный шаг")
    if layer.in_channels != 2 or layer.out_channels != 5:
        raise CheckerError("transposed2x2: неверное количество каналов")
        
def check_bce_loss(loss_class):
    loss_computer = loss_class()
    if not isinstance(loss_computer._bce, torch.nn.BCEWithLogitsLoss):
        raise CheckerError("Неверный класс функции потерь")
    def to_logits(probs):
        return np.log(probs) - np.log(1 - probs)
    logits = torch.from_numpy(to_logits(np.array([0.9, 0.2]))).reshape(2, 1, 1, 1).float()
    labels = torch.from_numpy(np.array([1, 0])).reshape(2, 1, 1).long()
    value = loss_computer(logits, labels).item()
    if abs(value + np.mean(np.log([0.9, 0.8]))) > 1e-3:
        raise CheckerError("Неверное значение функции потерь")
