import random
import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10
from matplotlib import pyplot as plt


import pytorch_lightning as pl


def show_images_dataset(dataset, n=5, collate_fn=lambda x: x[0]):
    """Plot images from dataset."""
    images = [collate_fn(random.choice(dataset)) for _ in range(n)]
    grid = torchvision.utils.make_grid(images)
    grid -= grid.min()
    grid /= grid.max()
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


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
        self.logger.experiment.add_scalars("accuracy", 
                                           {"train": self._accuarcy(logits, labels)}, 
                                           global_step=self.global_step)
        self.log("loss", loss)
        self.log("train_accuracy")
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self._model(x)
        loss = self._criterion(logits, y)
        self.log("valid_loss", loss)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self._trainset, batch_size=self._batch_size, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self._testset, batch_size=self._batch_size, shuffle=False)

    def configure_optimizers(self):
        optimizer = self._optimizer_fn(self._model)
        scheduler = self._lr_scheduler_fn(optimizer)
        return [optimizer], [scheduler]
    
    def _accuarcy(self, logits, labels):
        predictions = logits.argmax(-1)
        return (predictions == labels).mean()
    
    
def train_model(model, loss_fn, optimizer_fn, trainset, testset,
                lr_scheduler_fn=None,
                batch_size=16,
                n_iters=2000, eval_steps=250,
                use_cuda=False):
    model = Module(model, loss_fn, optimizer_fn, trainset, testset,
                   lr_scheduler_fn, batch_size)
    trainer = pl.Trainer(gpus=int(use_cuda), max_epochs=2, val_check_interval=250)
    trainer.fit(model)


def train_model(model, loss_fn, optimizer_fn, trainset, testset,
                lr_scheduler_fn=None,
                batch_size=16,
                n_iters=2000, eval_steps=250,
                use_cuda=False):
    model.initialize_weights()
    if use_cuda:
        model.cuda()
    loss = loss_fn()
    optimizer = optimizer_fn(model)
    lr_scheduler = lr_scheduler_fn(optimizer) if lr_scheduler_fn is not None else None
    # Включаем shuffle для тренировочных данных, чтобы
    # на каждом проходе корпуса получались разные батчи.
    kwargs = {}
    if use_cuda:
        kwargs["pin_memory"] = True
    train_loader = cvmade.torch.CyclicDataLoader(trainset, batch_size=batch_size,
                                                 num_workers=2, shuffle=True,
                                                 **kwargs)
    # Создаем интерактивный график. Считаем, что loss всегда меньше 3-х.
    eval_loss = eval_model(model, loss, testset, batch_size, use_cuda=use_cuda)
    i = 0
    while i < n_iters:
        print("Step", i)
        train_steps = min(eval_steps, n_iters - i)
        train_loop(model, optimizer, loss, train_loader, train_steps,
                   lr_scheduler=lr_scheduler,
                   plot=iplot, plot_kwargs=train_plot_kwargs,
                   use_cuda=use_cuda)
        i += train_steps
        eval_loss = eval_model(model, loss, testset, batch_size,
                               use_cuda=use_cuda)
        if plot:
            iplot.scatter([i], [eval_loss], **test_scatter_kwargs)


def show_augmenter_results(augmenter, data_root):
    trainset_notransform = CIFAR10(data_root, train=True, download=True)
    pil_image = random.choice(trainset_notransform)[0]
    transform = torchvision.transforms.Compose([
        augmenter,
        torchvision.transforms.ToTensor()
    ])
    show_images_dataset([pil_image], collate_fn=transform)


class CheckerError(Exception):
    pass


def check_ok():
    print("========")
    print("CHECK OK")


def check_vgg(model_class):
    model = model_class()
    layer = model._make_conv3x3(10, 20)
    if not isinstance(layer, torch.nn.Conv2d):
        raise CheckerError("_make_conv3x3: неверный тип слоя")
    if layer.kernel_size not in {3, (3, 3)}:
        raise CheckerError("_make_conv3x3: неверный размер фильта")
    if layer.padding != (1, 1):
        raise CheckerError("_make_conv3x3: неверный паддинг")

    layer = model._make_relu()
    if not isinstance(layer, torch.nn.ReLU):
        raise CheckerError("_make_relu: неверный тип слоя")

    layer = model._make_maxpool2x2()
    if not isinstance(layer, torch.nn.MaxPool2d):
        raise CheckerError("_make_maxpool2x2: неверный тип слоя")
    if layer.kernel_size not in {2, (2, 2)}:
        raise CheckerError("_make_maxpool3x3: неверный размер фильтра")

    layer = model._make_fully_connected(30, 50)
    if not isinstance(layer, torch.nn.Linear):
        raise CheckerError("_make_fully_connected: неверный тип слоя")
    check_ok()


def check_loss_fn(make_loss):
    loss = make_loss()
    if not isinstance(loss, torch.nn.CrossEntropyLoss):
        raise CheckerError("Неверный тип функции потерь")
    check_ok()


def check_optimizer_fn(make_optimizer):
    model = torch.nn.Linear(10, 20)
    optimizer = make_optimizer(model)
    if not isinstance(optimizer, torch.optim.SGD):
        raise CheckerError("Неверный тип оптимизатора")
    if abs(optimizer.defaults["lr"] - 0.01) > 1e-6:
        raise CheckerError("Неверный шаг обучения")
    if abs(optimizer.defaults["momentum"] - 0.9) > 1e-6:
        raise CheckerError("Неверное значение момента")
    check_ok()


def check_vgg_bn(model_class):
    model = model_class()
    layer = model._make_conv3x3(10, 20)
    if not isinstance(layer, torch.nn.Sequential):
        raise CheckerError("_make_conv3x3: ожидается Sequential")
    if len(layer) != 2:
        raise CheckerError("_make_conv3x3: ожидается Sequential из двух слоев")
    if not isinstance(layer[0], torch.nn.Conv2d):
        raise CheckerError("_make_conv3x3: неверный тип слоя")
    if layer[0].bias not in {None, False}:
        raise CheckerError("_make_conv3x3: свертка не должна иметь смещения")
    if layer[0].kernel_size not in {3, (3, 3)}:
        raise CheckerError("_make_conv3x3: неверный размер фильта")
    if layer[0].padding != (1, 1):
        raise CheckerError("_make_conv3x3: неверный паддинг")
    if not isinstance(layer[1], torch.nn.BatchNorm2d):
        raise CheckerError("_make_conv3x3: неверный тип BatchNorm слоя")
    if layer[1].num_features != 20:
        raise CheckerError("_make_conv3x3: неверное количество признаков в BatchNorm")
    check_ok()


def check_vgg_do(model_class):
    model = model_class()
    for last in [False, True]:
        layer = model._make_fully_connected(10, 20, last=last)
        if not isinstance(layer, torch.nn.Sequential):
            raise CheckerError("_make_fully_connected: ожидается Sequential")
        if not last:
            if len(layer) != 2:
                raise CheckerError("_make_fully_connected: ожидается Sequential из двух слоев")
            if not isinstance(layer[0], torch.nn.Linear):
                raise CheckerError("_make_fully_connected: неверный тип слоя или порядок")
            if not isinstance(layer[1], torch.nn.Dropout):
                raise CheckerError("_make_fully_connected: неверный тип слоя или порядок")
        else:
            if len(layer) != 1:
                raise CheckerError("_make_fully_connected: лишний dropout в конце сети")
            if not isinstance(layer[0], torch.nn.Linear):
                raise CheckerError("_make_fully_connected: неверный тип слоя")
    check_ok()
