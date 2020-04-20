import random
import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10

import cvmade


TRAIN_PLOT_KWARGS = {"c": "b"}
TEST_SCATTER_KWARGS = {"c": "y", "s": 100, "zorder": 1e10}


def train_loop(model, optimizer, loss, train_loader, n_iter, lr_scheduler=None, plot=None, plot_kwargs={}, use_cuda=False, plot_steps=10):
    model.train()
    losses = []
    for i, (images, masks) in enumerate(train_loader):
        if i == n_iter:
            break
        if use_cuda:
            images = images.cuda()
            masks = masks.cuda()
        optimizer.zero_grad()
        predicted = model(images)
        loss_value = loss(predicted, masks)
        loss_value.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        losses.append(loss_value.item())
        if (plot is not None) and (i > 0) and (i % plot_steps == 0):
            # Обновим график.
            prev_loss_value = np.mean(losses[-2 * plot_steps: -plot_steps])
            cur_loss_value = np.mean(losses[-plot_steps:])
            if prev_loss_value is not None:
                plot.plot([train_loader.step - plot_steps, train_loader.step], [prev_loss_value, cur_loss_value],
                          **plot_kwargs)
        else:
            if i % 10 == 0:
                    print("Step {} / {}, loss: {:.4f}, learning rate: {:.4f}\r".format(i, n_iter, loss_value.item(), optimizer.param_groups[0]["lr"]), end="")
    print(" " * 50 + "\r", end="")
    print("Train loss: {:.4f}, learning rate: {:.4f}".format(np.mean(losses[-plot_steps:]), optimizer.param_groups[0]["lr"]))


def eval_model(model, loss, testset, batch_size,
               use_cuda=False,
               num_workers=4):
    model.eval()
    kwargs = {}
    if use_cuda:
        kwargs["pin_memory"] = True
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              num_workers=num_workers,
                                              **kwargs)
    losses = []
    with torch.no_grad():
        for images, masks in test_loader:
            if use_cuda:
                images = images.cuda()
                masks = masks.cuda()
            predicted = model(images)
            # Посчитать функцию потерь для батча.
            loss_value = loss(predicted, masks)
            losses.append(loss_value.item())
    test_loss = np.mean(losses)
    print("Test loss:", test_loss)
    return test_loss


def train_model(model, loss_fn, optimizer_fn, trainset, testset,
                lr_scheduler_fn=None,
                batch_size=4,
                num_workers=4,
                n_iters=2000, eval_steps=250,
                plot=True,
                train_plot_kwargs=TRAIN_PLOT_KWARGS, test_scatter_kwargs=TEST_SCATTER_KWARGS,
                use_cuda=False,
                init=True):
    if init:
        model.initialize_weights()
    loss = loss_fn()
    if use_cuda:
        model.cuda()
        loss.cuda()
    optimizer = optimizer_fn(model)
    lr_scheduler = lr_scheduler_fn(optimizer) if lr_scheduler_fn is not None else None
    # Включаем shuffle для тренировочных данных, чтобы
    # на каждом проходе корпуса получались разные батчи.
    kwargs = {}
    if use_cuda:
        kwargs["pin_memory"] = True
    train_loader = cvmade.torch.CyclicDataLoader(trainset, batch_size=batch_size,
                                                 num_workers=num_workers, shuffle=True,
                                                 **kwargs)
    # Создаем интерактивный график. Считаем, что loss всегда меньше 3-х.
    iplot = cvmade.plot.InteractivePlot(0, -1, n_iters, 5) if plot else None
    eval_loss = eval_model(model, loss, testset, batch_size, use_cuda=use_cuda)
    if plot:
        iplot.scatter([0], [eval_loss], **test_scatter_kwargs)
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