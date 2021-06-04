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
    for i, (images, labels) in enumerate(train_loader):
        if i == n_iter:
            break
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        predicted = model(images)
        loss_value = loss(predicted, labels)
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


def eval_model(model, loss, testset, batch_size, use_cuda=False):
    model.eval()
    kwargs = {}
    if use_cuda:
        kwargs["pin_memory"] = True
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              num_workers=2,
                                              **kwargs)
    losses = []
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            predicted = model(images)
            # Посчитать функцию потерь для батча.
            loss_value = loss(predicted, labels)
            losses.append(loss_value.item())
            # Посчитать долю правильных ответов на батче.
            predicted_labels = predicted.detach().max(-1)[1]
            is_correct = predicted_labels == labels
            n_correct += is_correct.sum().item()
            n_samples += is_correct.numel()
    test_loss = np.mean(losses)
    print("Test loss:", test_loss)
    print("Test accuracy:", n_correct / n_samples)
    return test_loss


def train_model(model, loss_fn, optimizer_fn, trainset, testset,
                lr_scheduler_fn=None,
                batch_size=16,
                n_iters=2000, eval_steps=250,
                plot=True,
                train_plot_kwargs=TRAIN_PLOT_KWARGS, test_scatter_kwargs=TEST_SCATTER_KWARGS,
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
    iplot = cvmade.plot.InteractivePlot(0, 0, n_iters, 3) if plot else None
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


def show_augmenter_results(augmenter, data_root):
    trainset_notransform = CIFAR10(data_root, train=True, download=True)
    pil_image = random.choice(trainset_notransform)[0]
    transform = torchvision.transforms.Compose([
        augmenter,
        torchvision.transforms.ToTensor()
    ])
    cvmade.plot.torch.show_images_dataset([pil_image], collate_fn=transform)


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
