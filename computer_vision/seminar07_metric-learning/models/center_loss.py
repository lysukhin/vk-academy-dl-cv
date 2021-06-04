import torch
import torch.nn as nn
from torch.autograd.function import Function


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        "source: https://github.com/jxgu1016/MNIST_center_loss_pytorch/blob/master/CenterLoss.py"
        super(CenterLoss, self).__init__()
        # здесь мы храним наши центройды, как параметры которые учим, напрямую как среднее по классам мы их не вычисляем
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))

        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        "Save for backward,  needed for optimization"
        ctx.save_for_backward(feature, label, centers, batch_size)

        # берем нужные центройды согласно лейблам
        centers_batch = centers.index_select(0, label.long())

        # L2 between embeddings and centers
        loss = (feature - centers_batch).pow(2).sum() / 2.0 / batch_size
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # commit with the explanation: https://github.com/jxgu1016/MNIST_center_loss_pytorch/tree/dbeea5380de8a3c6b1b3b3f2c411b980e143dd87
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        # считаем формулу из Eq(4) из стат, нам надо поделить на кол-во центров
        counts = counts.scatter_add_(0, label.long(), ones) # counts += ones where label
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None

