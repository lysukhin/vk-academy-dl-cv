#-*- coding: utf8 -*-
from torch import nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math


class ArcFaceScoring(nn.Module):
    # pytorch 1.2 compatible
    def __init__(self, m, s=64, in_features=0, out_features=0, parameters=True, device=None):
        super(ArcFaceScoring, self).__init__()
        self._m = m
        self._s = s
        self.in_features = in_features
        self.out_features = out_features
        self._device = device or torch.cuda.is_available() and 'cuda' or 'cpu'

        if parameters:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            self.reset_parameters()

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, target=None):
        # нормализация весов и входного тензора
        input = F.normalize(input, p=2, dim=1).mul_(self._s)
        scores = F.linear(input, F.normalize(self.weight, p=2, dim=1))

        if target is None: # если evaluation
            return scores

        # нам нужно занизить скор по специальной формуле для таргет класса, для этого нам нужно заполнить маску 1,
        # где истинная персона
        index = torch.BoolTensor(*scores.size())
        index = index.to(self._device)

        # заполняем индекс/маску
        index.fill_(0)
        index.scatter_(1, target.data.view(-1, 1), 1) # (dim, index, src), performs: src -> self where index

        mask = index.byte().detach() # no gradient back, we use this tensor as output

        target_scores = scores[index] # берем только те скоры, которые соответствуют таргет классу
        target_cos_t = target_scores / self._s # cos (denormalize)

        # sin(t) = sqrt(1-cos^2(t))
        target_sin_t = target_cos_t.mul(target_cos_t).add(-1).mul(-1).sqrt()

        # scores = s * cos(t+m) =s * (cos(t)*cos(m) - sin(t)*sin(m))
        target_scores = target_scores * self.cos_m - target_sin_t * self.sin_m * self._s

        # replace scores
        new_scores = torch.zeros_like(scores)
        new_scores[index] = target_scores
        scores = torch.where(mask, new_scores, scores)

        return scores


class LinearScoring(nn.Linear):
    def __init__(self, in_features, out_features):
        super(LinearScoring, self).__init__(in_features, out_features, bias=True)

    def forward(self, input, target=None):
        "The only difference is normalization of the input and weights"
        input = F.normalize(input, p=2, dim=1)
        scores = F.linear(input, F.normalize(self.weight, p=2, dim=1))
        return scores
