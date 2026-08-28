import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """ArcFace margin softmax (Deng et al., 2019), khớp project_type=arc_margin
    trong config.yaml gốc của wespeaker (scale=32.0, easy_margin=false)."""

    def __init__(self, in_features, out_features, scale=32.0, margin=0.2, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.scale


class CosMarginProduct(nn.Module):
    """CosFace / AM-Softmax margin (Wang et al., 2018) -- margin_type='C' trong config
    goc cua wespeaker (vd redimnet2.yaml), khac ArcMarginProduct o cho tru margin truoc
    khi lay cos (additive trong khong gian cosine) thay vi cong margin goc (angular)."""

    def __init__(self, in_features, out_features, scale=32.0, margin=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        phi = cosine - self.margin

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.scale
