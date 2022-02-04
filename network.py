import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def make_mlp(l, act=nn.LeakyReLU(), tail=[], with_bn=False):
    """makes an MLP with no top layer activation"""
    net = nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + (([nn.BatchNorm1d(o), act] if with_bn else [act]) if n < len(l) - 2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []
    ) + tail))
    return net


def mlp_ebm(nin, nint=256, nout=1):
    return nn.Sequential(
        nn.Linear(nin, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nout),
    )