import torch
from torch import nn
from torch.nn import functional as F


class LinearSeq(nn.Module):
    def __init__(self, in_features, out_features_list, bias_list=True, batch_normalization_list=True,
                 activation_list='relu'):
        super(LinearSeq, self).__init__()
        if type(bias_list) == bool:
            bias_list = [bias_list] * len(out_features_list)
        if type(batch_normalization_list) == bool:
            batch_normalization_list = [batch_normalization_list] * len(out_features_list)
        if type(activation_list) == str:
            activation_list = [activation_list] * len(out_features_list)
        self.layers = nn.ModuleList([
            LinearNorm(i, o, b, bn, a)
            for (i, o, b, bn, a) in
            zip([in_features] + out_features_list[:-1], out_features_list, bias_list, batch_normalization_list,
                activation_list)
        ])

    def __len__(self):
        return len(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def transpose(x, dim0, dim1):
    '''
    Adaptive transpose function.
    :param x: Tensor
    :param dim0: int
    :param dim1: int
    :return: Tensor
    '''
    if max(dim0, dim1) < x.dim():
        return x.transpose(dim0, dim1)
    else:
        return x

class TransposeNorm(nn.Module):
    def __init__(self, dim0, dim1):
        super(TransposeNorm, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        if max(self.dim0, self.dim1) < x.dim():
            return x.transpose(self.dim0, self.dim1)
        else:
            return x


class BatchNorm1d(nn.Module):
    def __init__(self, out_features):
        super(BatchNorm1d, self).__init__()
        self.layer = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = transpose(x, 1, 2)
        if x.size(0) == 1:  # Sample size == 1
            if x.dim() == 2:
                x = x.unsqueeze(2)
            x = F.instance_norm(x)
            x = x.squeeze(2)
            x = transpose(x, 1, 2)
            return x
        else:
            x = self.layer(x)
            return transpose(x, 1, 2)


class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_normalization=True, activation=None):
        super(LinearNorm, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Linear(in_features, out_features, bias))  # Returns Size([batch_size(, time length), out_features])
        if batch_normalization:
            #self.layers.append(TransposeNorm(1, 2))  # Returns Size([batch_size, out_features(, time length)])
            self.layers.append(BatchNorm1d(out_features))  # Returns Size([batch_size, out_features(, time length)])
            #self.layers.append(TransposeNorm(1, 2))  # Returns Size([batch_size(, time length), out_features])
        if activation == 'relu':
            self.layers.append(nn.ReLU())
        elif activation == 'tanh':
            self.layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Conv2DSeq(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 stride_list=1,
                 dropout_list=.5,
                 batch_normalization_list=True,
                 activation_list='relu'):
        super(Conv2DSeq, self).__init__()
        if type(kernel_size_list) == int:
            kernel_size_list = [kernel_size_list] * len(out_channels_list)
        stride_list = stride_list
        if type(stride_list) == int:
            stride_list = [stride_list] * len(out_channels_list)
        elif type(stride_list) in (list, tuple):
            for i, stride in enumerate(stride_list):
                if type(stride) == int:
                    pass
                elif type(stride) == list:
                    stride_list[i] = tuple(stride)

                if type(stride) in (tuple, list):
                    assert len(stride) <= 2
                else:
                    raise ValueError('Argument not understood')
        if type(dropout_list) == float:
            dropout_list = [dropout_list] * len(out_channels_list)
        if type(batch_normalization_list) == bool:
            batch_normalization_list = [batch_normalization_list] * len(out_channels_list)
        if type(activation_list) == str:
            activation_list = [activation_list] * len(out_channels_list)

        self.layers = nn.ModuleList([
            Conv2DNorm(i, o, k, s, padding='SAME', dropout=d, batch_normalization=bn, activation=a) \
            for (i, o, k, s, d, bn, a) in zip(
                [in_channels] + out_channels_list[:-1],
                out_channels_list,
                kernel_size_list,
                stride_list,
                dropout_list,
                batch_normalization_list,
                activation_list)
        ])

    def __len__(self):
        return len(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Conv2DNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='SAME', dropout=.5,
                 batch_normalization=True, activation=None):
        super(Conv2DNorm, self).__init__()
        if type(kernel_size) == int:
            kernel_size = [kernel_size, 1]
        if padding == 'SAME':
            padding = (kernel_size[0] - 1) // 2, 0
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if dropout:
            self.layers.append(nn.Dropout2d(p=dropout))
        if batch_normalization:
            self.layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
