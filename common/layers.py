import torch
from torch import nn

class LinearSeq(nn.Module):
    def __init__(self, in_features, out_features_list, bias_list=True, batch_normalization_list=True, activation_list='relu'):
        super(LinearSeq, self).__init__()
        if type(bias_list) == bool:
            bias_list = [bias_list] * len(out_features_list)
        if type(batch_normalization_list) == bool:
            batch_normalization_list = [batch_normalization_list] * len(out_features_list)
        if type(activation_list) == str:
            activation_list = [activation_list] * len(out_features_list)
        self.layers = nn.Sequential([
            LinearNorm(i, o, b, bn, a)
            for (i, o, b, bn, a) in zip([in_features]+out_features_list[:-1], out_features_list, bias_list, batch_normalization_list, activation_list)
        ])

    def __len__(self):
        return len(self.layers)

    def forward(self, x):
        return self.layers(x)

class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_normalization=True, activation=None):
        super(LinearNorm, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, out_features, bias))
        if batch_normalization:
            self.layers.append(nn.BatchNorm1d(out_features))
        if activation == 'relu':
            self.layers.append(nn.ReLU())
        elif activation == 'tanh':
            self.layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)

class Conv1DSeq(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 stride_list=1,
                 batch_normalization_list=True,
                 activation_list='relu'):
        super(Conv1DSeq, self).__init__()
        for arg in [kernel_size_list, stride_list]:
            if type(arg) == int:
                arg = [arg for i in range(len(out_channels_list))]
        if type(batch_normalization_list) == bool:
            batch_normalization_list = [batch_normalization_list] * len(out_channels_list)
        if type(activation_list) == str:
            activation_list = [activation_list] * len(out_channels_list)

        self.layers = nn.Sequential([
            Conv1DNorm(i, o, k, s, padding='SAME', batch_normalization=bn, activation=a)
            for (i, o, k, s, bn, a)
            in zip([in_channels]+out_channels_list[:-1], out_channels_list, kernel_size_list, stride_list, batch_normalization_list, activation_list)
        ])

    def __len__(self):
        return len(self.layers)

    def forward(self, x):
        return self.layers(x)

class Conv1DNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='SAME', batch_normalization=True, activation=None):
        super(Conv1DNorm, self).__init__()
        if padding == 'SAME':
            padding # calculate
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
        if batch_normalization:
            self.layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            self.layers.append(nn.ReLU())

    def forward(self, x):
        return self.layers(x)
