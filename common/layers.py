import torch
from torch import nn

class Conv1DSeq(nn.Module):
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 kernel_size_list,
                 stride_list,
                 batch_normalization_list,
                 activation_list):
        super(Conv1DSeq, self).__init__()
        assert(len(in_channels_list) == len(out_channels_list) == len(kernel_size_list) == len(stride_list) == len(batch_normalization_list) == len(activation_list))
        assert(in_channels_list[1:] == out_channels_list[:-1])
        self.layers = nn.ModuleList([
            Conv1DNorm(i, o, k, s, padding='SAME')
            for i, o, k, s, bn, a
            in zip(in_channels_list, out_channels_list, kernel_size_list, stride_list, batch_normalization_list, activation_list)
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
