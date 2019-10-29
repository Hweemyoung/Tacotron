import numpy
import torch

import tacotron
import hparams

def train(model,num_epochs):
    # 1. set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 2. model to device
    model.to(device)
    # 3. loop over epoch
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs))

def main(args):
    if args.train == 'train':
        pass