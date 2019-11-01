import numpy
import torch

import tacotron.modules, tacotron.models
import common.layers
import hparams


def train(model, dataloaders_dict, criterion, optimizer, num_epochs=100):
    model.train(dataloaders_dict, criterion, optimizer, num_epochs=100)

def main(args):
    pass