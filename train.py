import numpy
import torch
import argparse

from tacotron import modules, models, loss_function, data_functions
import common.layers
import hparams


def train(model, dataloaders_dict, criterion, optimizer, num_epochs=100):
    model.train(dataloaders_dict, criterion, optimizer, num_epochs=100)

def main():
    parser = argparse.ArgumentParser(description='Taco2 Training')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    log_hardware()

    # Get model
    model_name = args.model_name
    parser = models.parse_model_args(model_name, parser)
    args = parser.parse_args()
    model_config = models.gel_model_config(model_name, args)
    model = models.get_model(model_name, model_config, to_cuda=True, initial_bn_weight=True)  # nn.Module instance

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Loss function
    criterion = loss_function.Taco2Loss()

    # Dataset, DataLoader
    # collate_fn =
    trainset = data_functions
    train_sampler
    train_loader


