import torch
from torch import nn
from torch import optim
import tqdm.tqdm

import tacotron.modules

class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()
        self.encoder = tacotron.modules.Encoder()
        self.attention = tacotron.modules.LocationSensitiveAttention()
        self.decoder = tacotron.modules.Decoder()
        self.net = nn.Sequential([
            self.encoder,
            self.attention,
            self.decoder
        ])

    def train(self, dataloaders_dict, criterion, optimizer, num_epochs=100):
        # 1. set device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device:', device)
        # 2. network to device
        self.net.to(device)
        # 3. loop over epoch
        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()

                # 5. initialize loss per phase
                epoch_loss = .0
                epoch_correct = 0

                # 7. iterate dataloader
                for inputs, labels in tqdm(dataloaders_dict[phase]):  # dataloader는 자체로 iterable
                    # 8. dataset to device
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 9. initialize grad
                    optimizer.zero_grad()

                    # 10. forward
                    with torch.set_grad_enabled(mode=(phase == 'train')):  # enable grad only when training # with + context_manager
                        outputs = self.net(inputs)
                        loss = criterion(outputs, labels)

                        # 11. (training)calc grad
                        if phase == 'train':
                            loss.backward()
                            # 12. update parameters
                            optimizer.step()

                        # 13. add loss and correct per minibatch per phase
                        epoch_loss += loss.item() * inputs.size(0)

            # 14. print epoch summary
            epoch_loss /= len(dataloaders_dict[phase].dataset)  ## len(dataloader): num of datum

            print('Epoch loss: {:.4f}'.format(epoch_loss))

