import torch
from torch import nn
from torch import optim
from tqdm import tqdm

import tacotron.modules, tacotron.loss_function

class Taco2ProsodyTransfer(nn.Module):
    def __init__(self, *args):
        super(Taco2ProsodyTransfer, self).__init__()
        self.encoder = tacotron.modules.Encoder(*args)
        self.attention = tacotron.modules.LocationSensitiveAttention

class Tacotron2(nn.Module):
    def __init__(self, *args):
        super(Tacotron2, self).__init__()
        self.encoder = tacotron.modules.Encoder(*args)
        self.attention = tacotron.modules.LocationSensitiveAttention(*args)
        self.decoder = tacotron.modules.Decoder(*args)

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
                for input_character_indices, spectrogram_labels in tqdm(
                        dataloaders_dict[phase]):  # dataloader는 자체로 iterable
                    # 8. dataset to device
                    input_character_indices = input_character_indices.to(device)
                    spectrogram_labels = spectrogram_labels.to(device)

                    # 9. initialize grad
                    optimizer.zero_grad()

                    # 10. forward
                    with torch.set_grad_enabled(
                            mode=(phase == 'train')):  # enable grad only when training # with + context_manager
                        # Encoder
                        encoder_output, (encoder_h_n, encoder_c_n) = self.encoder.forward(input_character_indices)
                        # Attention&Decoder
                        self.attention.h = encoder_output  # attention.h.Size([input length, batch, encoder output units])
                        self.decoder.reset(batch_size)
                        h_prev_1, stop_token_cum = self.decoder.h_prev_1, self.decoder.stop_token_cum  # Local variable to speed up
                        for decoder_step in range(self.decoder.max_output_time_length):
                            print('\n---------------------', 'decoder step: ', decoder_step + 1)
                            context_vector = self.attention.forward(h_prev_1, stop_token_cum)
                            h_prev_1, stop_token_cum = self.decoder.forward(context_vector)
                            if not any(stop_token_cum):  # stop decoding if no further prediction is needed for any samples in batch
                                break

                        # Calc loss
                        loss = criterion(self.decoder.spectrogram_pred, spectrogram_labels)

                        # 11. (training)calc grad
                        if phase == 'train':
                            loss.backward()
                            # 12. (training)update parameters
                            optimizer.step()

                        # 13. add loss and correct per minibatch per phase
                        epoch_loss += loss.item() * input_character_indices.size(0)

            # 14. print epoch summary
            epoch_loss /= len(dataloaders_dict[phase].dataset)  ## len(dataloader): num of datum

            print('Epoch loss: {:.4f}'.format(epoch_loss))


def checkup():
    taco = Tacotron2()
    criterion = tacotron.loss_function.Taco2Loss()
    optimizer = torch.optim.Adam(Tacotron2.parameters())
    taco.train(dataloaders_dict=,
               criterion=criterion,
               optimizer=optimizer,
               num_epochs=3)

checkup()