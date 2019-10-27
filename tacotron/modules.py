import torch
import torch.nn as nn

from common.layers import *
import hparams

class CharacterEmbedding(nn.Module):
    def __init__(self, max_input_text_length, embedding_dim, pretrained_embedding=None, initial_weights=None):
        '''
        :param max_input_text_length: int
        :param embedding_dim: int
        :param pretrained_embedding: 2-d LongTensor
        :param initial_weights: 2-d LongTensor
        '''

        super(CharacterEmbedding, self).__init__()
        if pretrained_embedding:
            self.character_embedding.from_pretrained(embeddings=pretrained_embedding, freeze=True)
        elif initial_weights:
            self.character_embedding = nn.Embedding(num_embeddings=max_input_text_length,
                                                    embedding_dim=embedding_dim,
                                                    _weight=initial_weights)
        else:
            self.character_embedding = nn.Embedding(num_embeddings=max_input_text_length,
                                                    embedding_dim=embedding_dim)

    def forward(self, input_text):
        '''
        :param input_text: 2-d LongTensor of arbitrary shape containing the indices to extract
            Size([batch_size, max_input_text_length])
        :return: 3-d LongTensor
            Size([batch_size, max_input_text_length, embedding_dim])
        '''
        return self.character_embedding(input_text)

class ConvLayers(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, kernel_size_list, stride_list, batch_normalization_list, activation_list):
        super(ConvLayers, self).__init__()
        self.layers = Conv1DSeq(in_channels_list,
                                out_channels_list,
                                kernel_size_list,
                                stride_list,
                                batch_normalization_list,
                                activation_list)

    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(CharacterEmbedding(max_input_text_length, embedding_dim, pretrained_embedding, initial_weights))
        self.layers.append(ConvLayers(in_channels_list, out_channels_list, kernel_size_list, stride_list, batch_normalization_list, activation_list))
        self.layers.append(nn.RNN(out_channels_list*embedding_dim, hidden_size=num_lstm, num_layers=num_layers, bias=True, bidirectional=True))

parser = hparams.parser