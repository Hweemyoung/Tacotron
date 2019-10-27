import torch
import torch.nn as nn
import torch.functional as F

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
    def __init__(self, in_channels_list, out_channels_list, kernel_size_list, stride_list, batch_normalization_list,
                 activation_list):
        super(ConvLayers, self).__init__()
        self.layers = Conv1DSeq(in_channels_list,
                                out_channels_list,
                                kernel_size_list,
                                stride_list,
                                batch_normalization_list,
                                activation_list)

    def forward(self, x):
        return self.layers(x)


class ARSG(nn.Module):
    """
    Attention-based Recurrent Sequence Generator(ARSG)
    """

    def __init__(self, dim_s, dim_h, dim_f, dim_w):
        super(ARSG, self).__init__()
        self.W = LinearNorm(dim_s, dim_w, bias=False)
        self.V = LinearNorm(dim_h, dim_w, bias=True)
        self.U = LinearNorm(dim_f, dim_w, bias=False)

    def calc_f(self, F, a_prev):
        """
        :param F: 2-d Tensor
            Size([input_time_length, dim_f])
        :param a_prev: Vector. Previous time alignment.
            Size([input_time_length, 1])
        :return f_current: 2-d Tensor.
            Size([input_time_length, dim_f])
        """
        padding = # calc same padding
        f_current = torch.conv1d(input=F, weight= a_prev, stride=1, padding=padding)
        return f_current

    def score(self, w, s_prev, h, f_current):
        """
        :param w: Vector
            Size([dim_w, 1])
        :param s_prev: Vector
            Size([dim_s, 1])
        :param h: 2-d Tensor. Encoder output.
            Size([input_time_length, dim_h])
        :param f_current: 2-d Tensor. calc_f output.
            Size([input_time_length, dim_f])
        :return e_current: Vector
            Size([input_time_length, 1])
        """
        e_current = torch.matmul(w.transpose, torch.tanh(self.W(s_prev)+self.V(h)+self.U(f_current))) #Broadcast s_prev
        return e_current

    def normalize(self, e_current, mode='sharpening', beta=1.0):
        """
        Normalize e_current. 'Sharpening' with beta=1.0 equals softmax normalization.
        :param e_current:
        :param mode:
        :param beta:
        :return:
        """
        if mode == 'sharpening':
            a_current = torch.softmax(beta * e_current, dim=0)
            return a_current

        elif mode == 'smoothing':
            a_current = torch.sigmoid(e_current)
            a_current /= a_current.sum()
            return a_current

    def forward(self, F, a_prev, w, s_prev, h, f_current, mode, beta):
        f_current = self.calc_f(F, a_prev)
        e_current = self.score(w, s_prev, h, f_current)
        a_current = self.normalize(e_current, mode, beta)
        return a_current

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            CharacterEmbedding(max_input_text_length, embedding_dim, pretrained_embedding, initial_weights))
        self.layers.append(
            ConvLayers(in_channels_list, out_channels_list, kernel_size_list, stride_list, batch_normalization_list,
                       activation_list))
        self.layers.append(
            nn.RNN(out_channels_list * embedding_dim, hidden_size=num_lstm, num_layers=num_layers, bias=True,
                   bidirectional=True))  # returns output.size([max_input_text_length, batch_size, 2(bidirection)*hidden_size): stack of output states, h_n.size([2(bidirection)*num_layers, batch_size, hidden_size]): last hidden state


parser = hparams.parser
