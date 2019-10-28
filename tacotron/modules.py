import torch
import torch.nn as nn
import torch.functional

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
    def __init__(self, in_channels, out_channels_list, kernel_size_list, stride_list, batch_normalization_list,
                 activation_list):
        super(ConvLayers, self).__init__()
        self.layers = Conv1DSeq(in_channels,
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
        self.w = LinearNorm(dim_w, 1, bias=False)

    def calc_f(self, F_matrix, a_prev):
        """
        :param F_matrix: 2-d Tensor
            Size([input_time_length, dim_f])
        :param a_prev: Vector. Previous time alignment.
            Size([input_time_length])
        :return f_current: 3-d Tensor.
            Size([batch_size, input_time_length, dim_f])
        """
        a_prev = a_prev.view(1, 1, -1) # (num_channels, divided by groups, kernel_size)
        padding = # calc same padding
        f_current = torch.conv1d(input=F_matrix, weight=a_prev, stride=1, padding=padding)
        return f_current

    def score(self, s_prev, h, f_current):
        """
        :param s_prev: 2-d Tensor.
            Size([batch_size, dim_s])
        :param h: 3-d Tensor. Encoder output.
            Size([batch_size, input_time_length, dim_h])
        :param f_current: 3-d Tensor. Output from calc_f.
            Size([batch_size, input_time_length, dim_f])
        :return e_current: 2-d Tensor.
            Size([batch_size, input_time_length])
        """

        e_current = self.w(torch.tanh(self.W(s_prev)+self.V(h)+self.U(f_current))) #Broadcast s_prev
        e_current = torch.squeeze(e_current)
        assert(e_current.dim() == 2)
        return e_current

    def normalize(self, e_current, mode='sharpening', beta=1.0):
        """
        Normalize e_current. 'Sharpening' with beta=1.0 equals softmax normalization.
        :param e_current: 2-d Tensor.
            Size([batch_size, input_time_length])
        :param mode: str.
        :param beta: float.
        :return a_current: 2-d Tensor.
            Size([batch_size, input_time_length])
        """
        if mode == 'sharpening':
            a_current = torch.softmax(beta * e_current, dim=1)
            return a_current

        elif mode == 'smoothing':
            a_current = torch.sigmoid(e_current)
            a_current /= a_current.sum()
            return a_current

    def forward(self, F_matrix, a_prev, s_prev, h, mode, beta):
        f_current = self.calc_f(F_matrix, a_prev)
        e_current = self.score(s_prev, h, f_current)
        a_current = self.normalize(e_current, mode, beta)
        return a_current

class LocationSensitiveAttention(nn.Module):
    def __init__(self, dim_s, dim_h, dim_f, dim_w, F_matrix, max_input_time_length, max_output_time_length, mode='sharpening', beta=1.0):
        super(LocationSensitiveAttention, self).__init__()
        self.ARSG = ARSG(dim_s, dim_h, dim_f, dim_w)
        self.alignments = torch.zeros([max_input_time_length, max_output_time_length], requires_grad=False) # Preserves alignments history
        self.decoder_time_step = 0
        self.F_matrix = F_matrix
        self.mode = mode
        self.beta = beta

        self.h = None

    def forward(self, s_prev):
        a_current = self.ARSG.forward(self.F_matrix, self.a_prev, s_prev, self.h, self.mode, self.beta)
        context_vector = torch.bmm(a_current.unsqueeze(2).transpose(1, 2), self.h)
        self.alignments[self.decoder_time_step] = a_current # Store alignment
        self.a_prev = a_current
        self.decoder_time_step += 1
        return context_vector

class Encoder(nn.Module):
    def __init__(self,
                 max_input_text_length,
                 encoder_embedding_dim,
                 pretrained_embedding,
                 initial_weights):
        super(Encoder, self).__init__()
        self.character_embedding = CharacterEmbedding(max_input_text_length,
                                                      encoder_embedding_dim,
                                                      pretrained_embedding,
                                                      initial_weights)
        self.conv_layers = ConvLayers(in_channels,
                                      out_channels_list,
                                      kernel_size_list,
                                      stride_list,
                                      batch_normalization_list,
                                      activation_list)
        self.rnn = nn.RNNBase(mode='LSTM',
                              input_size=out_channels_list * embedding_dim,
                              hidden_size=num_rnn_units,
                              num_layers=num_rnn_layers,
                              bias=True,
                              bidirectional=True) # returns output.size([max_input_text_length, batch_size, 2(bidirection)*hidden_size): stack of output states, h_n.size([2(bidirection)*num_layers, batch_size, hidden_size]): last hidden state
        self.layers = nn.Sequential([self.character_embedding,
                                     self.conv_layers,
                                     self.rnn
                                     ])

class Decoder(nn.Module):
    def __init__(self,
                 encoder_rnn_units=512,
                 decoder_rnn_units=1024,
                 decoder_rnn_layers=2,
                 decoder_prenet_in_features=80,
                 decoder_prenet_out_features_list=[256, 256],
                 decoder_prenet_bias_list=True,
                 decoder_prenet_batch_normalization_list=False,
                 decoder_prenet_activation_list='relu',
                 decoder_postnet_in_channels=1,
                 decoder_postnet_out_channels_list=[512, 512, 512, 512, 512],
                 decoder_postnet_kernel_size_list=5,
                 decoder_postnet_stride_list=1,
                 decoder_postnet_batch_normalization_list=True,
                 decoder_postnet_activation_list=['tanh', 'tanh', 'tanh', 'tanh', None],
                 num_mel_channels=80
                 ):
        super(Decoder, self).__init__()
        assert(decoder_prenet_in_features == num_mel_channels)

        lstm_cell = nn.LSTMCell(
            input_size=encoder_rnn_units,
            hidden_size=decoder_rnn_units,
            bias=True
        )
        self.lstm_cell_dict = {}
        for i in range(decoder_rnn_layers):
            self.lstm_cell_dict['lstm_cell_' + str(i)] = lstm_cell

        self.linear_projection_mel = LinearNorm(
            in_features=(encoder_rnn_units+decoder_rnn_units), #Concatenate attention context vector and LSTM output
            out_features=num_mel_channels,
            bias=True,
            batch_normalization= False,
        )

        self.linear_projection_stop = LinearNorm(
            in_features=(encoder_rnn_units+decoder_rnn_units),
            out_features=1,
            bias=True,
            batch_normalization=False,
            activation='sigmoid'
        )

        self.postnet = Conv1DSeq(
            in_channels=decoder_postnet_in_channels,
            out_channels_list=decoder_postnet_out_channels_list,
            kernel_size_list=decoder_postnet_kernel_size_list,
            stride_list=decoder_postnet_stride_list,
            batch_normalization_list=decoder_postnet_batch_normalization_list,
            activation_list=decoder_postnet_activation_list
        )

        self.prenet = LinearSeq(
            in_features=decoder_prenet_in_features,
            out_features_list=decoder_prenet_out_features_list,
            bias_list=decoder_prenet_bias_list,
            batch_normalization_list=decoder_prenet_batch_normalization_list,
            activation_list=decoder_prenet_activation_list
        )







parser = hparams.parser
