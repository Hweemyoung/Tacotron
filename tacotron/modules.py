import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import *
import hparams

class CharacterEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, pretrained_embedding=None, initial_weights=None):
        '''
        :param num_embeddings: int
        :param embedding_dim: int
        :param pretrained_embedding: 2-d LongTensor
        :param initial_weights: 2-d LongTensor
        '''
        super(CharacterEmbedding, self).__init__()
        if pretrained_embedding:
            self.character_embedding.from_pretrained(embeddings=pretrained_embedding, freeze=True)
        elif initial_weights:
            self.character_embedding = nn.Embedding(num_embeddings=num_embeddings,
                                                    embedding_dim=embedding_dim,
                                                    _weight=initial_weights)
        else:
            self.character_embedding = nn.Embedding(num_embeddings=num_embeddings,
                                                    embedding_dim=embedding_dim)

    def forward(self, input_indices):
        '''
        :param input_indices: 2-d LongTensor of arbitrary shape containing the indices to extract
            Size([batch_size, max_input_length])
        :return: 3-d LongTensor
            Size([batch_size, max_input_length, embedding_dim])
        '''
        return self.character_embedding(input_indices)

class ConvLayers(nn.Module):
    def __init__(self, in_channels, out_channels_list, kernel_size_list, stride_list, dropout_list, batch_normalization_list,
                 activation_list):
        super(ConvLayers, self).__init__()
        self.layers = Conv2DSeq(in_channels,
                                out_channels_list,
                                kernel_size_list,
                                stride_list,
                                dropout_list,
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
        :param a_prev: 2-d Tensor. Previous time alignment.
            Size([batch_size, input_time_length])
        :return f_current: 3-d Tensor.
            Size([batch_size, input_time_length, dim_f])
        """
        F_matrix = F_matrix.unsqueeze(0).unsqueeze(1)  # (minibatch,in_channels,iH,iW)
        # print('F_matrix shape: ', F_matrix.size())
        a_prev = a_prev.unsqueeze(1).unsqueeze(3)  # (out_channels, in_channel/1, kH,kW)
        # print('a_prev shape: ', a_prev.size())
        padding = (a_prev.shape[2]-1)//2 # SAME padding # (k-1)/2
        f_current = torch.conv2d(input=F_matrix, weight=a_prev, stride=1, padding=[padding, 0]) # (minibatch, out_channels, iH, iW)
        # print('f_current shape: ', f_current.size())
        f_current = f_current.squeeze(0)
        if f_current.size()[1] != F_matrix.size()[3]: # if time length shrinked due to padding
            f_current = F.pad(f_current, [0, 0, 1, 0], mode='constant', value=0)
        # print('f_current reshape: ', f_current.size())
        return f_current

    def score(self, s_prev, h, f_current):
        """
        :param s_prev: 2-d Tensor.
            Size([batch_size, decoder_rnn_units])
        :param h: 3-d Tensor. Encoder output.
            Size([batch_size, input_time_length, encoder_output_units])
        :param f_current: 3-d Tensor. Output from calc_f.
            Size([batch_size, input_time_length, dim_f])
        :return e_current: 2-d Tensor.
            Size([batch_size, input_time_length])
        """
        '''print('s_prev shape: ', s_prev.size())
        s_prev = s_prev[:, 1, :] #Use states of final layer only
        print('s_prev reshape: ', s_prev.size())
        Ws = self.W(s_prev).unsqueeze(1)
        print('Ws shape: ', Ws.size())
        print('h shape: ', h.size())
        h = h.transpose(0, 1)
        print('h reshape: ', h.size())
        Vh = self.V(h)
        print('Vh shape: ', Vh.size())
        print('f shape: ', f_current.size())
        Uf = self.U(f_current)
        print('Uf shape: ', Uf.size())
        x = torch.tanh(Ws + Vh + Uf)
        print('x shape: ', x.size())
        e_current = self.w(x)
        print('e shape: ', e_current.size())'''
        h = h.transpose(0, 1)
        e_current = self.w(torch.tanh(self.W(s_prev).unsqueeze(1)+self.V(h)+self.U(f_current))) #Broadcast s_prev
        e_current = e_current.squeeze(2)
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
    def __init__(self, batch_size, decoder_rnn_units, encoder_output_units, dim_f, dim_w, max_input_time_length, max_output_time_length, mode='sharpening', beta=1.0):
        super(LocationSensitiveAttention, self).__init__()
        self.ARSG = ARSG(decoder_rnn_units, encoder_output_units, dim_f, dim_w)
        self.alignments = torch.zeros([batch_size, max_input_time_length, max_output_time_length], requires_grad=False) # Preserves alignments history
        self.decoder_time_step = 0
        self.F_matrix = nn.Parameter(torch.rand([max_input_time_length, dim_f], requires_grad=True)) # Size([input_time_length, dim_f])
        self.mode = mode
        self.beta = beta

        self.h = None # Initial shape: (input length, batch, encoder_final_layer_output_units)

        #Initial alignment
        self.a_prev = torch.zeros([batch_size, max_input_time_length])
        self.a_prev[:, 0] = 1

    def forward(self, s_prev, stop_token_cum):
        """
        :param s_prev: 2-d Tensor. Previous decoder hidden states.
            Size([batch_size, decoder_rnn_units])
        :param stop_token_cum: 1-d Tensor. To skip calculation on stopped cases in batch.
            Size([batch_size]
        :return context_vector: 2-d Tensor.
            Size([batch_size, encoder_output_units])
        """
        h = self.h[:, stop_token_cum, :]
        s_prev = s_prev[stop_token_cum]
        a_prev = self.a_prev[stop_token_cum]
        a_current = self.ARSG.forward(self.F_matrix, a_prev, s_prev, h, self.mode, self.beta) # Size([batch, input_time_length])
        context_vector = torch.bmm(a_current.unsqueeze(1), h.transpose(0, 1)) # Size([batch, 1, encoder_output_units])
        context_vector = context_vector.squeeze(1) # Size([batch, encoder_output_units])
        self.alignments[stop_token_cum, :, self.decoder_time_step] = a_current # Store alignment
        self.a_prev[stop_token_cum] = a_current
        self.decoder_time_step += 1
        return context_vector

class Encoder(nn.Module):
    def __init__(self,
                 max_input_text_length=100,
                 encoder_embedding_dim=512,
                 pretrained_embedding=False,
                 initial_weights=False,
                 encoder_conv_out_channels_list=[512, 512, 1],
                 encoder_conv_kernel_size_list=5,
                 encoder_conv_stride_list=1,
                 encoder_conv_dropout_list=.5,
                 encoder_conv_batch_normalization_list=True,
                 encoder_conv_activation_list='relu',
                 encoder_rnn_units=256,
                 encoder_rnn_layers=1,
                 encoder_rnn_dropout=.5
                 ):
        super(Encoder, self).__init__()
        self.character_embedding = CharacterEmbedding(max_input_text_length,
                                                      encoder_embedding_dim,
                                                      pretrained_embedding,
                                                      initial_weights)
        self.conv_layers = ConvLayers(1,
                                      encoder_conv_out_channels_list,
                                      encoder_conv_kernel_size_list,
                                      encoder_conv_stride_list,
                                      encoder_conv_dropout_list,
                                      encoder_conv_batch_normalization_list,
                                      encoder_conv_activation_list)
        self.rnn = nn.RNNBase(mode='LSTM',
                              input_size=encoder_conv_out_channels_list[-1] * encoder_embedding_dim,
                              hidden_size=encoder_rnn_units,
                              num_layers=encoder_rnn_layers,
                              bias=True,
                              dropout=encoder_rnn_dropout,
                              bidirectional=True)  # returns output.size([num_embeddings, batch_size, 2(bidirection)*hidden_size): stack of output states, h_n.size([2(bidirection)*num_layers, batch_size, hidden_size]): last hidden state

    def forward(self, input_character_indices):
        # Encoder
        # Character embeddings
        character_embeddings = self.character_embeddings(input_character_indices)
        # Convolution layers
        character_embeddings = character_embeddings.unsqueeze(1)
        conv_embeddings = self.conv_layers(character_embeddings)
        # RNN
        conv_embeddings = conv_embeddings.squeeze(1).transpose(0,
                                                               1)  # (max_input_length, batch, input_dim)
        encoder_output, (encoder_h_n, encoder_c_n) = self.rnn(conv_embeddings)
        return encoder_output, (encoder_h_n, encoder_c_n)


class Decoder(nn.Module):
    def __init__(self,
                 batch_size=32,
                 encoder_output_units=512,
                 decoder_rnn_units=1024,
                 decoder_rnn_layers=2,
                 decoder_prenet_in_features=80,
                 decoder_prenet_out_features_list=[256, 256],
                 decoder_prenet_bias_list=True,
                 decoder_prenet_batch_normalization_list=False,
                 decoder_prenet_activation_list='relu',
                 decoder_postnet_in_channels=1,
                 decoder_postnet_out_channels_list=[512, 512, 512, 512, 1],
                 decoder_postnet_kernel_size_list=5,
                 decoder_postnet_stride_list=1,
                 decoder_postnet_batch_normalization_list=True,
                 decoder_postnet_activation_list=['tanh', 'tanh', 'tanh', 'tanh', None],
                 max_output_time_length=1024,
                 num_mel_channels=80
                 ):
        super(Decoder, self).__init__()
        assert(decoder_prenet_in_features == num_mel_channels)

        self.lstm_cell_0 = nn.LSTMCell(
            input_size=encoder_output_units + decoder_prenet_out_features_list[-1],
            hidden_size=decoder_rnn_units,
            bias=True
        )

        self.lstm_cell_1 = nn.LSTMCell(
            input_size=decoder_rnn_units,
            hidden_size=decoder_rnn_units,
            bias=True
        )

        self.linear_projection_mel = LinearNorm(
            in_features=(encoder_output_units + decoder_rnn_units), #Concatenate attention context vector and LSTM output
            out_features=num_mel_channels,
            bias=True,
            batch_normalization= False,
        )

        self.linear_projection_stop = LinearNorm(
            in_features=(encoder_output_units + decoder_rnn_units),
            out_features=1,
            bias=True,
            batch_normalization=True,
            activation='sigmoid'
        )

        self.postnet = Conv2DSeq(
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

        self.decoder_prenet_in_features = decoder_prenet_in_features
        self.decoder_rnn_units = decoder_rnn_units
        self.max_output_time_length = max_output_time_length
        self.num_mel_channels = num_mel_channels

        # Reset batch params
        self.frame_prev = torch.zeros([batch_size, self.decoder_prenet_in_features])  # Initial go-frame
        self.h_prev_0 = torch.zeros([batch_size, self.decoder_rnn_units])  # Initial hidden state
        self.c_prev_0 = torch.zeros([batch_size, self.decoder_rnn_units])  # Initial cell state
        self.h_prev_1 = torch.zeros([batch_size, self.decoder_rnn_units])  # Initial hidden state
        self.c_prev_1 = torch.zeros([batch_size, self.decoder_rnn_units])  # Initial cell state

        self.spectrogram_pred = torch.zeros([batch_size, self.max_output_time_length, self.num_mel_channels])
        self.stop_token_cum = torch.ones(batch_size, dtype=torch.uint8)
        self.decoder_time_step = 0

    def reset(self, batch_size):
        """
        Reset attributes for new batch.
        :param batch_size:
        :return:
        """
        self.frame_prev = torch.zeros([batch_size, self.decoder_prenet_in_features])  # Initial go-frame
        self.h_prev_0 = torch.zeros([batch_size, self.decoder_rnn_units])  # Initial hidden state
        self.c_prev_0 = torch.zeros([batch_size, self.decoder_rnn_units])  # Initial cell state
        self.h_prev_1 = torch.zeros([batch_size, self.decoder_rnn_units])  # Initial hidden state
        self.c_prev_1 = torch.zeros([batch_size, self.decoder_rnn_units])  # Initial cell state

        self.spectrogram_pred = torch.zeros([batch_size, self.max_output_time_length, self.num_mel_channels])
        self.stop_token_cum = torch.ones(batch_size, dtype=torch.uint8)
        self.decoder_time_step = 0

    def forward(self, context_vector):
        """
        :param context_vector: 3-d Tensor.
            Size([batch_size, encoder_output_units])
        :return self.frame_prev: 2-d Tensor
            Size([batch_size, num_mel_channels])
        :return _stop_token: 1-d Tensor
            Size([batch_size])
        """
        stop_token_cum = self.stop_token_cum
        #1.Prenet
        self.frame_prev = self.prenet(self.frame_prev) #Size([batch_size, decoder_prenet_out_features])
        #2.LSTMs
        h_prev_0 = self.h_prev_0[stop_token_cum]
        h_prev_1 = self.h_prev_1[stop_token_cum]
        c_prev_0 = self.c_prev_0[stop_token_cum]
        c_prev_1 = self.c_prev_1[stop_token_cum]
        _input = torch.cat([self.frame_prev, context_vector], dim=1) #Size([batch_size, encoder_output_units+decoder_prenet_in_features])

        (h_prev_0, c_prev_0) = self.lstm_cell_0(_input, (h_prev_0, c_prev_0))
        self.h_prev_0[stop_token_cum] = h_prev_0
        self.c_prev_0[stop_token_cum] = c_prev_0

        (h_prev_1, c_prev_1) = self.lstm_cell_1(h_prev_0, (h_prev_1, c_prev_1))
        self.h_prev_1[stop_token_cum] = h_prev_1
        self.c_prev_1[stop_token_cum] = c_prev_1

        #3.Linear projection for mel
        _input = torch.cat([h_prev_1, context_vector], dim=1)
        _frame_curr = self.linear_projection_mel(_input) #Size([batch_size, num_mel_channels])
        self.frame_prev = _frame_curr
        #4.Post-Net and add
        _frame_curr = _frame_curr.unsqueeze(1).unsqueeze(3) #Size([batch_size, 1, num_mel_channels, 1])
        _frame_curr = _frame_curr+self.postnet(_frame_curr) #Size([batch_size, 1, num_mel_channels, 1])
        _frame_curr = _frame_curr.squeeze(3).squeeze(1) #Size([batch_size, num_mel_channels])
        #5.Store predicted frame
        self.spectrogram_pred[stop_token_cum, self.decoder_time_step, :] = _frame_curr
        self.decoder_time_step += 1
        #6.Linear projection for stop token
        _stop_token = self.linear_projection_stop(_input) #Size([batch_size, 1])
        _stop_token = _stop_token.squeeze(1) #Size([batch_size])
        _stop_token = _stop_token < .5 # (<0.5) -> valid, (>0.5) -> switch off)
        #7. Update valid cases for further prediction
        self.stop_token_cum[self.stop_token_cum == 1] = _stop_token # Update stop token among still valid cases
        self.frame_prev = self.frame_prev[_stop_token] # Return newly valid cases only
        return self.h_prev_1, self.stop_token_cum

def checkup():
    #check CharacterEmbedding
    embedding = CharacterEmbedding(30, 128)
    batch_size = 32
    max_input_length = 100
    input_indices = torch.randint(0,30,[batch_size, max_input_length])
    character_embeddings = embedding(input_indices)
    print('character embeddings shape: ', character_embeddings.size())

    #check ConvLayers
    convlayers = ConvLayers(1, [512, 1], 5, 1, .5, True, 'relu')
    character_embeddings = character_embeddings.unsqueeze(1)
    print('unsqueezed character embeddings shape: ', character_embeddings.size())
    conv_embeddings = convlayers(character_embeddings)
    print('conv embeddings shape: ', conv_embeddings.size())

    #chack encoder rnn
    encoder_rnn = nn.LSTM(
        input_size=128,
        hidden_size=256,
        num_layers=1,
        bias=True,
        dropout=.5,
        bidirectional=True)  # returns output.size([num_embeddings, batch_size, 2(bidirection)*hidden_size): stack of output states, h_n.size([2(bidirection)*num_layers, batch_size, hidden_size]): last hidden state
    conv_embeddings = conv_embeddings.squeeze(1).transpose(0, 1) #(max_input_length, batch, input_dim)
    print('conv embeddings reshape: ', conv_embeddings.size())
    encoder_output, (encoder_h_n, encoder_c_n) = encoder_rnn(conv_embeddings)
    print('encoder final layer hidden states shape: ', encoder_output.size(),
          '\t encoder last step hidden states: ', encoder_h_n.size(),
          '\t encoder last step cell states: ', encoder_c_n.size())

    attention = LocationSensitiveAttention(32, 1024, 512, 128, 128, 100, 1024)
    attention.h = encoder_output #attention.h.Size([input length, batch, encoder output units])
    max_output_time_length = 30
    decoder = Decoder(max_output_time_length=max_output_time_length)
    h_prev_1 = decoder.h_prev_1
    stop_token_cum = decoder.stop_token_cum
    for decoder_step in range(max_output_time_length):
        print('\n---------------------', 'decoder step: ', decoder_step + 1)
        context_vector = attention.forward(h_prev_1, stop_token_cum)
        h_prev_1, stop_token_cum = decoder.forward(context_vector)
        if not any(stop_token_cum): # stop decoding if no further prediction is needed for any samples in batch
            break

checkup()




#parser = hparams.parser
