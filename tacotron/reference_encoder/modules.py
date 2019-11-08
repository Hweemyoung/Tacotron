"""
PyTorch implementation of:
R.J. Skerry-Ryan et al., Towards end-to-end prosody transfer for expressive speech synthesis with tacotron. In Proceedings of the 35th International Conference on Machine Learning, 2018.

H.M.Kim, 2019
http://github.com/Hweemyoung/Tacotron
"""

from common.layers import *

class SpeakerEmbeddings(nn.Module):
    """
    ... We follow a scheme similar to (Arik et al., 2017) to model multiple speakers.
    """

    def __init__(self, num_speakers, speaker_dim, pretrained_embeddings=None):
        super(SpeakerEmbeddings, self).__init__()
        if pretrained_embeddings:
            self.speaker_embeddings = nn.Embedding.from_pretrained(
                embeddings=pretrained_embeddings, freeze=True
            )
        else:
            """
            ... is initialized with Glorot(Glorot & Bengio, 2010) initialization.
            """
            self.speaker_embeddings = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty([num_speakers, speaker_dim], requires_grad=True),
                )
            )

    def forward(self, speaker_index):
        return self.speaker_embeddings[speaker_index]


class ReferenceEncoder(nn.Module):
    """
    R.J.Skerry-Ryan et al., Towards end-to-end prosody transfer for expressive speech synthesis with tacotron. In Proceedings of the 35th International Conference on Machine Learning, 2018.
    """

    def __init__(self,
                 batch_size,
                 ref_num_mel_channels=80,
                 ref_conv_out_channels_list=[32, 32, 64, 64, 128, 128],
                 ref_conv_kernel_size_list=3,
                 ref_conv_stride_list=2,
                 ref_conv_dropout_list=.5,
                 ref_conv_bn_list=True,
                 ref_conv_activation_list='relu',
                 ref_rnn_units=128,
                 prosody_dim=128):
        super(ReferenceEncoder, self).__init__()

        """
        6-Layer Strided Conv2D w/ BatchNorm.
        Returns Size([batch, 128, reference_input_length//64, num_mel_channels//64])
        Followed by concatenation.
        """
        self.conv_layers = Conv2DSeq(1,
                                     out_channels_list=ref_conv_out_channels_list,
                                     kernel_size_list=ref_conv_kernel_size_list,
                                     stride_list=ref_conv_stride_list,
                                     dropout_list=ref_conv_dropout_list,
                                     batch_normalization_list=ref_conv_bn_list,
                                     activation_list=ref_conv_activation_list
                                     )

        """
        128-unit GRU.
        Variable-length input.
        """
        self.rnn = nn.GRUCell(
            input_size=self.conv_layers[-1].out_channels * ref_num_mel_channels / ref_conv_stride_list ** len(
                self.conv_layers),
            hidden_size=ref_rnn_units,
            bias=True)
        self.h_prev = torch.zeros([batch_size, ref_rnn_units])

        """
        Linear projection
        """
        self.linear = LinearNorm(in_features=ref_rnn_units,
                                 out_features=prosody_dim,
                                 bias=True,
                                 batch_normalization=True,
                                 activation='tanh')

    def forward(self, ref_spectrogram):
        """
        :param ref_spectrogram: 3-d Tensor
            Size([batch, reference_input_length, num_mel_channels])
        :return:
        """

        # Conv layers
        ref_spectrogram = ref_spectrogram.unsqueeze(1)
        x = self.conv_layers(ref_spectrogram)

        # Reshape
        x = x.transpose(1, 2).view(x[0], x[1],
                                   -1)  # Size([batch, reference_input_length//64, 128*num_mel_channels//64])

        # RNN
        rnn_step = 0
        h_prev = self.h_prev
        while rnn_step < x.size(1):
            h_curr = self.rnn(x[:, rnn_step, :], h_prev)  # Size([batch, ref_rnn_units])
            h_prev = h_curr
            rnn_step += 1

        # Linear projection
        prosody_embedding = self.linear(h_curr)

        return prosody_embedding
