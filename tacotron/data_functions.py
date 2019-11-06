import argparse
import numpy as np
from scipy.io import wavfile
import torch
import torch.utils.data as data
from tqdm import tqdm

from tacotron.text import text_to_sequence
from common.data_functions import load_filepaths_and_texts

class TextMelSet(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, dataset_path, audiopaths_and_text, args):
        self.audiopaths_and_text = load_filepaths_and_text(dataset_path, audiopaths_and_text) # = [(path, text) for datum in dataset]
        self.text_cleaners = args.text_cleaners
        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        self.load_mel_from_disk = args.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.sampling_rate, args.mel_fmin,
            args.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        """
        :param audiopath_and_text: Tuple
            (path, text)
        :return:
        """
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        len_text = len(text)
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel, len_text)

    def get_mel(self, audio_path):
        if self.mel_from_disk:
            return torch.load(audio_path)
        else:
            sr, audio = wavfile.read(audio_path)
            audio = torch.FloatTensor(audio.astype(np.float32))
            if sr != self.stft.sr:
                raise ValueError('SR mismatch: {} for given audio, {} for STFT'.format(sr, self.stft.sr))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            mel_spectrogram = self.stft.mel_spectrogram(audio_norm)
            mel_spectrogram = mel_spectrogram.squeeze(0)
            return mel_spectrogram

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

class TextMelCollate():
    """
    batch = self.collate_fn([self.dataset[i] for i in indices])
    """
    def __init__(self, reduction_factor):
        self.reduction_factor = reduction_factor

    def __call__(self, batch):
        '''
        Collates training batch from normalized text and mel spectrogram.
        :param batch: 2-d Tensor
            Size([batch_size])
            Every sample contains (len(text), text_norm, mel_norm).
            Every text_norm may have different length.
        :return:
        '''
        input_lengths, ids_sorted_descending = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)  # sort text length
        max_input_length = input_lengths[0]

        text_padded = torch.LongTensor([len(batch), max_input_length])
        text_padded = torch.zeros([len(batch), max_input_length], dtype=torch.long)
        for i in range(len(ids_sorted_descending)):
            text = batch[ids_sorted_descending[i]][0]

        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        data.DataLoader
