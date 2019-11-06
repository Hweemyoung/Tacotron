import argparse
import numpy as np
from scipy.io import wavfile
import torch
import torch.utils.data as data
from tqdm import tqdm

from tacotron.text import text_to_sequence

class TextMelSet(data.Dataset):
    def __init__(self, model_name, dataset_path, training_files, args):
        self.load_mel_from_disk = args.load_mel_from_disk
        self.text_cleaners = args.text_cleaners
        self.audio_paths = self

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths[index], self.text_list[index])

    def __len__(self):
        return len(self.audiopaths)

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

    def text_to_sequence(self, text, text_cleaner, _curly_re):
        """ from https://github.com/keithito/tacotron """
        sequence = []
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence +=

    def get_mel_text_pair(self, audio_path, text):
        len_text = len(text)
        text = self.get_text(text)
        mel = self.get_mel(audio_path)
        return (text, mel, len_text)

def audio2mel(dataset_path, mel_paths, audio_paths):
    mel_paths_list = load_filepaths(dataset_path, mel_paths)
    audio_paths_list = load_filepaths(dataset_path, audio_paths)
    text_mel_loader = TextMelSet()

    for i in tqdm(range(len(mel_paths_list))):
        mel = text_mel_loader.get_mel(audio_paths_list[i][0])
        torch.save(mel, mel_paths_list[i][0])

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--wav-files', required=True,
                        type=str, help='Path to filelist with audio paths and text')
    parser.add_argument('--mel-files', required=True,
                        type=str, help='Path to filelist with mel paths and text')
    parser.add_argument('--text-cleaners', nargs='*',
                        default=['english_cleaners'], type=str,
                        help='Type of text cleaners for input text')
    parser.add_argument('--max-wav-value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--filter-length', default=1024, type=int,
                        help='Filter length')
    parser.add_argument('--hop-length', default=256, type=int,
                        help='Hop (stride) length')
    parser.add_argument('--win-length', default=1024, type=int,
                        help='Window length')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                        help='Minimum mel frequency')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                        help='Maximum mel frequency')
    parser.add_argument('--n-mel-channels', default=80, type=int,
                        help='Number of bins in mel-spectrograms')

    return parser

def main():
    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Training')
    parser = parse_args(parser)
    args = parser.parse_args()
    args.load_mel_from_disk = False

    audio2mel(args.dataset_path, args.wav_files, args.mel_files, args)
