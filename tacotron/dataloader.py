import torch
import torch.utils.data as data

class TextAudioDataset(data.Dataset):
    def __init__(self, file_list, preprocessor=None, phase='train'):
        super(TextAudioDataset, self).__init__()
        self.file_list = file_list
        self.preprocessor = preprocessor
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        text_path = self.file_list[index]
        if self.preprocessor:
            self.preprocessor