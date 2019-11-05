import torch
import torch.utils.data as data


class TextAudioDataset(data.Dataset):
    def __init__(self, file_list, phase='train'):
        self.file_list = file_list
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        text_path = self.file_list[index]

        # 画像のラベルをファイル名から抜き出す
        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]

        # ラベルを数値に変更する
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transformed, label