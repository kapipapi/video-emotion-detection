import os
import numpy as np
import torch
import torch.utils.data as data


class Data:
    def __init__(self, path: str):
        self.filepath: str = path
        self.label: int = self.get_label()

    def get_label(self) -> int:
        return int(self.filepath.split('-')[2]) - 1


class RAVDESS(data.Dataset):
    def __init__(self, dir_path="../dataset/", transforms=None):
        self.dir_path = dir_path
        self.data = self.load_data()
        self.transforms = transforms

    def load_data(self) -> [Data]:
        arr: [Data] = []
        for sess in sorted(os.listdir(self.dir_path)):
            for filename in os.listdir(os.path.join(self.dir_path, sess)):
                if filename.endswith('.npy'):
                    arr.append(Data(os.path.join(self.dir_path, sess, filename)))
        return arr

    def __getitem__(self, index) -> (np.array, int):
        d = self.data[index]

        clip = np.load(d.filepath)

        if self.transforms is not None:
            clip = [self.transforms(img) for img in clip]

        clip = torch.stack(clip, 0)

        return clip, d.label

    def __len__(self):
        return len(self.data)