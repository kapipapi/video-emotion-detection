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
    def __init__(self, dir_path="../dataset/", transforms=None, avoid_actor_numbers: [int] = None):
        self.dir_path = dir_path
        self.transforms = transforms
        self.avoid_actor_numbers = avoid_actor_numbers
        self.data = self.load_data()

    def load_data(self) -> [Data]:
        arr: [Data] = []
        for directory in sorted(os.listdir(self.dir_path)):
            actor_number = int(directory.split("_")[1])
            if self.avoid_actor_numbers is not None and actor_number in self.avoid_actor_numbers:
                print(f"Avoiding actor with number {actor_number}")
                continue
            for filename in os.listdir(os.path.join(self.dir_path, directory)):
                if filename.endswith('.npy'):
                    arr.append(Data(os.path.join(self.dir_path, directory, filename)))
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
