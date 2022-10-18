import torch.utils.data as data
from torchvision import transforms
import cv2

from model.dataloader import RAVDESS

transforms = transforms.Compose([
    transforms.ToTensor()
])

train = RAVDESS('./dataset/', transforms)
n_samples = len(train)
train, valid = data.random_split(train, [int(n_samples * 0.8), int(n_samples * 0.2)])
train_loader = data.DataLoader(train, batch_size=1)
valid_loader = data.DataLoader(valid, batch_size=1)

for data in train_loader:
    data, labels = data
    for i, d in enumerate(data[0]):
        print(labels)
        cv2.imshow(f'im', d.permute(1, 2, 0).numpy())
        cv2.waitKey()