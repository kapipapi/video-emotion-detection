import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import cv2

from model.cnn_model import VideoEmotionDetection
from model.dataloader import RAVDESS

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

model = VideoEmotionDetection()
model.load_state_dict(torch.load('/home/kacper/Documents/video-emotion-detection/saved_model_50epochs.pth'))
model.to(device)
if torch.cuda.is_available():
    model = model.cuda()

transforms = transforms.Compose([
    transforms.ToTensor()
])

test = RAVDESS('./dataset/', transforms)
n_samples = len(test)
batchSize = 1
test_loader = data.DataLoader(test, batch_size=batchSize)

test_output = []

model.eval()
for data in test_loader:
    data, labels = data

    if torch.cuda.is_available():
        data, labels = data.cuda(), labels.cuda()

    n_frames = data.shape[1]
    data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])
    labels = labels.reshape((batchSize * n_frames,))

    target = model(data)
    target = target.cpu().tolist()
    target = [np.argmax(i) for i in target]

    test_output.append(labels.tolist() == target)

    print(test_output)
