import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms

from model.cnn_model import VideoEmotionDetection

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=(720, 1280), device=device)
mtcnn.to(device)

model = VideoEmotionDetection()
model.load_state_dict(torch.load('/home/kacper/Documents/video-emotion-detection/saved_model_50epochs.pth'))
model.to(device)
if torch.cuda.is_available():
    model = model.cuda()
model.eval()

cap = cv2.VideoCapture("/home/kacper/Documents/video-emotion-detection/dataset/Actor_01/01-01-07-01-01-01-01.mp4")
# cap = cv2.VideoCapture(0)

emotions = ["neutral", "calm", "happy", "sad", "angry", 'fearful', 'disgust', 'surprised']

buff = []
while True:
    ret, im = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    temp = im[:, :, -1]
    im_rgb = im.copy()
    im_rgb[:, :, -1] = im_rgb[:, :, 0]
    im_rgb[:, :, 0] = temp
    im_rgb = torch.tensor(im_rgb)
    im_rgb = im_rgb.to(device)

    bbox = mtcnn.detect(im_rgb)
    if bbox[0] is not None:
        bbox = bbox[0][0]
        bbox = [round(x) for x in bbox]
        x1, y1, x2, y2 = bbox
        im = im[y1:y2, x1:x2, :]
        im = cv2.resize(im, (224, 224))
        cv2.imshow("test", im)

        im_tensor = transforms.ToTensor()(im)
        im_tensor = im_tensor.to(device)
        im_tensor = torch.stack([im_tensor])

        output = model(im_tensor)
        output = np.argmax(torch.nn.functional.softmax(output, dim=1).tolist())
        print(output, emotions[output])

        cv2.waitKey(1)
