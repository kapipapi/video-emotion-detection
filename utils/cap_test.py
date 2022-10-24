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
model.load_state_dict(
    torch.load('/home/kacper/Documents/video-emotion-detection/saved_models/saved_model_pretrained_50epochs.pth',
               map_location=device))
model.to(device)
if torch.cuda.is_available():
    model = model.cuda()
model.eval()

emotions = ["neutral", "calm", "happy", "sad", "angry", 'fearful', 'disgust', 'surprised']

cap = cv2.VideoCapture(0)

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

        im_tensor = transforms.ToTensor()(im)
        im_tensor = im_tensor.to(device)
        buff.append(im_tensor)
        if len(buff) == 15:
            input_tensor = torch.stack(buff)
            buff = buff[1:]
            output = model(input_tensor)
            output = np.argmax(torch.nn.functional.softmax(output, dim=1).tolist())
            cv2.putText(im, emotions[output], (0, 50), 1, 1, 1)

        cv2.imshow("test", im)

        cv2.waitKey(1)
