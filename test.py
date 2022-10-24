import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from model.cnn_model import VideoEmotionDetection
from model.dataloader import RAVDESS

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

model = VideoEmotionDetection()
model.load_state_dict(
    torch.load('/home/kacper/Documents/video-emotion-detection/saved_models/saved_model_pretrained_50epochs.pth',
               map_location=device))
model.to(device)
if torch.cuda.is_available():
    model = model.cuda()

transforms = transforms.Compose([
    transforms.ToTensor()
])

# avoid these actors because these are training and validation dataset
avoid_actor_number = list(range(1, 10 + 1))  # from 1 to 10

test = RAVDESS('./dataset/', transforms, avoid_actor_number)
n_samples = len(test)
batchSize = 4
test_loader = data.DataLoader(test, batch_size=batchSize)

y_true = []
y_pred = []

model.eval()
for i, data in enumerate(test_loader):
    data, labels = data

    if torch.cuda.is_available():
        data, labels = data.cuda(), labels.cuda()

    n_frames = data.shape[1]
    data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])

    prediction = model(data)
    prediction = [np.argmax(x) for x in torch.nn.functional.softmax(prediction, dim=1).tolist()]

    y_true += labels.tolist()
    y_pred += prediction

    print(f"{(i + 1) * batchSize} / {n_samples}")

cm = confusion_matrix(y_true, y_pred)
ac = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")

print(cm)
print("accuracy:", ac)
print("f1_score:", f1)

np.savetxt("confusion_matrix.txt", cm, fmt='%.d', delimiter="\t")
