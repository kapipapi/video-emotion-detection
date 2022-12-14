import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms

from model.dataloader import RAVDESS
from model.cnn_model import VideoEmotionDetection

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

print("Device:", device)

model = VideoEmotionDetection(
    efficient_face_pretrained_path='/home/kacper/Documents/video-emotion-detection/pretrained/EfficientFace_Trained_on_AffectNet7.pth.tar'
)
model.to(device)
if torch.cuda.is_available():
    model = model.cuda()

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20)
])

# save these actors for test purpose
avoid_actor_number = None
# avoid_actor_number = [11, 12, 13, 14, 15]

# generate dataset
train = RAVDESS('./dataset/', transforms, avoid_actor_number)
n_samples = len(train)
print("n_samples:", n_samples)

train, valid = data.random_split(train, [int(n_samples * 0.8), int(n_samples * 0.2)])

batchSize = 8
print("batch size:", batchSize)

train_loader = data.DataLoader(train, batch_size=batchSize)
valid_loader = data.DataLoader(valid, batch_size=batchSize)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 100
print("epochs:", epochs)

min_valid_loss = np.inf

for e in range(epochs):
    train_loss = 0.0
    model.train()
    for data in train_loader:
        data, labels = data

        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        n_frames = data.shape[1]
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])

        optimizer.zero_grad()
        target = model(data)

        loss = criterion(target, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()
    for data in valid_loader:
        data, labels = data

        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        n_frames = data.shape[1]
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])

        target = model(data)
        loss = criterion(target, labels)
        valid_loss = loss.item() * data.size(0)

    print(
        f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'testing/100epoch/saved_model_100epochs.pth')
