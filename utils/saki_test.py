import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from model.audio_model import AudioEmotionDetection


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    X_train, X_val, X_test, y_train, y_test, y_val = [], [], [], [], [], []
    with open('../audio_data.pickle', 'rb') as handle:
        dataset = pickle.load(handle)
        X_train = Variable(torch.Tensor(dataset["X_train"]).float())
        X_val = Variable(torch.Tensor(dataset["X_val"]).float())
        X_test = Variable(torch.Tensor(dataset["X_test"]).float())
        y_train = Variable(torch.Tensor(dataset["y_train"]).float())
        y_val = Variable(torch.Tensor(dataset["y_val"]).float())
        y_test = Variable(torch.Tensor(dataset["y_test"]).float())

    model = AudioEmotionDetection(torch.stack([X_train, X_val])[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model, lr=0.01)

    #TODO("validation dataset should be used somewhere in the future")
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(X_train)
        #here minibatches
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch+1} loss: {loss.item()}")


    #evaluate accuracy
    pred = model(X_test)
    _, predictions = torch.max(pred, axis=1)
    correct = accuracy_score(y_test, predictions)
    print(f"Accuracy: {correct}")