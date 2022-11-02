import torch.nn as nn


def conv1d_block(in_channels, out_channels, kernel_size=10, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                         #nn.BatchNorm1d(out_channels), #to be investigated what works better
                         nn.ReLU(inplace=True))


class AudioEmotionDetection(nn.Module):
    emotion_nb = 7

    #since input_shape=(181,1), in_count should be equal 1
    def __init__(self, in_count):
        super(AudioEmotionDetection, self).__init__()
        self.conv1d_0 = conv1d_block(in_count, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.maxpol1d_0 = nn.MaxPool1d(8)
        self.conv1d_2 = conv1d_block(64, 128)
        self.maxpol1d_1 = nn.MaxPool1d(8)
        self.dropout_0 = nn.Dropout1d(0.4)
        self.flatten_0 = nn.Flatten()
        self.classifier_0 = nn.Sequential(nn.Linear(128, 8),)

        # input_shape = (181, 1)
        # model = Sequential()
        # model.add(Conv1D(64, 10, activation='relu', input_shape=input_shape))
        # model.add(Conv1D(64, 10, activation='relu'))
        # model.add(MaxPooling1D(pool_size=8))
        # model.add(Conv1D(128, 10, activation='relu'))
        # model.add(MaxPooling1D(pool_size=8))
        # model.add(Dropout(0.4))
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(8, activation=tf.keras.activations.softmax))

    def forward_classifier(self, x):
        x = x.mean([-1])
        x = self.classifier_0(x)
        return x

    def forward(self, x):
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        x = self.maxpol1d_0(x)
        x = self.conv1d_2(x)
        x = self.maxpol1d_1(x)
        x = self.dropout_0(x)
        x = self.flatten_0(x)
        x = self.forward_classifier(x)
        return x
