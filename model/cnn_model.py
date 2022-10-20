import torch
import torch.nn as nn

import efficient.EfficientFace


def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True))


def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    print('Initializing efficientnet')
    model.load_state_dict(pre_trained_dict, strict=True)


def load_efficient_face():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    model = efficient.EfficientFace.efficient_face()
    init_feature_extractor(model,
                           '/home/kacper/Documents/video-emotion-detection/EfficientFace_Trained_on_AffectNet7.pth.tar')
    model.to(device)
    if torch.cuda.is_available():
        model = model.cuda()

    return model


class VideoEmotionDetection(nn.Module):
    def __init__(self):
        super(VideoEmotionDetection, self).__init__()

        self.n_classes = 8

        self.efficient_face = load_efficient_face()

        self.conv1d_0 = conv1d_block(1024, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        self.classifier_1 = nn.Sequential(
            nn.Linear(128, 8),
        )

    def forward_stage1(self, x):
        assert x.shape[0] % 15 == 0, "Batch size is not a multiple of sequence length."
        n_samples = x.shape[0] // 15
        x = x.view(n_samples, 15, x.shape[1])
        x = x.permute(0, 2, 1)

        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])
        x = self.classifier_1(x)
        return x

    def forward(self, x):
        x = self.efficient_face(x)
        x = self.forward_stage1(x)
        x = self.forward_classifier(x)
        return x
