import torch.nn as nn

from model.local import LocalFeatureExtractor, InvertedResidual
from model.modulator import Modulator


def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True))


class VideoEmotionDetection(nn.Module):
    def __int__(self):
        super(VideoEmotionDetection, self).__init__()

        self.n_classes = 8

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 29, 3, 2, 1, bias=False),
            nn.BatchNorm2d(29),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)

        self.stage2 = nn.Sequential(
            InvertedResidual(29, 116, 2),
            InvertedResidual(116, 116, 1),
            InvertedResidual(116, 116, 1),
            InvertedResidual(116, 116, 1),
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(116, 232, 2),
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
            InvertedResidual(232, 232, 1),
        )

        self.stage4 = nn.Sequential(
            InvertedResidual(232, 464, 2),
            InvertedResidual(464, 464, 1),
            InvertedResidual(464, 464, 1),
            InvertedResidual(464, 464, 1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(464, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.conv1d_0 = conv1d_block(1024, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        self.classifier_1 = nn.Sequential(
            nn.Linear(128, 8),
        )

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.modulator(self.stage2(x)) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        return x

    def forward_stage1(self, x):
        # Getting samples per batch
        assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        x = x.permute(0, 2, 1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])  # pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x
