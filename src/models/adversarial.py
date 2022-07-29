import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnode import TimeDistributed



class FrameClassifier(nn.Module):
    def __init__(self, device, in_channels,
                    activation=nn.ReLU()):
        super(FrameClassifier, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.activation = activation


        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=5, stride=2, padding=1),
                    self.activation,
                    nn.BatchNorm2d(32),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                    self.activation,
                    nn.BatchNorm2d(64),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                    self.activation,
                    nn.BatchNorm2d(128),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=1),
                    self.activation,
                    nn.BatchNorm2d(256)
        ).to(self.device)

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=3)
        self.fc = nn.Linear(in_features=256, out_features=1).to(self.device)


    def forward(self, x):
        # print("FrameClassifier: x.shape: ", x.shape)
        x = self.conv(x)
        # print("FrameClassifier: conv.shape: ", x.shape)
        x = self.avg_pool(x)
        # print("FrameClassifier: avg_pool.shape: ", x.shape)
        x = x.view(x.size(0), -1)
        # print("FrameClassifier: view.shape: ", x.shape)
        x = self.fc(x)
        # print("FrameClassifier: fc.shape: ", x.shape)
        x = torch.sigmoid(x)
        return x


class SequenceClassifier(nn.Module):
    def __init__(self, device, in_channels,
                    activation=nn.ReLU()):
        super(SequenceClassifier, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.activation = activation

        # The only difference between the two models is the number of channels
        # in the convolutional layers.
        # The number of channels in the convolutional layers is determined by
        # the number of frames in the input sequence by using all the grayscale frames and spatial encoding.

        # I may replace this 2D CNN classifier by a 3D CNN classifier
        self.conv = FrameClassifier(device, in_channels, activation)



    def forward(self, seq):        
        # # Seq is of shape (batch_size, seq_len, in_channels, height, width)
        # # in_channels = 3 for grayscale images and 5 for RGB images (spatial_encoding is used)
        # # spatial_encoding: (batch_size, 2, height, width)
        # spatial_encoding = seq[:,0,1:]
        # # grayscale_seq: (batch_size, seq_len, height, width)
        # grayscale_seq = seq[:,:,0,:,:]
        # # concatened_seq: (batch_size, seq_len + 2, height, width)
        # concatened_seq = torch.cat((grayscale_seq, spatial_encoding), dim=1)
        # # We need to reshape it to (batch_size, seq_len + 2, height, width)
        x = self.conv(seq)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class SequenceClassifier3D(nn.Module):
    def __init__(self, device, in_channels,
                    activation=nn.ReLU()):
        super(SequenceClassifier3D, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.activation = activation


        self.conv = nn.Sequential(
                    nn.Conv3d(in_channels=self.in_channels, out_channels=32, kernel_size=(4, 5, 5), stride=(2, 2, 2), padding=(0, 1, 1)),
                    self.activation,
                    nn.BatchNorm3d(32),
                    nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    self.activation,
                    nn.BatchNorm3d(64),
                    nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                    self.activation,
                    nn.BatchNorm3d(128),
                    nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1)),
                    self.activation,
                    nn.BatchNorm3d(256)
        ).to(self.device)

        self.avg_pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=256, out_features=1).to(self.device)


    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class Discriminators(nn.Module):
    def __init__(self, seq_discriminator, frame_discriminator) -> None:
        super(Discriminators, self).__init__()
        self.seq_discriminator = seq_discriminator
        self.frame_discriminator = frame_discriminator

    def forward_seq(self, seq):
        # print("seq shape", seq.shape)
        frames = seq[:,:,0,:,:]
        spatial_encoding = seq[:,0,1:,:,:]
        concatened_seq = torch.cat((frames, spatial_encoding), dim=1)
        # print("seq shape", concatened_seq.shape)
        return self.seq_discriminator(concatened_seq)

    def forward_frame(self, frame):
        # print("frame shape", frame.shape)
        return self.frame_discriminator(frame)
                

