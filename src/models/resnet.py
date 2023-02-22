import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt

EXPANSION = 4

class blockResNet(nn.Module):
    """
    ResNet block that could be used to create ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 or custom ResNet. 

    Parameters
    ----------
    in_channels : int, the number of input channels
    intermediate_channels : int, the number of intermediate channels
    identity_downsample : nn.Module, the downsample layer for the identity, default is None
    stride : int, the stride for the convolutional layers, default is 1
    expansion : int, the expansion factor for the output channels, default is 4
    """
    def __init__(
        self, in_channels, intermediate_channels, 
        identity_downsample=None, stride=1, expansion=4
    ):
        super(blockResNet, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
        # print("-"*70)
        # print("Number of parameters: ", sum([p.numel() for p in self.parameters()]))

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNetEncoder28(nn.Module):
    """
    ResNet Encoder for 28x28 images
    
    Parameters
    ----------
    layers : list, the number of residual blocks in each layer
    image_channels : int, the number of channels in the input image
    n_latent : int, the number of latent dimensions, default is 128
    """
    def __init__(self, layers, image_channels, n_latent=128):
        super(ResNetEncoder28, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.expansion = EXPANSION
        self.block = blockResNet

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
        blockResNet, layers[0], intermediate_channels=32, stride=1
        )
        self.layer2 = self._make_layer(
        blockResNet, layers[1], intermediate_channels=64, stride=2
        )
        # self.layer3 = self._make_layer(
        # block, layers[2], intermediate_channels=64, stride=2
        # )
        # self.layer4 = self._make_layer(
        # block, layers[3], intermediate_channels=128, stride=2
        # )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.expansion * 64, n_latent)
        print("-"*70)
        # print("Number of parameters: ", sum([p.numel() for p in self.layer1.parameters()])/1000, "k")
        # print("Number of parameters: ", sum([p.numel() for p in self.layer2.parameters()])/1000, "k")
        # print("Number of parameters: ", sum([p.numel() for p in self.layer3.parameters()])/1000, "k")
        # # print("Number of parameters: ", sum([p.numel() for p in self.layer4.parameters()])/1000, "k")
        
        print("Number of parameters: ", sum([p.numel() for p in self.parameters()])/1e6, "M")
        print("-"*70)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)         # 1x1
        x = torch.flatten(x, 1)     # remove 1 X 1 grid and make vector of tensor shape 
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        """
        Helper function to make a layer of residual blocks by stacking them on top of each other 
        and by downsampling the input space if needed
        """
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead


        if stride != 1 or self.in_channels != intermediate_channels * self.expansion:
            identity_downsample = nn.Sequential(
            nn.Conv2d(
            self.in_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=stride,
            bias=False
            ),
            nn.BatchNorm2d(intermediate_channels * self.expansion),
            )

            layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
            )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * self.expansion

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)



class ResNetDecoder28(nn.Module):
    """
    ResNet Decoder for 28x28 images using transposed convolutions, upsampling and residual blocks.

    Parameters
    ----------
    img_channel : int, the number of channels in the output image
    n_latent : int, the number of latent dimensions, default is 128
    """
    def __init__(self, img_channel=3, n_latent=128):
        super(ResNetDecoder28,self).__init__()
        self.img_channel = img_channel
        self.n_latent = n_latent
        self.conv_shape = (256, 1, 1)
        prods = self.conv_shape[0] * self.conv_shape[1] * self.conv_shape[2]
        self.dfc2 = nn.Linear(n_latent, prods)
        self.bn2 = nn.BatchNorm1d(prods)
        self.dfc1 = nn.Linear(prods, prods)
        self.bn1 = nn.BatchNorm1d(prods)
        self.upsample1=nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(self.conv_shape[0], 128, 3, padding = 0)
        self.dconv4 = nn.ConvTranspose2d(128, 64, 3, padding = 1)
        self.dconv3 = nn.ConvTranspose2d(64, 32, 3, padding = 1)
        self.dconv2 = nn.ConvTranspose2d(32, 16, 4, padding = 2)
        self.dconv1 = nn.ConvTranspose2d(16, img_channel, 6, stride = 2, padding = 2)

        print("Number of parameters in decoder: ", sum([p.numel() for p in self.parameters()])/1e6, "M")
        # for p in self.parameters():
        # 	print(p.numel())

    def forward(self,x):#,i1,i2,i3):
        batch_size = x.size(0)
        # print(x.size())
        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        # print(x.size())
        #x = F.relu(x)
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        # print(x.size())
        #x = F.relu(x)
        # print(x.size())

        x = x.view(batch_size,self.conv_shape[0],self.conv_shape[1],self.conv_shape[2])
        # print (x.size())
        x=self.upsample1(x)
        # print(x.size())
        x = self.dconv5(x)
        # print(x.size())
        x = F.relu(x)
        # print(x.size())
        x = F.relu(self.dconv4(x))
        # print(x.size())
        x = F.relu(self.dconv3(x))
        # print(x.size())
        x=self.upsample1(x)
        # print(x.size())
        x = self.dconv2(x)
        # print(x.size())
        x = F.relu(x)
        x=self.upsample1(x)
        # print(x.size())
        x = self.dconv1(x)
        # print(x.size())
        x = torch.sigmoid(x)
        #print x
        return x


class ResNetEncoder64(nn.Module):
    """
    ResNet Encoder for 64x64 images using convolutions, downsampling and residual blocks.
    
    Parameters
    ----------
    layers : list, the number of residual blocks in each layer
    image_channels : int, the number of channels in the input image
    n_latent : int, the number of latent dimensions, default is 128
    """
    def __init__(self, layers, image_channels, n_latent=128):
        super(ResNetEncoder64, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.expansion = EXPANSION

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
        blockResNet, layers[0], intermediate_channels=32, stride=1
        )
        self.layer2 = self._make_layer(
        blockResNet, layers[1], intermediate_channels=64, stride=2
        )
        self.layer3 = self._make_layer(
        blockResNet, layers[2], intermediate_channels=128, stride=2
        )
        # self.layer4 = self._make_layer(
        # block, layers[3], intermediate_channels=256, stride=2
        # )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.expansion * 128, self.expansion * 64)
        self.bnfc1 = nn.BatchNorm1d(self.expansion * 64)
        self.fc2 = nn.Linear(self.expansion * 64, n_latent)

        print("-"*70)
        # print("Number of parameters: ", sum([p.numel() for p in self.layer1.parameters()])/1000, "k")
        # print("Number of parameters: ", sum([p.numel() for p in self.layer2.parameters()])/1000, "k")
        # print("Number of parameters: ", sum([p.numel() for p in self.layer3.parameters()])/1000, "k")
        # # print("Number of parameters: ", sum([p.numel() for p in self.layer4.parameters()])/1000, "k")
        
        print("Number of parameters: ", sum([p.numel() for p in self.parameters() if p.requires_grad])/1e6, "M")
        print("-"*70)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)         # 1x1
        # print(x.shape)
        x = torch.flatten(x, 1)     # remove 1 X 1 grid and make vector of tensor shape 
        # print(x.shape)
        x = self.fc1(x)
        x = self.bnfc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        """
        Helper function to make a layer of residual blocks by stacking them on top of each other 
        and by downsampling the input space if needed        
        """
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead


        if stride != 1 or self.in_channels != intermediate_channels * self.expansion:
            identity_downsample = nn.Sequential(
            nn.Conv2d(
            self.in_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=stride,
            bias=False
            ),
            nn.BatchNorm2d(intermediate_channels * self.expansion),
            )

            layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
            )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * self.expansion

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


class ResNetDecoder64(nn.Module):
    """
    ResNet Decoder for 64x64 images using convolutions, upsampling and residual blocks.

    Parameters
    ----------
    img_channel : int, the number of channels in the output image, default is 3
    n_latent : int, the number of latent dimensions, default is 128
    """
    def __init__(self, img_channel=3, n_latent=128):
        super(ResNetDecoder64,self).__init__()
        self.img_channel = img_channel
        self.n_latent = n_latent
        # 256 before
        self.conv_shape = (512, 1, 1)
        prods = self.conv_shape[0] * self.conv_shape[1] * self.conv_shape[2]
        self.dfc2 = nn.Linear(n_latent, prods)
        self.bn2 = nn.BatchNorm1d(prods)
        self.dfc1 = nn.Linear(prods, prods)
        self.bn1 = nn.BatchNorm1d(prods)
        self.upsample1=nn.Upsample(scale_factor=2)
        self.dconv7 = nn.ConvTranspose2d(self.conv_shape[0], 256, 3, padding = 0)
        self.dconv6 = nn.ConvTranspose2d(256, 128, 3, padding = 1)
        self.dconv5 = nn.ConvTranspose2d(128, 64, 3, padding = 1)
        self.dconv4 = nn.ConvTranspose2d(64, 32, 3, padding = 1)
        self.dconv3 = nn.ConvTranspose2d(32, 16, 3, padding = 1)
        self.dconv2 = nn.ConvTranspose2d(16, 16, 4, padding = 2)
        self.dconv1 = nn.ConvTranspose2d(16, img_channel, 6, stride = 2, padding = 0)

        print("Number of parameters in decoder: ", sum([p.numel() for p in self.parameters()])/1e6, "M")
        # for p in self.parameters():
        # 	print(p.numel())

    def forward(self,x):#,i1,i2,i3):
        # Linear layers
        batch_size = x.size(0)
        # print(x.size())
        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        # print(x.size())
        #x = F.relu(x)
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        # print(x.size())
        #x = F.relu(x)
        # print(x.size())

        # Convolutional and Upsampling layers
        x = x.view(batch_size,self.conv_shape[0],self.conv_shape[1],self.conv_shape[2])
        # print (x.size())
        x=self.upsample1(x)
        # print(x.size())
        x = F.relu(self.dconv7(x))
        # print(x.size())
        x = F.relu(self.dconv6(x))
        # print(x.size())
        x = self.upsample1(x)
        # print(x.size())
        x = self.dconv5(x)
        # print(x.size())
        x = F.relu(x)
        # print(x.size())
        x = F.relu(self.dconv4(x))
        # print(x.size())
        x=self.upsample1(x)
        x = F.relu(self.dconv3(x))
        # print(x.size())
        x = self.dconv2(x)
        # print(x.size())
        x = F.relu(x)
        x=self.upsample1(x)
        # print(x.size())
        x = self.dconv1(x)
        # print(x.size())
        x = torch.sigmoid(x)
        #print x
        return x