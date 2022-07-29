import torch
import torch.nn as nn
import numpy as np



class Encoder(nn.Module):
    def __init__(self, device, latent_dim, in_channels,
                    activation=nn.ReLU(), relu=False):
        super(Encoder, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        # self.appearance_dim = appearance_dim
        self.in_channels = in_channels
        self.activation = activation
        self.relu = relu

        self.encoder = nn.Sequential(
            # first block
                    nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1, padding=0),
                    self.activation,
                    nn.BatchNorm2d(64),
                    # nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
                    self.activation,
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(kernel_size=2, stride=2),

            # second block
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
                    self.activation,
                    nn.BatchNorm2d(128),
                    # nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
                    self.activation,
                    nn.BatchNorm2d(128),
                    nn.MaxPool2d(kernel_size=2, stride=2),

            # third block
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
                    self.activation,
                    nn.BatchNorm2d(256),
                    # nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
                    self.activation,
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d(kernel_size=2, stride=2),

        ).to(device)
        
                    # nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=7, stride=2, padding=1),
                    # self.activation,
                    # # nn.MaxPool2d(kernel_size=2, stride=2),
                    # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1),
                    # self.activation,
                    # # nn.MaxPool2d(kernel_size=2, stride=2),
                    # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                    # self.activation,
                    # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                    # self.activation,
                    # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0),
                    # self.activation
                    # # nn.MaxPool2d(kernel_size=2, stride=2),
        self.encoder_dynamics = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features=256*4*4, out_features=self.latent_dim),
        ).to(device)

        # self.encoder_appearance = nn.Sequential(
        #             nn.Flatten(),
        #             nn.Linear(in_features=2*2*512, out_features=self.appearance_dim),
        # ).to(device)
        
        # print the number of parameters in the model
        print("Number of parameters in the encoder model: {}".format(np.sum([p.numel() for p in self.parameters() if p.requires_grad])))


    def forward(self, image):
        # print(image.shape)
        out = self.encoder(image)
        # print(out.shape)
        dyn = self.encoder_dynamics(out)
        stacked_tensor = dyn
        # appearance = self.encoder_appearance(out)

        # # In order to apply the class TimeDistributed
        # # dyn: N x D, appearance: N x A
        # stacked_tensor = torch.cat((dyn, appearance), dim=-1)

        return stacked_tensor

class Decoder(nn.Module):
    def __init__(self, device, latent_dim, out_channels,
                    activation=nn.ReLU()):
        super(Decoder, self).__init__()
        self.device = device
        # self.dynamics_dim = dynamics_dim
        # self.appearance_dim = appearance_dim
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.activation = activation

        self.decoder_linear = nn.Sequential(
                    nn.Linear(in_features=self.latent_dim, out_features=256*4*4),
                    self.activation
        ).to(device)

        self.decoder = nn.Sequential(
            # first block
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0),
                    self.activation,
                    nn.BatchNorm2d(128),
                    nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=0),
                    self.activation,
                    nn.BatchNorm2d(128),

            # second block
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0),
                    self.activation,
                    nn.BatchNorm2d(64),
                    nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=0),
                    self.activation,
                    nn.BatchNorm2d(64),
            
            # third block
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=0),
                    self.activation,
                    nn.BatchNorm2d(32),
                    nn.ConvTranspose2d(32, self.out_channels, kernel_size=4, stride=1, padding=0),
                    self.activation,
                #     nn.BatchNorm2d(32),

                #     nn.Conv2d(in_channels=32, out_channels=self.in_channels, kernel_size=1, stride=1, padding=1),
                    nn.Sigmoid()
                    

                    
        ).to(device)

                    # nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=0),
                    # self.activation,
                    # nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
                    # self.activation,
                    # nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
                    # self.activation,
                    # nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=1),
                    # self.activation,
                    # nn.ConvTranspose2d(in_channels=32, out_channels=self.in_channels, kernel_size=7, stride=2, padding=0),
                    # nn.Sigmoid()

        # print the number of parameters in the model
        print("Number of parameters in the decoder model: {}".format(np.sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, latent):

        out = self.decoder_linear(latent)
        # print(out.shape)
        out = out.view(latent.shape[0], 256, 4, 4)
        # print(out.shape)
        out = self.decoder(out)
        # print(out.shape)
        return out