from random import gauss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# import the MNIST dataset
import torchvision.datasets as datasets
from torchvision import transforms

import torchsummary

from sklearn.model_selection import train_test_split

# from torchdiffeq import odeint, odeint_adjoint

import cv2
import numpy as np

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from tqdm.notebook import trange



def gaussian_density(x, mu, sigma):
    return torch.exp(-torch.norm(torch.tensor(x - mu).float(), dim=1)**2/(2*sigma**2))/(2*np.pi*sigma**2)

def create_gaussian_dataset(r_min, r_max, n_samples, size, margin=1., n_balls=1):
    samples = []
    indices_matrix = np.array([[i,j] for i in range(size) for j in range(size)])
    # print(indices_matrix)
    # print(indices_matrix.shape)
    eps = 1E-5

    for i in range(n_samples):
        image = np.zeros((1, size, size))
        for _ in range(n_balls):
            sigma = np.random.uniform(r_min, r_max+eps)
            # create a gaussian ball
            mu = np.random.uniform(low=margin, high=size - 1 - margin, size=(2))
            # mu = np.random.randint(size, size=(2))
            # compute the density over the image and normalize it
            single_image = gaussian_density(indices_matrix, mu, sigma).numpy().copy()
            image += single_image.reshape(1, size, size)
        # standardization of the image
        image = (image - image.min()) / (image.max() - image.min())
        # image = (image - image.mean()) / (image.std() + eps)
        # image = (image - image.min()) / (image.max() - image.min()) 
        samples.append([image.reshape(1, size, size), np.array([mu[0]/(size ), mu[1]/(size)])])
        
    return samples

def add_spatial_encoding(gaussian_dataset):
    n_images = len(gaussian_dataset)
    size = gaussian_dataset[0][0].shape[1]
    # print(size)

    # create the spatial encoding
    # create the x encoding
    x_encoding = np.linspace(np.zeros(size), np.ones(size), size, dtype=np.float64, axis=1)
    y_encoding = np.linspace(np.zeros(size), np.ones(size), size, dtype=np.float64, axis=0)

    samples = []

    for i in range(n_images):
        new_image = np.stack([gaussian_dataset[i][0].squeeze(), x_encoding, y_encoding], axis=0)
        samples.append([new_image, gaussian_dataset[i][1]])

    return samples

def stack_dataset(gaussian_dataset):
    n_images = len(gaussian_dataset)
    size = gaussian_dataset[0][0].shape[1]

    samples = []

    for i in range(n_images):
        old_image = np.array(gaussian_dataset[i][0].squeeze())
        new_image = np.stack([old_image.copy(), old_image.copy(), old_image.copy()], axis=0)
        samples.append([new_image, gaussian_dataset[i][1]])

    return samples

class ConvAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.height = kwargs['height']
        self.width = kwargs['width']
        self.latent_dim = kwargs['latent_dim']

        self.encoder = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1),
                    nn.ReLU(),
                    # nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    # nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU()
                    # nn.MaxPool2d(kernel_size=2, stride=2),

        )
        
        self.encoder_linear = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features=3*3*128, out_features=self.latent_dim),
                    nn.ReLU()
        )

        self.decoder_linear = nn.Sequential(
                    nn.Linear(in_features=self.latent_dim, out_features=3*3*128),
                    nn.ReLU()
        )
        self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=0),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),
                    nn.Sigmoid()
        )

        # print the number of parameters in the model
        print("Number of parameters in the model: {}".format(np.sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, image):
        out = self.encoder(image)
        latent = self.encoder_linear(out)
        out = self.decoder_linear(latent)
        out = out.view(image.shape[0], 128, 3, 3)
        out = self.decoder(out)
        return out, latent

    def encode(self, image):
        out = self.encoder(image)
        return out

    def decode(self, latent_vector):
        out = self.decoder(latent_vector)
        return out
