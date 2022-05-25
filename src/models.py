import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from tqdm.notebook import trange


class ConvAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.height = kwargs['height']
        self.width = kwargs['width']
        self.latent_dim = kwargs['latent_dim']
        if not kwargs['in_channels'] is None:
            self.in_channels = kwargs['in_channels']
        else:
            self.in_channels = 3

        if not kwargs['relu'] is None:
            self.relu = True

        else:
            self.relu = False

        self.encoder = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=5, stride=2, padding=1),
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
                    nn.ConvTranspose2d(in_channels=32, out_channels=self.in_channels, kernel_size=4, stride=2, padding=1),
                    nn.Sigmoid()
        )

        # print the number of parameters in the model
        print("Number of parameters in the model: {}".format(np.sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, image):
        # print(image.shape)
        out = self.encoder(image)
        # print(out.shape)
        latent = self.encoder_linear(out)
        
        if self.relu:
            latent = torch.relu(latent)
        # print(latent.shape)
        out = self.decoder_linear(latent)
        # print(out.shape)
        out = out.view(image.shape[0], 128, 3, 3)
        # print(out.shape)
        out = self.decoder(out)
        # print(out.shape)
        return out

    def encode(self, image):
        out = self.encoder(image)
        out = self.encoder_linear(out)
        return out

    def decode(self, latent_vector):
        # print(latent_vector.shape)
        out = self.decoder_linear(latent_vector)
        out = out.view(out.shape[0], 128, 3, 3)
        out = self.decoder(out)
        return out

    def generate(self, n_samples=1):
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            img = self.decode(z)
            return img

    def plot(self, n_samples=1):
        self.eval()
        imgs = self.generate(n_samples)
        imgs = imgs[:,0].cpu().detach().numpy()
        imgs = np.reshape(imgs, (n_samples, 28, 28))
        fig, ax = plt.subplots(figsize=(10,5))
        for i in range(n_samples):
            plt.subplot(1, n_samples, i+1)
            plt.imshow(imgs[i], cmap='gray')
            plt.axis('off')
        plt.show()

    def reconstruction(self, plot_loader):
        self.eval()
        for i, data in enumerate(plot_loader):
            input_image, _ = data
            input_image = input_image.float()
            batch_size = input_image.shape[0]
            img = self(input_image)
            img = img[:,0].cpu().detach().numpy()
            img = np.reshape(img, (batch_size, 28, 28))
            input_image = input_image[:,0].cpu().detach().numpy()
            input_image = np.reshape(input_image, (batch_size, 28, 28))

            fig, ax = plt.subplots(figsize=(20,2))
            for i in range(batch_size):
                plt.subplot(2, batch_size, i+1)
                plt.imshow(img[i], cmap='gray')
                plt.axis('off')
                
            for i in range(batch_size):
                plt.subplot(2, batch_size, i+1+batch_size)
                plt.imshow(input_image[i], cmap='gray')
                plt.axis('off')
            plt.show()
            break

    def fit(self, dataloader_train, dataloader_test, optimizer, scheduler, criterion, epochs=10, display_step=1, n_plot=10):
        iterator = trange(epochs)
        losses_train = []
        losses_test = []
        for _ in iterator:
            self.plot(n_samples=n_plot)
            self.reconstruction(dataloader_test)
            
            loss_epoch = 0
            self.train()
            for i, data in enumerate(dataloader_train):
                input_image, _ = data
                input_image = input_image.float()
                optimizer.zero_grad()
                img = self(input_image)
                loss = criterion(input_image, img)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                if i % display_step == 0:
                    iterator.set_description(f'Batch: {i}/{len(dataloader_train)}, Loss: {loss_epoch/(i+1):.6f}')
                

            losses_train.append(loss_epoch/len(dataloader_train))
            # self.eval()
            # with torch.no_grad():
            #     loss_epoch = 0
            #     for i, data in enumerate(dataloader_test):
            #         input_image, _ = data
            #         input_image = input_image.float()
            #         img = self(input_image)
            #         loss = criterion(input_image, img)
            #         loss_epoch += loss.item()
            #         if i % display_step == 0:
            #             iterator.set_postfix_str(f'Test Batch: {i}/{len(dataloader_test)}, Loss: {loss_epoch/(i+1):.6f}')

            #     losses_test.append(loss_epoch/len(dataloader_test))
            
            scheduler.step()

        return losses_train, losses_test

class ConvAEwithResNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.height = kwargs['height']
        self.width = kwargs['width']
        self.latent_dim = kwargs['latent_dim']

        self.encoder = nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1], 
                                            nn.Flatten())
        self.encoder_linear = nn.Linear(512, self.latent_dim)

        self.decoder_linear = nn.Sequential(
                    nn.Linear(in_features=self.latent_dim, out_features=3*3*128),
                    nn.ReLU()
        )
        self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=0),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
                    nn.Sigmoid()
        )

        # print the number of parameters in the model
        print("Number of parameters in the model: {}".format(np.sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, image):
        # print(image.shape)
        out = self.encoder(image)
        # print(out.shape)
        latent = self.encoder_linear(out)
        # print(latent.shape)
        out = self.decoder_linear(latent)
        # print(out.shape)
        out = out.view(image.shape[0], 128, 3, 3)
        # print(out.shape)
        out = self.decoder(out)
        # print(out.shape)
        return out

    def encode(self, image):
        out = self.encoder(image)
        out = self.encoder_linear(out)
        return out

    def decode(self, latent_vector):
        # print(latent_vector.shape)
        out = self.decoder_linear(latent_vector)
        out = out.view(out.shape[0], 128, 3, 3)
        out = self.decoder(out)
        return out

    def generate(self, n_samples=1):
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            img = self.decode(z)
            return img

    def plot(self, n_samples=1):
        self.eval()
        imgs = self.generate(n_samples)
        imgs = imgs[:,0].cpu().detach().numpy()
        imgs = np.reshape(imgs, (n_samples, 28, 28))
        fig, ax = plt.subplots(figsize=(10,5))
        for i in range(n_samples):
            plt.subplot(1, n_samples, i+1)
            plt.imshow(imgs[i], cmap='gray')
            plt.axis('off')
        plt.show()

    def reconstruction(self, plot_loader):
        self.eval()
        for i, data in enumerate(plot_loader):
            input_image, _ = data
            input_image = input_image.float()
            batch_size = input_image.shape[0]
            img = self(input_image)
            img = img[:,0].cpu().detach().numpy()
            img = np.reshape(img, (batch_size, 28, 28))
            input_image = input_image[:,0].cpu().detach().numpy()
            input_image = np.reshape(input_image, (batch_size, 28, 28))

            fig, ax = plt.subplots(figsize=(20,2))
            for i in range(batch_size):
                plt.subplot(2, batch_size, i+1)
                plt.imshow(img[i], cmap='gray')
                plt.axis('off')
                
            for i in range(batch_size):
                plt.subplot(2, batch_size, i+1+batch_size)
                plt.imshow(input_image[i], cmap='gray')
                plt.axis('off')
            plt.show()
            break

    def fit(self, dataloader_train, dataloader_test, optimizer, scheduler, criterion, epochs=10, display_step=1, n_plot=10):
        iterator = trange(epochs)
        losses_train = []
        losses_test = []
        for _ in iterator:
            self.plot(n_samples=n_plot)
            self.reconstruction(dataloader_test)
            
            loss_epoch = 0
            self.train()
            for i, data in enumerate(dataloader_train):
                input_image, _ = data
                input_image = input_image.float()
                optimizer.zero_grad()
                img = self(input_image)
                loss = criterion(input_image, img)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                if i % display_step == 0:
                    iterator.set_description(f'Batch: {i}/{len(dataloader_train)}, Loss: {loss_epoch/(i+1):.6f}')
                

            losses_train.append(loss_epoch/len(dataloader_train))
            # self.eval()
            # with torch.no_grad():
            #     loss_epoch = 0
            #     for i, data in enumerate(dataloader_test):
            #         input_image, _ = data
            #         input_image = input_image.float()
            #         img = self(input_image)
            #         loss = criterion(input_image, img)
            #         loss_epoch += loss.item()
            #         if i % display_step == 0:
            #             iterator.set_postfix_str(f'Test Batch: {i}/{len(dataloader_test)}, Loss: {loss_epoch/(i+1):.6f}')

            #     losses_test.append(loss_epoch/len(dataloader_test))
            
            scheduler.step()

        return losses_train, losses_test


class custom_loss(nn.Module):
    def __init__(self):
        super(custom_loss, self).__init__()
    
    def forward(self, x, dec):
        # dec = dec[:,0]
        # x = x[:,0]
        unreshaped = torch.reshape(dec, [-1, 28*28])
        x = torch.reshape(x, [-1, 28*28])
        # print(unreshaped.shape)
        img_loss = torch.mean(torch.sum((unreshaped - x)**2, dim=1))
        # print(img_loss.shape)
        loss = torch.mean(img_loss)
        # print(loss.shape)
        return loss