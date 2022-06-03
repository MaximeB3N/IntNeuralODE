import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from tqdm import trange


class EncoderResNet(nn.Module):
    def __init__(self, n_latent):
        super(EncoderResNet, self).__init__()
        # remove the classification layer (last layer)
        self.encoder_model = nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1], 
                                            nn.Flatten())
        self.mean = nn.Linear(512, n_latent)
        self.sd = nn.Linear(512, n_latent)

    def forward(self, X_in):
        latent = self.encoder_model(X_in)
        # print(latent.shape)
        mn = self.mean(latent)
        sd = self.sd(latent)*0.5
        epsilon = torch.randn(mn.shape)
        z = mn + epsilon * torch.exp(sd)
        return z, mn, sd

class Encoder(nn.Module):
    def __init__(self, n_latent, in_channels):
        super(Encoder, self).__init__()
        activation = nn.LeakyReLU(0.3)
        
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=5, stride=2, padding=1),
                    activation,
                    # nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                    activation,
                    # nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                    activation,
                    # nn.MaxPool2d(kernel_size=2, stride=2),

        )
        
        self.encoder_linear = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features=3*3*128, out_features=512),
        )

        self.encoder_model = nn.Sequential(
                    self.encoder,
                    self.encoder_linear
        )
        self.mean = nn.Linear(512, n_latent)
        self.sd = nn.Linear(512, n_latent)

    def forward(self, X_in):
        # print(self.encoder(X_in).shape)
        latent = self.encoder_model(X_in)
        # print(latent.shape)
        mn = self.mean(latent)
        sd = self.sd(latent)*0.5
        epsilon = torch.randn(mn.shape)
        z = mn + epsilon * torch.exp(sd)
        return z, mn, sd

class Decoder(nn.Module):
    def __init__(self, n_latent, dec_in_channels):
        super(Decoder, self).__init__()

        self.n_latent = n_latent
        self.in_channels = dec_in_channels
        self.reshaped_dim = [-1, 7, 7, dec_in_channels]
        self.inputs_decoder = int(49 * dec_in_channels / 2)

        self.linear = nn.Sequential(
            nn.Linear(n_latent, self.inputs_decoder),
            nn.LeakyReLU(0.3),
            nn.Linear(self.inputs_decoder, self.inputs_decoder * 2 + 1),
            nn.LeakyReLU(0.3))
        self.decoder_model = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.last_linear = nn.Sequential(
            nn.Linear(3*28*28, self.in_channels*28*28),
            nn.Sigmoid()
        )
    def forward(self, sampled_z):
        decoded = self.linear(sampled_z)
        # print(decoded.shape)
        decoded = decoded.view(-1, self.in_channels, 7, 7)
        # print(decoded.shape)
        img = self.decoder_model(decoded)
        # print("out decoder", img.shape)
        img = self.last_linear(img)
        # print(img.shape)
        img = img.view(-1, self.in_channels, 28, 28)
        return img


class VAE(nn.Module):
    def __init__(self, n_latent=7, in_channels=1, dec_in_channels=1):
        super(VAE, self).__init__()
        
        self.n_latent = n_latent
        self.dec_in_channels = dec_in_channels
        self.in_channels = in_channels

        self.encoder = Encoder(n_latent, in_channels)
        self.decoder = Decoder(n_latent, dec_in_channels)

    def forward(self, X_in):
        z, mn, sd = self.encoder(X_in)
        img = self.decoder(z)
        return img, mn, sd

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
                img, mn, sd = self(input_image)
                loss = criterion(input_image, img, mn, sd)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                if i % display_step == 0:
                    iterator.set_description(f'Batch: {i}/{len(dataloader_train)}, Loss: {loss_epoch/(i+1):.6f}')
                

            losses_train.append(loss_epoch/len(dataloader_train))
            self.eval()
            with torch.no_grad():
                loss_epoch = 0
                for i, data in enumerate(dataloader_test):
                    input_image, _ = data
                    input_image = input_image.float()
                    img, mn, sd = self(input_image)
                    loss = criterion(input_image, img, mn, sd)
                    loss_epoch += loss.item()
                    if i % display_step == 0:
                        iterator.set_postfix_str(f'Test Batch: {i}/{len(dataloader_test)}, Loss: {loss_epoch/(i+1):.6f}')

                losses_test.append(loss_epoch/len(dataloader_test))
            
            scheduler.step()

        return losses_train, losses_test

    def generate(self, n_samples=1):
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.n_latent)
            img = self.decoder(z)
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
            img, _, _ = self(input_image)
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
                

class VAEwithResNet(nn.Module):
    def __init__(self, n_latent, dec_in_channels):
        super(VAEwithResNet, self).__init__()
        
        self.n_latent = n_latent
        self.dec_in_channels = dec_in_channels

        self.encoder = EncoderResNet(n_latent)
        self.decoder = Decoder(n_latent, dec_in_channels)

    def forward(self, X_in):
        z, mn, sd = self.encoder(X_in)
        img = self.decoder(z)
        return img, mn, sd

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
                img, mn, sd = self(input_image)
                loss = criterion(input_image, img, mn, sd)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                if i % display_step == 0:
                    iterator.set_description(f'Batch: {i}/{len(dataloader_train)}, Loss: {loss_epoch/(i+1):.6f}')
                

            losses_train.append(loss_epoch/len(dataloader_train))
            self.eval()
            with torch.no_grad():
                loss_epoch = 0
                for i, data in enumerate(dataloader_test):
                    input_image, _ = data
                    input_image = input_image.float()
                    img, mn, sd = self(input_image)
                    loss = criterion(input_image, img, mn, sd)
                    loss_epoch += loss.item()
                    if i % display_step == 0:
                        iterator.set_postfix_str(f'Test Batch: {i}/{len(dataloader_test)}, Loss: {loss_epoch/(i+1):.6f}')

                losses_test.append(loss_epoch/len(dataloader_test))
            
            scheduler.step()

        return losses_train, losses_test

    def generate(self, n_samples=1):
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.n_latent)
            img = self.decoder(z)
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
            img, _, _ = self(input_image)
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
                

class custom_loss(nn.Module):
    def __init__(self):
        super(custom_loss, self).__init__()
    
    def forward(self, x, dec, mu, sd):
        # dec = dec[:,0]
        # x = x[:,0]
        unreshaped = torch.reshape(dec, [-1, 28*28])
        x = torch.reshape(x, [-1, 28*28])
        # print(unreshaped.shape)
        img_loss = torch.mean(torch.sum((unreshaped - x)**2, dim=1))
        # print(img_loss.shape)
        latent_loss = -0.5 * torch.mean(1 + 2*sd - mu**2 - torch.exp(2*sd), dim=1)
        # print(latent_loss.shape)
        loss = torch.mean(img_loss + latent_loss)
        # print(loss.shape)
        return loss

