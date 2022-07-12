import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from tqdm.notebook import trange

from .ae import ConvAE, Decoder, Encoder
from .anode import ANODENet
from .resnet import ResNetCustomEncoder, ResNetCustomDecoder, blockResNet


class ConvNode(nn.Module):
    def __init__(self, device, size, latent_dim, in_channels,
    ode_hidden_dim, ode_out_dim, augment_dim=0, time_dependent=False, 
    ode_non_linearity='relu', conv_activation=nn.ReLU(),latent_activation=None, stack_size=1):
        super(ConvNode, self).__init__()
        self.device = device
        self.size = size
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.conv_activation = conv_activation
        self.latent_activation = latent_activation
        self.ode_hidden_dim = ode_hidden_dim
        self.out_dim = ode_out_dim
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.ode_non_linearity = ode_non_linearity

        print("-"*50)
        print("Creating ConvAE...")
        self.ae = ConvAE(height=size, width=size, latent_dim=latent_dim, in_channels=in_channels, 
            activation=conv_activation, relu=latent_activation).to(device)

        print("-"*50)
        print("Creating ANODENet...")
        self.node = ANODENet(device, latent_dim*(stack_size + 1), ode_hidden_dim, ode_out_dim, augment_dim, time_dependent=False,
            non_linearity=ode_non_linearity).to(device)

    def forward(self, images, times, dt):
        # images: [n_stack, in_channels, height, width]
        # latent_z: [n_stack, latent_dim]
        latent_z = self.ae.encode(images)
        # latent_z_stack: [1, latent_dim*(n_stack+1)]
        # for the moment n_stack = 1
        latent_z_stack = torch.cat([latent_z[:-1], (latent_z[1:]-latent_z[:-1])/dt], dim=-1)

        # sim : [times, 1, n_stack*ode_out_dim]
        sim = self.node(latent_z_stack, times)[:,0, :latent_z.shape[-1]]

        reconstructed_images = self.ae.decode(sim)

        return reconstructed_images, sim

    def encode(self, images):

        return self.ae.encode(images)

    def decode(self, latent_z):
        return self.ae.decode(latent_z)


class ConvNodeWithBatch(nn.Module):
    def __init__(self, device, size, latent_dim, in_channels,
    ode_hidden_dim, ode_out_dim, augment_dim=0, time_dependent=False, ode_linear_layer=False,
    ode_non_linearity='relu', conv_activation=nn.ReLU(),latent_activation=None, stack_size=1):
        super(ConvNodeWithBatch, self).__init__()
        self.device = device
        self.size = size
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.conv_activation = conv_activation
        self.latent_activation = latent_activation
        self.ode_hidden_dim = ode_hidden_dim
        self.out_dim = ode_out_dim
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.ode_linear_layer = ode_linear_layer
        self.ode_non_linearity = ode_non_linearity

        print("-"*50)
        print("Creating ConvAE...")
        self.encoder = TimeDistributed(
            Encoder(device=device, latent_dim=latent_dim, in_channels=in_channels,
            activation=conv_activation, relu=latent_activation).to(device), 
            len_shape_without_batch=4, # input without batch are (times, channels, height, width)
            batch_first=True
        )
        self.decoder = TimeDistributed(
            Decoder(device=device, latent_dim=latent_dim, in_channels=in_channels,
            activation=conv_activation).to(device),
            len_shape_without_batch=2, # input without batch are (times, latent_dim)
            batch_first=True
        )

        print("-"*50)
        print("Creating ANODENet...")
        self.node = ANODENet(device, latent_dim*(stack_size + 1), ode_hidden_dim, ode_out_dim, augment_dim, time_dependent=False,
            non_linearity=ode_non_linearity, linear_layer=ode_linear_layer).to(device)

    def forward(self, images, times, dt):
        # images: [(batch), n_stack, in_channels, height, width]
        # latent_z: [n_stack, latent_dim]
        # print("input_images: ", images.shape)
        latent_z = self.encoder(images)
        # print("latent_z: ", latent_z.shape)
        
        # latent_z_stack: [(batch), n_stack, latent_dim*(n_stack+1)]
        # for the moment n_stack = 1
        if len(latent_z.shape) == 3:
            latent_z_stack = torch.cat([latent_z[:, :-1], (latent_z[:, 1:]-latent_z[:, :-1])/dt], dim=-1).squeeze(1)
        

        elif len(latent_z.shape) == 2:
            latent_z_stack = torch.cat([latent_z[:-1], (latent_z[1:]-latent_z[:-1])/dt], dim=-1)

        # print("latent_z_stack: ", latent_z_stack.shape)

        # sim : [times, (batch),ode_out_dim]
        sim = self.node(latent_z_stack, times)
        # print("sim: ", sim.shape)
        # sim : [(batch), n_stack, ode_out_dim]
        if len(images.shape) == 5:
            sim = sim.swapdims(0,1)
        else:
            sim = sim.squeeze(1)
        # print("sim: ", sim.shape)

        reconstructed_images = self.decoder(sim)
        # print("reconstructed_images: ", reconstructed_images.shape)

        return reconstructed_images, sim

    def encode(self, images):

        return self.encoder(images)

    def decode(self, latent_z):
        return self.decoder(latent_z)



class TimeDistributed(nn.Module):
    def __init__(self, module, len_shape_without_batch, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self._len_shape_without_batch = len_shape_without_batch
        self.batch_first = batch_first

    def forward(self, x):
        # x: [batch, time, *]
        assert len(x.shape) == self._len_shape_without_batch or self._len_shape_without_batch + 1, f"Input must have shape {self._len_shape_without_batch}D or {self._len_shape_without_batch + 1}D, received {len(x.shape)}D"

        if len(x.size()) == self._len_shape_without_batch:
            return self.module(x)

        batch_flatten_shapes = list(x.shape[1:])
        batch_flatten_shapes[0] = -1
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().reshape(batch_flatten_shapes)  # (samples * timesteps, input_size)
        # print("TimeDistributed: x_reshape: ", x_reshape.shape)
        y = self.module(x_reshape)
        # print("TimeDistributed: y: ", y.shape)

        # We have to reshape Y
        
        if self.batch_first:
            final_shapes = [x.shape[0], -1] + list(y.shape[1:])
            y = y.contiguous().view(final_shapes)  # (samples, timesteps, output_size)
        else:
            final_shapes = [-1, x.shape[1]] + list(y.shape[1:])
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        # print("TimeDistributed: y return: ", y.shape)


        # print('TimeDistributed: y return: ', y.shape)    
        return y


class ResNodeWithBatch(nn.Module):
    def __init__(self, device, size, latent_dim, in_channels, layers,
    ode_hidden_dim, ode_out_dim, augment_dim=0, time_dependent=False, ode_linear_layer=False,
    ode_non_linearity='relu', conv_activation=nn.ReLU(),latent_activation=None, stack_size=1):
        super(ResNodeWithBatch, self).__init__()
        self.device = device
        self.size = size
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.layers = layers
        self.conv_activation = conv_activation
        self.latent_activation = latent_activation
        self.ode_hidden_dim = ode_hidden_dim
        self.out_dim = ode_out_dim
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.ode_linear_layer = ode_linear_layer
        self.ode_non_linearity = ode_non_linearity

        print("-"*50)
        print("Creating ConvAE...")
        self.encoder = TimeDistributed(
            ResNetCustomEncoder(layers, in_channels, n_latent=latent_dim, expansion=4)
            .to(device),
            len_shape_without_batch=4, # input without batch are (times, latent_dim)
            batch_first=True
        )
        self.decoder = TimeDistributed(
            ResNetCustomDecoder(img_channel=in_channels, n_latent=latent_dim).to(device),
            len_shape_without_batch=2, # input without batch are (times, latent_dim)
            batch_first=True
        )

        print("-"*50)
        print("Creating ANODENet...")
        self.node = ANODENet(device, latent_dim*(stack_size + 1), ode_hidden_dim, ode_out_dim, augment_dim, time_dependent=False,
            non_linearity=ode_non_linearity, linear_layer=ode_linear_layer).to(device)

    def forward(self, images, times, dt):
        # images: [(batch), n_stack, in_channels, height, width]
        # latent_z: [n_stack, latent_dim]
        # print("input_images: ", images.shape)
        latent_z = self.encoder(images)
        # print("latent_z: ", latent_z.shape)
        
        # latent_z_stack: [(batch), n_stack, latent_dim*(n_stack+1)]
        # for the moment n_stack = 1
        if len(latent_z.shape) == 3:
            latent_z_stack = torch.cat([latent_z[:, :-1], (latent_z[:, 1:]-latent_z[:, :-1])/dt], dim=-1).squeeze(1)
        

        elif len(latent_z.shape) == 2:
            latent_z_stack = torch.cat([latent_z[:-1], (latent_z[1:]-latent_z[:-1])/dt], dim=-1)

        # print("latent_z_stack: ", latent_z_stack.shape)

        # sim : [times, (batch),ode_out_dim]
        sim = self.node(latent_z_stack, times)
        # print("sim: ", sim.shape)
        # sim : [(batch), n_stack, ode_out_dim]
        if len(images.shape) == 5:
            sim = sim.swapdims(0,1)
        else:
            sim = sim.squeeze(1)
        # print("sim: ", sim.shape)

        reconstructed_images = self.decoder(sim)
        # print("reconstructed_images: ", reconstructed_images.shape)

        return reconstructed_images, sim

    def encode(self, images):

        return self.encoder(images)

    def decode(self, latent_z):
        return self.decoder(latent_z)

class ResNodeWithBatch(nn.Module):
    def __init__(self, device, size, latent_dim, in_channels, layers,
    ode_hidden_dim, ode_out_dim, augment_dim=0, time_dependent=False, ode_linear_layer=False,
    ode_non_linearity='relu', conv_activation=nn.ReLU(),latent_activation=None, stack_size=1):
        super(ResNodeWithBatch, self).__init__()
        self.device = device
        self.size = size
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.layers = layers
        self.conv_activation = conv_activation
        self.latent_activation = latent_activation
        self.ode_hidden_dim = ode_hidden_dim
        self.out_dim = ode_out_dim
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.ode_linear_layer = ode_linear_layer
        self.ode_non_linearity = ode_non_linearity

        print("-"*50)
        print("Creating ConvAE...")
        self.encoder = TimeDistributed(
            ResNetCustomEncoder(layers, in_channels, n_latent=latent_dim, expansion=4)
            .to(device),
            len_shape_without_batch=4, # input without batch are (times, latent_dim)
            batch_first=True
        )
        self.decoder = TimeDistributed(
            ResNetCustomDecoder(img_channel=in_channels, n_latent=latent_dim).to(device),
            len_shape_without_batch=2, # input without batch are (times, latent_dim)
            batch_first=True
        )

        print("-"*50)
        print("Creating ANODENet...")
        self.node = ANODENet(device, latent_dim*(stack_size + 1), ode_hidden_dim, ode_out_dim, augment_dim, time_dependent=False,
            non_linearity=ode_non_linearity, linear_layer=ode_linear_layer).to(device)

    def forward(self, images, times, dt):
        # images: [(batch), n_stack, in_channels, height, width]
        # latent_z: [n_stack, latent_dim]
        # print("input_images: ", images.shape)
        latent_z = self.encoder(images)
        # print("latent_z: ", latent_z.shape)
        
        # latent_z_stack: [(batch), n_stack, latent_dim*(n_stack+1)]
        # for the moment n_stack = 1
        if len(latent_z.shape) == 3:
            latent_z_stack = torch.cat([latent_z[:, :-1], (latent_z[:, 1:]-latent_z[:, :-1])/dt], dim=-1).squeeze(1)
        

        elif len(latent_z.shape) == 2:
            latent_z_stack = torch.cat([latent_z[:-1], (latent_z[1:]-latent_z[:-1])/dt], dim=-1)

        # print("latent_z_stack: ", latent_z_stack.shape)

        # sim : [times, (batch),ode_out_dim]
        sim = self.node(latent_z_stack, times)
        # print("sim: ", sim.shape)
        # sim : [(batch), n_stack, ode_out_dim]
        if len(images.shape) == 5:
            sim = sim.swapdims(0,1)
        else:
            sim = sim.squeeze(1)
        # print("sim: ", sim.shape)

        reconstructed_images = self.decoder(sim)
        # print("reconstructed_images: ", reconstructed_images.shape)

        return reconstructed_images, sim

    def encode(self, images):

        return self.encoder(images)

    def decode(self, latent_z):
        return self.decoder(latent_z)


class LatentRegularizerLoss(nn.Module):
    def __init__(self, device, reg_lambda, step_decay=1, decay_rate=0.9):
        super(LatentRegularizerLoss, self).__init__()
        self.device = device
        self.reg_lambda = reg_lambda
        self.image_loss = nn.MSELoss()
        self.step_decay = step_decay
        self.decay_rate = decay_rate
        self._step = 0

    def forward(self, latent_z, pred_images, true_images):
        # latent_z: [batch, latent_dim]
        # pred_images: [batch, n_stack, in_channels, height, width]
        # true_images: [batch, n_stack, in_channels, height, width]
        loss_img = self.image_loss(pred_images, true_images)
        loss_reg = torch.linalg.norm(latent_z, ord=2, dim=-1).mean(dim=-1).mean(dim=-1)
        # print("loss_img: ", loss_img)
        # print("loss_reg: ", loss_reg)
        return loss_img + self.reg_lambda * loss_reg
    

    def step(self):
        self._step +=1
        if self._step % self.step_decay == 0:
            self.reg_lambda *= self.decay_rate
            

    def forward_print(self, latent_z, pred_images, true_images):
        # latent_z: [batch, latent_dim]
        # pred_images: [batch, n_stack, in_channels, height, width]
        # true_images: [batch, n_stack, in_channels, height, width]
        loss_img = self.image_loss(pred_images, true_images)
        loss_reg = torch.linalg.norm(latent_z, ord=2, dim=-1).mean(dim=-1).mean(dim=-1)
        print("-"*30, "Loss prints", "-"*30)
        print("loss_img: ", loss_img)
        print("loss_reg: ", self.reg_lambda * loss_reg)
        print("reg_lambda: ",self.reg_lambda)
        print("-"*73)
        return None
