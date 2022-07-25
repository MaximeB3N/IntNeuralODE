import torch
import torch.nn as nn

from .anode import ANODENet


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

class ConvNodeApproxVelocity(nn.Module):
    def __init__(self, device, encoder, decoder, size, latent_dim,
    ode_hidden_dim, ode_out_dim, augment_dim=0, ode_linear_layer=False,
    ode_non_linearity='relu', stack_size=1):
        super(ConvNodeApproxVelocity, self).__init__()
        self.device = device
        self.size = size
        self.latent_dim = latent_dim
        self.ode_hidden_dim = ode_hidden_dim
        self.out_dim = ode_out_dim
        self.augment_dim = augment_dim
        self.ode_linear_layer = ode_linear_layer
        self.ode_non_linearity = ode_non_linearity

        print("-"*50)
        print("Creating ConvAE...")
        self.encoder = TimeDistributed(
            encoder.to(device),
            len_shape_without_batch=4, # input without batch are (times, latent_dim)
            batch_first=True
        ).to(device)
        self.decoder = TimeDistributed(
            decoder,
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


class ConvNodeAppearance(nn.Module):
    def __init__(self, device, encoder, decoder, layers, dynamics_dim, appearance_dim, in_channels, out_channels,
    ode_hidden_dim, ode_out_dim, augment_dim=0, time_dependent=False, ode_linear_layer=False,
    ode_non_linearity='relu', conv_activation=nn.ReLU(),latent_activation=None):
    
        super(ConvNodeAppearance, self).__init__()
        self.device = device
        self.layers = layers
        self.dynamics_dim = dynamics_dim
        self.appearance_dim = appearance_dim
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
        print("Creating ResNetAE...")
        self.encoder = encoder.to(device)
            
        self.decoder = TimeDistributed(
            decoder,
            len_shape_without_batch=2, # input without batch are (times, latent_dim)
            batch_first=True
        ).to(device)

        print("-"*50)
        print("Creating ANODENet...")
        self.node = ANODENet(device, 2*dynamics_dim, ode_hidden_dim, ode_out_dim, augment_dim, time_dependent=False,
            non_linearity=ode_non_linearity, linear_layer=ode_linear_layer).to(device)

    def forward(self, image_inputs, times):
        # images: [(batch), n_stack, in_channels, height, width]
        # latent_z: [n_stack, latent_dim]
        # print("input_images: ", images.shape)
        latent_z = self.encoder(image_inputs)
        # latent_dyn: [batch, n_stack, dynamics_dim]
        # print("latent_z shape", latent_z.shape)
        latent_dynamics = latent_z[..., :2*self.dynamics_dim]
        # print("latent_dynamics shape", latent_dynamics.shape)
        # latent_app: [batch, n_stack, appearance_dim]
        latent_appearance = latent_z[..., 2*self.dynamics_dim:].unsqueeze(1)
   
        sim = self.node(latent_dynamics, times)
        # print("sim shape", sim.shape)
        # print("sim: ", sim.shape)
        # sim : [(batch), n_stack, ode_out_dim]
        if len(image_inputs.shape) == 4:
            sim = sim.swapdims(0,1)
        else:
            sim = sim.squeeze(1)
        # print("sim: ", sim.shape)

        # add the latent_appearance to the sim to reconstruct
        # print("sim shape", sim.shape)
        # print("before", latent_appearance.shape)
        latent_appearance = latent_appearance.repeat(1, sim.shape[1], 1)
        # print("appearance shape", latent_appearance.shape)
        # print("after shape", latent_appearance.shape)
        # print("dynamics", sim.shape)

        latent_out = torch.cat([sim, latent_appearance], dim=-1)

        # print("latent_out: ", latent_out.shape)
        # print(latent_out)
        reconstructed_images = self.decoder(latent_out)
        # print("reconstructed_images: ", reconstructed_images.shape)

        return reconstructed_images, sim

    def forward_diff_appearance(self, images_dyn, images_app, times, dt):
        # Dynamics
        latent_z_dyn = self.encoder(images_dyn)
        latent_dynamics = latent_z_dyn[..., :2*self.dynamics_dim]
        
        # Appearance
        latent_z_app = self.encoder(images_app)
        latent_appearance = latent_z_app[..., 2*self.dynamics_dim:].unsqueeze(1)

        # sim : [times, (batch),ode_out_dim]
        sim = self.node(latent_dynamics, times)
        # print("sim: ", sim.shape)
        # sim : [(batch), n_stack, ode_out_dim]
        if len(images_dyn.shape) == 4:
            sim = sim.swapdims(0,1)
        else:
            sim = sim.squeeze(1)
        # print("sim: ", sim.shape)

        # add the latent_appearance to the sim to reconstruct
        # print("sim shape", sim.shape)
        latent_appearance = latent_appearance.repeat(1, sim.shape[1], 1)

        latent_out = torch.cat([sim, latent_appearance], dim=-1)

        reconstructed_images = self.decoder(latent_out)
        # print("reconstructed_images: ", reconstructed_images.shape)

        return reconstructed_images, sim

    def encode(self, images):

        return self.encoder(images)

    def decode(self, latent_z):
        return self.decoder(latent_z)