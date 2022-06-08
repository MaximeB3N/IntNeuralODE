import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import trange

from src.data.box import GravityHoleBall
from src.data.generate import generate_gravity_hole_ball_images, add_average_velocity

from src.utils.utils import add_spatial_encoding, gaussian_density
from src.utils.node import  BatchGetterMultiImages, train_convnode
from src.utils.viz import display_convnode_trajectory

from src.models.ae import ConvAE
from src.models.node import ODEnetSimple
from src.models.anode import ANODENet
from src.models.convnode import ConvNode


MARGIN_MIN = 5
MIN_INIT_VELOCITY = 200.
WIDTH, HEIGHT = 28, 28
RADIUS = 3

infos = {
    "MARGIN_MIN":MARGIN_MIN,
    "MIN_INIT_VELOCITY":MIN_INIT_VELOCITY,
    "WIDTH":WIDTH,
    "HEIGHT":HEIGHT,
    "RADIUS":RADIUS
}

x = WIDTH/4.
y = HEIGHT/4.
vx = 0.
vy = 0.

box = GravityHoleBall(x, y, vx, vy, (WIDTH, HEIGHT),RADIUS)


Num_pos_velocity = 1
N = 500
N_frames = 300 + Num_pos_velocity
dt = 1./N_frames

times = np.arange(0, N_frames*dt, dt)

# encoded_trajectory = generate_gravity_hole_ball_positions(box, N=N, N_frames=N_frames, dt=dt)[:,:,:]
# print(encoded_trajectory.shape)
print("-"*50)
print("Generating images...")
images = generate_gravity_hole_ball_images(box, N=N, N_frames=N_frames, dt=dt, infos=infos).reshape(-1, 1, HEIGHT, WIDTH)
print(images.shape)
# dataset = [(image, 0) for image in dataset]
# dataset = add_spatial_encoding(dataset)
# print(len(dataset), len(dataset[0]), dataset[0][0].shape)
images = torch.from_numpy(add_spatial_encoding(images)).float().reshape(N, -1, 3, HEIGHT, WIDTH)
print(images.shape)

print("-"*50)
print("Creating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size = HEIGHT
latent_dim = 25
in_channels = 3
ode_data_dim = 25
ode_hidden_dim = 128
augment_dim=0
time_dependent=False
ode_non_linearity='relu' 
conv_activation=nn.ReLU()
latent_activation=None
stack_size=1

conv_ode = ConvNode(device, size, latent_dim, in_channels,
    ode_hidden_dim, ode_data_dim, augment_dim=augment_dim, time_dependent=time_dependent,
    ode_non_linearity=ode_non_linearity, conv_activation=conv_activation,
    latent_activation=latent_activation, stack_size=stack_size)


print("-"*50)
print("Creating tools to train...")
batch_size = 16
batch_time = 200
n_stack = 1
total_length = N_frames - Num_pos_velocity
n_samples = 1
getter = BatchGetterMultiImages(batch_time, n_samples, n_stack, total_length, dt, images, frac_train=1.)

optimizer = torch.optim.Adam(conv_ode.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
loss_fn = nn.MSELoss()

print("-"*50)
print("Training...")
batch_size = 8
epochs = 10000
root = "images/AE_ODE/Gravity/MultiTrajectories/Together/"
name = "conv_ode_1_ball_latent_{}_hidden_ode_{}_stack_{}_conv_activation_{}".format(latent_dim, ode_hidden_dim, stack_size, conv_activation)

# to be lambdified
# def display_fn(i, model, out_display, getter, final_time, dt):
#     display_convnode_trajectory(i, model, out_display, getter, final_time, dt, root=root, name=name)

display_fn = lambda i, model, out_display, getter, final_time, dt: display_convnode_trajectory(i, model, out_display, getter, final_time, dt, root=root, name=name)
train_convnode(conv_ode, optimizer, scheduler, epochs, batch_size,
    getter, loss_fn=loss_fn, display=200, display_results_fn=display_fn)

print("-"*50)
print("Saving model...")
torch.save(conv_ode.state_dict(), 
"models/AE_ODE/ConvODE/conv_ode_1_ball_latent_{}_hidden_ode_{}_stack_{}_conv_activation_{}.pt".format(latent_dim, ode_hidden_dim, stack_size, conv_activation))
