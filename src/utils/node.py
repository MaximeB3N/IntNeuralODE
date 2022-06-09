import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from .viz import display_ode_trajectory



class batchGetterPositions:
    def __init__(self, batch_time, n_samples, total_length, dt, positions, frac_train, noise=-1):
        self.times = torch.linspace(0., total_length*dt, total_length, dtype=torch.float64).float()
        
        if isinstance(positions, torch.Tensor):
            self.true_positions = positions.float()

        elif isinstance(positions, np.ndarray):
            self.true_positions = torch.from_numpy(positions).float()

        else:
            assert False, "positions must be either a torch.Tensor or a np.ndarray"

        self.noise = noise
        self.N_train = int(positions.shape[0]*frac_train)
        if noise > 0 and noise < 1:
            # adding gaussian noise to the true positions
            self.true_positions = self.true_positions + torch.normal(0, noise, size=self.true_positions.shape)
            self.true_positions = self.true_positions.float()

        self.train_times = self.times[:self.N_train]
        self.test_times = self.times[self.N_train:]
        self.train_positions = self.true_positions[:self.N_train]
        self.test_positions = self.true_positions[self.N_train:]
        self.n_samples = n_samples
        self.batch_time = batch_time
        self.dt = dt
        self.total_length = total_length

    def get_batch(self):
        s = torch.from_numpy(np.random.choice(np.arange(self.N_train - self.batch_time, dtype=np.int64), self.n_samples, replace=False))
        batch_y0 = self.train_positions[s]  # (M, D)
        batch_t = self.train_times[:self.batch_time]  # (T)
        batch_y = torch.stack([self.train_positions[s + i] for i in range(self.batch_time)], dim=0)  # (T, M, D)
        return batch_y0, batch_t, batch_y

class BatchGetterMultiTrajectories:
    def __init__(self, batch_time, n_samples, total_length, dt, positions, frac_train):
        # N: number of trajectories
        # M: number of time steps
        # D: dimension of the state space
        # positions: (N, T, D)
        self.times = torch.linspace(0., total_length*dt, total_length, dtype=torch.float64).float()
        if isinstance(positions, torch.Tensor):
            self.true_positions = positions.float()

        elif isinstance(positions, np.ndarray):
            self.true_positions = torch.from_numpy(positions).float()

        else:
            assert False, "positions must be either a torch.Tensor or a np.ndarray"

        self.N_train = int(positions.shape[0]*frac_train)

        self.train_times = self.times #[:self.N_train]
        self.test_times = self.times #[self.N_train:]
        self.train_positions = self.true_positions[:self.N_train]
        self.test_positions = self.true_positions[self.N_train:]
        self.n_samples = n_samples
        self.batch_time = batch_time
        self.dt = dt
        self.total_length = total_length

    def get_batch(self):
        index = np.random.randint(0, self.N_train, self.n_samples)
        s = torch.from_numpy(np.random.choice(np.arange(self.train_times.shape[0] - self.batch_time, dtype=np.int64), self.n_samples, replace=False))
        batch_y0 = self.train_positions[index, s]  # (M, D)
        batch_t = self.train_times[:self.batch_time]  # (T)
        batch_y = torch.stack([self.train_positions[index, s + i] for i in range(self.batch_time)], dim=0)  # (T, M, D)
        return batch_y0, batch_t, batch_y

         
class BatchGetterMultiImages:
    def __init__(self, batch_time, batch_size, n_stack, total_length, dt, images, frac_train):
        # N: number of trajectories
        # M: number of time steps
        # D: dimension of the state space
        # positions: (N, T, D)
        self.times = torch.linspace(0., total_length*dt, total_length, dtype=torch.float64).float()
        if isinstance(images, torch.Tensor):
            self.true_images = images.float()

        elif isinstance(images, np.ndarray):
            self.true_images = torch.from_numpy(images).float()

        else:
            assert False, "positions must be either a torch.Tensor or a np.ndarray"

        self.N_train = int(images.shape[0]*frac_train)

        self.train_times = self.times #[:self.N_train]
        self.test_times = self.times #[self.N_train:]
        self.train_images = self.true_images[:self.N_train]
        self.test_images = self.true_images[self.N_train:]
        self.batch_size = batch_size
        self.n_stack = n_stack
        self.batch_time = batch_time
        self.dt = dt
        self.total_length = total_length

    
    def get_batch(self):
        index = np.random.randint(0, self.N_train, self.batch_size)
        s = torch.from_numpy(np.random.choice(np.arange(self.train_times.shape[0] - self.batch_time, dtype=np.int64), 1, replace=False))
        batch_y0 = self.train_images[index, s:s+self.n_stack+1].squeeze(0) # (M, D)
        batch_t = self.train_times[:self.batch_time]  # (T)
        batch_y = torch.stack([self.train_images[index, s + i] for i in range(self.batch_time)], dim=1).squeeze(1)  # (T, M, D)
        return batch_y0, batch_t, batch_y


def train(model, optimizer, scheduler, epochs, batch_size, getter, display=100, loss_fn=None, display_results_fn=display_ode_trajectory, out_display=-1):
    
    if out_display == -1:
        out_display = model.out_dim

    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    iterator = trange(1, epochs+1)
    # just for the plot part
    running_loss = 0.
    for i in iterator:
        # get a random time sample
        model.train()
        loss = 0.
        for _ in range(batch_size):
            batch_init_positions, batch_times, batch_true_positions = getter.get_batch()
            # compute the output of the model
            out = model(batch_init_positions, batch_times)
            # compute the loss
            # print(out.shape, out.view(-1, batch_init_positions.shape[-1]).shape)
            # print(batch_true_positions.shape, batch_true_positions.view(-1, batch_init_positions.shape[-1]).shape)
            loss += loss_fn(out[:], batch_true_positions[:])
            # .view(-1,batch_init_positions.shape[-1])
        loss /= batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the progress bar
        iterator.set_postfix_str(f'Loss: {loss.item():.8f}')
        running_loss += loss.item()

        if i % display == 0:
           display_results_fn(i, model, out_display, getter, getter.total_length, getter.dt)
           iterator.set_description_str(f'Display loss: {running_loss/display:.8f}')
           running_loss = 0.


        scheduler.step()
        
    return None

def train_convnode(model, optimizer, scheduler, epochs, batch_size, getter, display=100, loss_fn=None, display_results_fn=display_ode_trajectory, out_display=-1):
    
    if out_display == -1:
        out_display = model.out_dim

    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    iterator = trange(1, epochs+1)
    # just for the plot part
    running_loss = 0.
    for i in iterator:
        # get a random time sample
        model.train()
        loss = 0.
        for _ in range(batch_size):
            batch_init_images, batch_times, batch_true_images = getter.get_batch()
            # compute the output of the model
            out_images, _ = model(batch_init_images, batch_times, getter.dt)
            # compute the loss
            # print(out.shape, out.view(-1, batch_init_positions.shape[-1]).shape)
            # print(batch_true_positions.shape, batch_true_positions.view(-1, batch_init_positions.shape[-1]).shape)
            # print(out_images.shape, batch_true_images.shape)
            loss += loss_fn(out_images[:], batch_true_images[:])
            # .view(-1,batch_init_positions.shape[-1])
        loss /= batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the progress bar
        iterator.set_postfix_str(f'Loss: {loss.item():.8f}')
        running_loss += loss.item()

        if i % display == 0:
           display_results_fn(i, model, out_display, getter, getter.total_length, getter.dt)
           iterator.set_description_str(f'Display loss: {running_loss/display:.8f}')
           running_loss = 0.


        scheduler.step()
        
    return None


def train_convnode_with_batch(model, optimizer, scheduler, epochs, getter, display=100, loss_fn=None, display_results_fn=display_ode_trajectory, out_display=-1):
    
    device = model.device

    if out_display == -1:
        out_display = model.out_dim

    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    iterator = trange(1, epochs+1)
    # just for the plot part
    running_loss = 0.
    for i in iterator:
        # get a random time sample
        model.train()
        loss = 0.
        batch_init_images, batch_times, batch_true_images = getter.get_batch()
        batch_init_images = batch_init_images.to(device)
        batch_times = batch_times.to(device)
        batch_true_images = batch_true_images.to(device)
        # compute the output of the model
        out_images, latent = model(batch_init_images, batch_times, getter.dt)
        # compute the loss
        # print(out.shape, out.view(-1, batch_init_positions.shape[-1]).shape)
        # print(batch_true_positions.shape, batch_true_positions.view(-1, batch_init_positions.shape[-1]).shape)
        # print(out_images.shape, batch_true_images.shape)
        loss += loss_fn(latent, out_images[:], batch_true_images[:])
        # .view(-1,batch_init_positions.shape[-1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the progress bar
        iterator.set_postfix_str(f'Loss: {loss.item():.8f}')
        running_loss += loss.item()

        scheduler.step()
        loss_fn.step()

        if i % display == 0:
           display_results_fn(i, model, out_display, getter, getter.total_length, getter.dt)
           iterator.set_description_str(f'Display loss: {running_loss/display:.8f}')
           running_loss = 0.
           loss_fn.forward_print(latent, out_images[:], batch_true_images[:])
        
    return None