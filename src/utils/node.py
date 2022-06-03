import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from ..viz import display_results



class batchGetter:
    def __init__(self, batch_time, n_samples, total_length, dt, positions, frac_train, noise=-1):
        self.times = torch.linspace(0., total_length*dt, total_length, dtype=torch.float64).float()
        self.true_positions = torch.tensor(positions, dtype=torch.float64).float()
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


# class batchGetterMultiTrajectories:
#     def __init__(self, batch_time, n_samples, total_length, dt, positions, frac_train, noise=-1):
#         self.times = torch.linspace(0., total_length*dt, total_length, dtype=torch.float64).float()
#         self.true_positions = torch.tensor(positions, dtype=torch.float64).float()
#         self.noise = noise
#         self.N_train = int(positions.shape[0]*frac_train)
#         if noise > 0 and noise < 1:
#             # adding gaussian noise to the true positions
#             self.true_positions = self.true_positions + torch.normal(0, noise, size=self.true_positions.shape)
#             self.true_positions = self.true_positions.float()

#         self.train_times = self.times[:self.N_train]
#         self.test_times = self.times[self.N_train:]
#         self.train_positions = self.true_positions[:self.N_train]
#         self.test_positions = self.true_positions[self.N_train:]
#         self.n_samples = n_samples
#         self.batch_time = batch_time
#         self.dt = dt
#         self.total_length = total_length

#     def get_batch(self):
#         s = torch.from_numpy(np.random.choice(np.arange(self.N_train - self.batch_time, dtype=np.int64), self.n_samples, replace=False))
#         batch_y0 = self.train_positions[s]  # (M, D)
#         batch_t = self.train_times[:self.batch_time]  # (T)
#         batch_y = torch.stack([self.train_positions[s + i] for i in range(self.batch_time)], dim=0)  # (T, M, D)
#         return batch_y0, batch_t, batch_y

class BatchGetterMultiTrajectories:
    def __init__(self, batch_time, n_samples, total_length, dt, positions, frac_train):
        # N: number of trajectories
        # M: number of time steps
        # D: dimension of the state space
        # positions: (N, T, D)
        self.times = torch.linspace(0., total_length*dt, total_length, dtype=torch.float64).float()
        self.true_positions = torch.tensor(positions, dtype=torch.float64).float()
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


def train(model, optimizer, scheduler, epochs, batch_size, getter, display=100, display_results_fn=display_results, out_display=-1):
    
    if out_display == -1:
        out_display = model.out_dim
    
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
            loss += F.mse_loss(out[:].view(-1,batch_init_positions.shape[-1]), batch_true_positions[:].view(-1,batch_init_positions.shape[-1]))
            
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
