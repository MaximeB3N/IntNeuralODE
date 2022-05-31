import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
from tqdm.notebook import trange


# class ODEfunc(nn.Module):
#     def __init__(self, dim, mid_dim):
#         super(ODEfunc, self).__init__()
#         self.dim = dim
#         self.mid_dim = mid_dim
#         self.seq = nn.Sequential(
#             nn.Linear(dim, self.mid_dim),
#             nn.ReLU(),
#             nn.Linear(self.mid_dim, self.mid_dim),
#             nn.ReLU(),
#             nn.Linear(self.mid_dim, dim),
#             # nn.Tanh()
#         )

#     def forward(self, t, x):
#         return self.seq(x)

class ODEfunc(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ODEfunc, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mid_dim = mid_dim

        layers = [nn.Linear(in_dim, self.mid_dim[0]),
            nn.ReLU()]

        for i in range(len(self.mid_dim) - 1):
            layers.append(nn.Linear(self.mid_dim[i], self.mid_dim[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(self.mid_dim[-1], out_dim))
        self.seq = nn.Sequential(*layers)

    def forward(self, t, x):
        # from dim [1, 2] and [] to [1, 3]
        # t = t.reshape(x.shape[0], 1)
        # input_tensor = torch.cat((x, t), dim=1)
        input_tensor = x
        return self.seq(input_tensor)


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x, t):
        # integration_time = self.integration_time.type_as(x)
        
        out = odeint_adjoint(self.odefunc, x, t.reshape(-1), rtol=1e-4, atol=1e-4)
        return out


class ODEnetSimple(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ODEnetSimple, self).__init__()

        odefunc = ODEfunc(in_dim=in_dim, out_dim=out_dim, mid_dim=mid_dim)
        
        self.ode_block = ODEBlock(odefunc)
        
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        #Use ODE Block
        # self.norm2 = nn.BatchNorm1d(mid_dim)
        # self.fc2 = nn.Linear(mid_dim, out_dim)

        # count the number of parameters
        print("Number of parameters:", sum([p.numel() for p in self.parameters()]))

    def forward(self, x, t):
        # print(x.shape)
        # print(x.shape)
        batch_size = x.shape[0]
        x = x.view(-1, self.out_dim)
        # print("input ode shape", x.shape)
        out = self.ode_block(x, torch.flatten(t))
        # print(out.shape)
        # print(out)
        # out = self.norm2(out)
        # out = self.fc2(out)

        return out

class ODEnetWithZeros(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim, num_zeros):
        super(ODEnetSimple, self).__init__()

        odefunc = ODEfunc(in_dim=in_dim + num_zeros, out_dim=out_dim, mid_dim=mid_dim)
        
        self.ode_block = ODEBlock(odefunc)
        
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num_zeros = num_zeros
        self.mid_dim = mid_dim
        #Use ODE Block
        # self.norm2 = nn.BatchNorm1d(mid_dim)
        # self.fc2 = nn.Linear(mid_dim, out_dim)

        # count the number of parameters
        print("Number of parameters:", sum([p.numel() for p in self.parameters()]))

    def forward(self, x, t):
        # print(x.shape)
        # print(x.shape)
        batch_size = x.shape[0]
        x = x.view(-1, self.in_dim)

        # concatenate with zeros the initial conditions
        zeros = torch.zeros(batch_size, self.num_zeros)
        x = torch.cat((zeros, x), dim=1)
        # print("input ode shape", x.shape)
        out = self.ode_block(x, torch.flatten(t))
        # print(out.shape)
        # print(out)
        # out = self.norm2(out)
        # out = self.fc2(out)

        return out


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


def display_results(i, model, getter, final_time, dt):

    print("The graphs at epoch {}".format(i))
    with torch.no_grad():
        times = torch.linspace(0., final_time*dt, final_time, dtype=torch.float64).float()
        predicted_output = model(getter.train_positions[0].unsqueeze(0), times)
        # display in orange the predicted position and in blue the true position of the training set
        
        # ball.reset()
        # positions = get_positions(ball, final_time, dt)
        
        plt.plot(predicted_output[:,-1,0].cpu().detach().numpy(), 
                predicted_output[:,-1,1].cpu().detach().numpy(), 'orange', label="Predicted")

        # if getter.noise > 0 and getter.noise < 1:
        #     plt.scatter(getter.train_positions[:,0].cpu().detach().numpy(), 
        #         getter.train_positions[:,1].cpu().detach().numpy(), s=1, c='b', label="True train")

        #     plt.scatter(positions[N_train:,0].cpu().detach().numpy(), 
        #         positions[N_train:,1].cpu().detach().numpy(), s=1, c='cyan', label="True test")
        
        # else:
        plt.plot(getter.train_positions[:,0].cpu().detach().numpy(), 
            getter.train_positions[:,1].cpu().detach().numpy(), 'b', label="True train")

        plt.plot(getter.true_positions[getter.N_train:,0], 
                getter.true_positions[getter.N_train:,1], 'cyan', label="True test")

        plt.legend()
        plt.show()

        # print the X axis over the time
        plt.plot(times, getter.true_positions[:,0], 'r', label="True X")
        plt.plot(times, predicted_output[:,-1,0].cpu().detach().numpy(), 'orange', label="Predicted X")
        plt.plot(getter.train_times, getter.train_positions[:,0].cpu().detach().numpy(), 'b', label="True train X")
        plt.legend()
        plt.show()

        plt.plot(times, getter.true_positions[:,1], 'r', label="True X")
        plt.plot(times, predicted_output[:,-1,1].cpu().detach().numpy(), 'orange', label="Predicted X")
        plt.plot(getter.train_times, getter.train_positions[:,1].cpu().detach().numpy(), 'b', label="True train X")
        plt.legend()
        plt.show()



def train(model, optimizer, scheduler, epochs, batch_size, getter, display=100, display_results_fn=display_results):
    
    
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
            loss += 100.*F.mse_loss(out[:].view(-1,batch_init_positions.shape[-1]), batch_true_positions[:].view(-1,batch_init_positions.shape[-1]))
            
        loss /= batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the progress bar
        iterator.set_postfix_str(f'Loss: {loss.item():.8f}')
        running_loss += loss.item()

        if i % display == 0:
           display_results_fn(i, model, getter, getter.total_length, getter.dt)
           iterator.set_description_str(f'Display loss: {running_loss/display:.8f}')
           running_loss = 0.


        scheduler.step()
        
    return None

