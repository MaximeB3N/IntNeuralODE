import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint


class ODEfunc(nn.Module):

    """
    Neural ODE function f: R^d -> R^d used to learn the dynamics of the system.
    It is a MLP with linear layers and relu activation (not depending on time t)

    Parameters
    ----------
    in_dim : int, dimension of the input
    mid_dim : list of int, dimension of the hidden layers
    out_dim : int, dimension of the output

    Comment : 
       

        If we note f_theta the neural network where theta is the parameters of the MLP, 
        we have that f_theta is defined such that : 
            dx/dt = f_theta(x)
    """
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
    """
    Neural ODE block used to learn the dynamics of the system. It can use the ode function odefunc 
    to simulate the underlying dynamics by feeding feeding the initial position x and the output times t.

    Parameters
    ----------
    odefunc : ODEfunc, the neural network used to simulate the dynamics of the system
    """
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x, t):
        # integration_time = self.integration_time.type_as(x)
        
        out = odeint_adjoint(self.odefunc, x, t.reshape(-1), rtol=1e-4, atol=1e-4)
        return out


class ODEnetSimple(nn.Module):
    """
    Wrapper for the ODEBlock. It is used to learn the dynamics of the system and 
    make sure that we can batchify the data to parallelize the computation.

    Parameters
    ----------
    in_dim : int, dimension of the input
    mid_dim : list of int, dimension of the hidden layers
    out_dim : int, dimension of the output
    """
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
