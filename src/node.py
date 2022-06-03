import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

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
