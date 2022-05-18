import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from torch.nn.parameter import Parameter

class SmallRes(nn.Module):
    # a unet like concate between 64 dim layers
    def __init__(self, leaky=None, N1=None, h=None, net_type=None, indim=None, **kwargs):
        super(SmallRes, self).__init__()

        Layers = [32, 64, 128, 64, 1]

        in_dim1 = indim
        self.net1 = nn.Sequential()
        for i, out_dim in enumerate(Layers[:2]):
            self.net1.add_module('P1_conv_' + str(i), torch.nn.Conv1d(in_dim1, out_dim, 1))
            self.net1.add_module('P1_conv_leaky_' + str(i), nn.LeakyReLU(negative_slope=leaky))
            in_dim1 = out_dim

        self.net2 = nn.Sequential()
        for i, out_dim in enumerate(Layers[2:4]):
            self.net2.add_module('P2_conv_' + str(i), torch.nn.Conv1d(in_dim1, out_dim, 1))
            self.net2.add_module('P2_conv_leaky_' + str(i), nn.LeakyReLU(negative_slope=leaky))
            in_dim1 = out_dim

        in_dim1 *= 2
        self.net3 = nn.Sequential()
        self.net3.add_module('P3_conv', torch.nn.Conv1d(in_dim1, 1, 1))

        if net_type == 'h':
            self.h = torch.tensor(h, dtype=torch.float32, requires_grad=False).to('cuda')
        elif net_type == 'm':
            self.h = Parameter(torch.tensor(h, dtype=torch.float32), requires_grad=True)
        else:
            raise NotImplementedError

        self.N1 = N1

    def forward(self, x, clip=False, dual=True, neg=True):

        out1 = self.net1(x)
        out2 = self.net2(out1)
        p_out = self.net3(torch.cat([out1, out2], 1))

        if neg:
            out = -torch.abs(p_out)
        else:
            out = p_out

        h = -torch.abs(self.h)

        if clip == True and dual == True:
            R_p = out[:, :, :self.N1]
            F_p = out[:, :, self.N1:]
            F_p = nn.functional.relu(F_p - h.unsqueeze(-1)) + h.unsqueeze(-1)
            P = torch.cat([R_p, F_p], 2)
        elif clip == True and dual == False:
            P = nn.functional.relu(out - h.unsqueeze(-1)) + h.unsqueeze(-1)
        else:
            P = out

        return P, h