import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.nn.parameter import Parameter

"""
Ttype defines several types of transformations:
f: y=x+v
v: y=Ax+v
a: y=Ax
"""


class Transform(nn.Module):
    def __init__(self, indim=2, pointnum=100, Ttype='v'):
        super(Transform, self).__init__()
        self.Ttype = Ttype
        if Ttype == 'f':
            self.w = Parameter(torch.zeros((pointnum, indim), dtype=torch.float32, requires_grad=True))
        else:
            self.affineM = Parameter(torch.eye(indim, dtype=torch.float32, requires_grad=True))
            self.affinet = Parameter(torch.zeros((1, indim), dtype=torch.float32, requires_grad=True))
            if Ttype == 'v':
                self.w = Parameter(torch.zeros((pointnum, indim), dtype=torch.float32, requires_grad=True))
            elif Ttype == 'a':
                pass
            else:
                raise NotImplementedError
    
    def forward(self, x):
        if self.Ttype == 'f':
            out = x + self.w
            return out
        else:
            out = torch.mm(x, self.affineM) + self.affinet
            if self.Ttype == 'a':
                return out
            elif self.Ttype == 'v':
                out += self.w
                return out
            else:
                raise NotImplementedError