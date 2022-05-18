import numpy as np
import torch


def squared_distances(x, y):
    if x.dim() == 2:
        D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
        D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    else:
        print("x.shape : ", x.shape)
        raise ValueError("Incorrect number of dimensions")
    return D_xx - 2*D_xy + D_yy


def distances(x, y):
    return torch.sqrt( torch.clamp_min(squared_distances(x,y), 1e-8) )

def gaussian_kernel(x, y, beta):
    C2 = squared_distances(x / beta, y / beta)
    return (- 0.5 * C2 ).exp()

def laplacian_kernel(x, y, blur=.05):
    C = distances(x / blur, y / blur)
    return (- C ).exp()

def Gram_calc(X, beta, kernel='G', m=None, randomstate=0):
    # app: approximation flag. Set to 1 to use Nystrom method
    # m: approximated rank.
    
    if kernel == 'G':
        K = gaussian_kernel
    elif kernel == 'L':
        K = laplacian_kernel
    else:
        raise NotImplementedError

    n = X.shape[0]
    assert m <= n
    rng = np.random.RandomState(randomstate)
    
    ## Nystrom method, the approximated kernel is torch.matmul(Out1 * e.unsqueeze(0), Out1.transpose(1, 0))
    with torch.no_grad():
        Index = rng.choice(n, m, replace=False)
        Select = X[Index]
        Knm = K(X, Select, beta)
        e, v= torch.symeig(Knm[Index, ], eigenvectors=True, upper=True)
        
        Out1 = torch.matmul(Knm, v) * (m/n) ** 0.5 / e.unsqueeze(0)
        Out2 = e * n / m

        return [Out1, Out2]


class Nys_kernel:
    def __init__(self, X, beta, Lambda, sigma, trans_type='v', kernel='G', m=None, randomstate=0):
        # m: approximated rank.
        self.beta = beta
        self.Lambda = Lambda
        self.sigma = sigma
        self.trans_type = trans_type
        self.kernel = kernel
        self.m = m
        self.randomstate = randomstate
        self.Gram = Gram_calc(X, beta, kernel=self.kernel, m=self.m, randomstate=randomstate)
        if trans_type =='v' or trans_type == 'f':
            self.inner_inverse = self.in_inv()
        else:
            self.inner_inverse = None

    def in_inv(self):
        return torch.inverse(torch.diag(1 / self.Gram[1]) + torch.matmul(self.Gram[0].transpose(1, 0), self.Gram[0]) / self.sigma)

    def compute_grad(self, V):
        if self.trans_type =='v' or self.trans_type == 'f':
            add_grad = self.Lambda / self.sigma * (
                        V - torch.mm(self.Gram[0], torch.mm(self.inner_inverse, torch.mm(self.Gram[0].permute(1, 0), V))) / self.sigma)
        else:
            add_grad = torch.zeros_like(V, device='cuda')
        return add_grad
