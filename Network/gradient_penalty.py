import torch
from torch.autograd import grad

class Grad_Penalty:

    def __init__(self, lambdaGP, point_mass, gamma=1, device=torch.device('cpu') ):
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.device = device
        self.point_mass = point_mass

    def __call__(self, loss, All_points):

        gradients = grad(outputs=loss, inputs=All_points, grad_outputs=torch.ones(loss.size()).to(self.device), 
                    create_graph=True, retain_graph=True)[0].contiguous()
        gradient_penalty = ((torch.nn.functional.relu((gradients/self.point_mass).norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP
        
        with torch.no_grad():
            grad_norm = (gradients/self.point_mass).norm(2, dim=1, keepdim=True)
            
        return gradient_penalty, grad_norm