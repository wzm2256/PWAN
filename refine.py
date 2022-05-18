import numpy as np
import torch
from torch import optim
import logging
import utils
from pykeops.torch import Vi, Vj
import pdb
import os
from utils import EarlyStopping

def nn_pykeops_3d():
    # Encoding as KeOps LazyTensors:
    X_i = Vi(0, 3)  # Purely symbolic "i" variable, without any data array
    X_j = Vj(1, 3)  # Purely symbolic "j" variable, without any data array

    # Symbolic distance matrix:
    D_ij = ((X_i - X_j) ** 2).sum(-1)

    # K-NN query operator:
    KNN_fun = D_ij.argKmin(1, dim=1)

    return KNN_fun

def nn_pykeops_2d():
    # Encoding as KeOps LazyTensors:
    X_i = Vi(0, 2)  # Purely symbolic "i" variable, without any data array
    X_j = Vj(1, 2)  # Purely symbolic "j" variable, without any data array

    # Symbolic distance matrix:
    D_ij = ((X_i - X_j) ** 2).sum(-1)

    # K-NN query operator:
    KNN_fun = D_ij.argKmin(1, dim=1)

    return KNN_fun

def refine_gpu(G, scr_cloud, ref_cloud, step, train_type, mass, h, lr, clip, Transformed_src_gt, normalize, ref_mean, ref_var,
             name=3000, p=1, refine_curve=None, save=0, ny_kernel=None, current_name='.', f_d_loss=0.):

    optimizerG_refine = optim.RMSprop(G.parameters(), lr=lr)
    point_mass = 1. / scr_cloud.shape[0]

    if clip == True:
        h_ = h.detach()
    else:
        h_ = -50.

    logger = logging.getLogger('Refine')

    if scr_cloud.shape[1] == 3:
        Search_fun = nn_pykeops_3d()
    else:
        Search_fun = nn_pykeops_2d()

    final_mse = torch.tensor(0.)
    save_path = os.path.join(current_name, 'tmp.pt')
    es = EarlyStopping(patience=1, verbose=True, warm=0, save_path=save_path)

    for i in range(step):
        Transformed_src = G(scr_cloud)

        optimizerG_refine.zero_grad()

        indices = Search_fun(Transformed_src, ref_cloud)[:,0]
        nnDis = torch.sum((Transformed_src - ref_cloud[indices]) ** 2, 1) ** (1/2)

        if train_type == 'm':
            # pdb.set_trace()
            Loss = torch.mean(torch.topk(nnDis ** p , int(mass), largest=False)[0])
        elif train_type == 'h':
            Loss = torch.mean(torch.clamp_max(nnDis ** p, (-h_) ** p))
        else:
            raise NotImplementedError

        Stop = es(-Loss, G)
        if Stop == True:
            break

        #### evaluation
        if Transformed_src_gt == None:
            mse = torch.tensor(-1.)
        else:
            if normalize == 1:
                Transformed_src_de = utils.denormalize(Transformed_src, 
                                                        torch.tensor(ref_mean, dtype=torch.float).cuda(), 
                                                        torch.tensor(ref_var, dtype=torch.float).cuda())
            else:
                Transformed_src_de = Transformed_src
            mse, _ = utils.metric(Transformed_src_gt, Transformed_src_de)

        with torch.no_grad():
            if train_type == 'm':
                NN = torch.sum(torch.topk(nnDis, int(mass), largest=False)[0] * point_mass)
            elif train_type == 'h':
                NN = torch.sum(torch.clamp_max(nnDis, (-h_)) * point_mass)
            else:
                raise NotImplementedError


        if i == 0 and NN < -f_d_loss:
            # pdb.set_trace()
            logger.warning('D loss is too large. Consider re-running the program or decreasing L.')
            final_scr = G(scr_cloud)
            return final_scr, torch.tensor(np.NaN)

        if refine_curve is not None:
            refine_curve.write(str(mse.item()))
            refine_curve.write('\t')
            refine_curve.write(str(Loss.item()))
            refine_curve.write('\t')
            refine_curve.write(str(NN.item()))
            refine_curve.write('\n')

        #####
        Mdis = torch.max(nnDis)
        logger.info('Refine-step: ' + str(i) + 'Loss: ' + str(Loss.item()) + ' MSE: ' + str(np.array(mse.item()).round(6)) +
            ' Mdis: ' + str(np.array(Mdis.item()).round(6)))

        final_mse = mse

        Loss.backward()

        add_grad = ny_kernel.compute_grad(G.w)
        G.w.grad += add_grad

        optimizerG_refine.step()

    final_scr = G(scr_cloud)

    if save == 1:
        if normalize == 1:
            Transformed_tmp = utils.denormalize(final_scr, torch.tensor(ref_mean, dtype=torch.float).cuda(), torch.tensor(ref_var, dtype=torch.float).cuda()).detach().cpu().numpy()
        else:
            Transformed_tmp = final_scr.detach().cpu().numpy()

        pc_path = os.path.join(current_name, str(name).zfill(6) + '_refined.npy')
        np.save(pc_path, Transformed_tmp)

    return final_scr, final_mse