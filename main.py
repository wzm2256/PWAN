import argparse
import logging
import os

import numpy as np
import torch
from torch import optim

import data_util
import utils

from Network.Gen import Transform
from Network.gradient_penalty import Grad_Penalty
from kernel import Nys_kernel
from torch.utils.tensorboard import SummaryWriter
from Network.PotentialNet import SmallRes as PointNetPotential

args_ps = argparse.ArgumentParser()

## Data processing
args_ps.add_argument('--source', default='dataset/fish_x.txt', help='source path')
args_ps.add_argument('--reference', default='dataset/fish_y.txt', help='reference path')
args_ps.add_argument('--test_mode', type=int, default=1, help='synthesis reference data. 1:True (ignore parameter reference); 0:False')
args_ps.add_argument('--normalize', type=int, default=1, help='normalize the point sets or not')

args_ps.add_argument('--syn_subsample', type=int, default=500, help='Size of point set')
args_ps.add_argument('--syn_Lambda', type=float, default=10, help='synthesis parameter')
args_ps.add_argument('--syn_m', type=int, default=100, help='Nystrom method used in synthesis')
args_ps.add_argument('--syn_seed', type=int, default=None, help='random seed used in data synthesis')
args_ps.add_argument('--syn_noise_level', type=float, default=0.02, help='additive noise')
args_ps.add_argument('--syn_outlier', type=int, default=0, help='the number of outliers')
args_ps.add_argument('--syn_partial', type=float, default=None, help='percentage of kept point, e.g.0.7')
args_ps.add_argument('--syn_direction', type=int, default=None, help='choose the points to keep, default:random')

## Experiment name
args_ps.add_argument('--outfile', type=str, default='out_PW.txt', help='output summary file')
args_ps.add_argument('--name', type=str, default='desktop', help='Experiment name')

## model
args_ps.add_argument('--train_type', type=str, choices=['h', 'm'], default='h', help='h:distance type. m:mass type.')
args_ps.add_argument('--mass', type=float, default=100, help='mass threshold. It is used when train_type=m')
args_ps.add_argument('--h', type=float, default=0.1, help='distance threshold. It is used when train_type=h')
args_ps.add_argument('--mode', default='full', choices=['full', 'partial', 'part'], help='full: no outlier. part: only ref set has outliers. partial: both sets have outliers.')
args_ps.add_argument('--vis', type=int, default=1, help='visualize registration result.')

## Nonlinear regularizer
args_ps.add_argument('--G_type', type=str, choices=['v', 'a', 'f'], default='v', help='transformation type, see file for details')
args_ps.add_argument('--sigma', type=float, default=0.1, help='Non-rigid parameter')
args_ps.add_argument('--Lambda', type=float, default=2.0, help='Non-rigid parameter')
args_ps.add_argument('--beta', type=float, default=2.0, help='Non-rigid parameter')
args_ps.add_argument('--kernel', type=str, choices=['G', 'L'], default='G', help='Gaussian or Laplacian kernel')
args_ps.add_argument('--m', type=int, default=500, help='Nystrom parameter')
args_ps.add_argument('--randomstate', type=int, default=0)

## network
args_ps.add_argument('--leaky_D', type=float, default=0.2, help='leaky relu paramter in the network')
args_ps.add_argument('--d_iter', type=int, default=5, help='Number of iterations of the network at each step')

## training
args_ps.add_argument('--lr_D', type=float, default=1e-4, help='learning rate of the network')
args_ps.add_argument('--lr_G', type=float, default=1e-4, help='learning rate of point set')
args_ps.add_argument('--epoch', type=int, default=2001, help='Total iterations')
args_ps.add_argument('--disp', type=int, default=500, help='Dispay interval')
args_ps.add_argument('--save_pc', type=int, default=0, help='Save registered point sets')
args_ps.add_argument('--refine', type=int, default=1, help='refine the point sets at the end of the training process')
args_ps.add_argument('--Refine_step', type=int, default=1, help='Dispay interval')

args = args_ps.parse_args()

# Set level=logging.DEBUG to print more info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training")

root_f = 'LOG/' + args.name + '_' + args.train_type + '_' + str(args.mass) + '_' \
         + str(args.h) + '_' + str(args.Lambda) + '_' + str(args.beta) + '_' + str(args.sigma)

if not os.path.isdir(root_f):
    os.makedirs(root_f)

n = len(os.listdir(root_f))
current_name = os.path.join(root_f, str(n))
writer = SummaryWriter(current_name)

config_path = os.path.join(current_name, 'config.txt')

f = open(config_path, 'w')
config_Dict = vars(args)
for key, value in config_Dict.items():
    f.write(str(key))
    f.write('\t')
    f.write(str(value))
    f.write('\n')
f.close()

curve_path = os.path.join(current_name, 'curve.txt')
f_curve = open(curve_path, 'w')

loss_path = os.path.join(current_name, 'd_loss.txt')
f_loss = open(loss_path, 'w')

refine_path = os.path.join(current_name, 'refine.txt')
f_refine = open(refine_path, 'w')

#---------------------------------- preparing data
if args.test_mode == 0:
    # Load both sets directly from files
    src_cloud_ori = np.loadtxt(args.source, dtype=np.float32)
    ref_cloud_ori = np.loadtxt(args.reference, dtype=np.float32)
    T_src = None
else:
    # Artificially synthesize by deformation
    src_cloud_ori, ref_cloud_ori, T_src = data_util.load(args.source, beta=2., Lambda=args.syn_Lambda, 
                                                m=args.syn_m, seed=args.syn_seed, noise_level=args.syn_noise_level, 
                                                outlier=args.syn_outlier, partial=args.syn_partial, size=args.syn_subsample,
                                                direction=args.syn_direction)

# visualize point set
if args.vis == 1:
    if args.test_mode == 1:
        utils.vis_PC([T_src,  src_cloud_ori, ref_cloud_ori])
    else:
        utils.vis_PC([src_cloud_ori, ref_cloud_ori])

if args.normalize == 1:
    src_cloud_np, _, _ = utils.normalize(src_cloud_ori)
    ref_cloud_np, ref_mean, ref_var = utils.normalize(ref_cloud_ori)
else:
    src_cloud_np = src_cloud_ori
    ref_cloud_np = ref_cloud_ori
    ref_mean, ref_var = None, None

if args.save_pc == 1:
    pc_path = os.path.join(current_name, 'Ref.npy')
    np.save(pc_path, ref_cloud_ori)

src_cloud = torch.from_numpy(src_cloud_np).cuda()
ref_cloud = torch.from_numpy(ref_cloud_np).cuda()

if args.test_mode == 1:
    T_src_cloud = torch.from_numpy(T_src).cuda()
else:
    T_src_cloud = None

indim = src_cloud.shape[-1]

logger.info('PC dim: ' + str(indim))
logger.info('Source PC size: ' + str(src_cloud.shape[0]))
logger.info('Reference PC size: ' + str(ref_cloud.shape[0]))

#---------------------------------- training process setting
if args.mass < 0:
    raise ValueError('Mass threshold should be non-negative.')
if args.h < 0:
    raise ValueError('Distance threshold should be non-negative.')

if args.train_type == 'm':
    MinMass = min(src_cloud.shape[0], ref_cloud.shape[0])
    if args.mass > MinMass:
        mass = MinMass
        logger.warning('Transport mass exceeds {}, clip it to {}'.format(MinMass, MinMass))
    else:
        mass = args.mass
    logger.info('Training type: mass. Threshold mass:' + str(mass))
else:
    mass = 0.
    logger.info('Training type: distance. Threshold distance:' + str(args.h))

point_mass = 1. / src_cloud.shape[0]

# Network setting
D = PointNetPotential(leaky=args.leaky_D, net_type=args.train_type, h=args.h, indim=indim, N1=ref_cloud.shape[0])
D = D.to('cuda')

G = Transform(indim=indim, pointnum=src_cloud.shape[0], Ttype=args.G_type)
G = G.to('cuda')

G_P = Grad_Penalty(100, point_mass, device='cuda')

ny_kernel = Nys_kernel(src_cloud, args.beta, args.Lambda, args.sigma, trans_type=args.G_type, kernel=args.kernel, m=args.m, randomstate=args.randomstate)

optimizerD = optim.Adam(D.parameters(), lr=args.lr_D, betas=(0, 0.99))
optimizerG = optim.RMSprop(G.parameters(), lr=args.lr_G)

# Simplify network
if args.mode == 'partial':
    clip = True
    neg = True
else:
    clip = False
    if args.mode == 'part':
        neg = True
    elif args.mode == 'full':
        neg = False
    else:
        raise NotImplementedError

#---------------------------------- Start training
Loss_record = []
MSE = []

for epoch in range(args.epoch):
    D.train()

    Transformed_src = G(src_cloud)

    with torch.no_grad():
        if args.normalize == 1:
            Transformed_src_de = utils.denormalize(Transformed_src, torch.tensor(ref_mean, dtype=torch.float).cuda(),
                                                        torch.tensor(ref_var, dtype=torch.float).cuda())
        else:
            Transformed_src_de = Transformed_src

    # record loss
    if args.test_mode == 1:
        mse, _ = utils.metric(T_src_cloud, Transformed_src_de)
    else:
        mse = torch.tensor(-1.)
    
    MSE.append(mse)
    writer.add_scalar('MSE', mse, epoch)

    All_points_D = torch.cat([ref_cloud, Transformed_src], 0).unsqueeze(0).transpose(2, 1).detach()
    All_points_D.requires_grad_(True)

    # update the potential network
    for i in range(args.d_iter):
        potential, h = D(All_points_D, clip=clip, neg=neg)
        d_loss = utils.cal_dloss(potential, ref_cloud.shape[0], point_mass, args.train_type, mass, h)

        # Compute gradient penalty once each step
        if i == 0:
            gp_loss, grad_norm = G_P(d_loss, All_points_D)
            Max_grad = torch.max(grad_norm[0, 0, ref_cloud.shape[0]:])
            writer.add_scalar('Max_grad', Max_grad, epoch)
        else:
            gp_loss = torch.tensor(0.)
            Max_grad = torch.tensor(0.)

        d_loss_all = d_loss + gp_loss

        optimizerD.zero_grad()
        d_loss_all.backward()
        optimizerD.step()

        logger.debug('Iter: ' + str(epoch) + ' d_loss: ' + str(np.array(d_loss.item()).round(6)) + '\t' +
                        'gp_loss: ' + str(np.array(gp_loss.item()).round(6)) + '\t' +
                        'h: ' + str(np.array(h.mean().item()).round(6)) + ' max_grad: ' + str(np.array(Max_grad.item()).round(6)))

    # Update the point set with the fixed network
    D.eval()
    writer.add_scalar('Loss', d_loss, epoch)
    writer.add_scalar('h', h, epoch)

    Loss_record.append(d_loss.item())
    logger.info('Iter: ' + str(epoch) + ' d_loss: ' + str(np.array(d_loss.item()).round(6))
                + ' MSE: ' + str(np.array(mse.item()).round(6)))

    potential, h = D(Transformed_src.unsqueeze(0).transpose(2, 1), clip=clip, neg=neg, dual=False)

    G_fake = potential[:, 0, :]

    G_fake_int = torch.sum(G_fake * point_mass, 1)
    g_loss = - torch.mean(G_fake_int) 

    optimizerG.zero_grad()
    g_loss.backward()

    # add the gradient of the Nystrom term
    add_grad = ny_kernel.compute_grad(G.w)
    G.w.grad += add_grad

    torch.nn.utils.clip_grad_value_(G.parameters(), 1.)
    optimizerG.step()

    if epoch % args.disp == 0:
        Transformed_tmp = Transformed_src_de.detach().cpu().numpy()

        if args.save_pc == 1:

            All_points = torch.cat([ref_cloud, Transformed_src], 0).unsqueeze(0).transpose(2, 1)
            potential, h = D(All_points, clip=clip, neg=neg)
            d_loss = utils.cal_dloss(potential, ref_cloud.shape[0], point_mass, args.train_type, mass, h)
            grad_norm = utils.vis_gradnorm(All_points, d_loss, point_mass)

            np.save(os.path.join(current_name, str(epoch).zfill(6) + '_transformed.npy'), Transformed_tmp)
            np.save(os.path.join(current_name, str(epoch).zfill(6) + '_grad.npy'), grad_norm.detach().cpu().numpy())
            np.save(os.path.join(current_name, str(epoch).zfill(6) + '_potential.npy'), potential.detach().cpu().numpy())

        if args.vis == 1:
            utils.vis_PC([Transformed_tmp, ref_cloud_ori],
                                    viewpoint='ScreenCamera.json',
                                    savename=os.path.join(current_name, str(epoch).zfill(5)))

# visualize the gradient for 3D sets
if args.vis == 1 and src_cloud.shape[-1] == 3:

    All_points = torch.cat([ref_cloud, Transformed_src], 0).unsqueeze(0).transpose(2, 1)
    potential, h = D(All_points, clip=clip, neg=neg)
    d_loss = utils.cal_dloss(potential, ref_cloud.shape[0], point_mass, args.train_type, mass, h)
    grad_norm = utils.vis_gradnorm(All_points, d_loss, point_mass)

    if clip == True:
        potential = torch.clamp_min(potential, h.item())

    # pdb.set_trace()
    if args.normalize == 1:
        All_point_vis = utils.denormalize(All_points.squeeze(0).transpose(1, 0).detach().cpu(), ref_mean, ref_var).\
            unsqueeze(0).transpose(1, 2)
    else:
        All_point_vis = All_points

    utils.visualize_color(All_point_vis,
                            potential, ref_cloud.shape[0],
                            viewpoint='ScreenCamera.json',
                            savename=os.path.join(current_name, 'final_color'))

    grad_norm = torch.clamp(grad_norm, 0, 1)
    utils.visualize_color(All_point_vis,
                            grad_norm, ref_cloud.shape[0],
                            viewpoint='ScreenCamera.json',
                            savename=os.path.join(current_name, 'final_gradient'))

    utils.vis_PC([All_point_vis.squeeze(0).transpose(0,1)[ref_cloud.shape[0]:],
                    src_cloud_ori,
                    All_point_vis.squeeze(0).transpose(0,1)[:ref_cloud.shape[0]]],
                    viewpoint='ScreenCamera.json',
                    savename=os.path.join(current_name, 'final')
                    )

# Record loss and MSE to file
for i in range(len(MSE)):
    f_curve.write(str(MSE[i].item()))
    f_curve.write('\n')

for i in range(len(Loss_record)):
    f_loss.write(str(Loss_record[i]))
    f_loss.write('\n')
f_loss.close()

# Refinement
if args.refine > 0:
    import refine
    #---------------------------------- refine
    final_scr, final_mse = refine.refine_gpu(G, src_cloud, ref_cloud, args.Refine_step, args.train_type, mass, h, args.lr_G, clip, T_src_cloud, args.normalize,
                                ref_mean, ref_var, name=args.epoch, p=2, ny_kernel=ny_kernel, save=args.save_pc, current_name=current_name, f_d_loss=d_loss, 
                                refine_curve=f_refine)
    print(final_mse.item())
    f_curve.write(str(final_mse.item()))
    f_curve.write('\t')
    f_curve.write('0.')
    f_curve.write('\n')
    f_curve.close()

    f = open(args.name + '_' + args.outfile, 'a')
    f.write(str(final_mse.item()))
    f.write('\t')
    f.write('0.')
    f.write('\n')
    f.close()

    if args.vis == 1:
        if args.normalize == 1:
            vis_final = utils.denormalize(final_scr.detach().cpu().numpy(), ref_mean, ref_var)
        else:
            vis_final = final_scr
        
        utils.vis_PC([vis_final, src_cloud_ori, ref_cloud_ori],
                            viewpoint='ScreenCamera.json',
                            savename=os.path.join(current_name, 'final')
                            )
