import open3d as o3d
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import copy

def metric(gt_transformed, transformed):
    gt_shape = gt_transformed.shape[0]
    SquareD = ((gt_transformed - transformed[:gt_shape]) ** 2).sum(-1)
    mse = SquareD.mean()
    mae = (SquareD ** 0.5).mean()
    return mse, mae

def cal_dloss(potential, ref_shape, point_mass, train_type, mass, h):
    D_real_int = torch.sum(potential[:, 0, :ref_shape] * point_mass, 1)
    D_fake_int = torch.sum(potential[:, 0, ref_shape:] * point_mass, 1)
    if train_type == 'm':
        d_loss = torch.mean(D_fake_int - D_real_int + h * (mass - (potential.shape[2] - ref_shape)) * point_mass)
    elif train_type == 'h':
        d_loss = torch.mean(D_fake_int - D_real_int)
    else:
        raise NotImplementedError
    return d_loss

def normalize(x):
    """
    Translate and scale a point set to make it have zero mean and unit variance. Return the point set after normalization, its original centroid and scale.
    """
    centroid = x.mean(0)
    x = x - centroid
    scale = np.linalg.norm(x,'fro')/np.sqrt(x.shape[0])
    x = x/scale
    return x, centroid, scale


def denormalize(x,centroid,scale):
    """Denormalize a point set from saved centroid and scale."""
    x = x*scale + centroid
    return x


def view_with_direction(LandScape, parameters=None, savename=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    if isinstance(LandScape, list):
        for i in LandScape:
            vis.add_geometry(i)
    else:
        vis.add_geometry(LandScape)

    if parameters is not None:
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(parameters)

    render = vis.get_render_option()
    render.point_size = 1.

    vis.run()
    if savename is not None:
        vis.capture_screen_image(savename)


def visualize_3d_3(A, viewpoint=None, savename=None):
    template = o3d.geometry.PointCloud()
    template.points = o3d.utility.Vector3dVector(A[0])

    sample = o3d.geometry.PointCloud()
    sample.points = o3d.utility.Vector3dVector(A[1])

    sample2 = o3d.geometry.PointCloud()
    sample2.points = o3d.utility.Vector3dVector(A[2])

    template.paint_uniform_color([1, 0.706, 0])
    sample.paint_uniform_color([0, 0.651, 0.929])
    sample2.paint_uniform_color([0, 1., 0.])

    if viewpoint != None:
        parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
    else:
        parameters = None

    if savename is not None:
        current_savename = savename + '_vis3.png'
    else:
        current_savename = None

    view_with_direction([template, sample, sample2], parameters, savename=current_savename)


def visualize_2d_3(A, savename=None):
    plt.scatter(A[0][:,0], A[0][:,1], c='b', label='Transformed set')
    plt.scatter(A[1][:,0], A[1][:,1], c='g', label='Source set')
    plt.scatter(A[2][:,0], A[2][:,1], c='r', label='Reference set')
    plt.legend()
    if savename is not None:
        current_savename = savename + '_vis3.png'
    else:
        current_savename = None

    if current_savename != None:
        plt.tight_layout()
        plt.savefig(current_savename)
        plt.close()
    else: 
        plt.tight_layout()
        plt.show()


def visualize_3d_2(A, viewpoint=None, savename=None):

    template = o3d.geometry.PointCloud()
    template.points = o3d.utility.Vector3dVector(A[0])

    sample2 = o3d.geometry.PointCloud()
    sample2.points = o3d.utility.Vector3dVector(A[1])
    sample2.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0., 0., 1.]]), [A[1].shape[0], 1]))

    template.paint_uniform_color([1, 0, 0])
    sample2.paint_uniform_color([0, 0, 1.])

    if viewpoint != None:
        parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
    else:
        parameters = None

    if savename is not None:
        my_savename = savename + '_vis2.png'
    else:
        my_savename = None
    view_with_direction([template, sample2], parameters, savename=my_savename)


def visualize_2d_2(A, savename=None):
    plt.scatter(A[0][:,0], A[0][:,1], c='r', label='Source set')
    plt.scatter(A[1][:,0], A[1][:,1], c='b', label='Reference set')
    # plt.axis('off')
    # plt.xlim([-1, 1.22])
    # plt.ylim([-0.64, 0.92])

    plt.legend()

    if savename is not None:
        my_savename = savename + '_vis2.png'
    else:
        my_savename = None

    if my_savename != None:
        plt.tight_layout()
        plt.savefig(my_savename)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def vis_PC(A, savename=None, viewpoint=None):
    if len(A) == 2:
        if A[0].shape[-1] == 2:
            visualize_2d_2(A, savename=savename)
        elif A[0].shape[-1] == 3:
            visualize_3d_2(A, savename=savename, viewpoint=viewpoint)
    else:
        if A[0].shape[-1] == 2:
            visualize_2d_3(A, savename=savename)
        elif A[0].shape[-1] == 3:
            visualize_3d_3(A, savename=savename, viewpoint=viewpoint)

def visualize_color(All_points, potential, real_point_num, viewpoint=None, savename=None):
    All_points_np = All_points.detach().cpu().transpose(2, 1).numpy()

    F_np_ori = potential[:, 0].detach().cpu().numpy()

    colormap = plt.cm.viridis
    F_np = (F_np_ori - np.min(F_np_ori, 1, keepdims=True)) / (np.max(F_np_ori, 1, keepdims=True)
                                                              - np.min(F_np_ori, 1, keepdims=True))

    Color_np = np.zeros((All_points_np.shape[0], All_points_np.shape[1], 3))
    for b in range(All_points_np.shape[0]):
        for i in range(All_points_np.shape[1]):
            Color_np[b, i] = colormap(F_np[b, i])[:3]

    if viewpoint != None:
        parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
    else:
        parameters = None

    for i in range(All_points.shape[0]):


        LandScape = o3d.geometry.PointCloud()
        LandScape.points = o3d.utility.Vector3dVector(All_points_np[i])
        LandScape.colors = o3d.utility.Vector3dVector(Color_np[i])

        view_with_direction(LandScape, parameters=parameters, savename=savename + '_viscolor1.png')


        LandScape1 = o3d.geometry.PointCloud()
        LandScape1.points = o3d.utility.Vector3dVector(All_points_np[i][:real_point_num])
        LandScape1.colors = o3d.utility.Vector3dVector(Color_np[i][:real_point_num])

        view_with_direction(LandScape1, parameters=parameters, savename=savename + '_viscolor2.png')


        LandScape2 = o3d.geometry.PointCloud()
        LandScape2.points = o3d.utility.Vector3dVector(All_points_np[i][real_point_num:])
        LandScape2.colors = o3d.utility.Vector3dVector(Color_np[i][real_point_num:])

        view_with_direction(LandScape2, parameters=parameters, savename=savename + '_viscolor3.png')


        plt.hist(F_np_ori[i], bins=30)
        plt.show()


def visualize2d_color(All_points, potential, N, normalize=True, name=None, show=False):
    colormap = plt.cm.viridis
    All_points_np = All_points.detach().cpu().transpose(2, 1).numpy()

    F_np_ori = potential[:, 0].detach().cpu().numpy()

    if normalize == True:
        F_np = (F_np_ori - np.min(F_np_ori, 1, keepdims=True)) / (np.max(F_np_ori, 1, keepdims=True)
                - np.min(F_np_ori, 1, keepdims=True))
    else:
        F_np = F_np_ori * 0.9

    plt.scatter(All_points_np[0, :, 0], All_points_np[0,:,1], c=colormap(F_np[0]))
    plt.axis('off')
    plt.xlim([-1, 1.22])
    plt.ylim([-0.64, 0.92])

    if name != None:
        plt.savefig(name)
        plt.close()
    if show==True:
        plt.show()

def vis_gradnorm(All_points, d_loss, point_mass):
    gradients = grad(outputs=d_loss, inputs=All_points, grad_outputs=torch.ones(d_loss.size()).to('cuda'),
                create_graph=False, retain_graph=False)[0].contiguous()
    grad_norm = (gradients / point_mass).norm(2, dim=1, keepdim=True)
    return grad_norm


def visualize_3_2_step(A, viewpoint=None, savename=None):

    template = o3d.geometry.PointCloud()
    template.points = o3d.utility.Vector3dVector(A[0])


    sample2 = o3d.geometry.PointCloud()
    sample2.points = o3d.utility.Vector3dVector(A[1])
    sample2.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0., 0., 1.]]), [A[1].shape[0], 1]))

    template.paint_uniform_color([1, 0, 0])
    sample2.paint_uniform_color([0, 0, 1.])

    if viewpoint != None:
        parameters = o3d.io.read_pinhole_camera_parameters(viewpoint)
    else:
        parameters = None

    view_with_direction([template], parameters, savename=savename + '_vis3_step1.png')
    view_with_direction([sample2], parameters, savename=savename + '_vis3_step2.png')
    view_with_direction([template, sample2], parameters, savename=savename + '_vis3.png')



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, warm=0, save_path='.'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.global_step = -1
        self.warm = warm
        self.path = save_path

    def __call__(self, score, model):

        self.global_step += 1
        if self.global_step <= self.warm:
            # do nothing at warm stage
            return False

        if self.best_score is None:
            self.best_score = score
            self.Best_dict = copy.deepcopy(model.state_dict())
            self.best_step = self.global_step
            return False
        elif score < self.best_score:
            if self.verbose:
                print('best score', self.best_score)
                print('current score', score)
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f'EarlyStopping at step: {self.best_step}')
                self.early_stop = True
                torch.save(self.Best_dict, self.path)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.Best_dict = copy.deepcopy(model.state_dict())
            self.best_step = self.global_step
            return False
