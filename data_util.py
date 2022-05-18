import numpy as np
import kernel
import torch
import pdb

def wrap(X, beta, Lambda, m, seed=None):
    rng = np.random.RandomState(seed)
    nsample = torch.from_numpy(rng.randn(m, X.shape[1]).astype(np.float32))
    G = kernel.Gram_calc(X, beta, kernel='G', m=m, randomstate=seed)
    T = G[0] * (torch.clamp_min(G[1], 0) ** 0.5).unsqueeze(0)
    v = torch.matmul(T, nsample) / (Lambda ** 0.5)
    return v

def partial_crop(points, p_keep, rng, direction=None):
    points_np = points.numpy()

    if direction is None:
        direction_random = rng.randn(3)
        rand_xyz = direction_random / np.linalg.norm(direction_random)
    else:
        rand_xyz = np.zeros(3)
        rand_xyz[np.abs(direction)] = direction / np.abs(direction)

    centroid = np.mean(points_np[:, :3], axis=0)
    points_centered = points_np[:, :3] - centroid

    dist_from_plane = np.dot(points_centered, rand_xyz)
    mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)
    return points[mask, :], mask


def Select(PC, size, uniform=True, p_keep=None, rng=None, direction=None):
    n = PC.shape[0]

    if uniform == True:
        Index = rng.choice(n, size, replace=False)
        select = PC[Index]
    else:
        select = PC
        Index = np.asarray(range(PC.shape[0]))

    if p_keep is not None:
        select, mask = partial_crop(select, p_keep, rng, direction=direction)
        Index = Index[mask]

    return select, Index


def load(file, size=None, beta=None, Lambda=None, m=None, seed=None, noise_level=None, outlier=None, partial=None, direction=None):
    rng = np.random.RandomState(seed)

    src_cloud = np.loadtxt(file, dtype=np.float32)
    src_cloud = torch.from_numpy(src_cloud)

    if m > src_cloud.shape[0] or m is None:
        m = src_cloud.shape[0]

    v = wrap(src_cloud, beta=beta, Lambda=Lambda, m=m, seed=seed)
    ref_gt = src_cloud + v
    ref = ref_gt.clone()

    Ind_src = None

    if size is not None and size < src_cloud.shape[0]:
        if direction is None:
            direction1 = direction
            direction2 = direction
        else:
            direction1 = direction
            direction2 = -direction
            
        src_cloud, Ind_src = Select(src_cloud, size, p_keep=partial, rng=rng, direction=direction1)
        ref, Ind_ref = Select(ref, size, p_keep=partial, rng=rng, direction=direction2)

    if noise_level is not None:
        ref += torch.randn(ref.shape) * noise_level
        ref += torch.from_numpy(rng.randn(*(ref.shape)).astype(np.float32)) * noise_level

    if outlier is not None:
        M = torch.max(ref, 0)[0]
        m = torch.min(ref, 0)[0]
        Outlier = (torch.from_numpy(rng.rand(np.abs(outlier), src_cloud.shape[1]).astype(np.float32)) - 0.5) * (M.unsqueeze(0)-m.unsqueeze(0)) * 1.2 + \
                  (M.unsqueeze(0) + m.unsqueeze(0)) / 2.
        ref = torch.cat([ref, Outlier])

    # if outlier is not None and outlier < 0:
    #     M = torch.max(src_cloud, 0)[0]
    #     m = torch.min(src_cloud, 0)[0]
    #     Outlier_src = (torch.from_numpy(rng.rand(np.abs(outlier) // 2, src_cloud.shape[1]).astype(np.float32)) - 0.5) * (M.unsqueeze(0)-m.unsqueeze(0)) * 1.2 + \
    #               (M.unsqueeze(0) + m.unsqueeze(0)) / 2.
    #     src_cloud = torch.cat([src_cloud, Outlier_src])

    ref_np = ref.numpy()
    src_np = src_cloud.numpy()

    if Ind_src is None:
        Transformed_src = ref_gt.numpy()
    else:
        Transformed_src = ref_gt[Ind_src].numpy()

    return src_np, ref_np, Transformed_src




def load_bak(file, size=None, beta=None, Lambda=None, m=None, seed=None, noise_level=None, outlier=None, partial=None):
    rng = np.random.RandomState(seed)

    src_cloud = np.loadtxt(file, dtype=np.float32)

    if size is not None and size < src_cloud.shape[0]:
        rand_ind = rng.choice(src_cloud.shape[0], size, replace=False)
        src_cloud = src_cloud[rand_ind]

    src_cloud = torch.from_numpy(src_cloud)

    if m > src_cloud.shape[0] or m is None:
        m = src_cloud.shape[0]

    # pdb.set_trace()
    v = wrap(src_cloud, beta=beta, Lambda=Lambda, m=m, seed=seed)
    ref_gt = src_cloud + v
    ref = ref_gt.clone()

    Ind_src = None

    if partial is not None:
        src_cloud, Ind_src = Select(src_cloud)
        ref, Ind_ref = Select(ref)

    if noise_level is not None:
        ref += torch.randn(ref.shape) * noise_level
        ref += torch.from_numpy(rng.randn(*(ref.shape)).astype(np.float32)) * noise_level

    if outlier is not None:
        M = torch.max(ref, 0)[0]
        m = torch.min(ref, 0)[0]
        Outlier = (torch.from_numpy(rng.rand(outlier, src_cloud.shape[1]).astype(np.float32)) - 0.5) * (M.unsqueeze(0)-m.unsqueeze(0)) * 1.2 + \
                  (M.unsqueeze(0) + m.unsqueeze(0)) / 2.
        ref = torch.cat([ref, Outlier])

    ref_np = ref.numpy()
    src_np = src_cloud.numpy()

    if Ind_src is None:
        Transformed_src = ref_gt.numpy()
    else:
        Transformed_src = ref_gt[Ind_src].numpy()

    return src_np, ref_np, Transformed_src

if __name__ == "__main__":
    print('------Data Generation--------- ')
    import argparse
    import os
    args_ps = argparse.ArgumentParser()
    args_ps.add_argument('--test_file', help='', default='dataset/fish_x.txt')
    args_ps.add_argument('--syn_Lambda', type=float, default=10)
    args_ps.add_argument('--syn_subsample', type=int, default=500)
    args_ps.add_argument('--syn_m', type=int, default=100)
    args_ps.add_argument('--syn_seed', type=int, default=None)
    args_ps.add_argument('--syn_noise_level', type=float, default=0.02)
    args_ps.add_argument('--syn_outlier', type=int, default=50)
    args_ps.add_argument('--syn_partial', type=float, default=None)
    args_ps.add_argument('--out_path', type=str, default='')
    args = args_ps.parse_args()

    src_cloud, ref_cloud, T_src = load(args.test_file, beta=2., Lambda=args.syn_Lambda,
                                                    m=args.syn_m, seed=args.syn_seed, noise_level=args.syn_noise_level,
                                                    outlier=args.syn_outlier, partial=args.syn_partial, size=args.syn_subsample)
    
    src_file = os.path.join(args.out_path, 'current_src.txt')
    ref_file = os.path.join(args.out_path, 'current_ref.txt')
    srcT_file = os.path.join(args.out_path, 'current_srcT.txt')

    np.savetxt(src_file, src_cloud, delimiter='\t', fmt='%.8f')
    np.savetxt(ref_file, ref_cloud, delimiter='\t', fmt='%.8f')
    np.savetxt(srcT_file, T_src, delimiter='\t', fmt='%.8f')
