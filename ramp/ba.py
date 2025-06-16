import torch
from torch_scatter import scatter_sum

from . import lietorch
from .lietorch import SE3

from .utils import Timer

from . import projective_ops as pops

# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))

def block_matmul(A, B):
    """ block matrix multiply """
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)
    return torch.matmul(A, B).reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)


def block_show(A):
    import matplotlib.pyplot as plt
    b, n1, m1, p1, q1 = A.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    plt.imshow(A[0].detach().cpu().numpy())
    plt.show()

class CholeskySolver(torch.autograd.Function):
    """ Custom Cholesky solver to avoid crashes with non-positive-definite matrices """
    @staticmethod
    def forward(ctx, H, b):
        try:
            U = torch.linalg.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except torch.linalg.LinAlgError:
            ctx.failed = True
            return torch.zeros_like(b)
        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None
        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))
        return dH, dz

def block_solve(A, B, ep=1.0, lm=1e-4):
    """ 
    Block matrix solve with Levenberg-Marquardt damping.
    A is the block Hessian, B is the block residual vector.
    """
    # A is the Hessian S, shape (b, n_blocks, n_blocks, p, q) e.g., (1, 6, 6, 6, 6)
    # B is the residual vector y, shape (b, n_blocks, 1, p, 1) e.g., (1, 6, 1, 6, 1)
    b, n_blocks, m_blocks, p, q = A.shape
    A_flat = A.permute(0, 1, 3, 2, 4).reshape(b, n_blocks * p, m_blocks * q)

    # Its shape is (b, n_blocks, 1, p, 1), so we reshape it to (b, n_blocks * p, 1).
    B_flat = B.reshape(b, n_blocks * p, 1)

    # Apply damping directly to the diagonal of the flattened Hessian
    diag_indices = torch.arange(n_blocks * p, device=A.device)
    A_flat[:, diag_indices, diag_indices] += ep
    A_flat[:, diag_indices, diag_indices] += lm * A_flat[:, diag_indices, diag_indices].clone().detach()

    # Solve the flattened system
    X_flat = CholeskySolver.apply(A_flat, B_flat) # Result X_flat is shape (b, n_blocks * p, 1)
    
    # Reshape the result back to the block structure of B
    return X_flat.reshape(b, n_blocks, 1, p, 1)


def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations, eff_impl=False, imu_preintegrations=None):
    """
    Wrapper for the Python-based Bundle Adjustment.
    """
    
    for _ in range(iterations):
        # poses_window = poses[:, t0:t1].clone()
        poses_window = poses.clone()

        mask = (ii >= t0) & (ii < t1) & (jj >= t0) & (jj < t1)
        
        # ii_filt, jj_filt, kk_filt = ii[mask], jj[mask], kk[mask]
        # target_filt, weight_filt = target[:, mask], weight[:, mask]
        # ii_rel, jj_rel = ii_filt - t0, jj_filt - t0

        ii_filt, jj_filt, kk_filt = ii, jj, kk
        target_filt, weight_filt = target, weight
        ii_rel, jj_rel = ii_filt, jj_filt

        unique_kk_filt, kk_rel = torch.unique(kk_filt, return_inverse=True)
        patches_window = patches[:, unique_kk_filt].clone()

        fx, fy, cx, cy = intrinsics.unbind(dim=-1)
        ht = (cy.view(-1)[0].item() * 2)
        wd = (cx.view(-1)[0].item() * 2)
        bounds_arg = (0, 0, wd, ht)

        fixedp = 1
        
        poses_updated, patches_updated = _old_BA(
            poses=poses_window,
            patches=patches_window,
            intrinsics=intrinsics,
            targets=target_filt,
            weights=weight_filt,
            lmbda=lmbda,
            ii=ii_rel,
            jj=jj_rel,
            kk=kk_rel,
            bounds=bounds_arg,
            fixedp=fixedp, 
            imu_preintegrations=None)

        poses[:, t0:t1] = poses_updated[:, t0:t1]    
        patches[:, unique_kk_filt] = patches_updated

    return poses, patches


def _old_BA(poses, patches, intrinsics, targets, weights, lmbda, ii, jj, kk, bounds, ep=1.0, PRINT=False, fixedp=1, structure_only=False, imu_preintegrations=None):
    """ Original Python-based bundle adjustment implementation. """

    b = 1
    if ii.numel() > 0:
        n = max(ii.max().item(), jj.max().item()) + 1
    else:
        return poses, patches

    coords, v, (Ji, Jj, Jz) = \
        pops.transform(SE3(poses), patches, intrinsics, ii, jj, kk, jacobian=True)
    
    patch_size = coords.shape[3]
    r = targets - coords[..., patch_size//2, patch_size//2, :]

    v *= (r.norm(dim=-1) < 250).float()
    in_bounds = \
        (coords[...,patch_size//2,patch_size//2,0] > bounds[0]) & \
        (coords[...,patch_size//2,patch_size//2,1] > bounds[1]) & \
        (coords[...,patch_size//2,patch_size//2,0] < bounds[2]) & \
        (coords[...,patch_size//2,patch_size//2,1] < bounds[3])
    v *= in_bounds.float()

    if PRINT:
        print("Mean Error:", (r * v[...,None]).norm(dim=-1).mean().item())

    r = (v[...,None] * r).unsqueeze(dim=-1)    
    weights = (v[...,None] * weights).unsqueeze(dim=-1)

    wJiT = (weights * Ji).transpose(2,3)
    wJjT = (weights * Jj).transpose(2,3)
    wJzT = (weights * Jz).transpose(2,3)

    Bii = torch.matmul(wJiT, Ji)
    Bij = torch.matmul(wJiT, Jj)
    Bji = torch.matmul(wJjT, Ji)
    Bjj = torch.matmul(wJjT, Jj)
    Eik = torch.matmul(wJiT, Jz)
    Ejk = torch.matmul(wJjT, Jz)
    vi = torch.matmul(wJiT, r)
    vj = torch.matmul(wJjT, r)

    #TODO(?): fixed dp from 1 to 1..6 (cuda version)

    n_adj = n - fixedp
    ii_adj, jj_adj = ii - fixedp, jj - fixedp
    kx, kk_adj = torch.unique(kk, return_inverse=True, sorted=True)
    m = len(kx)
    
    if n_adj <= 0 or m == 0:
        return poses, patches

    B = safe_scatter_add_mat(Bii, ii_adj, ii_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
        safe_scatter_add_mat(Bij, ii_adj, jj_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
        safe_scatter_add_mat(Bji, jj_adj, ii_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
        safe_scatter_add_mat(Bjj, jj_adj, jj_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6)

    E = safe_scatter_add_mat(Eik, ii_adj, kk_adj, n_adj, m).view(b, n_adj, m, 6, 1) + \
        safe_scatter_add_mat(Ejk, jj_adj, kk_adj, n_adj, m).view(b, n_adj, m, 6, 1) 

    v = safe_scatter_add_vec(vi, ii_adj, n_adj).view(b, n_adj, 1, 6, 1) + \
        safe_scatter_add_vec(vj, jj_adj, n_adj).view(b, n_adj, 1, 6, 1)
    
    C = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk_adj, m)

    w = safe_scatter_add_vec(torch.matmul(wJzT,  r), kk_adj, m)
    Q = 1.0 / (C + lmbda)
    
    EQ = E * Q[:,None]

    if structure_only or n_adj == 0:
        dZ = (Q * w).view(b, -1, 1, 1)
        dX = torch.zeros(b, n_adj, 6, device=poses.device)
    else:
        S = B - block_matmul(EQ, E.permute(0,2,1,4,3))
        y = v - block_matmul(EQ, w.unsqueeze(dim=2))
        
        dX = block_solve(S, y, ep=ep, lm=1e-4)
        dZ = Q * (w - block_matmul(E.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        
        dX = dX.view(b, -1, 6)
        dZ = dZ.view(b, -1, 1, 1)

    x_coords, y_coords, disps = patches.unbind(dim=2)
    disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches = torch.stack([x_coords, y_coords, disps], dim=2)

    if not structure_only and n_adj > 0:
        update_indices = fixedp + torch.arange(n_adj, device=poses.device)
        poses = pose_retr(SE3(poses), dX, update_indices)

    if isinstance(poses, SE3):
        poses = poses.data

    return poses, patches