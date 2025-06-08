import torch
from torch_scatter import scatter_sum

from . import fastba
from . import lietorch
from .lietorch import SE3

from .utils import Timer

from . import projective_ops as pops

class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        U, info = torch.linalg.cholesky_ex(H)

        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(b)

        xs = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        ctx.failed = False

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

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

def block_solve(A, B, ep=1.0, lm=1e-4):
    """ block matrix solve """
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)

    A = A + (ep + lm * A) * torch.eye(n1*p1, device=A.device)

    X = CholeskySolver.apply(A, B)
    return X.reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)


def block_show(A):
    import matplotlib.pyplot as plt
    b, n1, m1, p1, q1 = A.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    plt.imshow(A[0].detach().cpu().numpy())
    plt.show()


def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations, eff_impl=False, **kwargs):
    """
    Wrapper for the Python-based Bundle Adjustment.

    This function matches the interface of the CUDA-based `fastba.BA`. It takes
    the full history of poses and patches and internally handles the windowing
    of data from index `t0` to `t1`. It then calls the original BA
    implementation (`_old_BA`) for the specified number of iterations.
    """
    
    # Run the bundle adjustment for the specified number of iterations
    for _ in range(iterations):
        # The BA window is defined by the range [t0, t1)
        poses_window = poses[:, t0:t1].clone()
        
        # Create a boolean mask to find all connections where both poses
        # are within the active optimization window.
        mask = (ii >= t0) & (ii < t1) & (jj >= t0) & (jj < t1)
        
        # Filter all connection-related tensors using the mask
        ii_filt = ii[mask]
        jj_filt = jj[mask]
        kk_filt = kk[mask]
        
        # Apply the mask to the second dimension (the connections dimension)
        target_filt = target[:, mask]
        weight_filt = weight[:, mask]

        # Remap the absolute pose indices to be relative to the window start (t0).
        # The new indices will be in the range [0, t1-t0).
        ii_rel = ii_filt - t0
        jj_rel = jj_filt - t0
        
        # Find the unique patches that are observed within this window and
        # remap the patch indices (kk) to be relative to this new subset of patches.
        unique_kk_filt, kk_rel = torch.unique(kk_filt, return_inverse=True)
        patches_window = patches[:, unique_kk_filt].clone()

        # Define image boundaries for the visibility check inside _old_BA.
        # We infer the image width and height from the camera intrinsics (cx, cy),
        # assuming the principal point is at the image center.
        # FIX: The intrinsics tensor might be a batch. We handle this by taking
        # the first element, as image size is constant within a sequence.
        fx, fy, cx, cy = intrinsics.unbind(dim=-1)
        ht = (cy.view(-1)[0].item() * 2)
        wd = (cx.view(-1)[0].item() * 2)
        bounds_arg = (0, 0, wd, ht)

        # In each window, the first pose is held fixed.
        fixedp = 1
        
        # Call the original BA implementation with the prepared windowed data
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
            **kwargs)

        # Update the original full tensors with the optimized results from the window
        poses[:, t0:t1] = poses_updated
        patches[:, unique_kk_filt] = patches_updated

    return poses, patches


def _old_BA(poses, patches, intrinsics, targets, weights, lmbda, ii, jj, kk, bounds, ep=100.0, PRINT=False, fixedp=1, structure_only=False):
    """ Original Python-based bundle adjustment implementation. """

    b = 1 # batch size
    # n is the number of poses in the current window. Add 1 because indices are 0-based.
    if ii.numel() > 0:
        n = max(ii.max().item(), jj.max().item()) + 1
    else:
        # If there are no connections, there's nothing to optimize.
        return poses, patches


    # Project patches into image space and compute jacobians
    coords, v, (Ji, Jj, Jz) = pops.transform(SE3(poses), patches, intrinsics, ii, jj, kk, jacobian=True)
    
    patch_size = coords.shape[3]
    # Compute reprojection error (residual)
    r = targets - coords[..., patch_size//2, patch_size//2, :]

    # --- Visibility and bounds checking ---
    # Invalidate points with very large reprojection error
    v *= (r.norm(dim=-1) < 250).float()
    # Invalidate points that project outside of the image boundaries
    in_bounds = \
        (coords[...,patch_size//2,patch_size//2,0] > bounds[0]) & \
        (coords[...,patch_size//2,patch_size//2,1] > bounds[1]) & \
        (coords[...,patch_size//2,patch_size//2,0] < bounds[2]) & \
        (coords[...,patch_size//2,patch_size//2,1] < bounds[3])
    v *= in_bounds.float()

    if PRINT:
        print("Mean Error:", (r * v[...,None]).norm(dim=-1).mean().item())

    # Weight the residuals
    r = (v[...,None] * r).unsqueeze(dim=-1)    
    weights = (v[...,None] * weights).unsqueeze(dim=-1)

    # --- Gauss-Newton System Construction ---
    # See paper supplementary for details on these matrices
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

    # Adjust pose indices to account for fixed poses
    n_adj = n - fixedp
    ii_adj = ii - fixedp
    jj_adj = jj - fixedp

    # Get unique patch indices for this window
    kx, kk_adj = torch.unique(kk, return_inverse=True, sorted=True)
    m = len(kx) # number of unique patches
    
    # If there are no adjustable poses or patches, return early.
    if n_adj <= 0 or m == 0:
        return poses, patches

    # Assemble the Hessian blocks using scatter operations
    B = safe_scatter_add_mat(Bii, ii_adj, ii_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
        safe_scatter_add_mat(Bij, ii_adj, jj_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
        safe_scatter_add_mat(Bji, jj_adj, ii_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
        safe_scatter_add_mat(Bjj, jj_adj, jj_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6)

    E = safe_scatter_add_mat(Eik, ii_adj, kk_adj, n_adj, m).view(b, n_adj, m, 6, 1) + \
        safe_scatter_add_mat(Ejk, jj_adj, kk_adj, n_adj, m).view(b, n_adj, m, 6, 1) 

    C = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk_adj, m)

    # Assemble the residual vector blocks
    v = safe_scatter_add_vec(vi, ii_adj, n_adj).view(b, n_adj, 1, 6, 1) + \
        safe_scatter_add_vec(vj, jj_adj, n_adj).view(b, n_adj, 1, 6, 1)

    w = safe_scatter_add_vec(torch.matmul(wJzT,  r), kk_adj, m)


    Q = 1.0 / (C + lmbda)
    
    # --- Solve the linear system using Schur Complement ---
    EQ = E * Q[:,None]

    if structure_only or n_adj == 0:
        # Solve only for structure (patches)
        dZ = (Q * w).view(b, -1, 1, 1)
        dX = torch.zeros(b, n_adj, 6, device=poses.device)
    else:
        # Solve for poses and structure
        S = B - block_matmul(EQ, E.permute(0,2,1,4,3))
        y = v - block_matmul(EQ, w.unsqueeze(dim=2))
        
        dX = block_solve(S, y, ep=ep, lm=1e-4)
        dZ = Q * (w - block_matmul(E.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        
        dX = dX.view(b, -1, 6)
        dZ = dZ.view(b, -1, 1, 1)

    # --- Update state estimates ---
    x_coords, y_coords, disps = patches.unbind(dim=2)
    # kx contains the unique indices into the windowed patch tensor
    disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches = torch.stack([x_coords, y_coords, disps], dim=2)

    if not structure_only and n_adj > 0:
        # Update poses using the computed delta
        update_indices = fixedp + torch.arange(n_adj, device=poses.device)
        poses = pose_retr(SE3(poses), dX, update_indices)
    
    # To rectify before returning to new ramp vo
    if isinstance(poses, SE3):
        poses = poses.data

    return poses, patches

