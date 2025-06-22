import torch
from torch_scatter import scatter_sum
import pypose as pp

# These are the original dependencies your code used
from . import lietorch
from .lietorch import SE3
from . import projective_ops as pops

# --- UNCHANGED UTILITY FUNCTIONS from ba_copy.py ---
# These functions are kept as they are essential for the memory-efficient
# block-sparse matrix assembly.

def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

def block_matmul(A, B):
    """ block matrix multiply """
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)
    return torch.matmul(A, B).reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)

class CholeskySolver(torch.autograd.Function):
    """ Custom Cholesky solver to avoid crashes with non-positive-definite matrices """
    @staticmethod
    def forward(ctx, H, b):
        try:
            # Add a small value to the diagonal for stability before Cholesky
            # This is a common practice in non-linear optimization
            H_stable = H + torch.eye(H.shape[-1], device=H.device, dtype=H.dtype) * 1e-6
            U = torch.linalg.cholesky(H_stable)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except torch.linalg.LinAlgError:
            ctx.failed = True
            # Return a zero update if the solve fails
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
    """ Block matrix solve with Levenberg-Marquardt damping. """
    b, n_blocks, m_blocks, p, q = A.shape
    A_flat = A.permute(0, 1, 3, 2, 4).reshape(b, n_blocks * p, m_blocks * q)
    B_flat = B.reshape(b, n_blocks * p, 1)

    diag_indices = torch.arange(n_blocks * p, device=A.device)
    # Add LM damping to the diagonal
    A_flat[:, diag_indices, diag_indices] += ep + lm * A_flat[:, diag_indices, diag_indices].clone().detach()
    
    X_flat = CholeskySolver.apply(A_flat, B_flat)
    return X_flat.reshape(b, n_blocks, 1, p, 1)


# --- NEW HYBRID BUNDLE ADJUSTMENT SOLVER ---

def BA_hybrid(poses, patches, intrinsics, targets, weights, lmbda, ii, jj, kk, bounds, fixedp=1, structure_only=False):
    """
    Hybrid Bundle Adjustment implementation.
    
    It uses the memory-efficient Schur-complement solver logic from the original
    `_old_BA` function but integrates `pypose` for robust pose representation
    and updates. This function should be memory-equivalent to the original.
    """

    b = 1
    if ii.numel() > 0:
        n = max(ii.max().item(), jj.max().item()) + 1
    else:
        return poses, patches

    # --- HYBRID STEP 1: Represent poses with pypose.SE3 ---
    # We convert the input tensor to a pypose object at the beginning.
    # The `poses` input is expected to be a raw tensor of shape (b, N, 7)
    poses_pp = pp.SE3(poses)

    # --- UNCHANGED SOLVER LOGIC (FORWARD PASS & JACOBIAN CALCULATION) ---
    # This entire block is preserved from your original code because it is
    # what makes the implementation memory-efficient. It computes analytical
    # jacobians and never forms the full J matrix.
    
    # We pass the pypose object directly to transform. `pops.transform` is
    # compatible as it can operate on the underlying tensor data.
    coords, v_vis, (Ji, Jj, Jz) = \
            pops.transform(SE3(poses_pp.data), patches, intrinsics, ii, jj, kk, jacobian=True)
    
    patch_size = coords.shape[3]
    r = targets - coords[..., patch_size//2, patch_size//2, :]

    v_vis *= (r.norm(dim=-1) < 250).float()
    in_bounds = \
        (coords[...,patch_size//2,patch_size//2,0] > bounds[0]) & \
        (coords[...,patch_size//2,patch_size//2,1] > bounds[1]) & \
        (coords[...,patch_size//2,patch_size//2,0] < bounds[2]) & \
        (coords[...,patch_size//2,patch_size//2,1] < bounds[3])
    v_vis *= in_bounds.float()

    r = (v_vis[...,None] * r).unsqueeze(dim=-1)    
    weights = (v_vis[...,None] * weights).unsqueeze(dim=-1)

    # --- UNCHANGED SOLVER LOGIC (HESSIAN & GRADIENT ASSEMBLY) ---
    # This avoids creating the full Jacobian by directly building the blocks
    # of the Gauss-Newton Hessian approximation (J^T * J).
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

    n_adj = n - fixedp
    ii_adj, jj_adj = ii - fixedp, jj - fixedp
    kx, kk_adj = torch.unique(kk, return_inverse=True, sorted=True)
    m = len(kx)
    
    if n_adj <= 0 or m == 0:
        return poses_pp.data, patches

    B_vis = safe_scatter_add_mat(Bii, ii_adj, ii_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
        safe_scatter_add_mat(Bij, ii_adj, jj_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
        safe_scatter_add_mat(Bji, jj_adj, ii_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
        safe_scatter_add_mat(Bjj, jj_adj, jj_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6)

    E = safe_scatter_add_mat(Eik, ii_adj, kk_adj, n_adj, m).view(b, n_adj, m, 6, 1) + \
        safe_scatter_add_mat(Ejk, jj_adj, kk_adj, n_adj, m).view(b, n_adj, m, 6, 1) 

    v_vis = safe_scatter_add_vec(vi, ii_adj, n_adj).view(b, n_adj, 1, 6, 1) + \
        safe_scatter_add_vec(vj, jj_adj, n_adj).view(b, n_adj, 1, 6, 1)
    
    C = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk_adj, m)
    w = safe_scatter_add_vec(torch.matmul(wJzT,  r), kk_adj, m)
    
    # --- UNCHANGED SOLVER LOGIC (SCHUR COMPLEMENT SOLVE) ---
    # This is the memory-saving step. We solve a smaller system for poses (dX)
    # and then back-substitute to find the structure update (dZ).
    Q = 1.0 / (C + lmbda)
    EQ = E * Q[:,None]

    if structure_only or n_adj == 0:
        dZ = (Q * w).view(b, -1, 1, 1)
        dX = torch.zeros(b, n_adj, 6, device=poses_pp.device)
    else:
        S = B_vis - block_matmul(EQ, E.permute(0,2,1,4,3))
        y = v_vis - block_matmul(EQ, w.unsqueeze(dim=2))
        
        dX = block_solve(S, y, ep=1.0, lm=1e-4)
        dZ = Q * (w - block_matmul(E.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        
        dX = dX.view(b, -1, 6)
        dZ = dZ.view(b, -1, 1, 1)

    # --- HYBRID STEP 2: Apply updates using pypose retraction ---
    # The structure update is a simple addition, so it remains unchanged.
    x_coords, y_coords, disps = patches.unbind(dim=2)
    disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches_updated = torch.stack([x_coords, y_coords, disps], dim=2)

    # The pose update uses pypose's `retr` method for a clean, robust update
    # on the SE(3) manifold. This replaces the old `pose_retr` function call.
    if not structure_only and n_adj > 0:
        with torch.no_grad():
            update_indices = fixedp + torch.arange(n_adj, device=poses_pp.device)
            # Create the sparse update vector for the retraction
            update_vector = torch.zeros(b, n, 6, device=poses_pp.device, dtype=dX.dtype)
            update_vector.scatter_add_(1, update_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 6), dX)
            
            # Apply the retraction to the full set of poses
            update = pp.se3(update_vector)
            poses_pp = poses_pp.Retr(update)

    # Return the raw tensor data to maintain compatibility with the rest of the system
    return poses_pp.data, patches_updated


# --- TOP-LEVEL WRAPPER (Largely Unchanged) ---
# This function handles windowing and iterations, calling the hybrid solver.

def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations, **kwargs):
    """
    Top-level wrapper for the Hybrid Bundle Adjustment.
    This version performs a full BA over all provided poses and factors.
    """
    
    # The `imu_preintegrations` etc. from the original are ignored as requested
    
    for _ in range(iterations):
        # --- FIX 1: Determine the actual number of poses in the graph ---
        # This ensures we don't pass more poses than are actually being optimized.
        if ii.numel() > 0:
            n = max(ii.max().item(), jj.max().item()) + 1
        else:
            # If there are no factors, there's nothing to do.
            return poses, patches
        
        
        # Pass only the active poses to the solver.
        poses_active = poses.data[:, :n].clone()
        
        # The rest of the inputs use all available factors, as you intended
        ii_filt, jj_filt, kk_filt = ii, jj, kk
        target_filt, weight_filt = target, weight

        # Patch selection logic remains the same
        unique_kk_filt, kk_rel = torch.unique(kk_filt, return_inverse=True)
        patches_window = patches[:, unique_kk_filt].clone()

        fx, fy, cx, cy = intrinsics.unbind(dim=-1)
        ht = (cy.view(-1)[0].item() * 2)
        wd = (cx.view(-1)[0].item() * 2)
        bounds_arg = (0, 0, wd, ht)

        fixedp = 1
        
        # Call the hybrid solver
        poses_updated, patches_updated = BA_hybrid(
            poses=poses_active,
            patches=patches_window,
            intrinsics=intrinsics,
            targets=target_filt,
            weights=weight_filt,
            lmbda=lmbda,
            ii=ii_filt,  # Note: indices are not relative anymore
            jj=jj_filt,
            kk=kk_rel,
            bounds=bounds_arg,
            fixedp=fixedp
        )
        
        # --- FIX 2: Update the full set of optimized poses ---
        # The result `poses_updated` has shape (1, n, 7). We update the
        # corresponding slice in the main `poses` object.
        poses.data[:, t0:n] = poses_updated[:, t0:n]
        patches.data[:, unique_kk_filt] = patches_updated

    return poses, patches
