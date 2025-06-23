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

def skew(v):
    """
    Convert a batch of 3-element vectors to a batch of 3x3 skew-symmetric matrices.
    Args:
        v: tensor of shape (*, 3)
    Returns:
        Tensor of shape (*, 3, 3)
    """
    # A zero tensor with the same batch shape as the input vector's components
    z = torch.zeros_like(v[..., 0])

    # Get components of the vector
    vx, vy, vz = v.unbind(dim=-1)

    # Build the skew-symmetric matrix by stacking the rows
    row1 = torch.stack([z, -vz, vy], dim=-1)
    row2 = torch.stack([vz, z, -vx], dim=-1)
    row3 = torch.stack([-vy, vx, z], dim=-1)
    
    S = torch.stack([row1, row2, row3], dim=-2)
    
    return S

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

def BA_hybrid(poses, velocities, patches, intrinsics, targets, weights, lmbda, ii, jj, kk, bounds, fixedp=1, structure_only=False,
              imu_preintegrations=None, imu_tensors = None):
    """
    Hybrid Bundle Adjustment implementation.
    
    It uses the memory-efficient Schur-complement solver logic from the original
    `_old_BA` function but integrates `pypose` for robust pose representation
    and updates. This function should be memory-equivalent to the original.
    """

    # --- State size is now 9 (6 for pose, 3 for velocity) ---
    STATE_SIZE = 9

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
    # --- CHANGE: Resize visual components for 9x9 state blocks ---
    # We pad the 6x6 visual matrices with zeros to fit into the new 9x9 state blocks.
    
    # z6x3 = torch.zeros(b, len(ii), 6, 3, device=poses_pp.device)
    # z3x6 = torch.zeros(b, len(ii), 3, 6, device=poses_pp.device)
    # z3x3 = torch.zeros(b, len(ii), 3, 3, device=poses_pp.device)

    # Ji9 = torch.cat([torch.cat([Ji, z6x3], dim=-1)], dim=-2)
    # Jj9 = torch.cat([torch.cat([Jj, z6x3], dim=-1)], dim=-2)

    v_vis_poses = safe_scatter_add_vec(vi, ii_adj, n_adj).view(b, n_adj, 1, 6, 1) + \
                  safe_scatter_add_vec(vj, jj_adj, n_adj).view(b, n_adj, 1, 6, 1)
    v_vis_vel = torch.zeros(b, n_adj, 1, 3, 1, device=poses_pp.device)
    v_vis9 = torch.cat([v_vis_poses, v_vis_vel], dim=-2) # Concatenate to form 9x1 gradient
    print("-----------")
    print(v_vis_poses.shape)
    print(v_vis9.shape)
    print("-----------")

    B_vis = safe_scatter_add_mat(Bii, ii_adj, ii_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
            safe_scatter_add_mat(Bij, ii_adj, jj_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
            safe_scatter_add_mat(Bji, jj_adj, ii_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
            safe_scatter_add_mat(Bjj, jj_adj, jj_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6)
    
    # Pad B_vis to be 9x9
    z_pad_B = torch.zeros(b, n_adj, n_adj, 9, 9, device=poses_pp.device)
    z_pad_B[..., :6, :6] = B_vis
    B_vis9 = z_pad_B

    E = safe_scatter_add_mat(Eik, ii_adj, kk_adj, n_adj, m).view(b, n_adj, m, 6, 1)
    # Pad E to be 9x1
    z_pad_E = torch.zeros(b, n_adj, m, 9, 1, device=poses_pp.device)
    z_pad_E[..., :6, :] = E
    E9 = z_pad_E

    C = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk_adj, m)
    w = safe_scatter_add_vec(torch.matmul(wJzT,  r), kk_adj, m)

    # --- 2. TIGHTLY-COUPLED IMU PART ---
    if imu_preintegrations is not None and not structure_only:
        g = torch.tensor([0, 0, -9.81], device=poses_pp.device, dtype=poses_pp.dtype)
        
        for idx_j in range(1,n): # Assuming imu_preintegrations is a list of factors

            idx_i = idx_j - 1 # The 'from' keyframe
            if idx_i < fixedp: # Don't apply constraints to the fixed frame
                continue

            # Adjust for the fixed frame
            ii_adj, jj_adj = idx_i - fixedp, idx_j - fixedp
            if ii_adj < 0: # Don't add factors connected to the fixed frame
                continue

            # Skip if states are outside the optimization window
            if idx_i < fixedp or idx_j >= n:
                continue


            imu_data = imu_preintegrations[idx_i]

            # Extract IMU measurements from your data structure
            delta_p_imu = imu_data['Dp'].cuda()
            delta_v_imu = imu_data['Dv'].cuda()
            delta_r_imu = pp.SO3(imu_data['Dr']).cuda() # Convert to SO3
            dt = imu_data['Dt'].cuda()
            
            
            pose_i, pose_j = poses_pp[:, idx_i], poses_pp[:, idx_j]
            vel_i, vel_j = velocities[:, idx_i], velocities[:, idx_j]
            # Note: biases (ba_i, bg_i, etc.) would also be extracted here
            

            
            # --- 2a. Calculate IMU Residual --- (You have likely done this)
            # R_i = pose_i.rotation()
            # t_i, t_j = pose_i.translation(), pose_j.translation()

            # res_r = (pimu.delta_r.Inv() * R_i.Inv() * pose_j.rotation()).Log()
            # res_v = R_i.Inv() @ (vel_j - vel_i - g*dt) - pimu.delta_v
            # res_p = R_i.Inv() @ (t_j - t_i - vel_i*dt - 0.5*g*dt**2) - pimu.delta_p

            R_i = pp.SE3(poses[:, idx_i]).rotation()
            R_j = pp.SE3(poses[:, idx_j]).rotation()
            R_i_inv = R_i.Inv()
            # Predicted rotation change, and (in body frame of i) velocity change and position change
            predicted_delta_r = R_i_inv * R_j.Inv()
            predicted_delta_v = R_i_inv @ (vel_j - vel_i - g * dt)
            predicted_delta_p = R_i_inv @ (
                pose_j.translation() - pose_i.translation() - vel_i * dt - 0.5 * g * dt**2
            )

            # --- 2. Calculate the 9-DOF IMU Residual (Error) ---
            res_r = (delta_r_imu.Inv() * predicted_delta_r).Log()
            res_v = predicted_delta_v - delta_v_imu
            res_p = predicted_delta_p - delta_p_imu 

            r_imu = torch.cat([res_r, res_v, res_p], dim=-1) # 9x1 residual

            R_i_mat = pose_i.rotation().matrix()
            R_j_mat = pose_j.rotation().matrix()
            R_i_inv_mat = pose_i.rotation().Inv().matrix()
            
                # # --- 2b. Calculate IMU Jacobians (The new part) ---
                # # This is the derivative of r_imu w.r.t state_i and state_j
                # # For simplicity, we use the standard VIO Jacobian formulations
                # # Let's build the 9x9 Jacobian J_i w.r.t state i [pose, vel]
                # J_r_pi = torch.zeros(3, 3); J_r_ri = -pose_j.rotation().Inv() * pose_i.rotation() # Approximated
                # J_v_pi = skew(R_i.Inv() @ (vel_j - vel_i - g*dt))
                # J_p_pi = skew(R_i.Inv() @ (pose_j.translation() - pose_i.translation() - vel_i*dt - 0.5*g*dt**2))
                
                # J_r_vi = torch.zeros(3, 3)
                # J_v_vi = -R_i.Inv().matrix()
                # J_p_vi = -R_i.Inv().matrix() * dt
                
                # J_i = torch.cat([
                #     torch.cat([J_r_ri, J_r_vi], dim=1),
                #     torch.cat([J_v_pi, J_v_vi], dim=1),
                #     torch.cat([J_p_pi, J_p_vi], dim=1),
                # ], dim=0) # This is a 9x6 Jacobian, needs padding to 9x9 if we add biases
                
                # # And the 9x9 Jacobian J_j w.r.t state j [pose, vel]
                # J_r_pj = torch.zeros(3,3); J_r_rj = torch.eye(3)
                # J_v_pj = torch.zeros(3,3); J_v_rj = torch.zeros(3,3)
                # J_p_pj = R_i.Inv().matrix(); J_p_rj = torch.zeros(3,3)
                
                # J_r_vj = torch.zeros(3,3)
                # J_v_vj = R_i.Inv().matrix()
                # J_p_vj = torch.zeros(3,3)

                # J_j = torch.cat([...]) # Assemble J_j similarly

            # --- 2b. Calculate IMU Jacobians (Corrected) ---
            # All Jacobians now use the explicit .matrix() tensors
            J_r_pi = torch.zeros(3, 3, device=poses_pp.device)
            # This Jacobian formulation is an approximation, a more exact one can be derived.
            J_r_ri = -R_j_mat.mT @ R_i_mat


            J_v_pi = skew( (R_i_inv_mat @ (vel_j - vel_i - g * dt).reshape(-1,1)).squeeze() )
            J_p_pi = skew( (R_i_inv_mat @ (pose_j.translation() - pose_i.translation() - vel_i * dt - 0.5 * g * dt**2).reshape(-1,1)).squeeze() )
            
            J_r_vi = torch.zeros(3, 3, device=poses_pp.device)
            J_v_vi = -R_i_inv_mat
            J_p_vi = -R_i_inv_mat * dt
            
            
            print(J_r_pi.shape, J_r_vi.shape)
            print(J_v_pi.shape, J_v_vi.shape)
            print(J_p_pi.shape, J_p_vi.shape)
            
            # Jacobian of residual w.r.t. state i (pose, velocity)
            J_i = torch.cat([
                torch.cat([J_r_pi, J_r_vi], dim=1),
                torch.cat([J_v_pi, J_v_vi.squeeze()], dim=1),
                torch.cat([J_p_pi, J_p_vi.squeeze()], dim=1),
            ], dim=0)
            
            # Jacobian of residual w.r.t. state j (pose, velocity)
            J_r_pj = torch.zeros(3, 3, device=poses_pp.device)
            J_r_rj = R_i_inv_mat @ R_j_mat
            
            J_v_pj = torch.zeros(3, 3, device=poses_pp.device)
            J_v_rj = torch.zeros(3, 3, device=poses_pp.device)
            
            J_p_pj = R_i_inv_mat
            J_p_rj = torch.zeros(3, 3, device=poses_pp.device)

            J_r_vj = torch.zeros(3, 3, device=poses_pp.device)
            J_v_vj = R_i_inv_mat
            J_p_vj = torch.zeros(3, 3, device=poses_pp.device)
            
            print(J_r_pj.shape, J_r_vj.shape)
            print(J_v_pj.shape, J_v_vj.shape)
            print(J_p_pj.shape, J_p_vj.shape)

            J_j = torch.cat([
                torch.cat([J_r_pj, J_r_vj], dim=1),
                torch.cat([J_v_pj, J_v_vj.squeeze()], dim=1),
                torch.cat([J_p_pj.squeeze(), J_p_vj], dim=1)
            ], dim=0)


            # --- 2c. Augment Hessian and Gradient ---
            # H_imu = J_imu^T * J_imu
            # g_imu = -J_imu^T * r_imu
            i_adj, j_adj = idx_i - fixedp, idx_j - fixedp
            
            # Add to gradient vector
            g_i = -J_i.T @ r_imu.T
            g_j = -J_j.T @ r_imu.T
            print("===========")
            print(g_i)
            print( v_vis9[:, i_adj])
            print("===========")
            v_vis9[:, i_adj] += g_i.view(1,1,STATE_SIZE,1)
            v_vis9[:, j_adj] += g_j.view(1,1,STATE_SIZE,1)

            # Add to Hessian matrix (THIS IS THE TIGHT COUPLING)
            H_ii = J_i.T @ J_i
            H_ij = J_i.T @ J_j
            H_ji = J_j.T @ J_i
            H_jj = J_j.T @ J_j
            B_vis9[:, i_adj, i_adj] += H_ii
            B_vis9[:, i_adj, j_adj] += H_ij
            B_vis9[:, j_adj, i_adj] += H_ji
            B_vis9[:, j_adj, j_adj] += H_jj
            
    # --- 3. SOLVE & UPDATE ---
    Q = 1.0 / (C + lmbda)
    # --- CHANGE: Use the 9-DoF E matrix ---
    EQ = block_matmul(E9, Q.view(b, m, 1, 1, 1))

    if structure_only or n_adj == 0:
        dZ = (Q * w).view(b, -1, 1, 1)
        dX = torch.zeros(b, n_adj, STATE_SIZE, device=poses_pp.device)
    else:
        # --- CHANGE: Use the 9-DoF B and v matrices ---
        S = B_vis9 - block_matmul(EQ, E9.permute(0,2,1,4,3))
        y = v_vis9 - block_matmul(EQ, w.unsqueeze(dim=2))
        
        dX = block_solve(S, y, ep=1.0, lm=1e-4) # block_solve now works on 9x9 blocks
        dZ = Q * (w - block_matmul(E9.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        
        dX = dX.view(b, -1, STATE_SIZE)
        dZ = dZ.view(b, -1, 1, 1)

    # --- 4. APPLY UPDATES ---
    # Structure update is unchanged
    x_coords, y_coords, disps = patches.unbind(dim=2)
    disps_updated = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches_updated = torch.stack([x_coords, y_coords, disps_updated], dim=2)

    if not structure_only and n_adj > 0:
        with torch.no_grad():
            # --- CHANGE: Parse 9-DoF update vector dX ---
            dx_poses = dX[..., :6] # First 6 elements are for pose
            dx_vels = dX[..., 6:]  # Next 3 elements are for velocity

            update_indices = fixedp + torch.arange(n_adj, device=poses_pp.device)
            
            # Update Poses
            pose_update_vector = torch.zeros(b, n, 6, device=poses_pp.device, dtype=dX.dtype)
            pose_update_vector.scatter_add_(1, update_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 6), dx_poses)
            poses_pp = poses_pp.Retr(pp.se3(pose_update_vector))

            # Update Velocities
            velocities.scatter_add_(-1, update_indices, dx_vels.squeeze(0))

    return poses_pp.data, velocities, patches_updated


# --- TOP-LEVEL WRAPPER (Largely Unchanged) ---
# This function handles windowing and iterations, calling the hybrid solver.

def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations,
       imu_preintegrations=None, imu_tensors = None, velocities = None, **kwargs):
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
        velocities_active = velocities[:, :n].clone()

        # n == t1
        
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
            velocities=velocities_active,
            patches=patches_window,
            intrinsics=intrinsics,
            targets=target_filt,
            weights=weight_filt,
            lmbda=lmbda,
            ii=ii_filt,
            jj=jj_filt,
            kk=kk_rel,
            bounds=bounds_arg,
            fixedp=fixedp,
            imu_preintegrations=imu_preintegrations
        )
        
        # --- FIX 2: Update the full set of optimized poses ---
        # The result `poses_updated` has shape (1, n, 7). We update the
        # corresponding slice in the main `poses` object.
        poses.data[:, t0:n] = poses_updated[:, t0:n]
        patches.data[:, unique_kk_filt] = patches_updated

    return poses, patches
