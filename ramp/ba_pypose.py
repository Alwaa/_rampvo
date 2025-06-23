import torch
from torch_scatter import scatter_sum
import pypose as pp

from . import lietorch
from .lietorch import SE3
from . import projective_ops as pops

# --- FUNCTIONS from ba_copy.py ---

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
              imu_preintegrations=None, imu_tensors = None, scale = None):
    """
    Hybrid Bundle Adjustment implementation.
    """

    # --- State size is now 9 (6 for pose, 3 for velocity) ---
    STATE_SIZE = 9

    b = 1
    if ii.numel() > 0:
        n = max(ii.max().item(), jj.max().item()) + 1
    else:
        return poses, patches

    # --- Represent poses with pypose.SE3 ---
    # from raw tensor of shape (b, N, 7)
    poses_pp = pp.SE3(poses)

    # Use LieTorch SE3 here
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
    
    # ---  9x9 state blocks ---
    v_vis_poses = safe_scatter_add_vec(vi, ii_adj, n_adj).view(b, n_adj, 1, 6, 1) + \
                  safe_scatter_add_vec(vj, jj_adj, n_adj).view(b, n_adj, 1, 6, 1)
    
    v_vis_vel = torch.zeros(b, n_adj, 1, 3, 1, device=poses_pp.device)
    v_vis9 = torch.cat([v_vis_poses, v_vis_vel], dim=-2) # Concatenate to form 9x1 gradient

    print("-----------")
    print("VIS:", torch.norm(r).item())
    print("-----------")

    B_vis = safe_scatter_add_mat(Bii, ii_adj, ii_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
            safe_scatter_add_mat(Bij, ii_adj, jj_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
            safe_scatter_add_mat(Bji, jj_adj, ii_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6) + \
            safe_scatter_add_mat(Bjj, jj_adj, jj_adj, n_adj, n_adj).view(b, n_adj, n_adj, 6, 6)
    
    # Pad B_vis to be 9x9
    z_pad_B = torch.zeros(b, n_adj, n_adj, 9, 9, device=poses_pp.device)
    z_pad_B[..., :6, :6] = B_vis
    B_vis9 = z_pad_B

    E = safe_scatter_add_mat(Eik, ii_adj, kk_adj, n_adj, m).view(b, n_adj, m, 6, 1)+ \
        safe_scatter_add_mat(Ejk, jj_adj, kk_adj, n_adj, m).view(b, n_adj, m, 6, 1) 
    

    # Pad E to be 9x1
    z_pad_E = torch.zeros(b, n_adj, m, 9, 1, device=poses_pp.device)
    z_pad_E[..., :6, :] = E
    E9 = z_pad_E

    C = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk_adj, m)
    w = safe_scatter_add_vec(torch.matmul(wJzT,  r), kk_adj, m)

    # --- TIGHTLY-COUPLED IMU PART ---
    if imu_preintegrations is not None and not structure_only:
        g = torch.tensor([0, 0, +9.81], device=poses_pp.device, dtype=poses_pp.dtype)

        r_, p_, v_ = 0,0,0 
        scale = torch.tensor([0.2], device=poses_pp.device, dtype=poses_pp.dtype)
        H_ss = torch.zeros(1, 1, device=poses_pp.device, dtype=poses_pp.dtype) # 1x1 Hessian for scale
        g_s = torch.zeros(1, 1, device=poses_pp.device, dtype=poses_pp.dtype)  # 1x1 Gradient for scale
        
        for idx_j in range(max(1, n-2000),n): # Assuming imu_preintegrations is a list of factors

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
            delta_r_imu = imu_data['Dr'].cuda()
            dt = imu_data['Dt'].cuda()
            
            
            pose_i = poses_pp[:, idx_i].clone()
            pose_j = poses_pp[:, idx_j].clone()
            vel_i = velocities[:, idx_i].clone()
            vel_j = velocities[:, idx_j].clone()
            # Note: biases (ba_i, bg_i, etc.) would also be extracted here

            metric_vel_i, metric_vel_j = vel_i.clone() * scale, vel_j.clone() * scale
            pose_i_data = pose_i.data
            pose_j_data = pose_j.data
            pose_i_data[:3] = pose_i_data[:3] * scale
            pose_j_data[:3] = pose_j_data[:3] * scale
            metric_pose_i = pp.SE3(pose_i_data).Inv()
            metric_pose_j = pp.SE3(pose_j_data).Inv()

            vel_i, vel_j = metric_vel_i, metric_vel_j
            pose_i, pose_j = metric_pose_i, metric_pose_j            

            
            # --- 2a. Calculate IMU Residual ---
            # R_i = pose_i.rotation()
            # t_i, t_j = pose_i.translation(), pose_j.translation()

            # res_r = (pimu.delta_r.Inv() * R_i.Inv() * pose_j.rotation()).Log()
            # res_v = R_i.Inv() @ (vel_j - vel_i - g*dt) - pimu.delta_v
            # res_p = R_i.Inv() @ (t_j - t_i - vel_i*dt - 0.5*g*dt**2) - pimu.delta_p

            R_i = pose_i.rotation()
            R_j = pose_j.rotation()
            R_i_inv = R_i.Inv()
            # Predicted rotation change, and (in body frame of i) velocity change and position change
            predicted_delta_r = R_i_inv * R_j
            predicted_delta_v = R_i_inv @ (metric_vel_j - metric_vel_i + g * dt)
            predicted_delta_p = R_i_inv @ (
                metric_pose_j.translation() - metric_pose_i.translation() - metric_vel_i * dt + 0.5 * g * dt**2
            )

            # --- 2. Calculate the 9-DOF IMU Residual (Error) ---
            res_r = (delta_r_imu.Inv() * predicted_delta_r).Log()
            res_v = predicted_delta_v - delta_v_imu
            res_p = predicted_delta_p - delta_p_imu 
            

            R_i_mat = pose_i.rotation().matrix()
            R_j_mat = pose_j.rotation().matrix()
            R_i_inv_mat = pose_i.rotation().Inv().matrix()
            
            r_imu = torch.cat([res_r.data, res_v.data, res_p.data], dim=-1).unsqueeze(-1) # Shape (b, 9, 1)
            r_imu /= n
            # print(r_imu)

            # 2b. Calculate 9x9 IMU Jacobians w.r.t. states i and j

            delta_v_world = vel_j - vel_i + g * dt
            delta_p_world = pose_j.translation() - pose_i.translation() - vel_i * dt + 0.5 * g * dt**2
            
            # Create rotated delta vectors for skew-symmetric matrices
            v_in_body_frame = (R_i_inv_mat @ delta_v_world.unsqueeze(-1)).squeeze(-1)
            p_in_body_frame = (R_i_inv_mat @ delta_p_world.unsqueeze(-1)).squeeze(-1)

            # --- Jacobian of r_imu w.r.t. state i ---
            # Deriv of residual wrt POSE_i update (9x6)

            dr_dpose = torch.cat([(-R_j_mat.mT @ pose_i.rotation().matrix()).squeeze(), torch.zeros(3,3, device=poses_pp.device)], dim=-1) # Approximation
            dv_dpose = torch.cat([skew(v_in_body_frame.squeeze()), torch.zeros(3,3, device=poses_pp.device)], dim=-1)
            dp_dpose = torch.cat([skew(p_in_body_frame.squeeze()), torch.zeros(3,3, device=poses_pp.device)], dim=-1)
            J_pose_i = torch.cat([dr_dpose, dv_dpose, dp_dpose], dim=0)

            # Deriv of residual wrt VELOCITY_i update (9x3)
            dr_dvel = torch.zeros(3, 3, device=poses_pp.device)
            dv_dvel = -R_i_inv_mat.squeeze()
            dp_dvel = -R_i_inv_mat.squeeze() * dt
            J_vel_i = torch.cat([dr_dvel, dv_dvel, dp_dvel], dim=0)
            
            # Full 9x9 Jacobian for state i
            J_i = torch.cat([J_pose_i, J_vel_i], dim=1)
            # --- .............................. ---

            # --- Jacobian of r_imu w.r.t. state j ---

            # Deriv of residual wrt POSE_j update (9x6)
            dr_dpose = torch.cat([torch.zeros(3,3, device=poses_pp.device), (R_i_inv_mat @ R_j_mat).squeeze()], dim=-1)
            dv_dpose = torch.zeros(6, 6, device=poses_pp.device) # Vel_j does not affect vel residual's rotation part
            J_pose_j = torch.cat([dr_dpose, dv_dpose], dim=0)

            # Deriv of residual wrt VELOCITY_j update (9x3)
            dr_dvel = torch.zeros(3, 3, device=poses_pp.device)
            dv_dvel = R_i_inv_mat.squeeze()
            dp_dvel = torch.zeros(3, 3, device=poses_pp.device)
            J_vel_j = torch.cat([dr_dvel, dv_dvel, dp_dvel], dim=0)

            # Full 9x9 Jacobian for state j
            J_j = torch.cat([J_pose_j, J_vel_j], dim=1)
            # --- .............................. ---

            # --- Jacobian of r_imu w.r.t. scale ---
            dr_ds = torch.zeros(3, 1, device=poses_pp.device, dtype=poses_pp.dtype) # Rotation residual is not affected by scale
            dv_ds = (predicted_delta_v / scale).reshape(-1, 1)
            dp_ds = (predicted_delta_p / scale).reshape(-1, 1)

            J_s = torch.cat([dr_ds, dv_ds, dp_ds], dim=0)
            # --- .............................. ---
            
            g_s += J_s.mT @ r_imu.squeeze()
            H_ss += J_s.mT @ J_s

            # --- 2c. Augment Hessian and Gradient (Corrected) ---
            i_adj, j_adj = idx_i - fixedp, idx_j - fixedp
            
            # Calculate 9-dof gradient contributions
            # (9,9).T @ (9,1) -> (9,1)
            g_i = -J_i.mT @ r_imu
            g_j = -J_j.mT @ r_imu

            #should be tensors of shape (..., 9, 1)
            # print("===========")
            # print(g_i.shape)
            # print(v_vis9[:, i_adj].shape)
            # print("===========")
            factor = 1.5
            
            # Add to gradient vector
            v_vis9[:, i_adj] += g_i.view(1, 1, STATE_SIZE, 1) * factor
            v_vis9[:, j_adj] += g_j.view(1, 1, STATE_SIZE, 1) * factor

            # Add to Hessian matrix
            H_ii = J_i.mT @ J_i
            H_ij = J_i.mT @ J_j
            H_ji = J_j.mT @ J_i
            H_jj = J_j.mT @ J_j

            #Check TODO: Remove
            assert not torch.isnan(H_ii).any(), f"NaN in IMU H_ii for factor {idx_i}->{idx_j}"
            assert not torch.isnan(H_ij).any(), f"NaN in IMU H_ij for factor {idx_i}->{idx_j}"
            assert not torch.isnan(H_ji).any(), f"NaN in IMU H_ji for factor {idx_i}->{idx_j}"
            assert not torch.isnan(H_jj).any(), f"NaN in IMU H_jj for factor {idx_i}->{idx_j}"

            B_vis9[:, i_adj, i_adj] += H_ii * factor
            B_vis9[:, i_adj, j_adj] += H_ij * factor
            B_vis9[:, j_adj, i_adj] += H_ji * factor
            B_vis9[:, j_adj, j_adj] += H_jj * factor

            #For logging
            p_ += torch.norm(res_p).item()
            v_ += torch.norm(res_v).item()
            r_ += torch.norm(res_r).item()
        
        print("\n\nPos Residual" ,p_)
        print("Vel Residual" ,v_)
        print("Rot Residual" ,r_)

            
    # --- 3. SOLVE & UPDATE ---
    Q = 1.0 / (C + lmbda)
    # print("\nE,Q", E9.shape, Q[:,None].shape, "\n")
    EQ = E9 * Q[:,None]

    if structure_only or n_adj == 0:
        dZ = (Q * w).view(b, -1, 1, 1)
        dX = torch.zeros(b, n_adj, STATE_SIZE, device=poses_pp.device)
    else:
        # --- 9-DoF B and v matrices ---

        # print(B_vis9.shape)
        # print(EQ.shape, E9.permute(0,2,1,4,3).shape)
        # print(E9.shape)

        S = B_vis9 - block_matmul(EQ, E9.permute(0,2,1,4,3))
        y = v_vis9 - block_matmul(EQ, w.unsqueeze(dim=2))
        
        # print(f"Solver matrix 'S' norm: {torch.norm(S).item()}")
        # print(f"Solver vector 'y' norm: {torch.norm(y).item()}")
        
        dX = block_solve(S, y, ep=1.0, lm=1e-4) # block_solve now works on 9x9 blocks

        # print(f"Update vector 'dX' norm: {torch.norm(dX).item()}")
        assert not torch.isnan(dX).any(), "NaN in update vector 'dX'"

        dZ = Q * (w - block_matmul(E9.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        
        dX = dX.view(b, -1, STATE_SIZE)
        dZ = dZ.view(b, -1, 1, 1)

        lmbda_s = 1e-2 # Use a small damping factor
        ds = -g_s / (H_ss + lmbda_s)

    # --- APPLY UPDATES ---
    
    x_coords, y_coords, disps = patches.unbind(dim=2)
    disps_updated = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches_updated = torch.stack([x_coords, y_coords, disps_updated], dim=2)

    if not structure_only and n_adj > 0:
        
        # --- Parse 9-DoF update vector dX ---
        dx_poses = dX[..., :6] # First 6 elements are for pose
        dx_vels = dX[..., 6:9]  # Next 3 elements are for velocity

        update_indices = fixedp + torch.arange(n_adj, device=poses_pp.device)
        
        # Update Poses
        pose_update_vector = torch.zeros(b, n, 6, device=poses_pp.device, dtype=dX.dtype)
        #print(pose_update_vector.shape, update_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 6).shape, dx_poses.shape)

        pose_update_vector.scatter_add_(1, update_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 6), dx_poses)
        poses_pp = poses_pp.Retr(pp.se3(pose_update_vector))

        # Update Velocities
        #print(velocities.shape, update_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 3).shape, dx_vels.shape)
        velocities.scatter_add_(1, update_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, 3), dx_vels)

        # Update scale #TODO: MOVE/INITIALIZE
        scale.data += ds.squeeze()
        scale.data.clamp_(min=0.1)

    return poses_pp.data, patches_updated, velocities


def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations,
       imu_preintegrations=None, imu_tensors = None, velocities = None, scale = None, **kwargs):
    """
    Top-level wrapper for the Hybrid Bundle Adjustment.
    This version performs a full BA over all provided poses and factors.
    Handles windowing and iterations, calling the hybrid solver
    """
    
    if ii.numel() > 0:
        n = t1
    else:
        # If there are no factors, there's nothing to do.
        return poses, patches

    for _ in range(iterations):
        
        # Pass only the active poses to the solver.
        poses_active = poses.data[:, :n].clone()
        velocities_active = velocities[:, :n].clone()

        # n == t1
        
        # The rest of the inputs use all available factors, as you intended
        ii_filt, jj_filt, kk_filt = ii, jj, kk.clone()
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
        poses_updated, patches_updated, velocities_updated = BA_hybrid(
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
            imu_preintegrations=imu_preintegrations,
            scale=scale if scale is not None else torch.tensor([1.0], device=poses.device)
        )
        #t0 = 1

        poses.data[:, t0:n] = poses_updated[:, t0:n]
        velocities[:, t0:n] = velocities_updated[:, t0:n]
        patches.data[:, unique_kk_filt] = patches_updated

    return poses, patches, velocities
