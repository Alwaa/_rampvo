import torch
from torch_scatter import scatter_sum

from . import lietorch
from .lietorch import SE3, SO3
import pypose as pp

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


def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations, eff_impl=False, 
       imu_preintegrations=None, imu_tensors = None, velocities = None):
    """
    Wrapper for the Python-based Bundle Adjustment.
    """
    
    for _ in range(iterations):
        # poses_window = poses[:, t0:t1].clone()
        poses_window = poses.clone()
        velocities_window = velocities.clone() if velocities is not None else None

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
        
        ba_ret = _old_BA(
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
            t1 = t1,
            imu_preintegrations=imu_preintegrations,
            imu_tensors=imu_tensors,
            velocities=velocities_window
            )
        
        if velocities is None:
            poses_updated, patches_updated = ba_ret
        else:
            poses_updated, patches_updated, velocities_updated = ba_ret
            velocities[:, t0:t1] = velocities_updated[:, t0:t1]  

        poses[:, t0:t1] = poses_updated[:, t0:t1]    
        patches[:, unique_kk_filt] = patches_updated

    return poses, patches


def _old_BA(poses, patches, intrinsics, targets, weights, lmbda, ii, jj, kk, bounds, t1,
            ep=1.0, PRINT=False, fixedp=1, structure_only=False, 
            imu_preintegrations=None, imu_tensors = None, velocities = None):
    """ Original Python-based bundle adjustment implementation. """

    b = 1
    if ii.numel() > 0:
        n = max(ii.max().item(), jj.max().item()) + 1
    else:
        if velocities is None:
            return poses, patches
        else: 
            return poses, patches, velocities

    coords, v_vis, (Ji, Jj, Jz) = \
        pops.transform(SE3(poses), patches, intrinsics, ii, jj, kk, jacobian=True)
    
    patch_size = coords.shape[3]
    r = targets - coords[..., patch_size//2, patch_size//2, :]

    v_vis *= (r.norm(dim=-1) < 250).float()
    in_bounds = \
        (coords[...,patch_size//2,patch_size//2,0] > bounds[0]) & \
        (coords[...,patch_size//2,patch_size//2,1] > bounds[1]) & \
        (coords[...,patch_size//2,patch_size//2,0] < bounds[2]) & \
        (coords[...,patch_size//2,patch_size//2,1] < bounds[3])
    v_vis *= in_bounds.float()

    if PRINT:
        print("Mean Error:", (r * v_vis[...,None]).norm(dim=-1).mean().item())

    r = (v_vis[...,None] * r).unsqueeze(dim=-1)    
    weights = (v_vis[...,None] * weights).unsqueeze(dim=-1)

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
        if velocities is None:
            return poses, patches
        else: 
            return poses, patches, velocities

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
    Q = 1.0 / (C + lmbda)
    
    print("\nE,Q", E.shape, Q[:,None].shape, "\n")
    EQ = E * Q[:,None]

    if imu_preintegrations is not None and not structure_only:
        # These are your tuning parameters for the IMU factors
        imu_rot_weight = 1.0
        imu_vel_weight = 1.0
        imu_pos_weight = 1.0
        
        g = torch.tensor([0, 0, -9.81], device=poses.device, dtype=poses.dtype)

        if imu_tensors is not None:


            # --- 1. Calculate Predicted Relative Motion from BA States ---
            delta_pose_tensor, delta_vel_tensor, delta_t_tensor = imu_tensors
        
            # Define the range of indices for the IMU measurements
            num_imu_measurements = t1 - 1

            #TODO: CHECK INDECIES
            i_indices = torch.arange(1 + fixedp - 1, num_imu_measurements)
            j_indices = torch.arange(1 + fixedp, num_imu_measurements + 1)
            
            # --- Batch extract states using advanced indexing ---
            pose_i = pp.SE3(poses[:, i_indices])
            pose_j = pp.SE3(poses[:, j_indices])
            vel_i = velocities[:, i_indices]
            vel_j = velocities[:, j_indices]

            # --- Batch extract IMU measurements from tensors ---
            delta_p_imu = delta_pose_tensor[i_indices, :3]
            delta_r_imu = pp.SO3(delta_pose_tensor[i_indices, 3:])
            delta_v_imu = delta_vel_tensor[i_indices, :]

            dt = delta_t_tensor[i_indices].unsqueeze(0)

            # --- 1. Vectorized Calculation of Predicted Relative Motion ---
            R_i = pose_i.rotation()
            
            predicted_delta_r = R_i.Inv() * pose_j.rotation()
            predicted_delta_v = R_i.Inv() @ (vel_j - vel_i - g * dt)

            predicted_delta_p = R_i.Inv() @ (
                pose_j.translation() - pose_i.translation() - vel_i * dt - 0.5 * g * dt**2
            )


            # --- 2. Vectorized Calculation of the 9-DOF IMU Residual ---

            # print((delta_r_imu.Inv() * predicted_delta_r))
            # print((delta_r_imu.Inv() * predicted_delta_r).Log())

            residual_r = (delta_r_imu.Inv() * predicted_delta_r).Log()
            residual_v = predicted_delta_v - delta_v_imu
            residual_p = predicted_delta_p - delta_p_imu


            # --- 3. Vectorized Addition of IMU cost to the Gauss-Newton system ---
            
            pose_error_tangent = torch.cat([
                residual_p * imu_pos_weight,
                residual_r * imu_rot_weight
            ], dim=-1)

            # Use scatter_add to update the 'v' vector in a vectorized way
            v_flat = v_vis.view(b, n_adj, 6).cpu()

            # pose_error_tangent has shape (k, 6) where k is the number of valid measurements
            # We need to add the batch dimension to match v_flat.
            src_tangent = pose_error_tangent #.unsqueeze(0)  # Shape becomes (1, k, 6)
            
            # Indices need to be adjusted by -fixedp for the 'v' and 'B' matrices
            i_adj = i_indices - fixedp
            j_adj = j_indices - fixedp

            i_adj_idx = i_adj.unsqueeze(0).unsqueeze(-1) # Shape becomes (1, k, 1)
            j_adj_idx = j_adj.unsqueeze(0).unsqueeze(-1) # Shape becomes (1, k, 1)


            v_flat.scatter_add_(1, i_adj_idx.cpu(), -src_tangent.cpu())
            v_flat.scatter_add_(1, j_adj_idx.cpu(), src_tangent.cpu())
            
            v_vis = v_flat.view(b, n_adj, 1, 6, 1).cuda()

        else:

            # Store gradients for the velocity update step later
            velocity_gradients = torch.zeros_like(velocities)

            # Identity matrices for convenience
            I3 = torch.eye(3, device=poses.device, dtype=poses.dtype)
            I6 = torch.eye(6, device=poses.device, dtype=poses.dtype)


            # You should get this from your IMU preintegration data, inverse of the covariance
            W_imu = torch.diag(torch.cat([
                torch.full([3], 0.05), # Rotation weight
                torch.full([3], 0.05),# Velocity weight
                torch.full([3], 0.05)  # Position weight
            ], dim=0)).to(poses.device)
            

            # Loop through each preintegrated measurement
            # 'j' is the index of the 'to' keyframe for the measurement
            for j, imu_data in enumerate(imu_preintegrations):
                i = j - 1 # The 'from' keyframe
                if i < fixedp: # Don't apply constraints to the fixed frame
                    continue

                # Adjust for the fixed frame
                ii_adj, jj_adj = i - fixedp, j - fixedp
                if ii_adj < 0: # Don't add factors connected to the fixed frame
                    continue

                # Extract states for the two keyframes
                pose_i = pp.SE3(poses[:, i])
                pose_j = pp.SE3(poses[:, j])
                vel_i = velocities[:, i]
                vel_j = velocities[:, j]

                # Extract IMU measurements from your data structure
                delta_p_imu = imu_data['Dp'].cuda()
                delta_v_imu = imu_data['Dv'].cuda()
                delta_r_imu = pp.SO3(imu_data['Dr']).cuda() # Convert to SO3
                dt = imu_data['Dt'].cuda()

                # --- 1. Calculate Predicted Relative Motion from BA States ---
                R_i = pp.SE3(poses[:, i]).rotation()
                R_j = pp.SE3(poses[:, j]).rotation()

                R_i_inv = R_i.Inv()
                # Predicted rotation change, and (in body frame of i) velocity change and position change
                predicted_delta_r = R_i_inv * R_j.Inv()
                predicted_delta_v = R_i_inv @ (vel_j - vel_i - g * dt)
                predicted_delta_p = R_i_inv @ (
                    pose_j.translation() - pose_i.translation() - vel_i * dt - 0.5 * g * dt**2
                )

                # --- 2. Calculate the 9-DOF IMU Residual (Error) ---
                residual_r = (delta_r_imu.Inv() * predicted_delta_r).Log()
                residual_v = predicted_delta_v - delta_v_imu
                residual_p = predicted_delta_p - delta_p_imu           
                

                imu_residual = torch.cat([residual_r, residual_v, residual_p], dim=-1).unsqueeze(-1) # Shape (9, 1)


                # --- 3. Calculate Jacobians of the residual w.r.t. pose_i and pose_j ---
                # This is the "expert" part. These are standard VIO Jacobians.
                
                # # Helper: skew-symmetric matrix for cross products
                # skew_v = skew(predicted_delta_p)

                # # Jacobian of the 9-dof residual w.r.t the 6-dof pose_i
                # J_res_pose_i = torch.zeros(1, 9, 6, device=poses.device)
                # J_res_pose_i[:, 0:3, 3:6] = -predicted_delta_r.Inv().matrix()  # d(res_r)/d(rot_i)
                # J_res_pose_i[:, 3:6, 3:6] = skew_v                         # d(res_v)/d(rot_i)
                # J_res_pose_i[:, 6:9, 0:3] = -I3                            # d(res_p)/d(pos_i)
                # J_res_pose_i[:, 6:9, 3:6] = skew(R_i_inv @ (pose_j.translation() - pose_i.translation())) # d(res_p)/d(rot_i)

                # # Jacobian of the 9-dof residual w.r.t the 6-dof pose_j
                # J_res_pose_j = torch.zeros(1, 9, 6, device=poses.device)
                # J_res_pose_j[:, 0:3, 3:6] = I3                             # d(res_r)/d(rot_j)
                # J_res_pose_j[:, 6:9, 0:3] = R_i_inv.matrix()               # d(res_p)/d(pos_j)

# ####
#                 # Calculate Jacobians of residual w.r.t. pose_i and pose_j
#                 J_res_pose_i = torch.zeros(9, 6, device=poses.device)
#                 J_res_pose_i[0:3, 3:6] = -predicted_delta_r.Inv().matrix()
#                 J_res_pose_i[3:6, 3:6] = skew(predicted_delta_v)
#                 J_res_pose_i[6:9, 0:3] = -R_i_inv.matrix()
#                 J_res_pose_i[6:9, 3:6] = skew(R_i_inv @ (pose_j.translation()))

#                 J_res_pose_j = torch.zeros(9, 6, device=poses.device)
#                 J_res_pose_j[0:3, 3:6] = predicted_delta_r.Inv().matrix()
#                 J_res_pose_j[6:9, 0:3] = R_i_inv.matrix()


#                 # --- 4. Form Hessian and Residual Contributions ---
#                 J_i_T = J_res_pose_i.transpose(-1, -2)
#                 J_j_T = J_res_pose_j.transpose(-1, -2)

#                 # H = J.T * W * J
#                 H_ii = J_i_T @ W_imu @ J_res_pose_i
#                 H_ij = J_i_T @ W_imu @ J_res_pose_j
#                 H_jj = J_j_T @ W_imu @ J_res_pose_j
                
#                 # r = J.T * W * e
#                 r_i = J_i_T @ W_imu @ imu_residual
#                 r_j = J_j_T @ W_imu @ imu_residual


#                 # --- 5. Add Contributions Directly to the System ---
                
#                 # Update Hessian diagonal blocks
#                 B[:, ii_adj, ii_adj] += H_ii.squeeze(0)
#                 B[:, jj_adj, jj_adj] += H_jj.squeeze(0)

#                 # Update Hessian off-diagonal blocks
#                 B[:, ii_adj, jj_adj] += H_ij.squeeze(0)
#                 B[:, jj_adj, ii_adj] += H_ij.transpose(-1, -2).squeeze(0)

#                 # Update residual vector
#                 v[:, ii_adj] -= r_i.squeeze(0)
#                 v[:, jj_adj] -= r_j.squeeze(0)       
# ####         
                
                # # --- 5. Scatter Add to the Main System ---
                # ii_adj, jj_adj = i - fixedp, j - fixedp

                # # Update Hessian diagonal blocks
                # B = B + safe_scatter_add_mat(H_ii, torch.tensor([ii_adj]), torch.tensor([ii_adj]), n_adj, n_adj).view(b, n_adj, n_adj, 6, 6)
                # B = B + safe_scatter_add_mat(H_jj, torch.tensor([jj_adj]), torch.tensor([jj_adj]), n_adj, n_adj).view(b, n_adj, n_adj, 6, 6)

                # # Update Hessian off-diagonal blocks
                # B = B + safe_scatter_add_mat(H_ij, torch.tensor([ii_adj]), torch.tensor([jj_adj]), n_adj, n_adj).view(b, n_adj, n_adj, 6, 6)
                # B = B + safe_scatter_add_mat(H_ij.transpose(-1, -2), torch.tensor([jj_adj]), torch.tensor([ii_adj]), n_adj, n_adj).view(b, n_adj, n_adj, 6, 6)

                # # Update residual vector
                # v = v - safe_scatter_add_vec(r_i.squeeze(-1), torch.tensor([ii_adj]), n_adj).view(b, n_adj, 1, 6, 1)
                # v = v - safe_scatter_add_vec(r_j.squeeze(-1), torch.tensor([jj_adj]), n_adj).view(b, n_adj, 1, 6, 1)


                # # Form Hessian and Residual contributions
                # J_i_T = J_res_pose_i.transpose(-1, -2)
                # J_j_T = J_res_pose_j.transpose(-1, -2)

                # H_ii = J_i_T @ W_imu @ J_res_pose_i
                # H_ij = J_i_T @ W_imu @ J_res_pose_j
                # H_jj = J_j_T @ W_imu @ J_res_pose_j
                
                # r_i = J_i_T @ W_imu @ imu_residual
                # r_j = J_j_T @ W_imu @ imu_residual

                # # Add contributions directly to the system
                # B[:, ii_adj, ii_adj] += H_ii
                # B[:, jj_adj, jj_adj] += H_jj
                # B[:, ii_adj, jj_adj] += H_ij
                # B[:, jj_adj, ii_adj] += H_ij.transpose(-1, -2)

                # v_vec[:, ii_adj] -= r_i
                # v_vec[:, jj_adj] -= r_j


                # --- Calculate ALL Jacobians (Pose and Velocity) ---
                J_res_pose_i, J_res_pose_j = torch.zeros(9, 6, device=poses.device), torch.zeros(9, 6, device=poses.device)
                J_res_pose_i[0:3, 3:6] = -predicted_delta_r.Inv().matrix()
                J_res_pose_i[3:6, 3:6] = skew(predicted_delta_v)
                J_res_pose_i[6:9, 0:3] = -R_i_inv.matrix()
                J_res_pose_i[6:9, 3:6] = skew(R_i_inv @ pose_j.translation())
                J_res_pose_j[0:3, 3:6] = predicted_delta_r.Inv().matrix()
                J_res_pose_j[6:9, 0:3] = R_i_inv.matrix()

                # ** Jacobians w.r.t. Velocity **
                J_res_vel_i, J_res_vel_j = torch.zeros(9, 3, device=poses.device), torch.zeros(9, 3, device=poses.device)
                J_res_vel_i[3:6, 0:3] = -R_i_inv.matrix()
                J_res_vel_i[6:9, 0:3] = -R_i_inv.matrix() * dt
                J_res_vel_j[3:6, 0:3] = R_i_inv.matrix()

                # --- Update Pose System (Hessian B and residual v_vec) ---
                J_i_T, J_j_T = J_res_pose_i.transpose(-1, -2), J_res_pose_j.transpose(-1, -2)
                H_ii, H_ij, H_jj = J_i_T @ W_imu @ J_res_pose_i, J_i_T @ W_imu @ J_res_pose_j, J_j_T @ W_imu @ J_res_pose_j
                r_i, r_j = J_i_T @ W_imu @ imu_residual, J_j_T @ W_imu @ imu_residual
                
                B_vis[:, ii_adj, ii_adj] += H_ii
                B_vis[:, jj_adj, jj_adj] += H_jj
                B_vis[:, ii_adj, jj_adj] += H_ij
                B_vis[:, jj_adj, ii_adj] += H_ij.transpose(-1, -2)
                v_vis[:, ii_adj] -= r_i
                v_vis[:, jj_adj] -= r_j

                # ** Calculate and Store Velocity Gradients **
                # Gradient = J.T * W * residual
                grad_v_i = J_res_vel_i.transpose(-1, -2) @ W_imu @ imu_residual
                grad_v_j = J_res_vel_j.transpose(-1, -2) @ W_imu @ imu_residual
                
                # Accumulate gradients for each velocity state
                velocity_gradients[:,i] += grad_v_i.squeeze(-1)
                velocity_gradients[:,j] += grad_v_j.squeeze(-1)

    if structure_only or n_adj == 0:
        dZ = (Q * w).view(b, -1, 1, 1)
        dX = torch.zeros(b, n_adj, 6, device=poses.device)
    else:

        S = B_vis - block_matmul(EQ, E.permute(0,2,1,4,3))
        y = v_vis - block_matmul(EQ, w.unsqueeze(dim=2))
        
        
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

    if velocities is None:
        return poses, patches
    else: 
        vel_learning_rate = 0.2
        velocities -= vel_learning_rate * velocity_gradients
        return poses, patches, velocities