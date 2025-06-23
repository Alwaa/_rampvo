import torch
import pypose as pp
from torch import nn


from .lietorch import SE3 # May need this with pops.transform

# The projective_ops are still needed for back-projection from patches
from . import projective_ops as pops


class BundleAdjustmentResidual(nn.Module):
    """
    PyPose-based Residual Module for Bundle Adjustment.

    This version correctly wraps the user's existing `projective_ops.transform`
    function to calculate the reprojection error.
    """
    def __init__(self, poses, patches):
        """
        Initializes the optimizable parameters.

        Args:
            poses (torch.Tensor): Initial poses, shape (b, N, 7) (lietorch convention).
            patches (torch.Tensor): Initial patches, shape (b, M, 3) -> (x, y, inv_depth).
        """
        super().__init__()
        
        # All poses and patches are optimizable parameters.
        # Pypose's optimizers can work directly on pypose types like pp.SE3
        self.poses = pp.Parameter(pp.SE3(poses))
        self.patches = nn.Parameter(patches)

    def forward(self, intrinsics, targets, weights, ii, jj, kk):
        """
        Calculates the weighted reprojection error by calling pops.transform.

        Args:
            intrinsics (torch.Tensor): Camera intrinsics.
            targets (torch.Tensor): Target 2D coordinates for each patch in frame `j`.
            weights (torch.Tensor): Weights for each residual.
            ii (torch.Tensor): Indices for the reference camera pose (where patch is defined).
            jj (torch.Tensor): Indices for the target camera pose (where patch is observed).
            kk (torch.Tensor): Indices mapping observations to patches.
        """
        
        # The `projective_ops.transform` function expects a lietorch.SE3 object.
        # We can pass the raw tensor data from our pypose object.
        # Or, for full compatibility, we can cast it. Let's assume passing the raw tensor works,
        # but if not, we would convert: `SE3(self.poses.data)`
        
        # We call the existing transform function to get the projected coordinates.
        # CRUCIALLY, we set jacobian=False because pypose will handle this.
        projected_coords = pops.transform(
            poses=self.poses, # SE3(self.poses.data),#
            patches=self.patches, 
            intrinsics=intrinsics, 
            ii=ii, 
            jj=jj, 
            kk=kk,
            jacobian=False
        )
        
        # The target is the observed coordinate in frame `j`.
        # The shape of projected_coords will be (b, num_obs, patch_size, patch_size, 2)
        # We take the center of the patch, as in your original _old_BA code.
        patch_center_coords = projected_coords[..., projected_coords.shape[-2]//2, projected_coords.shape[-2]//2, :]

        # Calculate the residual
        residual = patch_center_coords.squeeze(0) - targets
        
        # Apply weights and flatten for the optimizer
        weighted_residual = (torch.sqrt(weights) * residual).view(-1)
        
        return weighted_residual

class __BundleAdjustmentResidual(nn.Module):
    """
    PyPose-based Residual Module for Bundle Adjustment.

    This module defines the reprojection error. The Pypose optimizer will
    automatically compute the Jacobians of this error and minimize its squared norm.
    """
    def __init__(self, poses, patches, fixedp):
        """
        Initializes the optimizable parameters.

        Args:
            poses (torch.Tensor): Initial poses of shape (b, N, 7) from lietorch.
            patches (torch.Tensor): Initial patches of shape (b, M, 3), where the last
                                     dimension is (x, y, inv_depth).
            fixedp (int): The number of initial poses to keep fixed during optimization.
        """
        super().__init__()
        
        # We need to separate the fixed poses from the optimizable ones.
        # The optimizer will only see and update the non-fixed poses.
        self.fixedp = fixedp
        self.poses_fixed = pp.Parameter(pp.SE3(poses[:, :fixedp]), requires_grad=False)
        self.poses_optim = pp.Parameter(pp.SE3(poses[:, fixedp:]))

        # The patches (structure) are fully optimizable parameters.
        self.patches = nn.Parameter(patches)

    def forward(self, intrinsics, targets, weights, ii, jj, kk):
        """
        Calculates the weighted reprojection error (residual).

        Args:
            intrinsics (torch.Tensor): Camera intrinsics (fx, fy, cx, cy).
            targets (torch.Tensor): Target 2D coordinates for each patch.
            weights (torch.Tensor): Weights for each residual.
            ii (torch.Tensor): Indices for the first camera pose.
            jj (torch.Tensor): Indices for the second camera pose (not used in this simplified model,
                                 but kept for signature consistency if needed).
            kk (torch.Tensor): Indices mapping observations to patches.
        """
        b = self.patches.shape[0]
        
        # Reconstruct full pose tensor
        all_poses = pp.SE3(torch.cat([self.poses_fixed.data, self.poses_optim.data], dim=1))

        # Select the poses and patches corresponding to each observation
        poses_obs = all_poses[:, ii]
        patches_obs = self.patches[:, kk]
        
        # --- 1. Unproject Patches to 3D Points ---
        # The `patches` tensor contains (x, y, inv_depth). We need to convert this to 3D points.
        # We use the backprojection logic from your original implementation.
        # Note: pops.backproject expects lietorch.SE3, but we can pass the raw tensor data.
        # Let's assume a simplified back-projection for clarity here. A more robust implementation
        # would use a dedicated function.
        x, y, inv_depth = patches_obs.unbind(-1)
        # Assuming intrinsics are (fx, fy, cx, cy)
        fx, fy, cx, cy = intrinsics.unbind(-1)

        print(cx.shape, x.shape)

        fx = fx.unsqueeze(1)
        fy = fy.unsqueeze(1)
        cx = cx.unsqueeze(1)
        cy = cy.unsqueeze(1)
        

        print(cx.shape, x.shape)


        # Create a grid of normalized coordinates
        # This reconstruction assumes patches are defined by their top-left corner and a fixed size.
        # For simplicity, we only project the center point, as in the original BA.
        points_3d_cam = torch.stack([
            (x - cx) / fx * (1.0 / inv_depth),
            (y - cy) / fy * (1.0 / inv_depth),
            1.0 / inv_depth
        ], dim=-1)
        
        # --- 2. Project 3D points to 2D image plane ---
        # `pypose.point2pixel` handles the transformation from world to camera to pixel coordinates.
        # Since `points_3d_cam` are already in the camera frame of the patch-defining pose,
        # we can use them directly. The BA optimizes the poses of observing cameras.
        projected_coords = pp.point2pixel(points_3d_cam, poses_obs, intrinsics.squeeze(0))

        # --- 3. Calculate the Residual ---
        residual = projected_coords - targets
        
        # Apply the weights. Pypose's LM optimizer minimizes the squared norm of this vector.
        weighted_residual = torch.sqrt(weights) * residual
        
        # The optimizer expects a 1D vector of residuals
        return weighted_residual.view(-1)

def __BA_pypose(poses_lietorch, patches, intrinsics, target, weight, ii, kk, iterations=5, fixedp=1, lm_lambda=1e-4):
    """
    Bundle Adjustment implementation using the Pypose library.

    Args:
        poses_lietorch (torch.Tensor): Poses from lietorch SE3 object, shape (1, N, 7).
        patches (torch.Tensor): Patches tensor, shape (1, M, 3) -> (x, y, inv_depth).
        intrinsics (torch.Tensor): Intrinsics tensor, shape (1, 4) -> (fx, fy, cx, cy).
        target (torch.Tensor): Target 2D coordinates, shape (1, P, 2).
        weight (torch.Tensor): Observation weights, shape (1, P, 1).
        ii (torch.Tensor): Indices mapping observations to poses, shape (P,).
        kk (torch.Tensor): Indices mapping observations to patches, shape (P,).
        iterations (int): Number of optimization iterations.
        fixedp (int): Number of initial poses to keep fixed.
        lm_lambda (float): Initial damping factor for the LM optimizer.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The updated poses and patches.
    """
    
    # 1. Instantiate the Residual Model
    # We pass the full set of poses and patches to be optimized.
    # The model internally handles which poses are fixed.
    model = BundleAdjustmentResidual(poses=poses_lietorch.clone(), patches=patches.clone(), fixedp=fixedp)

    # 2. Set up the Optimizer
    # Use the Levenberg-Marquardt optimizer, the standard for BA.
    # The TrustRegion strategy is robust for non-linear problems.
    strategy = None #pp.optim.strategy.TrustRegion(damping=lm_lambda)
    optimizer = pp.optim.LM(model, strategy=strategy)

    # 3. Set up the Scheduler
    # This will run the optimization for a fixed number of steps.
    # You could also use StopOnPlateau for convergence-based termination.
    scheduler = pp.optim.scheduler.StopOnPlateau(optimizer, steps=10) #pp.optim.scheduler.FixedStep(optimizer, steps=iterations)
    
    # Prepare the input data for the optimizer.step() call. This must match
    # the arguments of the model's forward() method.
    # Note: We can probably simplify jj away if it's not needed.
    jj = torch.zeros_like(ii) # Dummy jj if not used.
    input_data = (intrinsics, target.squeeze(0), weight.squeeze(0), ii, jj, kk)

    # 4. Run the Optimization Loop
    while scheduler.continual():
        loss = optimizer.step(input_data)
        scheduler.step(loss)
        # Optional: print progress
        # print(f"Step: {scheduler.current_step}, Loss: {loss.item():.4f}")

    # 5. Retrieve the Optimized Parameters
    # The model's parameters have been updated in-place by the optimizer.
    optimized_poses_fixed = model.poses_fixed.data
    optimized_poses_optim = model.poses_optim.data
    
    # Combine fixed and optimized poses and convert back to lietorch's tensor format
    poses_updated_pp = pp.SE3(torch.cat([optimized_poses_fixed, optimized_poses_optim], dim=1))
    poses_updated_lietorch = poses_updated_pp.data # Shape (1, N, 7)
    
    patches_updated = model.patches.data

    return poses_updated_lietorch, patches_updated

def BA_pypose(poses_lietorch, patches, intrinsics, target, weight, ii, jj, kk, iterations=5, lm_lambda=1e-4, fixedp = 1):
    """
    Bundle Adjustment implementation using the Pypose library,
    correctly wrapping the existing projective_ops.transform function.
    """
    
    # 1. Instantiate the Residual Model
    model = BundleAdjustmentResidual(poses=poses_lietorch.clone(), patches=patches.clone())

    # 2. Set up the Optimizer
    strategy = None # pp.optim.strategy.TrustRegion(damping=lm_lambda)
    optimizer = pp.optim.LM(model, strategy=strategy)

    # 3. Set up the Scheduler for a fixed number of iterations
    scheduler = pp.optim.scheduler.StopOnPlateau(optimizer, steps=iterations) #pp.optim.scheduler.FixedStep(optimizer, steps=iterations)
    
    # Prepare the input data tuple. It must match the model's forward() signature.
    # Note: We now pass the real `jj` indices.
    input_data = (intrinsics, target.squeeze(0), weight.squeeze(0), ii, jj, kk)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 4. Run the Optimization Loop
    while scheduler.continual():
        print("\n\n\nCHECK\n\n\n")
        loss = optimizer.step(input_data)
        print("\n\n\nSTEPPED\n\n\n")
        scheduler.step(loss)

    # 5. Retrieve the Optimized Parameters
    # The model's parameters are updated in-place.
    poses_updated_pp = model.poses.data
    patches_updated = model.patches.data

    # Return in the original lietorch tensor format
    return poses_updated_pp, patches_updated


def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations, eff_impl=False, 
       imu_preintegrations=None, imu_tensors = None, velocities = None):
    """
    Wrapper for the Python-based Bundle Adjustment.
    """
    
    for _ in range(iterations):
        # poses_window = poses[:, t0:t1].clone()

        poses_window = poses.clone()

        # poses is a lietorch.SE3 object, so we use its .data tensor
        # poses_window = poses[:, t0:t1].data
        # For simplicity, let's assume the full window is passed
        # poses_data = poses.data


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

        # Call the new PyPose BA function
        ba_ret = BA_pypose(
            poses_lietorch=poses_window, # Pass the relevant window
            patches=patches_window.clone(),
            intrinsics=intrinsics,
            target=target_filt,
            weight=weight_filt,
            ii=ii_rel,
            jj=jj_rel,
            kk=kk_rel,
            iterations=iterations,
            fixedp=fixedp 
        )
        
        if velocities is None:
            poses_updated, patches_updated = ba_ret
        else:
            poses_updated, patches_updated, velocities_updated = ba_ret
            velocities[:, t0:t1] = velocities_updated[:, t0:t1]  

        poses[:, t0:t1] = poses_updated[:, t0:t1]    
        patches[:, unique_kk_filt] = patches_updated

    return poses, patches
