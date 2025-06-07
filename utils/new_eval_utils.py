import numpy as np
import os
import os.path as osp
import cv2
import torch

FEATURE_TRACK_WINDOW = "Feature Tracks - VO"
TAJECTORY_WINDOW = "Trajectory"
PATCH_WINDOW = 'Patches Visualization'


def save_results(
    traj_ref, traj_est, scene, j=0, eval_type="None"
):
    # save poses for finer evaluations
    save_dir = osp.join(
        os.getcwd(),
        "trajectory_evaluation",
        f"{eval_type}",
        "trial_" + str(j),
        scene,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    time_ref = (traj_ref.timestamps * 10 ** -9)[..., np.newaxis]
    time_est = (traj_est.timestamps * 10 ** -9)[..., np.newaxis]
    np.savetxt(
        osp.join(save_dir, "stamped_groundtruth.txt"),
        np.concatenate((time_ref, traj_ref.positions_xyz, traj_ref.orientations_quat_wxyz), axis=1),
    )
    np.savetxt(
        osp.join(save_dir, "stamped_traj_estimate.txt"),
        np.concatenate((time_est, traj_est.positions_xyz, traj_est.orientations_quat_wxyz), axis=1),
    )


# --- Visualization Class ---
class Visualizer:
    def __init__(self, traj_img_width=400, traj_img_height=600, track_refresh_interval=30):
        self.traj_img_width = traj_img_width
        self.traj_img_height = traj_img_height
        self.track_visualization_mask = None
        self._track_refresh_interval = track_refresh_interval
        self._frame_count_since_last_refresh = 0

        # Trajectory 
        self.gt_points = []
        self.est_points = []

        self.pre_update_points = []
        self.post_update_points = []


        # setting up OpenCV windows here
        cv2.namedWindow(FEATURE_TRACK_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(TAJECTORY_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TAJECTORY_WINDOW, self.traj_img_width, self.traj_img_height)
        cv2.namedWindow(PATCH_WINDOW, cv2.WINDOW_NORMAL)


        print(f"Visualizer initialized. Trajectory view: {traj_img_width}x{traj_img_height}, Track refresh: {track_refresh_interval} frames.")
    
    def update_trajectories(self, gt_pose, pre_update_buffer, post_update_buffer, num_valid_poses):
        """
        Updates all three trajectory lists for visualization.
        """
        # 1. Add new ground truth point
        if gt_pose is not None:
            # Raw tensor
            self.gt_points.append(gt_pose[:3].cpu().numpy().flatten())

        # 2. Refresh the pre-update ("old") trajectory
        if pre_update_buffer is not None and num_valid_poses > 0:
            valid_poses = pre_update_buffer[:num_valid_poses, :3]
            self.pre_update_points = [row.cpu().numpy() for row in valid_poses]

        # 3. Refresh the post-update ("new") trajectory
        if post_update_buffer is not None and num_valid_poses > 0:
            valid_poses = post_update_buffer[:num_valid_poses, :3]
            self.post_update_points = [row.cpu().numpy() for row in valid_poses]
    
    def overrite_est_traj_(self, curr_traj_est):
        self.est_points = curr_traj_est[:,:3]
    
    def add_pose(self, pose_gt=None, pose_est=None):
        """
        Adds new ground truth and/or estimated poses to the trajectory lists.
        Handles both lietorch.SE3 objects and 7-dimensional tensors.
        """
        # --- Handle Ground Truth Pose ---
        if pose_gt is not None:
            # Raw tensor [tx, ty, tz, qx, qy, qz, qw]
            # We only need the translation part (the first 3 elements)
            self.gt_points.append(pose_gt[:3].cpu().numpy().flatten())
        
        # --- Handle Estimated Pose ---
        if pose_est is not None:
            # Raw tensor [tx, ty, tz, qx, qy, qz, qw]
            # We only need the translation part (the first 3 elements)
            self.est_points.append(pose_est[:3].cpu().numpy().flatten())

    def newer_plot_trajectory_2d(self):
        """
        Plots all three trajectories:
        - Ground Truth (Blue)
        - Pre-Update Estimate (Grey, Dashed)
        - Post-Update Estimate (Green, Solid)
        """
        traj_img = np.ones((self.traj_img_height, self.traj_img_width, 3), dtype=np.uint8) * 255
        center_x, center_y = self.traj_img_width // 2, self.traj_img_height // 2
        
        all_points = self.post_update_points
        if not all_points:
            print("NONE TRAJ")
            cv2.imshow(TAJECTORY_WINDOW, traj_img)
            return traj_img

        # Auto-scaling logic
        all_points_np = np.array([p for p in self.post_update_points if p is not None])
        max_coord = np.max(np.abs(all_points_np[:, [0, 2]])) if all_points_np.shape[0] > 0 else 1.0
        scale = (min(self.traj_img_width, self.traj_img_height) / (2.5 * max_coord)) if max_coord > 0 else 1.0

        # --- Draw All Trajectories ---
        # Ground Truth (Blue)
        if len(self.gt_points) > 1:
            gt_pts = np.array([(center_x + p[0]*scale, center_y + p[2]*scale) for p in self.gt_points], dtype=np.int32)
            cv2.polylines(traj_img, [gt_pts], isClosed=False, color=(255, 0, 0), thickness=2)

        # Pre-Update/Old Estimate (Grey, Dashed)
        if len(self.pre_update_points) > 1:
            pre_pts = np.array([(center_x + p[0]*scale, center_y + p[2]*scale) for p in self.pre_update_points], dtype=np.int32)
            # Draw dashed line by drawing circles
            cv2.polylines(traj_img, [pre_pts], isClosed=False, color=(200, 200, 200), thickness=2)

        # Post-Update/New Estimate (Green, Solid)
        if len(self.post_update_points) > 1:
            post_pts = np.array([(center_x + p[0]*scale, center_y + p[2]*scale) for p in self.post_update_points], dtype=np.int32)
            cv2.polylines(traj_img, [post_pts], isClosed=False, color=(0, 255, 0), thickness=2)
            cv2.circle(traj_img, tuple(post_pts[-1]), 5, (0, 0, 255), -1) # Mark current position

        cv2.imshow(TAJECTORY_WINDOW, traj_img)
        return traj_img
    
    def plot_trajectory_2d(self):
        """
        Plots the stored ground truth (blue) and estimated (green) trajectories
        on a white background. It automatically scales the plot to fit the window.
        """
        X_IDX = 0
        Y_IDX = 2
        traj_img = np.ones((self.traj_img_height, self.traj_img_width, 3), dtype=np.uint8) * 255
        center_x, center_y = self.traj_img_width // 2, self.traj_img_height // 2


        if len(self.est_points) == 0:
            cv2.imshow(TAJECTORY_WINDOW, traj_img)
            return traj_img

        scale_pints = np.array(self.est_points)
        # Use the max of absolute x and u coordinates for scaling
        max_coord = np.max(np.abs(scale_pints[:, [X_IDX, Y_IDX]])) if scale_pints.shape[0] > 0 else 1.0

        # Calculate scale to fit
        scale = (min(self.traj_img_width, self.traj_img_height) / (2.5 * max_coord)) if max_coord > 0 else 1.0

        # --- Draw Ground Truth Trajectory (Blue) ---
        if len(self.gt_points) > 1:
            gt_screen_pts = np.array([(center_x + p[X_IDX] * scale, center_y + p[Y_IDX] * scale) for p in self.gt_points], dtype=np.int32)
            cv2.polylines(traj_img, [gt_screen_pts], isClosed=False, color=(255, 0, 0), thickness=2)
            
        # --- Draw Estimated Trajectory (Green) ---
        if len(self.est_points) > 1:
            est_screen_pts = np.array([(center_x + p[X_IDX] * scale, center_y + p[Y_IDX] * scale) for p in self.est_points], dtype=np.int32)
            cv2.polylines(traj_img, [est_screen_pts], isClosed=False, color=(0, 255, 0), thickness=2)
            # Mark the current estimated position with a red circle
            cv2.circle(traj_img, tuple(est_screen_pts[-1]), 5, (0, 0, 255), -1)

        # scale legend
        cv2.putText(traj_img, f"Scale: {scale:.2f} px/m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        
        cv2.imshow(TAJECTORY_WINDOW, traj_img)
        return traj_img

    def _initialize_track_mask(self, frame_shape_with_channels):
        """Initializes or re-initializes the track mask if needed based on frame dimensions."""
        if self.track_visualization_mask is None or \
           self.track_visualization_mask.shape[0] != frame_shape_with_channels[0] or \
           self.track_visualization_mask.shape[1] != frame_shape_with_channels[1] or \
           self.track_visualization_mask.shape[2] != frame_shape_with_channels[2]:
            self.track_visualization_mask = np.zeros(frame_shape_with_channels, dtype=np.uint8)
            # print("Track visualization mask initialized/resized.")

    def _manage_track_mask_refresh(self):
        """Manages the refresh logic for the track visualization mask."""
        if self._track_refresh_interval and self._track_refresh_interval > 0:
            self._frame_count_since_last_refresh += 1
            if self._frame_count_since_last_refresh >= self._track_refresh_interval:
                if self.track_visualization_mask is not None:
                    self.track_visualization_mask.fill(0)
                self._frame_count_since_last_refresh = 0

    def draw_feature_tracks(self, display_frame_bgr, prev_points, current_points, track_color=(0, 255, 0)):
        """
        Draws feature tracks on the display frame.
        Manages an internal track_visualization_mask which accumulates tracks.
        """
        self._initialize_track_mask(display_frame_bgr.shape)
        self._manage_track_mask_refresh()

        vis_frame = display_frame_bgr.copy()

        if prev_points is not None and current_points is not None and len(prev_points) == len(current_points):
            for _, (new, old) in enumerate(zip(current_points, prev_points)):
                pt_new = new.ravel().astype(int)
                pt_old = old.ravel().astype(int)
                # Draw line on the persistent mask
                cv2.line(self.track_visualization_mask, tuple(pt_new), tuple(pt_old), track_color, 2)
                # Draw current feature point on the current frame copy
                cv2.circle(vis_frame, tuple(pt_new), 3, track_color, -1)

        # Add the accumulated tracks from the mask to the current visual frame
        vis_img_tracks = cv2.add(vis_frame, self.track_visualization_mask)
        cv2.imshow(FEATURE_TRACK_WINDOW, vis_img_tracks)
        return vis_img_tracks
    
    def draw_patch_locations(self, 
                             display_frame_bgr, 
                             patch_centers_xy, 
                             patch_size_pixels, 
                             color=(0, 0, 255), 
                             thickness=2):
        """
        Draws patch bounding boxes on the display frame.
        Args:
            display_frame_bgr: The BGR image (NumPy array) to draw on.
            patch_centers_xy: NumPy array of shape (M, 2) with (x, y) centers of M patches in image pixel coordinates.
            patch_size_pixels: The size (width and height) of the patch square in image pixels.
            color: Tuple for patch color (B, G, R).
            thickness: Thickness of the rectangle lines.
        """
        vis_frame = display_frame_bgr.copy()
        P_half = patch_size_pixels // 2

        for i in range(patch_centers_xy.shape[0]):
            center_x = patch_centers_xy[i, 0]
            center_y = patch_centers_xy[i, 1]
            
            x1 = int(center_x - P_half)
            y1 = int(center_y - P_half)
            x2 = int(center_x + P_half)
            y2 = int(center_y + P_half)
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)

        # --- Scale the output for PATCH_WINDOW by 2 ---
        display_scale_factor = 4
        if vis_frame.shape[0] > 0 and vis_frame.shape[1] > 0: # Check if frame is not empty
            # Calculate new dimensions
            new_width = vis_frame.shape[1] * display_scale_factor
            new_height = vis_frame.shape[0] * display_scale_factor
            # Resize the frame
            vis_frame_display = cv2.resize(vis_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            vis_frame_display = vis_frame # Fallback to original if empty or problematic
        

        cv2.imshow(PATCH_WINDOW, vis_frame_display)
        return vis_frame_display

    def draw_patch_motion_tracks(self, display_frame_bgr,
                                 current_patch_centers_feat_th, # Starting points of flows, shape (M, 2)
                                 patch_flow_vectors_feat_th,    # Flow vectors (deltas), shape (M, 2)
                                 P_feat, RES,                   # Patch size in feature map, Resolution scale
                                 box_color=(0, 255, 0), track_color=(255, 0, 0),
                                 patch_confidences_feat_th=None, # Confidences, shape (M, 2) or (M, 1)
                                 thickness=2):
        vis_frame = display_frame_bgr.copy()

        if current_patch_centers_feat_th is None or patch_flow_vectors_feat_th is None:
            # Fallback: if flow is missing, just draw current patch locations if available
            if current_patch_centers_feat_th is not None:
                patch_centers_img_np = (current_patch_centers_feat_th * RES).cpu().numpy()
                patch_size_img = P_feat * RES
                return self.draw_patch_locations(vis_frame,
                                                 patch_centers_img_np,
                                                 patch_size_img,
                                                 color=box_color, thickness=thickness)
            else:
                cv2.imshow(PATCH_WINDOW, vis_frame)
                return vis_frame
        
        num_patches = current_patch_centers_feat_th.shape[0]
        if num_patches == 0 or num_patches != patch_flow_vectors_feat_th.shape[0]:
            print(f"Warning: Mismatch in patch numbers or zero patches. Centers: {num_patches}, Flows: {patch_flow_vectors_feat_th.shape[0]}")
            cv2.imshow(PATCH_WINDOW, vis_frame)
            return vis_frame

        # Calculate previous (start of flow) and current (end of flow) positions in feature map scale
        # The 'current_patch_centers_feat_th' are the P_start for the flow vectors.
        # The 'P_end' would be P_start + flow.
        start_positions_feat = current_patch_centers_feat_th
        end_positions_feat = current_patch_centers_feat_th + patch_flow_vectors_feat_th

        # Convert to image scale for drawing
        start_centers_img_np = (start_positions_feat * RES).cpu().numpy()
        end_centers_img_np = (end_positions_feat * RES).cpu().numpy()

        patch_size_img = P_feat * RES # Patch display size in image pixels
        P_half_img = patch_size_img // 2

        # --- Drawing ---
        for i in range(num_patches):
            sx, sy = int(start_centers_img_np[i, 0]), int(start_centers_img_np[i, 1]) # Start of arrow
            ex, ey = int(end_centers_img_np[i, 0]), int(end_centers_img_np[i, 1])     # End of arrow (current pos)

            current_track_color_tuple = tuple(track_color) # Default track color

            if patch_confidences_feat_th is not None and i < patch_confidences_feat_th.shape[0]:
                # Use confidence to modulate color (e.g., brighter for higher confidence)
                # Assuming confidence is (M, 2), take the mean. If (M,1) just use it.
                confidence = patch_confidences_feat_th[i].mean().item() if patch_confidences_feat_th.shape[1]==2 else patch_confidences_feat_th[i].item()
                # Example: Scale color intensity by confidence. Min intensity 0.2 to keep it visible.
                alpha = 0.2 + 0.8 * confidence
                current_track_color_tuple = tuple(int(c * alpha) for c in track_color)

            # Draw the flow vector (line from start to end)
            cv2.line(vis_frame, (sx, sy), (ex, ey), current_track_color_tuple, thickness)
            # Draw a small circle at the start of the flow vector
            cv2.circle(vis_frame, (sx, sy), radius=max(1, int(thickness/2)), color=current_track_color_tuple, thickness=-1)

            # Draw the patch box at the END of the flow vector (current estimated position)
            box_x1, box_y1 = ex - P_half_img, ey - P_half_img
            box_x2, box_y2 = ex + P_half_img, ey + P_half_img
            cv2.rectangle(vis_frame, (box_x1, box_y1), (box_x2, box_y2), box_color, thickness)

        cv2.imshow(PATCH_WINDOW, vis_frame)
        return vis_frame



    def draw_trajectory_map(self, trajectory_points_list, scale=10):
        """Draws the 2D trajectory map (viewed from top-down, X-Z plane)."""
        traj_img = np.zeros((self.traj_img_height, self.traj_img_width, 3), dtype=np.uint8)
        center_x, center_y = self.traj_img_width // 2, self.traj_img_height // 2

        if not trajectory_points_list:
            return traj_img

        # Convert 3D points (x, y, z) to 2D screen points (plotting x and z)
        screen_points = []
        for pt3d in trajectory_points_list:
            if pt3d is not None and len(pt3d) >= 3: # Expects (x,y,z)
                 screen_points.append( (int(center_x + pt3d[0] * scale), int(center_y + pt3d[2] * scale)) )
            # Add handling for other formats if necessary, e.g. if points are already (x,z)

        if len(screen_points) > 1:
            for i in range(len(screen_points) - 1):
                cv2.line(traj_img, screen_points[i], screen_points[i+1], (0, 255, 0), 2)

        if screen_points:
            cv2.circle(traj_img, screen_points[-1], 5, (0, 0, 255), -1)  # Current position in red
            cv2.circle(traj_img, screen_points[0], 5, (255, 0, 0), -1)    # Start position in blue

        cv2.putText(traj_img, f"Scale: {scale:.1f} pix/m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow(TAJECTORY_WINDOW, traj_img)
        return traj_img

    def reset_tracks_visualization(self):
        """Resets the feature track visualization mask and its refresh counter."""
        if self.track_visualization_mask is not None:
            self.track_visualization_mask.fill(0)
        self._frame_count_since_last_refresh = 0
        print("Visualizer track mask and counter reset.")