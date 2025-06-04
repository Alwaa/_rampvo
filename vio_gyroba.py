import cv2
import numpy as np
import os 
from pathlib import Path
import glob
from scipy.spatial.transform import Rotation as R_sci
from scipy.optimize import least_squares
import time # For simulating timestamps

# --- Configuration: ---
DATASET_DIR = "/run/media/alexander/T5 EVO/datasets" 
CAMERA_VIEW = "left" 
MAX_FRAMES_TO_PROCESS = None 
TRACK_REFRESH_INTERVAL = 30 
INTRINSICS = (320.0, 320.0, 320.0, 240.0) 
OUTPUT_POSE_FILE_NAME = "stamped_traj_estimate_gyro_ba.txt" 
Y_FACTOR = 1 
FB_ERROR_THRESHOLD = 1.0 

# IMU Simulation Parameters
SIM_IMU_RATE = 200.0  # Hz
SIM_GYRO_NOISE_STD = np.deg2rad(0.1) # rad/s
SIM_TRUE_GYRO_BIAS = np.array([np.deg2rad(0.05), np.deg2rad(-0.03), np.deg2rad(0.02)]) # rad/s
# --- End of Configuration ---


# --- Parameters for feature detection and tracking (More Aggressive) ---
FEATURE_PARAMS = dict(maxCorners=500, qualityLevel=0.005, minDistance=5, blockSize=7)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

IMAGE_FOLDER_TO_USE = Path(DATASET_DIR) / "seasonsforest" / "Easy" / "P001" / f"image_{CAMERA_VIEW}"

# --- Helper Functions ---
def load_camera_intrinsics():
    fx, fy, cx, cy = INTRINSICS
    K = np.array([[fx, 0,  cx], [0,  fy, cy], [0,  0,  1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32) 
    print(f"Using hardcoded intrinsics (fx, fy, cx, cy): {fx}, {fy}, {cx}, {cy}")
    return K, dist

def load_tartanair_image_paths(): 
    if not IMAGE_FOLDER_TO_USE.exists() or not IMAGE_FOLDER_TO_USE.is_dir():
        print(f"Error: IMAGE_FOLDER_TO_USE does not exist or is not a directory: {IMAGE_FOLDER_TO_USE}"); return []
    print(f"Loading images from: {IMAGE_FOLDER_TO_USE}")
    image_files = sorted(glob.glob(str(IMAGE_FOLDER_TO_USE / "*.png")))
    if not image_files: print(f"Warning: No PNG images found in {IMAGE_FOLDER_TO_USE}")
    return image_files

def draw_tracks(display_frame, prev_points, current_points, track_mask, track_color=(0, 255, 0)):
    vis_frame = display_frame.copy() 
    if prev_points is not None and current_points is not None and len(prev_points) == len(current_points):
        for _, (new, old) in enumerate(zip(current_points, prev_points)): 
            pt_new, pt_old = new.ravel().astype(int), old.ravel().astype(int)
            track_mask = cv2.line(track_mask, tuple(pt_new), tuple(pt_old), track_color, 2)
            vis_frame = cv2.circle(vis_frame, tuple(pt_new), 3, track_color, -1)
    return cv2.add(vis_frame, track_mask), track_mask

def draw_trajectory(trajectory_points_list, traj_img_width, traj_img_height, scale=10, window_name="Trajectory"):
    traj_img = np.zeros((traj_img_height, traj_img_width, 3), dtype=np.uint8)
    center_x, center_y = traj_img_width // 2, traj_img_height // 2
    if not trajectory_points_list: cv2.imshow(window_name, traj_img); return traj_img
    screen_points = [ (int(center_x + pt3d[0] * scale), int(center_y + pt3d[2] * scale)) for pt3d in trajectory_points_list]
    for i in range(len(screen_points) - 1): cv2.line(traj_img, screen_points[i], screen_points[i+1], (0, 255, 0), 2)
    if screen_points: 
        cv2.circle(traj_img, screen_points[-1], 5, (0, 0, 255), -1) 
        cv2.circle(traj_img, screen_points[0], 5, (255,0,0), -1)
    cv2.putText(traj_img, f"Scale: {scale:.1f} pix/m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imshow(window_name, traj_img); return traj_img

def get_simulated_imu_data(prev_img_timestamp, current_img_timestamp, true_angular_velocity_body, gyro_bias_true, noise_std):
    """ Simulates IMU gyro data for the interval. Returns list of (gyro_xyz, dt)."""
    imu_data = []
    dt_imu = 1.0 / SIM_IMU_RATE
    current_imu_time = prev_img_timestamp
    while current_imu_time < current_img_timestamp:
        # Simulate some change in true angular velocity if desired, for now constant
        noisy_gyro = true_angular_velocity_body + gyro_bias_true + np.random.normal(0, noise_std, 3)
        imu_data.append({'ts': current_imu_time, 'gyro': noisy_gyro})
        current_imu_time += dt_imu
    # Ensure at least one measurement if interval is very short
    if not imu_data and prev_img_timestamp < current_img_timestamp :
         noisy_gyro = true_angular_velocity_body + gyro_bias_true + np.random.normal(0, noise_std, 3)
         imu_data.append({'ts': prev_img_timestamp, 'gyro': noisy_gyro}) # Add one at start of interval
    
    # Convert to (gyro_sample, dt_segment)
    processed_imu = []
    if len(imu_data) > 1:
        for i in range(len(imu_data) -1):
            dt_segment = imu_data[i+1]['ts'] - imu_data[i]['ts']
            # Use midpoint or start gyro for the segment
            processed_imu.append( ( (imu_data[i]['gyro'] + imu_data[i+1]['gyro'])/2.0 , dt_segment) )
    elif len(imu_data) == 1: # Single IMU reading for the whole interval
        dt_segment = current_img_timestamp - prev_img_timestamp
        if dt_segment > 0:
             processed_imu.append( (imu_data[0]['gyro'], dt_segment) )

    return processed_imu

def log_map_SO3(R):
    """SO(3) logarithm map: converts rotation matrix to 3-vector."""
    angle = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(angle, 0): return np.zeros(3)
    # Check for numerical instability near pi
    if np.isclose(angle, np.pi): # Special case for 180-degree rotations
        # Determine axis from columns of R + I
        # For example, if R = diag([-1, -1, 1]), axis is [0,0,1] * pi
        # This can be tricky. A robust way is to find eigenvector for eigenvalue 1 of R.
        # For simplicity here, if trace is -1 (angle is pi), we might need a more robust axis extraction.
        # However, scipy's Rotation.as_rotvec() handles this.
        return R_sci.from_matrix(R).as_rotvec()
    return (angle / (2 * np.sin(angle))) * np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])


# --- Visual Odometry Pipeline Class ---
class VisualOdometryPipeline:
    def __init__(self, K_matrix, feature_params, lk_params, fb_thresh):
        self.K_matrix = K_matrix; self.fx = K_matrix[0,0]; self.fy = K_matrix[1,1]
        self.cx = K_matrix[0,2]; self.cy = K_matrix[1,2]
        self.feature_params = feature_params; self.lk_params = lk_params
        self.fb_thresh = fb_thresh; self.next_feature_id = 0
        self.gyro_bias = np.zeros(3) # Initialize gyro bias
        self.current_frame_time = 0.0 # Keep track of current image timestamp
        self.reset()

    def reset(self):
        self.R_global_pose = np.eye(3); self.t_global_pos = np.zeros((3, 1))
        self.trajectory_vo_points = [self.t_global_pos.flatten().copy()]
        self.prev_gray = None; self.prev_features_with_ids = [] 
        self.is_first_frame = True; self.camera_poses_ba = [] 
        self.map_points_ba = {}; self.point_observations_ba = [] 
        self.last_successful_pose_R = np.eye(3); self.last_successful_pose_t = np.zeros((3,1))
        self.gyro_bias = np.zeros(3) # Reset bias too
        self.current_frame_time = 0.0
        print("VO Pipeline Reset.")

    def _get_feature_points_array(self, features_list):
        if not features_list: return np.array([], dtype=np.float32).reshape(0,1,2)
        return np.array([f['pt'] for f in features_list], dtype=np.float32)

    def _assign_new_feature_ids(self, points_arr):
        new_feats = []
        if points_arr is None: return new_feats
        for pt in points_arr:
            new_feats.append({'id': self.next_feature_id, 'pt': pt.reshape(1,2)})
            self.next_feature_id += 1
        return new_feats
    
    def _preintegrate_gyro(self, gyro_data_with_dt_list, current_gyro_bias):
        delta_R_imu = np.eye(3)
        for gyro_sample, dt_segment in gyro_data_with_dt_list:
            calibrated_gyro = gyro_sample - current_gyro_bias
            delta_R_segment = R_sci.from_rotvec(calibrated_gyro * dt_segment).as_matrix()
            delta_R_imu = delta_R_imu @ delta_R_segment
        return delta_R_imu

    def _process_features_internal(self, current_gray, frame_idx_str=""):
        # ... (Feature processing logic remains the same as previous version) ...
        prev_points_arr = self._get_feature_points_array(self.prev_features_with_ids)
        good_old_for_vo_pts, good_new_for_vo_pts, corresponding_ids_for_vo = None, None, None
        if len(prev_points_arr) >= 1:
            p1_fwd, st_fwd, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, prev_points_arr, None, **self.lk_params)
            fwd_success_mask = (st_fwd.flatten() == 1)
            p0_fwd_good_pts, p1_fwd_good_pts = prev_points_arr[fwd_success_mask], p1_fwd[fwd_success_mask]
            ids_fwd_good = [self.prev_features_with_ids[i]['id'] for i, success in enumerate(fwd_success_mask) if success]
            if len(p1_fwd_good_pts) > 0:
                p0_reprojected, st_bwd, _ = cv2.calcOpticalFlowPyrLK(current_gray, self.prev_gray, p1_fwd_good_pts, None, **self.lk_params)
                bwd_success_mask = (st_bwd.flatten() == 1)
                p0_fwd_bwd_good_pts, p1_fwd_bwd_good_pts = p0_fwd_good_pts[bwd_success_mask], p1_fwd_good_pts[bwd_success_mask]
                ids_fwd_bwd_good = [ids_fwd_good[i] for i, success in enumerate(bwd_success_mask) if success]
                p0_reprojected_final_pts = p0_reprojected[bwd_success_mask]
                if len(p0_reprojected_final_pts) > 0:
                    error_fb = np.linalg.norm(p0_fwd_bwd_good_pts.reshape(-1, 2) - p0_reprojected_final_pts.reshape(-1, 2), axis=1)
                    consistent_mask = error_fb < self.fb_thresh
                    good_old_for_vo_pts, good_new_for_vo_pts = p0_fwd_bwd_good_pts[consistent_mask], p1_fwd_bwd_good_pts[consistent_mask]
                    corresponding_ids_for_vo = [ids_fwd_bwd_good[i] for i, consistent in enumerate(consistent_mask) if consistent]
        newly_detected_points_arr = cv2.goodFeaturesToTrack(current_gray, mask=None, **self.feature_params)
        newly_detected_with_ids = self._assign_new_feature_ids(newly_detected_points_arr) if newly_detected_points_arr is not None else []
        temp_combined_list = []
        if good_new_for_vo_pts is not None and corresponding_ids_for_vo is not None:
            temp_combined_list.extend([{'id': fid, 'pt': pt_arr.reshape(1,2)} for fid, pt_arr in zip(corresponding_ids_for_vo, good_new_for_vo_pts)])
        if newly_detected_with_ids: temp_combined_list.extend(newly_detected_with_ids)
        final_features_for_next_iter = []
        if temp_combined_list:
            unique_pts_dict = {tuple(feat_dict['pt'].ravel()): feat_dict for feat_dict in reversed(temp_combined_list)}
            final_features_for_next_iter = list(unique_pts_dict.values())
            if len(final_features_for_next_iter) > self.feature_params['maxCorners']:
                np.random.shuffle(final_features_for_next_iter)
                final_features_for_next_iter = final_features_for_next_iter[:self.feature_params['maxCorners']]
        return good_old_for_vo_pts, good_new_for_vo_pts, corresponding_ids_for_vo, final_features_for_next_iter

    def _estimate_relative_pose_internal(self, points_prev, points_curr):
        # ... (Remains the same as previous version) ...
        if points_prev is None or points_curr is None or len(points_prev) < 8 or len(points_curr) < 8: return None, None, None
        E, mask_e = cv2.findEssentialMat(points_curr, points_prev, self.K_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or np.sum(mask_e) < 5: return None, None, None
        inlier_mask = mask_e.flatten().astype(bool)
        inlier_new, inlier_old = points_curr[inlier_mask], points_prev[inlier_mask] 
        retval, R_mat, t_vec, _ = cv2.recoverPose(E, inlier_new, inlier_old, self.K_matrix)
        if retval > 0 and R_mat is not None and t_vec is not None: return R_mat.T, -R_mat.T @ t_vec, inlier_mask 
        return None, None, None

    def _triangulate_and_update_map(self, pose_id_prev, pose_id_curr, old_pts_arr, new_pts_arr, feature_ids_arr, inlier_mask_from_essential):
        # ... (Remains the same as previous version) ...
        if old_pts_arr is None or new_pts_arr is None or feature_ids_arr is None or not np.any(inlier_mask_from_essential): return
        old_pts_inliers, new_pts_inliers = old_pts_arr[inlier_mask_from_essential], new_pts_arr[inlier_mask_from_essential]
        feature_ids_inliers = [fid for i, fid in enumerate(feature_ids_arr) if inlier_mask_from_essential[i]]
        P1 = self.K_matrix @ np.hstack((self.last_successful_pose_R.T, -self.last_successful_pose_R.T @ self.last_successful_pose_t))
        P2 = self.K_matrix @ np.hstack((self.R_global_pose.T, -self.R_global_pose.T @ self.t_global_pos))
        pts4D_h = cv2.triangulatePoints(P1, P2, old_pts_inliers.T.reshape(2,-1), new_pts_inliers.T.reshape(2,-1))
        valid_depth_mask = pts4D_h[3,:] > 1e-5
        if not np.any(valid_depth_mask): return
        pts4D_h = pts4D_h[:, valid_depth_mask]
        old_pts_inliers_valid, new_pts_inliers_valid = old_pts_inliers[valid_depth_mask], new_pts_inliers[valid_depth_mask]
        feature_ids_inliers_valid = [fid for i, fid in enumerate(feature_ids_inliers) if valid_depth_mask[i]]
        pts3D = pts4D_h[:3, :] / pts4D_h[3, :]
        for i, feature_id in enumerate(feature_ids_inliers_valid):
            pt3d_world = pts3D[:, i]
            if feature_id not in self.map_points_ba:
                self.map_points_ba[feature_id] = {'pt3d': pt3d_world, 'first_observed_pose_id': pose_id_prev}
                self.point_observations_ba.append((feature_id, pose_id_prev, old_pts_inliers_valid[i,0,0], old_pts_inliers_valid[i,0,1]))
            self.point_observations_ba.append((feature_id, pose_id_curr, new_pts_inliers_valid[i,0,0], new_pts_inliers_valid[i,0,1]))

    def process_frame(self, current_gray_frame, frame_timestamp, frame_idx): # Added frame_timestamp
        pose_updated, vis_old_pts_arr, vis_new_pts_arr = False, None, None 
        
        # Simulate getting IMU data for the interval
        # In a real system, you'd fetch actual IMU data based on timestamps
        # For simulation, assume some true angular velocity (e.g., small rotation)
        true_angular_vel_for_sim = np.array([0.0, np.deg2rad(2.0), 0.0]) # rad/s, e.g. yawing slowly
        raw_gyro_data_for_interval = []
        if not self.is_first_frame:
            raw_gyro_data_for_interval = get_simulated_imu_data(
                self.current_frame_time, frame_timestamp, 
                true_angular_vel_for_sim, SIM_TRUE_GYRO_BIAS, SIM_GYRO_NOISE_STD
            )

        if self.is_first_frame:
            detected_features_arr = cv2.goodFeaturesToTrack(current_gray_frame, mask=None, **self.feature_params)
            if detected_features_arr is None or len(detected_features_arr) < 10:
                self.prev_gray = current_gray_frame.copy(); self.current_frame_time = frame_timestamp; return pose_updated, None, None 
            self.prev_features_with_ids = self._assign_new_feature_ids(detected_features_arr)
            self.is_first_frame = False
            vis_new_pts_arr = self._get_feature_points_array(self.prev_features_with_ids)
            self.camera_poses_ba.append({'pose_id': frame_idx, 'R': self.R_global_pose.copy(), 't': self.t_global_pos.copy(), 
                                         'timestamp': frame_timestamp, 'raw_gyro_data': [], 'dt_interval': 0.0}) # No gyro for first frame
            self.last_successful_pose_R, self.last_successful_pose_t = self.R_global_pose.copy(), self.t_global_pos.copy()
        else:
            vis_old_pts_arr, vis_new_pts_arr, corresponding_ids, features_for_next = \
                self._process_features_internal(current_gray_frame, frame_idx_str=str(frame_idx))
            self.prev_features_with_ids = features_for_next

            if vis_new_pts_arr is not None and vis_old_pts_arr is not None and corresponding_ids is not None:
                R_relative, t_relative, essential_inlier_mask = self._estimate_relative_pose_internal(vis_old_pts_arr, vis_new_pts_arr)
                if R_relative is not None and t_relative is not None:
                    self.t_global_pos = self.t_global_pos + self.R_global_pose @ t_relative
                    self.R_global_pose = self.R_global_pose @ R_relative
                    self.trajectory_vo_points.append(self.t_global_pos.flatten().copy())
                    pose_updated = True
                    current_pose_id = frame_idx
                    
                    # Store raw gyro data for this interval with the *new* pose
                    # The preintegrated rotation will be computed using this in BA
                    dt_interval = frame_timestamp - self.current_frame_time
                    self.camera_poses_ba.append({
                        'pose_id': current_pose_id, 'R': self.R_global_pose.copy(), 't': self.t_global_pos.copy(),
                        'timestamp': frame_timestamp, 
                        'raw_gyro_data': raw_gyro_data_for_interval, # List of (gyro_sample, dt_segment)
                        'dt_interval': dt_interval 
                    })
                    
                    if essential_inlier_mask is not None: 
                        self._triangulate_and_update_map(
                            self.camera_poses_ba[-2]['pose_id'], current_pose_id,
                            vis_old_pts_arr, vis_new_pts_arr, 
                            corresponding_ids, essential_inlier_mask 
                        )
                    self.last_successful_pose_R, self.last_successful_pose_t = self.R_global_pose.copy(), self.t_global_pos.copy()
        
        self.prev_gray = current_gray_frame.copy()
        self.current_frame_time = frame_timestamp
        return pose_updated, vis_old_pts_arr, vis_new_pts_arr

    def get_ba_data(self):
        return {"camera_poses": self.camera_poses_ba, "map_points_3d_dict": self.map_points_ba, 
                "point_observations": self.point_observations_ba, "gyro_bias": self.gyro_bias.copy()}

    def _ba_residuals(self, params, n_cameras_to_optimize, n_points, point_obs_formatted, 
                      K_matrix, fixed_pose_R, fixed_pose_t, 
                      raw_gyro_data_for_opt_poses_list, dt_intervals_for_opt_poses_list):
        cam_param_size = 6 
        gyro_bias_size = 3
        
        opt_camera_params = params[:n_cameras_to_optimize * cam_param_size].reshape((n_cameras_to_optimize, cam_param_size))
        opt_gyro_bias = params[n_cameras_to_optimize * cam_param_size : n_cameras_to_optimize * cam_param_size + gyro_bias_size]
        opt_points_3d_params = params[n_cameras_to_optimize * cam_param_size + gyro_bias_size:].reshape((n_points, 3))
        
        residuals = []
        
        all_Rs_ba = [fixed_pose_R] + [R_sci.from_rotvec(opt_camera_params[i, :3]).as_matrix() for i in range(n_cameras_to_optimize)]
        all_ts_ba = [fixed_pose_t] + [opt_camera_params[i, 3:].reshape(3,1) for i in range(n_cameras_to_optimize)]

        # Visual Reprojection Errors
        for point_ba_idx, pose_ba_idx_in_active_list, x_obs, y_obs in point_obs_formatted:
            if not (0 <= pose_ba_idx_in_active_list < len(all_Rs_ba)): continue 
            R_glob, t_glob = all_Rs_ba[pose_ba_idx_in_active_list], all_ts_ba[pose_ba_idx_in_active_list]
            P_world = opt_points_3d_params[point_ba_idx]
            P_cam = R_glob.T @ (P_world.reshape(3,1) - t_glob)
            if P_cam[2,0] < 1e-4: residuals.extend([1e6, 1e6]); continue 
            x_proj = self.fx * P_cam[0,0] / P_cam[2,0] + self.cx
            y_proj = self.fy * P_cam[1,0] / P_cam[2,0] + self.cy
            residuals.extend([x_proj - x_obs, y_proj - y_obs])

        # IMU Gyro Preintegration Errors
        # Iterate through consecutive optimizable poses (pose_idx from 0 to n_cameras_to_optimize-1)
        # This corresponds to intervals between all_Rs_ba[pose_idx] and all_Rs_ba[pose_idx+1]
        for i in range(n_cameras_to_optimize): 
            # R_i is all_Rs_ba[i], R_j is all_Rs_ba[i+1]
            # If i=0, R_i is fixed_pose_R, R_j is the first optimized pose
            # If i>0, R_i is opt_camera_params[i-1], R_j is opt_camera_params[i]
            
            R_i_world = all_Rs_ba[i] 
            R_j_world = all_Rs_ba[i+1]

            # Relative rotation from visual poses
            delta_R_visual = R_i_world.T @ R_j_world # Rotation from body_i to body_j

            # Preintegrated rotation from IMU using current bias estimate
            # raw_gyro_data_for_opt_poses_list[i] corresponds to interval between pose i and pose i+1
            # dt_intervals_for_opt_poses_list[i] also for this interval
            if i < len(raw_gyro_data_for_opt_poses_list): # Ensure data exists
                delta_R_imu = self._preintegrate_gyro(raw_gyro_data_for_opt_poses_list[i], opt_gyro_bias)
                
                # Rotational error: log(delta_R_imu_transpose * delta_R_visual)
                error_rot_mat = delta_R_imu.T @ delta_R_visual
                error_rot_vec = log_map_SO3(error_rot_mat) 
                residuals.extend(error_rot_vec * 1.0) # Weighting factor for gyro error (e.g. 1.0)
            # else:
                # This case should not happen if data is prepared correctly
                # residuals.extend([0,0,0]) # Or a large error if it's unexpected

        return np.array(residuals)

    def run_bundle_adjustment(self, window_size=None):
        print("\n--- Starting Bundle Adjustment ---")
        if len(self.camera_poses_ba) <=1 or not self.map_points_ba or not self.point_observations_ba:
            print("Not enough data for BA."); return False

        active_camera_poses_data = self.camera_poses_ba
        if window_size and len(self.camera_poses_ba) > window_size:
            active_camera_poses_data = self.camera_poses_ba[-window_size:]
        
        n_cameras_total_active = len(active_camera_poses_data)
        if n_cameras_total_active <= 1: print("Not enough cameras in window for BA."); return False
        
        n_cameras_to_optimize = n_cameras_total_active - 1 
        fixed_pose_data = active_camera_poses_data[0] 

        active_pose_ids = {p['pose_id'] for p in active_camera_poses_data}
        active_observations = [(fid, pid, x, y) for fid, pid, x, y in self.point_observations_ba if pid in active_pose_ids]
        relevant_feature_ids = {fid for fid, _, _, _ in active_observations}
        
        if not active_observations or not relevant_feature_ids: print("No relevant obs/pts in window for BA."); return False

        active_map_point_ids_list = sorted([fid for fid in relevant_feature_ids if fid in self.map_points_ba])
        active_feature_id_to_ba_point_idx = {fid: i for i, fid in enumerate(active_map_point_ids_list)}
        n_points_ba_active = len(active_map_point_ids_list)
        if n_points_ba_active == 0: print("No map points in window for BA."); return False

        initial_camera_params_active = []
        raw_gyro_data_for_active_intervals = [] # For intervals between active_camera_poses_data[i] and [i+1]
        dt_intervals_for_active_intervals = []

        for i in range(1, n_cameras_total_active): 
            pose_data = active_camera_poses_data[i] # This is the j-th pose in R_i.T @ R_j
            initial_camera_params_active.extend(R_sci.from_matrix(pose_data['R']).as_rotvec())
            initial_camera_params_active.extend(pose_data['t'].flatten())
            # Gyro data is associated with the *end* pose of an interval
            raw_gyro_data_for_active_intervals.append(pose_data['raw_gyro_data'])
            dt_intervals_for_active_intervals.append(pose_data['dt_interval'])


        initial_points_3d_active = np.array([self.map_points_ba[fid]['pt3d'] for fid in active_map_point_ids_list]).flatten()
        initial_gyro_bias = self.gyro_bias.copy()
        params0 = np.concatenate([initial_camera_params_active, initial_gyro_bias, initial_points_3d_active])

        ba_observations_formatted_active = []
        original_pose_id_to_active_idx = {p_data['pose_id']: i for i, p_data in enumerate(active_camera_poses_data)}

        for feature_id, obs_original_pose_id, x_obs, y_obs in active_observations:
            if feature_id in active_feature_id_to_ba_point_idx and obs_original_pose_id in original_pose_id_to_active_idx:
                point_ba_idx = active_feature_id_to_ba_point_idx[feature_id]
                pose_ba_idx_in_active = original_pose_id_to_active_idx[obs_original_pose_id]
                ba_observations_formatted_active.append((point_ba_idx, pose_ba_idx_in_active, x_obs, y_obs))
        
        if not ba_observations_formatted_active: print("No valid formatted observations for BA window."); return False
        
        print(f"Optimizing {n_cameras_to_optimize} cameras, {n_points_ba_active} points, gyro_bias in window with {len(ba_observations_formatted_active)} visual obs.")
        
        res = least_squares(
            self._ba_residuals, params0, 
            args=(n_cameras_to_optimize, n_points_ba_active, ba_observations_formatted_active, 
                  self.K_matrix, fixed_pose_data['R'], fixed_pose_data['t'],
                  raw_gyro_data_for_active_intervals, dt_intervals_for_active_intervals), # Pass raw gyro data
            verbose=0, ftol=2e-3, xtol=2e-3, gtol=2e-3, max_nfev=60, method='lm' # Further reduced max_nfev
        )

        optimized_params = res.x; cam_param_size = 6; gyro_bias_size = 3
        opt_camera_params = optimized_params[:n_cameras_to_optimize * cam_param_size].reshape((n_cameras_to_optimize, cam_param_size))
        self.gyro_bias = optimized_params[n_cameras_to_optimize * cam_param_size : n_cameras_to_optimize * cam_param_size + gyro_bias_size]
        opt_points_3d = optimized_params[n_cameras_to_optimize * cam_param_size + gyro_bias_size:].reshape((n_points_ba_active, 3))
        print(f"Optimized Gyro Bias: {np.rad2deg(self.gyro_bias)} deg/s")

        for i in range(n_cameras_to_optimize):
            target_pose_id = active_camera_poses_data[i+1]['pose_id']
            try:
                original_pose_idx_in_pipeline = next(idx for idx,p_dict in enumerate(self.camera_poses_ba) if p_dict['pose_id'] == target_pose_id)
                self.camera_poses_ba[original_pose_idx_in_pipeline]['R'] = R_sci.from_rotvec(opt_camera_params[i, :3]).as_matrix()
                self.camera_poses_ba[original_pose_idx_in_pipeline]['t'] = opt_camera_params[i, 3:].reshape(3,1)
            except StopIteration: pass # Should not happen if logic is correct

        for i, feature_id in enumerate(active_map_point_ids_list):
            if feature_id in self.map_points_ba: self.map_points_ba[feature_id]['pt3d'] = opt_points_3d[i]
        
        print("Bundle Adjustment finished for window.")
        self.trajectory_vo_points = [p['t'].flatten().copy() for p in self.camera_poses_ba] 
        return True

# --- Main Execution Logic ---
def run_visual_odometry_main(K_matrix, dist_coeffs_val, max_frames, track_refresh_interval, output_pose_file, fb_error_thresh):
    BA_WINDOW_SIZE = 8  # Further reduced window size
    BA_TRIGGER_INTERVAL = 12 # Run BA less frequently

    vo_pipeline = VisualOdometryPipeline(K_matrix, FEATURE_PARAMS, LK_PARAMS, fb_error_thresh)
    image_paths = load_tartanair_image_paths()
    if not image_paths: return

    if max_frames is not None and max_frames > 0 : image_paths = image_paths[:max_frames]
    print(f"Processing a maximum of {len(image_paths)} frames.")

    track_visualization_mask, traj_img_width, traj_img_height = None, 400, 600
    cv2.namedWindow('Feature Tracks - VO', cv2.WINDOW_NORMAL); cv2.namedWindow('Trajectory (VO)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trajectory (VO)', traj_img_width, traj_img_height)
    frame_count_since_last_refresh = 0
    
    current_loop_time = 0.0 # Simulated time for images / IMU
    
    pose_file_writer = None
    if output_pose_file:
        try:
            pose_file_writer = open(output_pose_file, 'w') 
            pose_file_writer.write("# timestamp tx ty tz qx qy qz qw (Post-SWBA)\n")
            print(f"Opened pose output file: {output_pose_file}")
        except IOError as e: print(f"Error opening pose output file '{output_pose_file}': {e}"); pose_file_writer = None

    for frame_idx in range(len(image_paths)):
        # Simulate timestamp for current image frame (e.g., assuming 10 FPS for VO)
        # This timestamp is passed to the VO pipeline
        # In a real system, image_paths might be tuples of (path, timestamp)
        image_timestamp = frame_idx * 0.1 # Example: 10 FPS
        
        current_frame_bgr = cv2.imread(image_paths[frame_idx])
        if current_frame_bgr is None: continue
        if track_visualization_mask is None: track_visualization_mask = np.zeros_like(current_frame_bgr)
        current_frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)

        pose_updated, vis_old_pts, vis_new_pts = vo_pipeline.process_frame(current_frame_gray, image_timestamp, frame_idx)

        if track_refresh_interval and track_refresh_interval > 0 and frame_count_since_last_refresh >= track_refresh_interval:
            track_visualization_mask.fill(0); frame_count_since_last_refresh = 0
        else: frame_count_since_last_refresh += 1
        
        vis_img_tracks, track_visualization_mask = draw_tracks(current_frame_bgr, vis_old_pts, vis_new_pts, track_visualization_mask)
        cv2.imshow('Feature Tracks - VO', vis_img_tracks)
        
        if frame_idx > 0 and frame_idx % BA_TRIGGER_INTERVAL == 0 and len(vo_pipeline.camera_poses_ba) >= BA_WINDOW_SIZE:
            print(f"\n--- Triggering SWBA at frame {frame_idx} ---")
            vo_pipeline.run_bundle_adjustment(window_size=BA_WINDOW_SIZE)

        if pose_file_writer and vo_pipeline.camera_poses_ba:
            latest_pose_in_pipeline = vo_pipeline.camera_poses_ba[-1]
            if latest_pose_in_pipeline['pose_id'] == frame_idx : 
                # Use the timestamp stored with the pose for consistency
                ts_to_write = latest_pose_in_pipeline['timestamp'] 
                current_quat = R_sci.from_matrix(latest_pose_in_pipeline['R']).as_quat()
                t_curr = latest_pose_in_pipeline['t']
                pose_file_writer.write(f"{ts_to_write:.6f} {t_curr[0,0]:.6f} {Y_FACTOR*t_curr[1,0]:.6f} {t_curr[2,0]:.6f} "
                                    f"{current_quat[0]:.6f} {current_quat[1]:.6f} {current_quat[2]:.6f} {current_quat[3]:.6f}\n")

        dynamic_scale = 10.0
        if len(vo_pipeline.trajectory_vo_points) > 1:
            coords = np.array(vo_pipeline.trajectory_vo_points)
            max_val = np.max(np.abs(coords[:, [0,2]])) if coords.size > 0 and coords.shape[1] >=3 else 1.0 
            current_max_range = max(max_val, 1e-5)
            dynamic_scale = (min(traj_img_width, traj_img_height) / 2.5) / current_max_range if current_max_range > 0 else 10.0
            dynamic_scale = max(1.0, min(dynamic_scale, 200.0))
        
        draw_trajectory(vo_pipeline.trajectory_vo_points, traj_img_width, traj_img_height, scale=dynamic_scale, window_name="Trajectory (VO)")
        
        key = cv2.waitKey(1) & 0xff # Reduced waitKey for faster processing if possible
        if key == 27: print("ESC pressed, stopping."); break
        if key == ord('r'): 
            print("User requested VO reset ('r' key)."); vo_pipeline.reset() 
            if track_visualization_mask is not None: track_visualization_mask.fill(0)
            frame_count_since_last_refresh = 0
            if pose_file_writer: 
                reset_timestamp = image_timestamp + 0.001 # Ensure distinct timestamp
                if not vo_pipeline.camera_poses_ba or vo_pipeline.camera_poses_ba[-1]['pose_id'] != frame_idx : 
                     vo_pipeline.camera_poses_ba.append({'pose_id': frame_idx, 'R': vo_pipeline.R_global_pose.copy(), 
                                                         't': vo_pipeline.t_global_pos.copy(), 'timestamp': reset_timestamp,
                                                         'raw_gyro_data': [], 'dt_interval': 0.0})
                initial_quat = R_sci.from_matrix(vo_pipeline.R_global_pose).as_quat()
                t_reset = vo_pipeline.t_global_pos
                pose_file_writer.write(f"{reset_timestamp:.6f} {t_reset[0,0]:.6f} {Y_FACTOR*t_reset[1,0]:.6f} {t_reset[2,0]:.6f} "
                                   f"{initial_quat[0]:.6f} {initial_quat[1]:.6f} {initial_quat[2]:.6f} {initial_quat[3]:.6f}\n")
        current_loop_time = image_timestamp # Update loop time


    if len(vo_pipeline.camera_poses_ba) > 1 :
        print(f"\n--- Running Final BA ---")
        final_window_size = BA_WINDOW_SIZE if len(vo_pipeline.camera_poses_ba) >= BA_WINDOW_SIZE else len(vo_pipeline.camera_poses_ba)
        if final_window_size > 1 : 
             vo_pipeline.run_bundle_adjustment(window_size=final_window_size)
             if pose_file_writer: pose_file_writer.close() 
             if output_pose_file:
                try:
                    with open(output_pose_file, 'w') as final_pf_writer: 
                        final_pf_writer.write("# timestamp tx ty tz qx qy qz qw (Final BA)\n")
                        for pose_data in vo_pipeline.camera_poses_ba:
                            ts_to_write = pose_data['timestamp']
                            quat_opt = R_sci.from_matrix(pose_data['R']).as_quat()
                            t_opt = pose_data['t']
                            final_pf_writer.write(f"{ts_to_write:.6f} {t_opt[0,0]:.6f} {Y_FACTOR*t_opt[1,0]:.6f} {t_opt[2,0]:.6f} "
                                            f"{quat_opt[0]:.6f} {quat_opt[1]:.6f} {quat_opt[2]:.6f} {quat_opt[3]:.6f}\n")
                        print(f"Final BA optimized pose data saved to {output_pose_file}")
                except IOError as e: print(f"Error saving final BA pose data: {e}")
        else:
            if pose_file_writer: pose_file_writer.close()

    cv2.destroyAllWindows()
    print("Visual Odometry finished.")
    if vo_pipeline.trajectory_vo_points: print("Final camera position (X,Y,Z):", vo_pipeline.trajectory_vo_points[-1])

    ba_data = vo_pipeline.get_ba_data()
    print(f"\n--- Data for Bundle Adjustment (Final State) ---")
    print(f"Number of camera poses: {len(ba_data['camera_poses'])}")
    print(f"Number of 3D map points: {len(ba_data['map_points_3d_dict'])}")
    print(f"Number of 2D observations: {len(ba_data['point_observations'])}")
    print(f"Final Gyro Bias: {np.rad2deg(ba_data['gyro_bias'])} deg/s")


# --- Main Execution ---
if __name__ == '__main__':
    camera_matrix_K_main, dist_coeffs_main = load_camera_intrinsics() 
    if camera_matrix_K_main is None: exit(1)
    
    output_file_path_main = Path(OUTPUT_POSE_FILE_NAME) if OUTPUT_POSE_FILE_NAME else None
    print(f"Starting Visual Odometry for images in: {IMAGE_FOLDER_TO_USE}")
    print(f"FB Error Threshold: {FB_ERROR_THRESHOLD}")
    if output_file_path_main: print(f"Output pose file: {output_file_path_main.resolve()}")
    
    run_visual_odometry_main( 
        K_matrix=camera_matrix_K_main, dist_coeffs_val=dist_coeffs_main, 
        max_frames=MAX_FRAMES_TO_PROCESS, track_refresh_interval=TRACK_REFRESH_INTERVAL,
        output_pose_file=str(output_file_path_main) if output_file_path_main else None,
        fb_error_thresh=FB_ERROR_THRESHOLD
    )
