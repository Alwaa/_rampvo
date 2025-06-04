import cv2
import numpy as np
import os
from pathlib import Path
import glob
from scipy.spatial.transform import Rotation as R
from utils.new_eval_utils import Visualizer

# --- Configuration: ---
DATASET_DIR = "/run/media/alexander/T5 EVO/datasets"
CAMERA_VIEW = "left"
MAX_FRAMES_TO_PROCESS = None
TRACK_REFRESH_INTERVAL = 30  # This will be passed to the Visualizer
INTRINSICS = (320.0, 320.0, 320.0, 240.0)
OUTPUT_POSE_FILE_NAME = "stamped_traj_estimate_ba_prep.txt"
Y_FACTOR = 1

FB_ERROR_THRESHOLD = 1.0
# --- End of Configuration ---


# --- Parameters for feature detection and tracking (More Aggressive) ---
FEATURE_PARAMS = dict(maxCorners=500, qualityLevel=0.005, minDistance=5, blockSize=7)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

#Folder to use
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

# --- Visual Odometry Pipeline Class (Unchanged from original) ---
class VisualOdometryPipeline:
    def __init__(self, K_matrix, feature_params, lk_params, fb_thresh):
        self.K_matrix = K_matrix
        self.feature_params = feature_params
        self.lk_params = lk_params
        self.fb_thresh = fb_thresh
        self.next_feature_id = 0
        self.reset()

    def reset(self):
        self.R_global_pose = np.eye(3)
        self.t_global_pos = np.zeros((3, 1))
        self.trajectory_vo_points = [self.t_global_pos.flatten().copy()]
        self.prev_gray = None
        self.prev_features_with_ids = []
        self.is_first_frame = True
        self.camera_poses_ba = []
        self.map_points_ba = {}
        self.point_observations_ba = []
        self.last_successful_pose_R = np.eye(3)
        self.last_successful_pose_t = np.zeros((3,1))
        print("VO Pipeline Reset.")

    def _get_feature_points_array(self, features_with_ids_list):
        if not features_with_ids_list:
            return np.array([], dtype=np.float32).reshape(0,1,2)
        return np.array([f['pt'] for f in features_with_ids_list], dtype=np.float32)

    def _assign_new_feature_ids(self, points_array):
        features_with_ids = []
        for pt in points_array:
            features_with_ids.append({'id': self.next_feature_id, 'pt': pt.reshape(1,2)})
            self.next_feature_id += 1
        return features_with_ids

    def _process_features_internal(self, current_gray, frame_idx_str=""):
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
            temp_combined_list.extend([{'id': fid, 'pt': pt_arr.reshape(1,2)}
                                       for fid, pt_arr in zip(corresponding_ids_for_vo, good_new_for_vo_pts)])
        if newly_detected_with_ids:
            temp_combined_list.extend(newly_detected_with_ids)

        final_features_for_next_iter = []
        if temp_combined_list:
            unique_pts_dict = {tuple(feat_dict['pt'].ravel()): feat_dict for feat_dict in reversed(temp_combined_list)}
            final_features_for_next_iter = list(unique_pts_dict.values())
            if len(final_features_for_next_iter) > self.feature_params['maxCorners']:
                np.random.shuffle(final_features_for_next_iter)
                final_features_for_next_iter = final_features_for_next_iter[:self.feature_params['maxCorners']]

        return good_old_for_vo_pts, good_new_for_vo_pts, corresponding_ids_for_vo, final_features_for_next_iter

    def _estimate_relative_pose_internal(self, points_prev, points_curr):
        if points_prev is None or points_curr is None or len(points_prev) < 8 or len(points_curr) < 8: return None, None, None
        E, mask_e = cv2.findEssentialMat(points_curr, points_prev, self.K_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or np.sum(mask_e) < 5: return None, None, None

        inlier_mask = mask_e.flatten().astype(bool)
        inlier_new, inlier_old = points_curr[inlier_mask], points_prev[inlier_mask]
        retval, R_mat, t_vec, _ = cv2.recoverPose(E, inlier_new, inlier_old, self.K_matrix)

        if retval > 0 and R_mat is not None and t_vec is not None:
            return R_mat.T, -R_mat.T @ t_vec, inlier_mask
        return None, None, None

    def _triangulate_and_update_map(self, pose_id_prev, pose_id_curr,
                                   old_pts_arr, new_pts_arr, feature_ids_arr, inlier_mask_from_essential):
        if old_pts_arr is None or new_pts_arr is None or feature_ids_arr is None or not np.any(inlier_mask_from_essential): return

        old_pts_inliers, new_pts_inliers = old_pts_arr[inlier_mask_from_essential], new_pts_arr[inlier_mask_from_essential]
        feature_ids_inliers = [fid for i, fid in enumerate(feature_ids_arr) if inlier_mask_from_essential[i]]

        P1 = self.K_matrix @ np.hstack((self.last_successful_pose_R.T, -self.last_successful_pose_R.T @ self.last_successful_pose_t))
        P2 = self.K_matrix @ np.hstack((self.R_global_pose.T, -self.R_global_pose.T @ self.t_global_pos))

        pts4D_h = cv2.triangulatePoints(P1, P2, old_pts_inliers.T.reshape(2,-1), new_pts_inliers.T.reshape(2,-1))
        pts3D = pts4D_h[:3, :] / pts4D_h[3, :]

        for i, feature_id in enumerate(feature_ids_inliers):
            pt3d_world = pts3D[:, i]
            if feature_id not in self.map_points_ba:
                self.map_points_ba[feature_id] = {'pt3d': pt3d_world, 'first_observed_pose_id': pose_id_prev}
                self.point_observations_ba.append((feature_id, pose_id_prev, old_pts_inliers[i,0,0], old_pts_inliers[i,0,1]))
            self.point_observations_ba.append((feature_id, pose_id_curr, new_pts_inliers[i,0,0], new_pts_inliers[i,0,1]))


    def process_frame(self, current_gray_frame, frame_idx):
        pose_updated, vis_old_pts_arr, vis_new_pts_arr = False, None, None

        if self.is_first_frame:
            detected_features = cv2.goodFeaturesToTrack(current_gray_frame, mask=None, **self.feature_params)
            if detected_features is None or len(detected_features) < 10:
                print(f"Frame {frame_idx}: Not enough initial features. VO cannot start.")
                self.prev_gray = current_gray_frame.copy(); return pose_updated, None, None

            self.prev_features_with_ids = self._assign_new_feature_ids(detected_features)
            self.is_first_frame = False
            vis_new_pts_arr = self._get_feature_points_array(self.prev_features_with_ids) # For visualization
            self.camera_poses_ba.append({'pose_id': frame_idx, 'R': self.R_global_pose.copy(), 't': self.t_global_pos.copy()})
            self.last_successful_pose_R, self.last_successful_pose_t = self.R_global_pose.copy(), self.t_global_pos.copy()
        else:
            vis_old_pts_arr, vis_new_pts_arr, corresponding_ids, features_for_next = \
                self._process_features_internal(current_gray_frame, frame_idx_str=str(frame_idx))
            self.prev_features_with_ids = features_for_next

            if vis_new_pts_arr is not None and vis_old_pts_arr is not None:
                R_relative, t_relative, essential_inlier_mask = self._estimate_relative_pose_internal(vis_old_pts_arr, vis_new_pts_arr)
                if R_relative is not None and t_relative is not None:
                    self.t_global_pos = self.t_global_pos + self.R_global_pose @ t_relative
                    self.R_global_pose = self.R_global_pose @ R_relative
                    self.trajectory_vo_points.append(self.t_global_pos.flatten().copy())
                    pose_updated = True

                    current_pose_id = frame_idx
                    self.camera_poses_ba.append({'pose_id': current_pose_id, 'R': self.R_global_pose.copy(), 't': self.t_global_pos.copy()})
                    if essential_inlier_mask is not None and corresponding_ids is not None: # Ensure data for triangulation exists
                        self._triangulate_and_update_map(
                            self.camera_poses_ba[-2]['pose_id'], current_pose_id,
                            vis_old_pts_arr, vis_new_pts_arr,
                            corresponding_ids, essential_inlier_mask
                        )
                    self.last_successful_pose_R, self.last_successful_pose_t = self.R_global_pose.copy(), self.t_global_pos.copy()


        self.prev_gray = current_gray_frame.copy()
        return pose_updated, vis_old_pts_arr, vis_new_pts_arr

    def get_ba_data(self):
        return {"camera_poses": self.camera_poses_ba, "map_points_3d_dict": self.map_points_ba, "point_observations": self.point_observations_ba}


# --- Main Execution Logic (Modified to use Visualizer class) ---
def run_visual_odometry_main(K_matrix, dist_coeffs_val, max_frames, vo_track_refresh_interval, output_pose_file, fb_error_thresh):
    vo_pipeline = VisualOdometryPipeline(K_matrix, FEATURE_PARAMS, LK_PARAMS, fb_error_thresh)
    image_paths = load_tartanair_image_paths()
    if not image_paths: return

    if max_frames is not None and max_frames > 0 : image_paths = image_paths[:max_frames]
    print(f"Processing a maximum of {len(image_paths)} frames.")

    # Initialize the Visualizer
    traj_img_width, traj_img_height = 400, 600
    visualizer = Visualizer(traj_img_width=traj_img_width,
                            traj_img_height=traj_img_height,
                            track_refresh_interval=vo_track_refresh_interval)


    pose_file_writer = None
    if output_pose_file:
        try:
            pose_file_writer = open(output_pose_file, 'w')
            pose_file_writer.write("# timestamp tx ty tz qx qy qz qw\n")
            print(f"Opened pose output file: {output_pose_file}")
        except IOError as e: print(f"Error opening pose output file '{output_pose_file}': {e}"); pose_file_writer = None

    for frame_idx in range(len(image_paths)):
        current_frame_bgr = cv2.imread(image_paths[frame_idx])
        if current_frame_bgr is None:
            print(f"Warning: Could not read frame {image_paths[frame_idx]}"); continue

        current_frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)

        pose_updated, vis_old_pts, vis_new_pts = vo_pipeline.process_frame(current_frame_gray, frame_idx)

        # Use Visualizer to draw feature tracks
        # The Visualizer class now handles its internal track mask and refresh logic
        _ = visualizer.draw_feature_tracks(current_frame_bgr, vis_old_pts, vis_new_pts)

        # Write pose to file if updated
        if pose_file_writer and (pose_updated or (frame_idx == 0 and vo_pipeline.camera_poses_ba)):
            latest_pose_ba = vo_pipeline.camera_poses_ba[-1]
            if latest_pose_ba['pose_id'] == frame_idx: # Ensure we are writing the pose for the current frame
                timestamp = frame_idx * 0.1 # Example timestamp
                current_quat = R.from_matrix(latest_pose_ba['R']).as_quat()
                t_curr = latest_pose_ba['t']
                pose_file_writer.write(f"{timestamp:.6f} {t_curr[0,0]:.6f} {Y_FACTOR*t_curr[1,0]:.6f} {t_curr[2,0]:.6f} "
                                    f"{current_quat[0]:.6f} {current_quat[1]:.6f} {current_quat[2]:.6f} {current_quat[3]:.6f}\n")

        # Dynamic scaling for trajectory visualization
        dynamic_scale = 10.0
        if len(vo_pipeline.trajectory_vo_points) > 1:
            coords = np.array(vo_pipeline.trajectory_vo_points)
            if coords.ndim == 2 and coords.shape[1] >=3: # Ensure it's a list of [x,y,z] or similar
                max_val = np.max(np.abs(coords[:, [0,2]])) # Use X and Z for scale
                current_max_range = max(max_val, 1e-5)
                dynamic_scale = (min(traj_img_width, traj_img_height) / 2.5) / current_max_range if current_max_range > 0 else 10.0
                dynamic_scale = max(1.0, min(dynamic_scale, 200.0)) # Clamp scale

        # Use Visualizer to draw trajectory
        _ = visualizer.draw_trajectory_map(vo_pipeline.trajectory_vo_points, scale=dynamic_scale)

        key = cv2.waitKey(30) & 0xff
        if key == 27: print("ESC pressed, stopping."); break
        if key == ord('r'):
            print("User requested VO reset ('r' key).")
            vo_pipeline.reset()
            visualizer.reset_tracks_visualization() # Reset visualizer's track mask
            # Log reset pose to file
            if pose_file_writer:
                reset_timestamp = (frame_idx + 0.05) * 0.1 # Slightly offset timestamp for reset event
                # Ensure there's a pose logged for this frame_idx if VO was reset
                if not vo_pipeline.camera_poses_ba or vo_pipeline.camera_poses_ba[-1]['pose_id'] != frame_idx :
                     vo_pipeline.camera_poses_ba.append({'pose_id': frame_idx, 'R': vo_pipeline.R_global_pose.copy(), 't': vo_pipeline.t_global_pos.copy()})

                initial_quat = R.from_matrix(vo_pipeline.R_global_pose).as_quat()
                t_reset = vo_pipeline.t_global_pos
                pose_file_writer.write(f"{reset_timestamp:.6f} {t_reset[0,0]:.6f} {Y_FACTOR*t_reset[1,0]:.6f} {t_reset[2,0]:.6f} "
                                   f"{initial_quat[0]:.6f} {initial_quat[1]:.6f} {initial_quat[2]:.6f} {initial_quat[3]:.6f}\n")


    if pose_file_writer: pose_file_writer.close(); print(f"Pose data saved to {output_pose_file}")
    cv2.destroyAllWindows()
    print("Visual Odometry finished.")
    if vo_pipeline.trajectory_vo_points: print("Final VO camera position (X,Y,Z):", vo_pipeline.trajectory_vo_points[-1])

    ba_data = vo_pipeline.get_ba_data()
    print(f"\n--- Data for Bundle Adjustment ---")
    print(f"Number of camera poses: {len(ba_data['camera_poses'])}")
    print(f"Number of 3D map points: {len(ba_data['map_points_3d_dict'])}")
    print(f"Number of 2D observations: {len(ba_data['point_observations'])}")

# --- Main Execution ---
if __name__ == '__main__':
    camera_matrix_K_main, dist_coeffs_main = load_camera_intrinsics()
    if camera_matrix_K_main is None: exit(1)

    output_file_path_main = Path(OUTPUT_POSE_FILE_NAME) if OUTPUT_POSE_FILE_NAME else None
    print(f"Starting Visual Odometry for images in: {IMAGE_FOLDER_TO_USE}")
    print(f"FB Error Threshold: {FB_ERROR_THRESHOLD}")
    if output_file_path_main: print(f"Outputting poses to: {output_file_path_main.resolve()}")

    run_visual_odometry_main(
        K_matrix=camera_matrix_K_main, dist_coeffs_val=dist_coeffs_main,
        max_frames=MAX_FRAMES_TO_PROCESS,
        vo_track_refresh_interval=TRACK_REFRESH_INTERVAL, # Pass the constant
        output_pose_file=str(output_file_path_main) if output_file_path_main else None,
        fb_error_thresh=FB_ERROR_THRESHOLD
    )