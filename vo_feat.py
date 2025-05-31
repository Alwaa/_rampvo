import cv2
import numpy as np
import os # Retained for Path, though os direct functions might not be used
from pathlib import Path
import glob
from scipy.spatial.transform import Rotation as R 

# --- Configuration: EDIT THESE VALUES ---
DATASET_DIR = "/run/media/alexander/T5 EVO/datasets" 
CAMERA_VIEW = "left" 
MAX_FRAMES_TO_PROCESS = None 
TRACK_REFRESH_INTERVAL = 50 
INTRINSICS = (320.0, 320.0, 320.0, 240.0) 
OUTPUT_POSE_FILE_NAME = "stamped_traj_estimate.txt"
Y_FACTOR = 1 

# Threshold for forward-backward optical flow error (in pixels)
FB_ERROR_THRESHOLD = 1.0
# --- End of Configuration ---


# --- Parameters for feature detection and tracking (More Aggressive) ---
FEATURE_PARAMS = dict(
    maxCorners=500,      
    qualityLevel=0.005,  
    minDistance=5,       
    blockSize=7
)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Set the image folder to use. 
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
        print(f"Error: IMAGE_FOLDER_TO_USE does not exist or is not a directory: {IMAGE_FOLDER_TO_USE}")
        return []
    print(f"Loading images from: {IMAGE_FOLDER_TO_USE}")
    image_files = sorted(glob.glob(str(IMAGE_FOLDER_TO_USE / "*.png")))
    if not image_files: print(f"Warning: No PNG images found in {IMAGE_FOLDER_TO_USE}")
    return image_files

def draw_tracks(display_frame, prev_points, current_points, track_mask, track_color=(0, 255, 0)):
    vis_frame = display_frame.copy() 
    if prev_points is not None and current_points is not None and len(prev_points) == len(current_points):
        for _, (new, old) in enumerate(zip(current_points, prev_points)):
            a, b = new.ravel().astype(int); c, d = old.ravel().astype(int)
            # cv2.line modifies track_mask in place if it's a numpy array.
            # It returns the image it drew on (which is track_mask itself).
            track_mask = cv2.line(track_mask, (a, b), (c, d), track_color, 2)
            vis_frame = cv2.circle(vis_frame, (a, b), 3, track_color, -1)
    # The cv2.add operation combines the visualization with the original frame copy.
    return cv2.add(vis_frame, track_mask), track_mask

def draw_trajectory(trajectory_points_list, traj_img_width, traj_img_height, scale=10):
    traj_img = np.zeros((traj_img_height, traj_img_width, 3), dtype=np.uint8)
    center_x, center_y = traj_img_width // 2, traj_img_height // 2
    if not trajectory_points_list: return traj_img
    screen_points = [ (int(center_x + pt3d[0] * scale), int(center_y + pt3d[2] * scale)) for pt3d in trajectory_points_list]
    for i in range(len(screen_points) - 1): cv2.line(traj_img, screen_points[i], screen_points[i+1], (0, 255, 0), 2)
    if screen_points: 
        cv2.circle(traj_img, screen_points[-1], 5, (0, 0, 255), -1) 
        cv2.circle(traj_img, screen_points[0], 5, (255,0,0), -1)
    cv2.putText(traj_img, f"Scale: {scale:.1f} pix/m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return traj_img

# --- Visual Odometry Pipeline Class ---
class VisualOdometryPipeline:
    def __init__(self, K_matrix, feature_params, lk_params, fb_thresh):
        self.K_matrix = K_matrix
        self.feature_params = feature_params
        self.lk_params = lk_params
        self.fb_thresh = fb_thresh
        self.reset()

    def reset(self):
        self.R_global_pose = np.eye(3)
        self.t_global_pos = np.zeros((3, 1))
        self.trajectory_3d_points = [self.t_global_pos.flatten().copy()]
        self.prev_gray = None
        self.prev_features = None 
        self.is_first_frame = True
        print("VO Pipeline Reset.")

    def _process_features_internal(self, current_gray, frame_idx_str=""): # Takes current_gray, uses self.prev_gray, self.prev_features
        good_old_for_vo, good_new_for_vo = None, None
        
        if self.prev_features is not None and len(self.prev_features) >= 5:
            p1_fwd, st_fwd, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, self.prev_features, None, **self.lk_params)
            fwd_success_mask = (st_fwd.flatten() == 1)
            p0_fwd_good, p1_fwd_good = self.prev_features[fwd_success_mask], p1_fwd[fwd_success_mask]

            if len(p1_fwd_good) > 0:
                p0_reprojected, st_bwd, _ = cv2.calcOpticalFlowPyrLK(current_gray, self.prev_gray, p1_fwd_good, None, **self.lk_params)
                bwd_success_mask = (st_bwd.flatten() == 1)
                p0_fwd_bwd_good, p1_fwd_bwd_good = p0_fwd_good[bwd_success_mask], p1_fwd_good[bwd_success_mask]
                p0_reprojected_final = p0_reprojected[bwd_success_mask]

                if len(p0_reprojected_final) > 0:
                    error_fb = np.linalg.norm(p0_fwd_bwd_good.reshape(-1, 2) - p0_reprojected_final.reshape(-1, 2), axis=1)
                    consistent_mask = error_fb < self.fb_thresh
                    good_old_for_vo = p0_fwd_bwd_good[consistent_mask]
                    good_new_for_vo = p1_fwd_bwd_good[consistent_mask]
                    # print(f"Frame {frame_idx_str}: Tracked In {len(self.prev_features)} -> FwdOK {len(p1_fwd_good)} -> FwdBwdOK {len(good_new_for_vo if good_new_for_vo is not None else [])}")
        
        newly_detected_features = cv2.goodFeaturesToTrack(current_gray, mask=None, **self.feature_params)
        
        # Prepare features for the next iteration
        # Start with successfully tracked points in the current frame (good_new_for_vo)
        # These become the "previous" points for the next frame.
        current_valid_for_next = good_new_for_vo if good_new_for_vo is not None else np.array([])
        newly_detected = newly_detected_features if newly_detected_features is not None else np.array([])

        # Ensure correct shape for vstack
        if newly_detected.ndim == 2 and newly_detected.size > 0: newly_detected = newly_detected.reshape(-1,1,2)
        if current_valid_for_next.ndim == 2 and current_valid_for_next.size > 0: current_valid_for_next = current_valid_for_next.reshape(-1,1,2)

        # Combine and ensure uniqueness
        combined_list = [arr for arr in (current_valid_for_next, newly_detected) if len(arr) > 0]
        features_for_next = None
        if combined_list:
            combined_arr = np.vstack(combined_list)
            if len(combined_arr) > 0:
                unique_2d = np.unique(combined_arr.reshape(-1, 2), axis=0)
                features_for_next = unique_2d.reshape(-1, 1, 2).astype(np.float32)
                if len(features_for_next) > self.feature_params['maxCorners']:
                    np.random.shuffle(features_for_next)
                    features_for_next = features_for_next[:self.feature_params['maxCorners']]
        
        # Fallback if no features are generated
        if features_for_next is None or len(features_for_next) == 0:
            features_for_next = cv2.goodFeaturesToTrack(current_gray, mask=None, **self.feature_params)
            if features_for_next is None: features_for_next = np.array([]) # Ensure it's an array
        
        # print(f"Frame {frame_idx_str}: Features for next iter: {len(features_for_next)}")
        return good_old_for_vo, good_new_for_vo, features_for_next

    def _estimate_relative_pose_internal(self, points_prev, points_curr):
        if points_prev is None or points_curr is None or len(points_prev) < 8 or len(points_curr) < 8:
            return None, None
        E, mask_e = cv2.findEssentialMat(points_curr, points_prev, self.K_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or np.sum(mask_e) < 5: return None, None
        
        inlier_new = points_curr[mask_e.flatten().astype(bool)]
        inlier_old = points_prev[mask_e.flatten().astype(bool)]
        retval, R_mat, t_vec, _ = cv2.recoverPose(E, inlier_new, inlier_old, self.K_matrix)

        if retval > 0 and R_mat is not None and t_vec is not None:
            return R_mat.T, -R_mat.T @ t_vec 
        return None, None

    def process_frame(self, current_gray_frame, frame_idx_str=""):
        pose_updated = False
        vis_old_pts, vis_new_pts = None, None 

        if self.is_first_frame:
            self.prev_features = cv2.goodFeaturesToTrack(current_gray_frame, mask=None, **self.feature_params)
            if self.prev_features is None or len(self.prev_features) < 10:
                print(f"Frame {frame_idx_str}: Not enough initial features. VO cannot start.")
                self.prev_gray = current_gray_frame.copy() 
                return pose_updated, self.prev_features, None 
            self.is_first_frame = False
            vis_new_pts = self.prev_features # For drawing initial points
        else:
            # _process_features_internal uses self.prev_gray and self.prev_features
            vis_old_pts, vis_new_pts, features_for_next = self._process_features_internal(
                current_gray_frame, frame_idx_str=frame_idx_str 
            )
            self.prev_features = features_for_next # Update features for the *next* frame's *previous*

            if vis_new_pts is not None and vis_old_pts is not None:
                R_relative, t_relative = self._estimate_relative_pose_internal(vis_old_pts, vis_new_pts)
                if R_relative is not None and t_relative is not None:
                    self.t_global_pos = self.t_global_pos + self.R_global_pose @ t_relative
                    self.R_global_pose = self.R_global_pose @ R_relative
                    self.trajectory_3d_points.append(self.t_global_pos.flatten().copy())
                    pose_updated = True
        
        self.prev_gray = current_gray_frame.copy()
        return pose_updated, vis_old_pts, vis_new_pts


# --- Main Execution Logic ---
def run_visual_odometry_main(K_matrix, dist_coeffs_val, # dist_coeffs_val currently unused by VO class
                             max_frames, track_refresh_interval, 
                             output_pose_file, fb_error_thresh):
    
    vo_pipeline = VisualOdometryPipeline(K_matrix, FEATURE_PARAMS, LK_PARAMS, fb_error_thresh)
    image_paths = load_tartanair_image_paths()
    if not image_paths: return

    if max_frames is not None and max_frames > 0 : image_paths = image_paths[:max_frames]
    print(f"Processing a maximum of {len(image_paths)} frames.")

    track_visualization_mask = None
    traj_img_width, traj_img_height = 400, 600
    cv2.namedWindow('Feature Tracks - VO', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Trajectory', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trajectory', traj_img_width, traj_img_height)
    frame_count_since_last_refresh = 0
    
    pose_file_writer = None
    if output_pose_file:
        try:
            pose_file_writer = open(output_pose_file, 'w')
            pose_file_writer.write("# timestamp tx ty tz qx qy qz qw\n")
            initial_quat = R.from_matrix(vo_pipeline.R_global_pose).as_quat() 
            t_init = vo_pipeline.t_global_pos
            pose_file_writer.write(f"0.000000 {t_init[0,0]:.6f} {Y_FACTOR*t_init[1,0]:.6f} {t_init[2,0]:.6f} "
                                   f"{initial_quat[0]:.6f} {initial_quat[1]:.6f} {initial_quat[2]:.6f} {initial_quat[3]:.6f}\n")
            print(f"Opened pose output file: {output_pose_file}")
        except IOError as e:
            print(f"Error opening pose output file '{output_pose_file}': {e}")
            pose_file_writer = None

    for frame_idx in range(len(image_paths)):
        current_frame_bgr = cv2.imread(image_paths[frame_idx])
        if current_frame_bgr is None: continue
        
        if track_visualization_mask is None: track_visualization_mask = np.zeros_like(current_frame_bgr)
        current_frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)

        pose_updated, vis_old_pts, vis_new_pts = vo_pipeline.process_frame(current_frame_gray, frame_idx_str=str(frame_idx))

        if track_refresh_interval and track_refresh_interval > 0 and frame_count_since_last_refresh >= track_refresh_interval:
            track_visualization_mask.fill(0)
            frame_count_since_last_refresh = 0
        else: frame_count_since_last_refresh += 1
        
        vis_img_tracks, track_visualization_mask = draw_tracks(current_frame_bgr, vis_old_pts, vis_new_pts, track_visualization_mask)
        cv2.imshow('Feature Tracks - VO', vis_img_tracks)

        if pose_updated and pose_file_writer:
            timestamp = frame_idx * 0.1 
            current_quat = R.from_matrix(vo_pipeline.R_global_pose).as_quat()
            t_curr = vo_pipeline.t_global_pos
            pose_file_writer.write(f"{timestamp:.6f} "
                                   f"{t_curr[0,0]:.6f} {Y_FACTOR*t_curr[1,0]:.6f} {t_curr[2,0]:.6f} " # Y_FACTOR used here
                                   f"{current_quat[0]:.6f} {current_quat[1]:.6f} {current_quat[2]:.6f} {current_quat[3]:.6f}\n")

        dynamic_scale = 10.0
        if len(vo_pipeline.trajectory_3d_points) > 1:
            coords = np.array(vo_pipeline.trajectory_3d_points)
            # Consider X and Z for scale, avoid error on empty coords
            max_val = np.max(np.abs(coords[:, [0,2]])) if coords.size > 0 and coords.shape[1] >=3 else 1.0 
            current_max_range = max(max_val, 1e-5)
            dynamic_scale = (min(traj_img_width, traj_img_height) / 2.5) / current_max_range if current_max_range > 0 else 10.0
            dynamic_scale = max(1.0, min(dynamic_scale, 200.0))
        
        traj_display_img = draw_trajectory(vo_pipeline.trajectory_3d_points, traj_img_width, traj_img_height, scale=dynamic_scale)
        cv2.imshow('Trajectory', traj_display_img)
        
        key = cv2.waitKey(30) & 0xff
        if key == 27: print("ESC pressed, stopping."); break
        if key == ord('r'): 
            print("User requested VO reset ('r' key).")
            vo_pipeline.reset() 
            if track_visualization_mask is not None: track_visualization_mask.fill(0)
            frame_count_since_last_refresh = 0
            if pose_file_writer: 
                reset_timestamp = (frame_idx + 0.05) * 0.1 
                initial_quat = R.from_matrix(vo_pipeline.R_global_pose).as_quat()
                t_reset = vo_pipeline.t_global_pos
                pose_file_writer.write(f"{reset_timestamp:.6f} {t_reset[0,0]:.6f} {Y_FACTOR*t_reset[1,0]:.6f} {t_reset[2,0]:.6f} " # Y_FACTOR
                                   f"{initial_quat[0]:.6f} {initial_quat[1]:.6f} {initial_quat[2]:.6f} {initial_quat[3]:.6f}\n")

    if pose_file_writer: pose_file_writer.close(); print(f"Pose data saved to {output_pose_file}")
    cv2.destroyAllWindows()
    print("Visual Odometry finished.")
    if vo_pipeline.trajectory_3d_points: print("Final camera position (X,Y,Z):", vo_pipeline.trajectory_3d_points[-1])

# --- Main Execution ---
if __name__ == '__main__':
    camera_matrix_K_main, dist_coeffs_main = load_camera_intrinsics() 
    if camera_matrix_K_main is None: exit(1)
    
    # Use OUTPUT_POSE_FILE_NAME directly from config
    output_file_path_main = Path(OUTPUT_POSE_FILE_NAME) if OUTPUT_POSE_FILE_NAME else None

    print(f"Starting Visual Odometry for images in: {IMAGE_FOLDER_TO_USE}")
    print(f"FB Error Threshold: {FB_ERROR_THRESHOLD}")
    if output_file_path_main: print(f"Outputting poses to: {output_file_path_main.resolve()}")
    
    run_visual_odometry_main( 
        K_matrix=camera_matrix_K_main, 
        dist_coeffs_val=dist_coeffs_main, 
        max_frames=MAX_FRAMES_TO_PROCESS, 
        track_refresh_interval=TRACK_REFRESH_INTERVAL,
        output_pose_file=str(output_file_path_main) if output_file_path_main else None,
        fb_error_thresh=FB_ERROR_THRESHOLD
    )
