import cv2
import numpy as np
import os
from pathlib import Path
import glob
from scipy.spatial.transform import Rotation as R 

# --- Configuration: EDIT THESE VALUES ---
DATASET_DIR = "/run/media/alexander/T5 EVO/datasets" 
CAMERA_VIEW = "left"
MAX_FRAMES_TO_PROCESS = None 
TRACK_REFRESH_INTERVAL = 50 
INTRINSICS = (320.0, 320.0, 320.0, 240.0) 
OUTPUT_POSE_FILE_NAME = f"stamped_traj_estimate.txt" # Changed suffix
Y_FACTOR = 1

# Threshold for forward-backward optical flow error (in pixels)
FB_ERROR_THRESHOLD = 1.0 # User's value
# --- End of Configuration ---


# --- Parameters for feature detection and tracking (More Aggressive) ---
FEATURE_PARAMS = dict(
    maxCorners=500,      # Increased: Detect more features
    qualityLevel=0.005,  # Decreased: Accept features of slightly lower quality
    minDistance=5,       # Decreased: Allow features to be closer
    blockSize=7
)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# A selection
# IMAGE_FOLDER_TO_USE = Path(DATASET_DIR) / "ocean" / "Easy" / "P000" / f"image_{CAMERA_VIEW}"
# IMAGE_FOLDER_TO_USE = Path(DATASET_DIR) / "abandonedfactory" / "Easy" / "P002" / f"image_{CAMERA_VIEW}"
IMAGE_FOLDER_TO_USE = Path(DATASET_DIR) / "seasonsforest" / "Easy" / "P001" / f"image_{CAMERA_VIEW}"

# --- Helper Functions ---

def load_camera_intrinsics():
    fx, fy, cx, cy = INTRINSICS
    K = np.array([[fx, 0,  cx], [0,  fy, cy], [0,  0,  1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32) 
    print(f"Using hardcoded intrinsics (fx, fy, cx, cy): {fx}, {fy}, {cx}, {cy}")
    return K, dist

def load_tartanair_image_paths(): # Uses global IMAGE_FOLDER_TO_USE
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
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            track_mask = cv2.line(track_mask, (a, b), (c, d), track_color, 2)
            vis_frame = cv2.circle(vis_frame, (a, b), 3, track_color, -1)
    img_with_tracks = cv2.add(vis_frame, track_mask)
    return img_with_tracks, track_mask

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

# --- Main Visual Odometry Logic ---

def run_visual_odometry(K_matrix=None, dist_coeffs_val=None, # Removed dataset_root_dir etc. as IMAGE_FOLDER_TO_USE is global
                        max_frames=None, track_refresh_interval=None, 
                        output_pose_file=None, fb_error_thresh=1.0):
    if K_matrix is None:
        print("Error: Camera intrinsic matrix K is not available.")
        return

    image_paths = load_tartanair_image_paths() # Now uses global IMAGE_FOLDER_TO_USE
    if not image_paths: return

    if max_frames is not None and max_frames > 0 :
        image_paths = image_paths[:max_frames]
        print(f"Processing a maximum of {len(image_paths)} frames.")

    R_global_pose, t_global_pos = np.eye(3), np.zeros((3, 1))
    trajectory_3d_points = [t_global_pos.flatten().copy()]
    
    vo_prev_gray, vo_p0_prev_features = None, None
    track_visualization_mask = None
    traj_img_width, traj_img_height = 400, 600
    cv2.namedWindow('Feature Tracks - VO', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Trajectory', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trajectory', traj_img_width, traj_img_height)

    frame_count_since_last_refresh, first_frame = 0, True
    
    pose_file_writer = None
    if output_pose_file:
        try:
            pose_file_writer = open(output_pose_file, 'w')
            pose_file_writer.write("# timestamp tx ty tz qx qy qz qw\n")
            initial_quat = R.from_matrix(R_global_pose).as_quat() 
            pose_file_writer.write(f"0.000000 {t_global_pos[0,0]:.6f} {Y_FACTOR*t_global_pos[1,0]:.6f} {t_global_pos[2,0]:.6f} "
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
        current_frame_gray_undistorted = current_frame_gray 

        good_new_points, good_old_points = None, None 

        if first_frame:
            vo_p0_prev_features = cv2.goodFeaturesToTrack(current_frame_gray_undistorted, mask=None, **FEATURE_PARAMS)
            if vo_p0_prev_features is None or len(vo_p0_prev_features) < 10: 
                print(f"Frame {frame_idx}: Not enough initial features. Skipping.")
                cv2.imshow('Feature Tracks - VO', current_frame_bgr) 
                if (cv2.waitKey(30) & 0xff) == 27: break
                continue
            vo_prev_gray = current_frame_gray_undistorted.copy()
            first_frame = False
            vis_img_tracks = current_frame_bgr.copy() 
            if vo_p0_prev_features is not None:
                for pt in vo_p0_prev_features: cv2.circle(vis_img_tracks, tuple(pt.ravel().astype(int)), 3, (0,0,255), -1)
        else: 
            tracked_features_current = None
            tracked_features_prev = None
            if vo_p0_prev_features is not None and len(vo_p0_prev_features) >= 5: # Need some points to track
                p1_current_features, st_fwd, _ = cv2.calcOpticalFlowPyrLK(vo_prev_gray, current_frame_gray_undistorted, vo_p0_prev_features, None, **LK_PARAMS)
                fwd_success_mask = (st_fwd.flatten() == 1)
                p0_fwd_good = vo_p0_prev_features[fwd_success_mask]
                p1_fwd_good = p1_current_features[fwd_success_mask]

                if len(p1_fwd_good) > 0:
                    p0_reprojected, st_bwd, _ = cv2.calcOpticalFlowPyrLK(current_frame_gray_undistorted, vo_prev_gray, p1_fwd_good, None, **LK_PARAMS)
                    bwd_success_mask = (st_bwd.flatten() == 1)
                    p0_fwd_bwd_good = p0_fwd_good[bwd_success_mask]
                    p1_fwd_bwd_good = p1_fwd_good[bwd_success_mask] 
                    p0_reprojected_final = p0_reprojected[bwd_success_mask]

                    if len(p0_reprojected_final) > 0:
                        error_fb = np.linalg.norm(p0_fwd_bwd_good.reshape(-1, 2) - p0_reprojected_final.reshape(-1, 2), axis=1)
                        consistent_mask = error_fb < fb_error_thresh
                        tracked_features_prev = p0_fwd_bwd_good[consistent_mask]
                        tracked_features_current = p1_fwd_bwd_good[consistent_mask]
                        print(f"Frame {frame_idx}: Tracked In {len(vo_p0_prev_features)} -> FwdOK {len(p1_fwd_good)} -> FwdBwdOK {len(tracked_features_current if tracked_features_current is not None else [])}")
                    else: print(f"Frame {frame_idx}: No points survived backward tracking pass.")
                else: print(f"Frame {frame_idx}: No points survived forward tracking pass.")
            
            good_old_points = tracked_features_prev
            good_new_points = tracked_features_current
            
            if track_refresh_interval and track_refresh_interval > 0:
                if frame_count_since_last_refresh >= track_refresh_interval:
                    track_visualization_mask = np.zeros_like(current_frame_bgr)
                    frame_count_since_last_refresh = 0
                else: frame_count_since_last_refresh += 1
            vis_img_tracks, track_visualization_mask = draw_tracks(current_frame_bgr, good_old_points, good_new_points, track_visualization_mask)

        pose_updated_this_frame = False
        if good_new_points is not None and good_old_points is not None and len(good_new_points) >= 8: # Min points for E matrix
            E, mask_e = cv2.findEssentialMat(good_new_points, good_old_points, K_matrix, 
                                             method=cv2.RANSAC, prob=0.999, threshold=1.0) 
            
            if E is not None and np.sum(mask_e) >= 5 : 
                inlier_new_pts = good_new_points[mask_e.flatten().astype(bool)]
                inlier_old_pts = good_old_points[mask_e.flatten().astype(bool)]
                retval, R_21, t_21, mask_rp = cv2.recoverPose(E, inlier_new_pts, inlier_old_pts, K_matrix)

                if retval > 0 and R_21 is not None and t_21 is not None and (mask_rp is None or np.sum(mask_rp) >=5) :
                    R_relative_pose, t_relative_pose = R_21.T, -R_21.T @ t_21
                    t_global_pos = t_global_pos + R_global_pose @ t_relative_pose
                    R_global_pose = R_global_pose @ R_relative_pose
                    trajectory_3d_points.append(t_global_pos.flatten().copy())
                    pose_updated_this_frame = True
                else: print(f"Frame {frame_idx}: recoverPose failed or not enough inliers.")
            else: print(f"Frame {frame_idx}: findEssentialMat failed or not enough inliers.")
        
        if 'vis_img_tracks' not in locals(): vis_img_tracks = current_frame_bgr.copy()
        cv2.imshow('Feature Tracks - VO', vis_img_tracks)
        
        if pose_updated_this_frame and pose_file_writer:
            timestamp = frame_idx * 0.1 
            current_quat = R.from_matrix(R_global_pose).as_quat()
            pose_file_writer.write(f"{timestamp:.6f} "
                                   f"{t_global_pos[0,0]:.6f} {-t_global_pos[1,0]:.6f} {t_global_pos[2,0]:.6f} "
                                   f"{current_quat[0]:.6f} {current_quat[1]:.6f} {current_quat[2]:.6f} {current_quat[3]:.6f}\n")

        vo_prev_gray = current_frame_gray_undistorted.copy()
        
        # --- Aggressive Feature Re-detection and Merging ---
        # Always detect new features in the current frame
        newly_detected_features = cv2.goodFeaturesToTrack(current_frame_gray_undistorted, mask=None, **FEATURE_PARAMS)
        
        current_valid_tracked_points = good_new_points # Points that were successfully tracked and passed FB
        
        if newly_detected_features is None: newly_detected_features = np.array([]) # Ensure it's an array
        if current_valid_tracked_points is None: current_valid_tracked_points = np.array([]) # Ensure it's an array

        # Ensure both are 3D (N, 1, 2) or empty before vstack
        if newly_detected_features.ndim == 2: newly_detected_features = newly_detected_features.reshape(-1,1,2)
        if current_valid_tracked_points.ndim == 2 : current_valid_tracked_points = current_valid_tracked_points.reshape(-1,1,2)

        # Combine tracked points with newly detected points
        if len(current_valid_tracked_points) > 0 and len(newly_detected_features) > 0:
            combined_features = np.vstack((current_valid_tracked_points, newly_detected_features))
        elif len(current_valid_tracked_points) > 0:
            combined_features = current_valid_tracked_points
        elif len(newly_detected_features) > 0:
            combined_features = newly_detected_features
        else:
            combined_features = np.array([]) # No features at all

        if len(combined_features) > 0:
            # Remove duplicates (reshape to 2D for unique, then back to 3D)
            unique_points_2d = np.unique(combined_features.reshape(-1, 2), axis=0)
            vo_p0_prev_features = unique_points_2d.reshape(-1, 1, 2).astype(np.float32)
            
            # Shuffle and cap at maxCorners
            if len(vo_p0_prev_features) > FEATURE_PARAMS['maxCorners']:
                np.random.shuffle(vo_p0_prev_features) # Shuffle in place
                vo_p0_prev_features = vo_p0_prev_features[:FEATURE_PARAMS['maxCorners']]
            print(f"Frame {frame_idx}: Features for next iter: {len(vo_p0_prev_features)} (after merge & cap)")
        else:
            vo_p0_prev_features = None # No features to carry to next iteration
            print(f"Frame {frame_idx}: No features for next iteration. Will attempt full re-detection.")
            first_frame = True # This will trigger re-initialization logic in the next iteration

        current_max_range = 0
        if len(trajectory_3d_points) > 1:
            coords = np.array(trajectory_3d_points)
            max_x, max_z = np.max(np.abs(coords[:, 0])), np.max(np.abs(coords[:, 2]))
            current_max_range = max(max_x, max_z, 1e-5) 
        dynamic_scale = (min(traj_img_width, traj_img_height) / 2.5) / current_max_range if current_max_range > 0 else 10.0
        dynamic_scale = max(1.0, min(dynamic_scale, 200.0)) 
        traj_display_img = draw_trajectory(trajectory_3d_points, traj_img_width, traj_img_height, scale=dynamic_scale)
        cv2.imshow('Trajectory', traj_display_img)
        
        key = cv2.waitKey(30) & 0xff
        if key == 27: print("ESC pressed, stopping."); break
        if key == ord('r'): 
            print("User requested VO reset ('r' key).")
            R_global_pose, t_global_pos = np.eye(3), np.zeros((3, 1))
            trajectory_3d_points = [t_global_pos.flatten().copy()]
            first_frame, vo_p0_prev_features = True, None
            if track_visualization_mask is not None: track_visualization_mask.fill(0)
            frame_count_since_last_refresh = 0
            print("VO reset.")
            if pose_file_writer: 
                reset_timestamp = (frame_idx + 0.05) * 0.1 
                initial_quat = R.from_matrix(R_global_pose).as_quat()
                pose_file_writer.write(f"{reset_timestamp:.6f} {Y_FACTOR*t_global_pos[0,0]:.6f} {t_global_pos[1,0]:.6f} {t_global_pos[2,0]:.6f} "
                                   f"{initial_quat[0]:.6f} {initial_quat[1]:.6f} {initial_quat[2]:.6f} {initial_quat[3]:.6f}\n")

    if pose_file_writer: pose_file_writer.close(); print(f"Pose data saved to {output_pose_file}")
    cv2.destroyAllWindows()
    print("Visual Odometry finished.")
    if trajectory_3d_points: print("Final camera position (X,Y,Z):", trajectory_3d_points[-1])

# --- Main Execution ---
if __name__ == '__main__':
    camera_matrix_K_main, dist_coeffs_main = load_camera_intrinsics() 

    if camera_matrix_K_main is None: exit(1)
    # No need to validate DATASET_DIR, SEQUENCE_NAME, CAMERA_VIEW here as IMAGE_FOLDER_TO_USE is now global
    
    output_file_path_main = Path(OUTPUT_POSE_FILE_NAME) if OUTPUT_POSE_FILE_NAME else None

    print(f"Starting Visual Odometry for images in: {IMAGE_FOLDER_TO_USE}") # Changed log
    print(f"FB Error Threshold: {FB_ERROR_THRESHOLD}")
    if output_file_path_main: print(f"Outputting poses to: {output_file_path_main.resolve()}")
    
    run_visual_odometry( # Removed unused args
        K_matrix=camera_matrix_K_main, 
        dist_coeffs_val=dist_coeffs_main,
        max_frames=MAX_FRAMES_TO_PROCESS, 
        track_refresh_interval=TRACK_REFRESH_INTERVAL,
        output_pose_file=str(output_file_path_main) if output_file_path_main else None,
        fb_error_thresh=FB_ERROR_THRESHOLD
    )

