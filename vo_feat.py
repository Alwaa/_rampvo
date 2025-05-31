import cv2
import numpy as np
import os
from pathlib import Path
import glob
from scipy.spatial.transform import Rotation as R # For converting rotation matrix to quaternion

# --- Configuration: EDIT THESE VALUES ---
DATASET_DIR = "/run/media/alexander/T5 EVO/datasets" 
SEQUENCE_NAME = "ocean"
CAMERA_VIEW = "left"
MAX_FRAMES_TO_PROCESS = None # Example: 300
TRACK_REFRESH_INTERVAL = 50 # Example: clear tracks every 50 frames

# fx 0 cx
# 0 fy cy
# 0 0 1
INTRINSICS = (320.0, 320.0, 320.0, 240.0) # format: fx fy cx cy (Made them float for consistency)

# timestamp tx ty tz qx qy qz qw # Set to None to disable pose file output.
OUTPUT_POSE_FILE_NAME = f"stamped_traj_estimate.txt" 
# --- End of Configuration ---


# --- Parameters for feature detection and tracking ---
FEATURE_PARAMS = dict(maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# --- Global Variables for VO ---
# These are set in __main__ after calling load_camera_intrinsics
# camera_matrix_K = None 
# dist_coeffs = None 

# --- Helper Functions ---

def load_camera_intrinsics():
    """
    Loads camera intrinsic parameters using the global INTRINSICS tuple.
    fx 0 cx
    0 fy cy
    0 0 1
    """
    fx, fy, cx, cy = INTRINSICS
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32) # Default to no distortion
    
    print(f"Using hardcoded intrinsics (fx, fy, cx, cy): {fx}, {fy}, {cx}, {cy}")
    print(f"K matrix:\n{K}")
    return K, dist


def load_tartanair_image_paths(dataset_dir_str, sequence_name, camera='left'):
    dataset_dir_path = Path(dataset_dir_str)
    image_folder_to_use = dataset_dir_path / sequence_name / "Easy" / "P000" / f"image_{camera}"
    
    if not (image_folder_to_use.exists() and image_folder_to_use.is_dir()):
        print(f"Warning: User-specified image folder not found: {image_folder_to_use}")
        print("  Attempting fallback to common TartanAir structures...")
        image_folder_attempt_orig1 = dataset_dir_path / sequence_name / sequence_name / f"image_{camera}"
        image_folder_attempt_orig2 = dataset_dir_path / sequence_name / f"image_{camera}"
        if image_folder_attempt_orig1.exists() and image_folder_attempt_orig1.is_dir():
            image_folder_to_use = image_folder_attempt_orig1
        elif image_folder_attempt_orig2.exists() and image_folder_attempt_orig2.is_dir():
            image_folder_to_use = image_folder_attempt_orig2
        else:
            print(f"Error: Image folder not found for sequence '{sequence_name}', camera '{camera}'.")
            return []
            
    print(f"Loading images from: {image_folder_to_use}")
    image_files = sorted(glob.glob(str(image_folder_to_use / "*.png")))
    if not image_files: print(f"Warning: No PNG images found in {image_folder_to_use}")
    return image_files

def draw_tracks(display_frame, prev_points, current_points, track_mask, track_color=(0, 255, 0)):
    vis_frame = display_frame.copy() 
    for _, (new, old) in enumerate(zip(current_points, prev_points)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        track_mask = cv2.line(track_mask, (a, b), (c, d), track_color, 2)
        vis_frame = cv2.circle(vis_frame, (a, b), 3, track_color, -1) # Smaller radius for points
    img_with_tracks = cv2.add(vis_frame, track_mask)
    return img_with_tracks, track_mask

def draw_trajectory(trajectory_points_list, traj_img_width, traj_img_height, scale=10):
    """Draws the 2D trajectory (X-Z plane, assuming Z is forward, X is right)"""
    traj_img = np.zeros((traj_img_height, traj_img_width, 3), dtype=np.uint8)
    center_x, center_y = traj_img_width // 2, traj_img_height // 2

    if not trajectory_points_list:
        return traj_img

    screen_points = []
    for pt3d in trajectory_points_list:
        screen_x = int(center_x + pt3d[0] * scale)
        screen_y = int(center_y + pt3d[2] * scale) 
        screen_points.append((screen_x, screen_y))

    for i in range(len(screen_points) - 1):
        cv2.line(traj_img, screen_points[i], screen_points[i+1], (0, 255, 0), 2) # Green line for trajectory
    
    if screen_points: # Draw current position
        cv2.circle(traj_img, screen_points[-1], 5, (0, 0, 255), -1) # Red circle for current pos
        cv2.circle(traj_img, screen_points[0], 5, (255,0,0), -1) # Blue circle for start

    cv2.putText(traj_img, f"Scale: {scale:.1f} pix/m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return traj_img

# --- Main Visual Odometry Logic ---

def run_visual_odometry(dataset_root_dir, current_sequence_name, current_camera, 
                        max_frames=None, track_refresh_interval=None, K_matrix=None, dist_coeffs_val=None,
                        output_pose_file=None): # Added output_pose_file argument
    if K_matrix is None:
        print("Error: Camera intrinsic matrix K is not available. Cannot proceed with VO.")
        return

    image_paths = load_tartanair_image_paths(dataset_root_dir, current_sequence_name, current_camera)
    if not image_paths: return

    if max_frames is not None and max_frames > 0 :
        image_paths = image_paths[:max_frames]
        print(f"Processing a maximum of {len(image_paths)} frames.")

    R_global_pose = np.eye(3) 
    t_global_pos = np.zeros((3, 1)) 
    trajectory_3d_points = [t_global_pos.flatten().copy()] 
    
    vo_prev_gray = None
    vo_p0_prev_features = None 

    track_visualization_mask = None
    traj_img_width, traj_img_height = 400, 600 
    cv2.namedWindow('Feature Tracks - VO', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Trajectory', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trajectory', traj_img_width, traj_img_height)

    frame_count_since_last_refresh = 0
    first_frame = True
    
    pose_file_writer = None
    if output_pose_file:
        try:
            pose_file_writer = open(output_pose_file, 'w')
            pose_file_writer.write("# timestamp tx ty tz qx qy qz qw\n")
            initial_quat = R.from_matrix(R_global_pose).as_quat() 
            pose_file_writer.write(f"0.000000 {t_global_pos[0,0]:.6f} {t_global_pos[1,0]:.6f} {t_global_pos[2,0]:.6f} "
                                   f"{initial_quat[0]:.6f} {initial_quat[1]:.6f} {initial_quat[2]:.6f} {initial_quat[3]:.6f}\n")
            print(f"Opened pose output file: {output_pose_file}")
        except IOError as e:
            print(f"Error opening pose output file '{output_pose_file}': {e}")
            pose_file_writer = None

    for frame_idx in range(len(image_paths)):
        current_frame_bgr = cv2.imread(image_paths[frame_idx])
        if current_frame_bgr is None:
            print(f"Warning: Could not read image {image_paths[frame_idx]}. Skipping.")
            continue
        
        if track_visualization_mask is None: 
            track_visualization_mask = np.zeros_like(current_frame_bgr)

        current_frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
        current_frame_gray_undistorted = current_frame_gray # Assuming undistorted

        if first_frame:
            vo_p0_prev_features = cv2.goodFeaturesToTrack(current_frame_gray_undistorted, mask=None, **FEATURE_PARAMS)
            if vo_p0_prev_features is None or len(vo_p0_prev_features) < 10: 
                print(f"Frame {frame_idx}: Not enough initial features. Skipping.")
                cv2.imshow('Feature Tracks - VO', current_frame_bgr) 
                key = cv2.waitKey(30) & 0xff
                if key == 27: break
                continue
            vo_prev_gray = current_frame_gray_undistorted.copy()
            first_frame = False
            vis_img = current_frame_bgr.copy()
            for pt in vo_p0_prev_features:
                cv2.circle(vis_img, tuple(pt.ravel().astype(int)), 3, (0,0,255), -1)
            cv2.imshow('Feature Tracks - VO', vis_img)
        else:
            if vo_p0_prev_features is None or len(vo_p0_prev_features) < 10:
                print(f"Frame {frame_idx}: Too few features. Re-detecting.")
                vo_p0_prev_features = cv2.goodFeaturesToTrack(current_frame_gray_undistorted, mask=None, **FEATURE_PARAMS)
                if vo_p0_prev_features is None or len(vo_p0_prev_features) < 10:
                    print(f"Frame {frame_idx}: Re-detection failed. Skipping VO.")
                    vo_prev_gray = current_frame_gray_undistorted.copy() 
                    cv2.imshow('Feature Tracks - VO', current_frame_bgr)
                    key = cv2.waitKey(30) & 0xff
                    if key == 27: break
                    continue 
            
            p1_current_features, st, err = cv2.calcOpticalFlowPyrLK(vo_prev_gray, current_frame_gray_undistorted, vo_p0_prev_features, None, **LK_PARAMS)

            good_new_points = None
            good_old_points = None
            if p1_current_features is not None and st is not None:
                good_new_points = p1_current_features[st == 1]
                good_old_points = vo_p0_prev_features[st == 1]

            if track_refresh_interval and track_refresh_interval > 0:
                if frame_count_since_last_refresh >= track_refresh_interval:
                    track_visualization_mask = np.zeros_like(current_frame_bgr)
                    frame_count_since_last_refresh = 0
                else: frame_count_since_last_refresh += 1
            
            vis_img = current_frame_bgr.copy() 
            pose_updated_this_frame = False
            if good_new_points is not None and len(good_new_points) > 5: 
                E, mask_e = cv2.findEssentialMat(good_new_points, good_old_points, K_matrix, 
                                                 method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None and np.sum(mask_e) >= 5 : 
                    retval, R_21, t_21, mask_rp = cv2.recoverPose(E, good_new_points, good_old_points, K_matrix, mask=mask_e)

                    if retval > 0 and R_21 is not None and t_21 is not None and np.sum(mask_rp) >=5:
                        R_relative_pose = R_21.T 
                        t_relative_pose = -R_21.T @ t_21
                        t_global_pos = t_global_pos + R_global_pose @ t_relative_pose
                        R_global_pose = R_global_pose @ R_relative_pose
                        trajectory_3d_points.append(t_global_pos.flatten().copy())
                        pose_updated_this_frame = True
                    else:
                        print(f"Frame {frame_idx}: recoverPose failed or not enough inliers ({np.sum(mask_rp) if mask_rp is not None else 0}).")
                else:
                    print(f"Frame {frame_idx}: findEssentialMat failed or not enough inliers ({np.sum(mask_e) if mask_e is not None else 0}).")
                
                vis_img, track_visualization_mask = draw_tracks(current_frame_bgr, good_old_points, good_new_points, track_visualization_mask)
            
            cv2.imshow('Feature Tracks - VO', vis_img)
            
            if pose_updated_this_frame and pose_file_writer:
                timestamp = frame_idx * 0.1 
                current_quat = R.from_matrix(R_global_pose).as_quat()
                pose_file_writer.write(f"{timestamp:.6f} "
                                       f"{t_global_pos[0,0]:.6f} {t_global_pos[1,0]:.6f} {t_global_pos[2,0]:.6f} "
                                       f"{current_quat[0]:.6f} {current_quat[1]:.6f} {current_quat[2]:.6f} {current_quat[3]:.6f}\n")

            vo_prev_gray = current_frame_gray_undistorted.copy()
            if good_new_points is not None and len(good_new_points) > FEATURE_PARAMS['maxCorners'] * 0.25: 
                 vo_p0_prev_features = good_new_points.reshape(-1, 1, 2)
            else: 
                print(f"Frame {frame_idx}: Feature count low or tracking failed. Re-detecting.")
                vo_p0_prev_features = cv2.goodFeaturesToTrack(current_frame_gray_undistorted, mask=None, **FEATURE_PARAMS)
                if vo_p0_prev_features is None or len(vo_p0_prev_features) < 10:
                    print(f"Frame {frame_idx}: Critical re-detection failure.")
                    first_frame = True 
                    if track_visualization_mask is not None: 
                        track_visualization_mask = np.zeros_like(track_visualization_mask)

        current_max_range = 0
        if len(trajectory_3d_points) > 1:
            coords = np.array(trajectory_3d_points)
            max_x = np.max(np.abs(coords[:, 0]))
            max_z = np.max(np.abs(coords[:, 2]))
            current_max_range = max(max_x, max_z, 1e-5) 
        
        dynamic_scale = (min(traj_img_width, traj_img_height) / 2.5) / current_max_range if current_max_range > 0 else 10.0
        dynamic_scale = max(1.0, min(dynamic_scale, 200.0)) 

        traj_display_img = draw_trajectory(trajectory_3d_points, traj_img_width, traj_img_height, scale=dynamic_scale)
        cv2.imshow('Trajectory', traj_display_img)
        
        key = cv2.waitKey(30) & 0xff
        if key == 27:  
            print("ESC pressed, stopping.")
            break
        if key == ord('r'): 
            print("User requested VO reset ('r' key).")
            R_global_pose = np.eye(3)
            t_global_pos = np.zeros((3, 1))
            trajectory_3d_points = [t_global_pos.flatten().copy()]
            first_frame = True 
            vo_p0_prev_features = None
            if track_visualization_mask is not None:
                 track_visualization_mask = np.zeros_like(track_visualization_mask)
            frame_count_since_last_refresh = 0
            print("VO reset.")
            if pose_file_writer: # Write reset pose to file
                # Use a slightly incremented timestamp for the reset event
                reset_timestamp = (frame_idx + 0.05) * 0.1 # Ensure it's distinct if reset happens on same frame_idx
                initial_quat = R.from_matrix(R_global_pose).as_quat()
                pose_file_writer.write(f"{reset_timestamp:.6f} {t_global_pos[0,0]:.6f} {t_global_pos[1,0]:.6f} {t_global_pos[2,0]:.6f} "
                                   f"{initial_quat[0]:.6f} {initial_quat[1]:.6f} {initial_quat[2]:.6f} {initial_quat[3]:.6f}\n")


    if pose_file_writer:
        pose_file_writer.close()
        print(f"Pose data saved to {output_pose_file}")
    
    cv2.destroyAllWindows()
    print("Visual Odometry finished.")
    if trajectory_3d_points:
        print("Final camera position (X,Y,Z):", trajectory_3d_points[-1])

# --- Main Execution ---
if __name__ == '__main__':
    camera_matrix_K_main, dist_coeffs_main = load_camera_intrinsics() # Use your function

    if camera_matrix_K_main is None:
        print("Could not load camera intrinsics. Exiting.")
        exit(1)
    
    if not Path(DATASET_DIR).exists() or not Path(DATASET_DIR).is_dir():
        print(f"Error: DATASET_DIR ('{DATASET_DIR}') not found or is not a directory.")
        exit(1)
    
    if not SEQUENCE_NAME:
        print(f"Error: SEQUENCE_NAME is not configured.")
        exit(1)

    output_file_path_main = None
    if OUTPUT_POSE_FILE_NAME:
        output_file_path_main = Path(OUTPUT_POSE_FILE_NAME)

    print(f"Starting Visual Odometry for sequence: {SEQUENCE_NAME} in {DATASET_DIR}")
    print(f"Camera: {CAMERA_VIEW}, Max Frames: {MAX_FRAMES_TO_PROCESS if MAX_FRAMES_TO_PROCESS is not None else 'All'}")
    print(f"Track Refresh Interval: {TRACK_REFRESH_INTERVAL if TRACK_REFRESH_INTERVAL and TRACK_REFRESH_INTERVAL > 0 else 'Disabled'}")
    if output_file_path_main:
        print(f"Outputting poses to: {output_file_path_main.resolve()}")
    
    run_visual_odometry(
        DATASET_DIR, 
        SEQUENCE_NAME, 
        CAMERA_VIEW, 
        MAX_FRAMES_TO_PROCESS,
        TRACK_REFRESH_INTERVAL,
        camera_matrix_K_main, 
        dist_coeffs_main,
        str(output_file_path_main) if output_file_path_main else None
    )
