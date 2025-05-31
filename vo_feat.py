import cv2
import numpy as np
import os
from pathlib import Path
import glob

# --- Configuration: EDIT THESE VALUES ---
DATASET_DIR = "/run/media/alexander/T5 EVO/datasets" 
SEQUENCE_NAME = "ocean"
CAMERA_VIEW = "left"
MAX_FRAMES_TO_PROCESS = None # Example: 300
TRACK_REFRESH_INTERVAL = 50 # Example: clear tracks every 50 frames

# fx 0 cx
# 0 fy cy
# 0 0 1
INTRINSICS = (320,320,320,240) # format: fx fy cx cy

# --- End of Configuration ---


# --- Parameters for feature detection and tracking ---
FEATURE_PARAMS = dict(maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# --- Global Variables for VO ---
camera_matrix_K = None
dist_coeffs = None # Assuming undistorted images for now

# --- Helper Functions ---

def load_camera_intrinsics():
    """
    fx 0 cx
    0 fy cy
    0 0 1
    """

    K = np.eye(3)
    dist = np.zeros(5) # Default to no distortion
    K[0,0], K[1,1] = 320, 320
    K[0,2], K[1,2] = 320, 240

    # K = ⎡ 320.0   0.0   320.0 ⎤  
    # ⎢  0.0  320.0  240.0 ⎥  
    # ⎢  0.0    0.0    1.0 ⎥ 
    
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

    # Convert points to screen coordinates (X -> screen_x, Z -> screen_y)
    # Assuming input trajectory points are [x, y, z] and we plot (x, z)
    # Positive Z is forward, positive X is right.
    # For display, positive Z (forward) often goes upwards or downwards. Let's make it go downwards.
    screen_points = []
    for pt3d in trajectory_points_list:
        # Use pt3d[0] for X, pt3d[2] for Z
        screen_x = int(center_x + pt3d[0] * scale)
        screen_y = int(center_y + pt3d[2] * scale) # If Z is depth, and positive Z is away from camera.
                                                 # Or use -pt3d[2] if Z is forward and you want forward to be up.
                                                 # Let's assume Z is depth (away), so positive Z maps to positive Y on screen (down).
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
                        max_frames=None, track_refresh_interval=None, K_matrix=None, dist_coeffs_val=None):
    if K_matrix is None:
        print("Error: Camera intrinsic matrix K is not available. Cannot proceed with VO.")
        return

    image_paths = load_tartanair_image_paths(dataset_root_dir, current_sequence_name, current_camera)
    if not image_paths: return

    if max_frames is not None and max_frames > 0 :
        image_paths = image_paths[:max_frames]
        print(f"Processing a maximum of {len(image_paths)} frames.")

    # VO State
    R_global_pose = np.eye(3) # Camera orientation in world frame (Identity initially)
    t_global_pos = np.zeros((3, 1)) # Camera position in world frame (Origin initially)
    trajectory_3d_points = [t_global_pos.flatten().copy()] # Store 3D positions [x,y,z]
    
    # For feature tracking
    vo_prev_gray = None
    vo_p0_prev_features = None # Features from the previous VO frame

    track_visualization_mask = None
    traj_img_width, traj_img_height = 400, 600 # Dimensions for trajectory plot window
    cv2.namedWindow('Feature Tracks - VO', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Trajectory', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trajectory', traj_img_width, traj_img_height)

    frame_count_since_last_refresh = 0
    first_frame = True

    for frame_idx in range(len(image_paths)):
        current_frame_bgr = cv2.imread(image_paths[frame_idx])
        if current_frame_bgr is None:
            print(f"Warning: Could not read image {image_paths[frame_idx]}. Skipping.")
            continue
        
        if track_visualization_mask is None: # Initialize once we have frame dimensions
            track_visualization_mask = np.zeros_like(current_frame_bgr)

        current_frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Undistort (if distortion coeffs are provided)
        # For now, assuming undistorted or dist_coeffs_val is None/zeros
        # if dist_coeffs_val is not None and np.any(dist_coeffs_val):
        # current_frame_gray_undistorted = cv2.undistort(current_frame_gray, K_matrix, dist_coeffs_val)
        # else:
        current_frame_gray_undistorted = current_frame_gray # Use as is

        # --- Feature Tracking ---
        if first_frame:
            vo_p0_prev_features = cv2.goodFeaturesToTrack(current_frame_gray_undistorted, mask=None, **FEATURE_PARAMS)
            if vo_p0_prev_features is None or len(vo_p0_prev_features) < 10: # Need enough features
                print(f"Frame {frame_idx}: Not enough initial features found. Skipping frame.")
                cv2.imshow('Feature Tracks - VO', current_frame_bgr) # Show frame even if no features
                key = cv2.waitKey(30) & 0xff
                if key == 27: break
                continue
            vo_prev_gray = current_frame_gray_undistorted.copy()
            first_frame = False
            # Display current frame with initial points
            vis_img = current_frame_bgr.copy()
            for pt in vo_p0_prev_features:
                cv2.circle(vis_img, tuple(pt.ravel().astype(int)), 3, (0,0,255), -1)
            cv2.imshow('Feature Tracks - VO', vis_img)
        else:
            if vo_p0_prev_features is None or len(vo_p0_prev_features) < 10:
                print(f"Frame {frame_idx}: Too few features from previous step. Re-detecting.")
                vo_p0_prev_features = cv2.goodFeaturesToTrack(current_frame_gray_undistorted, mask=None, **FEATURE_PARAMS)
                if vo_p0_prev_features is None or len(vo_p0_prev_features) < 10:
                    print(f"Frame {frame_idx}: Re-detection failed. Skipping VO for this frame.")
                    vo_prev_gray = current_frame_gray_undistorted.copy() # Update prev_gray anyway
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

            # Periodically refresh the track visualization mask
            if track_refresh_interval and track_refresh_interval > 0:
                if frame_count_since_last_refresh >= track_refresh_interval:
                    track_visualization_mask = np.zeros_like(current_frame_bgr)
                    frame_count_since_last_refresh = 0
                else: frame_count_since_last_refresh += 1
            
            vis_img = current_frame_bgr.copy() # Default display if no good points
            if good_new_points is not None and len(good_new_points) > 5: # Need at least 5 points for Essential Matrix
                # --- Visual Odometry Estimation ---
                E, mask_e = cv2.findEssentialMat(good_new_points, good_old_points, K_matrix, 
                                                 method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None and np.sum(mask_e) >= 5 : # Check if E is found and enough inliers
                    # Decompose Essential Matrix and recover pose
                    # R_21: Rotation from cam2 to cam1, t_21: Translation of cam2 origin in cam1 coords
                    retval, R_21, t_21, mask_rp = cv2.recoverPose(E, good_new_points, good_old_points, K_matrix, mask=mask_e)

                    if retval > 0 and R_21 is not None and t_21 is not None and np.sum(mask_rp) >=5:
                        # Relative pose (transform from cam1 to cam2)
                        R_relative_pose = R_21.T 
                        t_relative_pose = -R_21.T @ t_21

                        # Update global pose
                        # t_global_pos is the position of the camera in the world
                        # R_global_pose is the orientation of the camera in the world
                        t_global_pos = t_global_pos + R_global_pose @ t_relative_pose
                        R_global_pose = R_global_pose @ R_relative_pose
                        
                        trajectory_3d_points.append(t_global_pos.flatten().copy())
                    else:
                        print(f"Frame {frame_idx}: recoverPose failed or not enough inliers ({np.sum(mask_rp) if mask_rp is not None else 0}).")
                else:
                    print(f"Frame {frame_idx}: findEssentialMat failed or not enough inliers ({np.sum(mask_e) if mask_e is not None else 0}).")

                # Drawing tracks (use original good_old/new before RANSAC filtering for visualization continuity)
                vis_img, track_visualization_mask = draw_tracks(current_frame_bgr, good_old_points, good_new_points, track_visualization_mask)
            
            cv2.imshow('Feature Tracks - VO', vis_img)
            
            # Update for next iteration (tracking)
            vo_prev_gray = current_frame_gray_undistorted.copy()
            if good_new_points is not None and len(good_new_points) > FEATURE_PARAMS['maxCorners'] * 0.25: # If enough points remain
                 vo_p0_prev_features = good_new_points.reshape(-1, 1, 2)
            else: # Re-detect if too few points or tracking failed
                print(f"Frame {frame_idx}: Feature count low or tracking failed. Re-detecting for next frame.")
                vo_p0_prev_features = cv2.goodFeaturesToTrack(current_frame_gray_undistorted, mask=None, **FEATURE_PARAMS)
                if vo_p0_prev_features is None or len(vo_p0_prev_features) < 10:
                    print(f"Frame {frame_idx}: Critical re-detection failure. May affect VO.")
                    first_frame = True # Reset to re-initialize detector on next available frame
                    track_visualization_mask = np.zeros_like(current_frame_bgr) # Also reset tracks
                # else vo_p0_prev_features is now set for next iteration

        # Draw trajectory
        # Determine dynamic scale for trajectory plotting
        current_max_range = 0
        if len(trajectory_3d_points) > 1:
            coords = np.array(trajectory_3d_points)
            max_x = np.max(np.abs(coords[:, 0]))
            max_z = np.max(np.abs(coords[:, 2]))
            current_max_range = max(max_x, max_z, 1e-5) # Avoid division by zero
        
        # Heuristic for scaling: try to fit max_range into about half the trajectory image dimension
        dynamic_scale = (min(traj_img_width, traj_img_height) / 2.5) / current_max_range if current_max_range > 0 else 10.0
        dynamic_scale = max(1.0, min(dynamic_scale, 200.0)) # Clamp scale

        traj_display_img = draw_trajectory(trajectory_3d_points, traj_img_width, traj_img_height, scale=dynamic_scale)
        cv2.imshow('Trajectory', traj_display_img)
        
        key = cv2.waitKey(30) & 0xff
        if key == 27:  # ESC
            print("ESC pressed, stopping.")
            break
        if key == ord('r'): # Reset VO
            print("User requested VO reset ('r' key).")
            R_global_pose = np.eye(3)
            t_global_pos = np.zeros((3, 1))
            trajectory_3d_points = [t_global_pos.flatten().copy()]
            first_frame = True # Re-initialize feature detection and VO
            vo_p0_prev_features = None
            track_visualization_mask = np.zeros_like(current_frame_bgr)
            frame_count_since_last_refresh = 0
            print("VO reset.")
    
    cv2.destroyAllWindows()
    print("Visual Odometry finished.")
    if trajectory_3d_points:
        print("Final camera position (X,Y,Z):", trajectory_3d_points[-1])

# --- Main Execution ---
if __name__ == '__main__':
    camera_matrix_K, dist_coeffs = load_camera_intrinsics()

    if camera_matrix_K is None:
        print("Could not load camera intrinsics. Exiting.")
        exit(1)
    
    # Validate dataset directory
    if not Path(DATASET_DIR).exists() or not Path(DATASET_DIR).is_dir():
        print(f"Error: DATASET_DIR ('{DATASET_DIR}') not found or is not a directory.")
        exit(1)
    
    if not SEQUENCE_NAME:
        print(f"Error: SEQUENCE_NAME is not configured.")
        exit(1)

    print(f"Starting Visual Odometry for sequence: {SEQUENCE_NAME} in {DATASET_DIR}")
    print(f"Camera: {CAMERA_VIEW}, Max Frames: {MAX_FRAMES_TO_PROCESS if MAX_FRAMES_TO_PROCESS is not None else 'All'}")
    print(f"Track Refresh Interval: {TRACK_REFRESH_INTERVAL if TRACK_REFRESH_INTERVAL and TRACK_REFRESH_INTERVAL > 0 else 'Disabled'}")
    
    run_visual_odometry(
        DATASET_DIR, 
        SEQUENCE_NAME, 
        CAMERA_VIEW, 
        MAX_FRAMES_TO_PROCESS,
        TRACK_REFRESH_INTERVAL,
        camera_matrix_K, # Pass loaded intrinsics
        dist_coeffs      # Pass loaded distortion coefficients
    )
