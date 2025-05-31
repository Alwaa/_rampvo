# feature_tracker.py

import cv2
import numpy as np
import os
from pathlib import Path
import glob

# Root directory of the TartanAir dataset.
DATASET_DIR = "/run/media/alexander/T5 EVO/datasets"
SEQUENCE_NAME = "ocean"
# Camera view to use ('left' or 'right').
CAMERA_VIEW = "left"

# Maximum number of frames to process from the sequence.
# Set to None to process all frames.
MAX_FRAMES_TO_PROCESS = None # Example: process first 550 frames


# --- Parameters for feature detection and tracking ---
# Parameters for ShiTomasi corner detection (goodFeaturesToTrack)
FEATURE_PARAMS = dict(maxCorners=200,    # Maximum number of corners to return.
                      qualityLevel=0.01, # Minimal accepted quality of image corners.
                      minDistance=10,    # Minimum possible Euclidean distance between corners.
                      blockSize=7)       # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.

# Parameters for Lucas-Kanade optical flow (calcOpticalFlowPyrLK)
LK_PARAMS = dict(
    winSize=(21, 21),  # Size of the search window at each pyramid level.
    maxLevel=3,        # 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on.
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), # Termination criteria for iterative search algorithm.
)

# --- Helper Functions ---

def load_tartanair_image_paths(dataset_dir_str, sequence_name, camera='left'):
    """
    Loads sorted image file paths for a given TartanAir sequence and camera.
    Tries a couple of common directory structures for TartanAir.
    
    Args:
        dataset_dir_str (str): Root directory of the TartanAir dataset.
        sequence_name (str): Name of the sequence (e.g., 'abandonedfactory', 'office').
        camera (str): 'left' or 'right'.
        
    Returns:
        list: A list of sorted image file paths. Returns empty if not found.
    """
    dataset_dir_path = Path(dataset_dir_str) # Renamed to avoid conflict with global
    
    # Common TartanAir structure: <dataset_root>/<sequence_name>/<sequence_name>/image_left/
    image_folder_attempt1 = dataset_dir_path / sequence_name / "Easy" / "P000" / f"image_{camera}"
    

    image_folder_to_use = None
    if image_folder_attempt1.exists() and image_folder_attempt1.is_dir():
        image_folder_to_use = image_folder_attempt1
    else:
        print(f"Error: Image folder not found for sequence '{sequence_name}', camera '{camera}' in '{dataset_dir_str}'.")
        print(f"  Attempted: {image_folder_attempt1}")
        return []
    
    print(f"Loading images from: {image_folder_to_use}")
    image_files = sorted(glob.glob(str(image_folder_to_use / "*.png")))
    
    if not image_files:
        print(f"Warning: No PNG images found in {image_folder_to_use}")
        
    return image_files

def draw_tracks(display_frame, prev_points, current_points, track_mask, track_color=(0, 255, 0)):
    """
    Draws the feature tracks and current feature points on the frame.
    """
    vis_frame = display_frame.copy() 

    for i, (new, old) in enumerate(zip(current_points, prev_points)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        track_mask = cv2.line(track_mask, (a, b), (c, d), track_color, 2)
        vis_frame = cv2.circle(vis_frame, (a, b), 5, track_color, -1)
    
    img_with_tracks = cv2.add(vis_frame, track_mask)
    return img_with_tracks, track_mask


# --- Main Tracking Logic ---

def track_features_in_sequence(dataset_root_dir, current_sequence_name, current_camera, max_frames=None):
    """
    Performs feature tracking on a TartanAir image sequence.
    """
    image_paths = load_tartanair_image_paths(dataset_root_dir, current_sequence_name, current_camera)
    if not image_paths:
        print(f"No images found for sequence '{current_sequence_name}'. Exiting.")
        return

    if max_frames is not None and max_frames > 0 : # Ensure max_frames is positive
        image_paths = image_paths[:max_frames]
        print(f"Processing a maximum of {len(image_paths)} frames.")

    old_frame_bgr = cv2.imread(image_paths[0])
    if old_frame_bgr is None:
        print(f"Error: Could not read the first image: {image_paths[0]}")
        return
        
    old_gray = cv2.cvtColor(old_frame_bgr, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **FEATURE_PARAMS)

    if p0 is None or len(p0) == 0:
        print("No initial features found. Try adjusting FEATURE_PARAMS or check image content.")
        cv2.imshow('First Frame (No Features)', old_frame_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    print(f"Initial features detected: {len(p0)}")

    track_visualization_mask = np.zeros_like(old_frame_bgr)
    cv2.namedWindow('Feature Tracks - TartanAir', cv2.WINDOW_NORMAL)

    for i in range(1, len(image_paths)):
        current_frame_bgr = cv2.imread(image_paths[i])
        if current_frame_bgr is None:
            print(f"Warning: Could not read image {image_paths[i]}. Skipping.")
            continue
        
        current_frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, current_frame_gray, p0, None, **LK_PARAMS)

        if p1 is not None and st is not None:
            good_new_points = p1[st == 1]
            good_old_points = p0[st == 1]
        else:
            print(f"Optical flow failed for frame {i}. Resetting features.")
            p0 = cv2.goodFeaturesToTrack(current_frame_gray, mask=None, **FEATURE_PARAMS)
            if p0 is None or len(p0) == 0:
                print("No features found after reset. Stopping.")
                break
            old_gray = current_frame_gray.copy()
            track_visualization_mask = np.zeros_like(current_frame_bgr)
            continue
        
        if len(good_new_points) > 0:
            frame_with_tracks, track_visualization_mask = draw_tracks(
                current_frame_bgr, good_old_points, good_new_points, track_visualization_mask
            )
            cv2.imshow('Feature Tracks - TartanAir', frame_with_tracks)
        else:
            cv2.imshow('Feature Tracks - TartanAir', current_frame_bgr)
        
        key = cv2.waitKey(30) & 0xff
        if key == 27:  # ESC
            print("ESC pressed, stopping.")
            break
        if key == ord('r'):
            print("User requested feature reset ('r' key).")
            p0 = cv2.goodFeaturesToTrack(current_frame_gray, mask=None, **FEATURE_PARAMS)
            if p0 is None or len(p0) == 0:
                print("No features found after user reset. Stopping.")
                break
            track_visualization_mask = np.zeros_like(current_frame_bgr)
            print(f"Features reset. New count: {len(p0) if p0 is not None else 0}")

        old_gray = current_frame_gray.copy()
        p0 = good_new_points.reshape(-1, 1, 2)

        if len(p0) < FEATURE_PARAMS['maxCorners'] / 2 : 
            print(f"Feature count low ({len(p0)}). Re-detecting.")
            newly_detected_features = cv2.goodFeaturesToTrack(old_gray, mask=None, **FEATURE_PARAMS)
            
            if newly_detected_features is not None:
                if len(p0) > 0:
                    p0 = np.vstack((p0, newly_detected_features))
                else:
                    p0 = newly_detected_features
                
                if p0 is not None and len(p0) > 0:
                    unique_points_2d = np.unique(p0.reshape(-1, 2), axis=0)
                    p0 = unique_points_2d.reshape(-1, 1, 2)
                    if len(p0) > FEATURE_PARAMS['maxCorners']:
                        np.random.shuffle(p0)
                        p0 = p0[:FEATURE_PARAMS['maxCorners']]
                print(f"  Total features after re-detection: {len(p0) if p0 is not None else 0}")

            if p0 is None or len(p0) == 0:
                print("Re-detection failed. Stopping.")
                break
    
    cv2.destroyAllWindows()
    print("Tracking finished.")

# --- Main Execution ---
if __name__ == '__main__':
    # Validate dataset directory
    if DATASET_DIR == "/path/to/your/tartanair_dataset" or not Path(DATASET_DIR).exists() or not Path(DATASET_DIR).is_dir():
        print(f"Error: DATASET_DIR ('{DATASET_DIR}') is not configured, not found, or is not a directory.")
        print("Please edit the DATASET_DIR variable at the top of the script.")
        exit(1)
    
    if not SEQUENCE_NAME:
        print(f"Error: SEQUENCE_NAME is not configured. Please edit it at the top of the script.")
        exit(1)

    print(f"Starting feature tracker for sequence: {SEQUENCE_NAME} in {DATASET_DIR}")
    print(f"Camera: {CAMERA_VIEW}, Max Frames: {MAX_FRAMES_TO_PROCESS if MAX_FRAMES_TO_PROCESS is not None else 'All'}")
    
    # Run the tracking
    track_features_in_sequence(
        DATASET_DIR, 
        SEQUENCE_NAME, 
        CAMERA_VIEW, 
        MAX_FRAMES_TO_PROCESS
    )
