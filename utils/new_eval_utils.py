import numpy as np
import os
import os.path as osp
import cv2

FEATURE_TRACK_WINDOW = "Feature Tracks - VO"
TAJECTORY_WINDOW = "Trajectory"


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

        # setting up OpenCV windows here
        cv2.namedWindow(FEATURE_TRACK_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(TAJECTORY_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TAJECTORY_WINDOW, self.traj_img_width, self.traj_img_height)
        print(f"Visualizer initialized. Trajectory view: {traj_img_width}x{traj_img_height}, Track refresh: {track_refresh_interval} frames.")

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
        return 

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