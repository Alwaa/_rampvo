import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

from scipy.spatial.transform import Rotation as SRot
import g2o
from g2o import (
    CameraParameters,
    SparseOptimizer,
    LinearSolverDenseSE3, # Updated import if using a different solver later
    BlockSolverSE3,
    OptimizationAlgorithmLevenberg,
    SE3Quat,
    VertexSE3Expmap,
    VertexPointXYZ,
    EdgeProjectXYZ2UV,
    EdgeSE3, # Added for IMU factors
    Isometry3d # Added for SE3 edge measurement
)

def load_camera_params(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    K = np.array(config['camera']['intrinsics'])
    dist_coeffs = np.array(config['camera']['distortion'])
    return K, dist_coeffs


def skew(v):
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],    0]])


class IMUPreintegrator:
    def __init__(self, dt):
        self.dt = dt
        # Store delta_R, delta_v, delta_p for BA
        self.delta_R_raw = np.eye(3)
        self.delta_v_raw = np.zeros(3)
        self.delta_p_raw = np.zeros(3)
        self.reset()

    def reset(self):
        # These are the values used by BA, copied from raw values when an integration period ends.
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        
        # Reset raw values for the next integration period
        self.delta_R_raw = np.eye(3)
        self.delta_v_raw = np.zeros(3)
        self.delta_p_raw = np.zeros(3)


    def integrate_batch(self, acc_batch, gyro_batch):
        # Integrate into _raw variables
        for acc, gyro in zip(acc_batch, gyro_batch):
            dR = R.from_rotvec(gyro * self.dt).as_matrix()
            self.delta_R_raw = self.delta_R_raw.dot(dR) # Corrected: was self.delta_R
            acc_world = self.delta_R_raw.dot(acc) # Use the evolving delta_R_raw for acc transformation
            self.delta_p_raw += self.delta_v_raw * self.dt + 0.5 * acc_world * self.dt**2
            self.delta_v_raw += acc_world * self.dt
        
        # At the end of the batch (i.e., when process_frame calls this and then uses the preint object)
        # copy the integrated values to be used by BA.
        self.delta_R = self.delta_R_raw.copy()
        self.delta_v = self.delta_v_raw.copy()
        self.delta_p = self.delta_p_raw.copy()


class FeatureTracker:
    def __init__(self, detector='ORB', max_features=2000):
        self.det        = (cv2.ORB_create(max_features)
                           if detector=='ORB'
                           else cv2.GFTTDetector_create(max_features))
        self.lk_params  = dict(winSize=(21,21),maxLevel=3,
                               criteria=(cv2.TERM_CRITERIA_EPS|
                                         cv2.TERM_CRITERIA_COUNT,30,0.01))
        self.prev_img   = None
        self.prev_pts   = None     # shape (M,2)
        self.prev_ids   = None     # shape (M,) integer IDs
        self.next_id    = 0

    def detect(self, img):
        kps = self.det.detect(img, None)
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        return pts

    def init_frame(self, img):
        # For your very first frame: assign fresh IDs
        pts = self.detect(img)
        self.prev_pts = pts
        if pts is not None and len(pts) > 0:
            self.prev_ids = np.arange(len(pts)) + self.next_id
            self.next_id += len(pts)
        else:
            self.prev_ids = np.array([], dtype=int)
        self.prev_img = img


    def track(self, img):
        """
        Tracks self.prev_pts→current points, maintains IDs.
        Returns:
          - pts_prev (K×2), pts_cur (K×2) for two-view init
          - cur_pts (M×2), cur_ids (M,)       for PnP in sliding-window
        """
        if self.prev_pts is None or len(self.prev_pts) == 0 or self.prev_img is None:
            # nothing to track or no previous points
            self.prev_img = img
            # It's important to initialize prev_pts and prev_ids here if they are None
            # This typically happens on the very first call to track after init_frame
            # or if all points were lost.
            # For robustness, we can call detect here if prev_pts is empty.
            if self.prev_pts is None or len(self.prev_pts) == 0:
                 self.init_frame(img) # Detect new features if none to track

            return None, None, self.prev_pts, self.prev_ids


        # calc optical flow
        pts_cur_all, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_img, img, self.prev_pts, None, **self.lk_params
        )
        
        if pts_cur_all is None: # Handle case where optical flow fails
            mask = np.zeros(len(self.prev_pts), dtype=bool)
        else:
            mask = status.flatten() == 1

        # matched prev→cur for two-view
        pts_prev = self.prev_pts[mask]
        pts_cur  = pts_cur_all[mask] if pts_cur_all is not None else np.array([])


        # filter cur for next PnP
        cur_ids  = self.prev_ids[mask]
        # cur_pts  = pts_cur.copy() # This was original
        
        # Detect new features to maintain a consistent number of trackable points
        # This is a common strategy but can be adapted.
        # For now, we just update with successfully tracked points.
        # If len(pts_cur) < some_threshold, one might run self.detect(img)
        # and append new features. This part is kept simple for now.

        # update history
        self.prev_pts = pts_cur # Update with successfully tracked points
        self.prev_ids = cur_ids
        self.prev_img = img

        # Return values:
        # pts_prev, pts_cur: for two-view geometry (e.g. initialization)
        # self.prev_pts, self.prev_ids: current tracked points and their IDs for PnP
        return pts_prev, pts_cur, self.prev_pts, self.prev_ids


class XRVIO:
    def __init__(self, dt, window=10, cam_config = None):
        if cam_config is not None:
            self.K, self.dist = load_camera_params(cam_config)
        else:
            # Default K and dist if no config provided (e.g. for testing)
            self.K = np.eye(3) 
            self.dist = np.zeros(5)
            print("Warning: Camera config not provided. Using default identity K and zero distortion.")

        self.dt = dt
        self.preint = IMUPreintegrator(dt) # This is the current preintegrator for the LATEST interval
        self.tracker = FeatureTracker()
        
        # SLIDING WINDOW IMPLEMENTATION
        self.window = window # Max number of states in the window
        self.states = []         # each: {'R','t','v', 'preint': IMUPreintegrator_object_for_this_state_to_next}
                                 # 'preint' stores the preintegration FROM this state TO THE NEXT
        self.landmarks = {}      # id->3D point. Not pruned in this version for simplicity.
        # self.imu_slices = []   # This will be replaced by storing preint objects within self.states
        
        self.initialized = False
        self.logs = {'reproj': [], 'imu': []}
        self.tlist = [] # For timestamping poses, mainly for evaluation

        # Store the preintegrator for the interval LEADING TO the current state
        # When a new state is added, the self.preint (which has just integrated up to this new state)
        # will be associated with the *previous* state, as it defines the transform from prev to current.
        # This needs careful handling. Let's store preintegration *results* with each state.
        # Each state will store the delta_R, delta_v, delta_p that led *from the previous state to itself*.

    def undistort_points(self, pts):
        if pts is None or len(pts) == 0:
            return np.array([])
        # If no distortion, return pts unchanged
        if self.dist is None or np.all(self.dist == 0):
            return pts.copy()
        pts_n = cv2.undistortPoints(pts.reshape(-1,1,2), self.K, self.dist, None, self.K) # Undistort and reproject with K
        return pts_n.reshape(-1,2)

    
    def vg_pnp_solve(self, pts_3d, pts_2d, R_pred, K, dist_coeffs=None):
        """
        Visual-Gyro PnP: solve for pose given 3D-2D correspondences and gyro-based prior rotation.
        """
        if len(pts_3d) < 4 or len(pts_2d) < 4:
            print("Not enough points for PnP, using prediction.")
            return R_pred, None # Return None for t_opt if PnP fails

        # 1) Rodrigues init from R_pred
        rvec_prior, _ = cv2.Rodrigues(R_pred.astype(np.float64))
        # 2) Prepare inputs
        obj = pts_3d.astype(np.float64)
        img = pts_2d.astype(np.float64) # These should be undistorted
        dc  = dist_coeffs if dist_coeffs is not None else np.zeros((4,1))
        
        # 3) solvePnP
        # Try with RANSAC first for robustness
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K, dc,
            useExtrinsicGuess=True, rvec=rvec_prior, tvec=None, # Let tvec be solved
            iterationsCount=100, reprojectionError=8.0, confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE # Using iterative with guess
        )
        
        if not success:
            # Fallback to simple solvePnP if RANSAC fails
            success, rvec, tvec = cv2.solvePnP(
                obj, img, K, dc,
                rvec_prior, None, True, # useExtrinsicGuess=True
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        
        if not success: # If still fails, fallback to non-guess iterative
             success, rvec, tvec = cv2.solvePnP(
                obj, img, K, dc,
                None, None, False, 
                flags=cv2.SOLVEPNP_ITERATIVE 
            )

        if not success: # If still fails, PnP failed
            print("PnP failed, using prediction.")
            return R_pred, None


        # 4) back to matrix
        R_opt, _ = cv2.Rodrigues(rvec)
        t_opt = tvec.flatten()
        return R_opt, t_opt

    def process_frame(self, img, imu_t, imu_acc, imu_gyro):
        current_timestamp = imu_t[-1] if len(imu_t) > 0 else (self.tlist[-1] + self.dt if self.tlist else 0)
        self.tlist.append(current_timestamp)

        # 0. IMU Preintegration for the interval leading to this new frame
        # This preintegrator (self.preint) integrates measurements from the time of the
        # previous frame up to the time of the current frame.
        self.preint.reset() # Reset raw values in preintegrator for new batch
        self.preint.integrate_batch(imu_acc, imu_gyro)
        # Now self.preint.delta_R, .delta_v, .delta_p hold the integrated values for this interval.

        # 1. Feature Tracking
        # pts_prev_matched, pts_cur_matched: for 2-view geometry if needed (e.g. init)
        # tracked_pts_cur, tracked_ids_cur: current set of tracked 2D points and their IDs for PnP
        pts_prev_matched, pts_cur_matched, tracked_pts_cur, tracked_ids_cur = self.tracker.track(img)
        
        # Undistort current tracked points for PnP and triangulation
        undistorted_tracked_pts_cur = self.undistort_points(tracked_pts_cur)


        if not self.initialized:
            if len(self.states) == 0: # First frame
                self.tracker.init_frame(img) # Initialize features for the first frame
                # Initial state: Origin, identity rotation, zero velocity
                # The preintegration for this state is from a hypothetical state -1 to 0, so it's zero.
                initial_state = {
                    'R': np.eye(3), 't': np.zeros(3), 'v': np.zeros(3),
                    'delta_R': np.eye(3), 'delta_v': np.zeros(3), 'delta_p': np.zeros(3), # Preint *to* this state
                    'timestamp': current_timestamp
                }
                self.states.append(initial_state)
                return None # No pose to return yet
            else: # Second frame (or subsequent, if init failed) - Attempt initialization
                if pts_prev_matched is None or len(pts_prev_matched) < 2 or pts_cur_matched is None or len(pts_cur_matched) < 2:
                    print("Not enough matched points for initialization.")
                    # Update tracker's previous image, but don't add a state yet if init fails
                    self.tracker.prev_img = img 
                    return None

                # Use preintegration from state 0 to state 1 (current self.preint)
                R0_to_R1_imu = self.preint.delta_R 
                
                # Undistort points for two_point_ransac
                undistorted_pts0 = self.undistort_points(pts_prev_matched)
                undistorted_pts1 = self.undistort_points(pts_cur_matched)

                if len(undistorted_pts0) <2 or len(undistorted_pts1) < 2:
                    print("Not enough undistorted points for initialization.")
                    self.tracker.prev_img = img
                    return None

                # Estimate translation direction (t_dir is normalized)
                # R0_to_R1_imu is R_imu in two_point_ransac context
                t_dir_normalized = self.two_point_ransac(undistorted_pts0, undistorted_pts1, R0_to_R1_imu)

                if np.all(t_dir_normalized == 0): # RANSAC failed
                    print("Two-point RANSAC failed to estimate translation direction.")
                    self.tracker.prev_img = img
                    return None
                
                # For initialization, we need a scale. Assume a small initial scale for t.
                # This scale will be refined by va_align later.
                initial_scale = 0.1 # Heuristic, VINS-Mono uses depth median of SfM
                t1_in_R0_frame = t_dir_normalized * initial_scale

                # State 1 (current frame)
                R1 = self.states[0]['R'].dot(R0_to_R1_imu) # R_world_to_B1 = R_world_to_B0 * R_B0_to_B1
                t1 = self.states[0]['t'] + self.states[0]['R'].dot(t1_in_R0_frame) # p_B1_in_W = p_B0_in_W + R_W_B0 * p_B1_in_B0
                
                # Velocity V1 (in world frame)
                # v1 = v0 + R0*delta_v01 - g*dt. For init, assume g=0, v0=0.
                # delta_v is in frame of R0.
                v1 = self.states[0]['v'] + self.states[0]['R'].dot(self.preint.delta_v)


                new_state = {
                    'R': R1, 't': t1, 'v': v1,
                    'delta_R': self.preint.delta_R.copy(), 
                    'delta_v': self.preint.delta_v.copy(), 
                    'delta_p': self.preint.delta_p.copy(),
                    'timestamp': current_timestamp
                }
                self.states.append(new_state)
                
                # Triangulate landmarks using R0 (identity), t0 (zero) and R1_in_R0 (R0_to_R1_imu), t1_in_R0
                # P0 is K @ [I|0]
                # P1 is K @ [R0_to_R1_imu | t1_in_R0_frame]
                # Landmarks are in world frame (which is B0 frame initially)
                self.triangulate_landmarks(undistorted_pts0, undistorted_pts1, 
                                           R0_to_R1_imu, t1_in_R0_frame, 
                                           self.tracker.prev_ids[cv2.calcOpticalFlowPyrLK(self.tracker.prev_img, img, self.tracker.prev_pts, None, **self.tracker.lk_params)[1].flatten() == 1]) # Pass corresponding IDs

                self.initialized = True
                print("System Initialized.")
                # After init, tracker's prev_img, prev_pts, prev_ids are already updated by track()

        else: # System is initialized, perform VIO tracking
            prev_state = self.states[-1]
            
            # Predicted Rotation: R_pred = R_prev_world * delta_R_imu
            R_pred = prev_state['R'].dot(self.preint.delta_R)
            
            # Predicted Translation (simplified, full prediction involves gravity and prev_v)
            # t_pred = prev_state['t'] + prev_state['R'].dot(self.preint.delta_p) # Simplified: p_k = p_k-1 + R_k-1 * delta_p_k-1_k
            # More complete: t_pred = prev_state['t'] + prev_state['v'] * dt_interval + prev_state['R'].dot(self.preint.delta_p)
            # dt_interval needs to be calculated based on timestamps. For now, use simplified.
            dt_interval = current_timestamp - prev_state['timestamp'] if 'timestamp' in prev_state else self.dt # Approx
            t_pred = prev_state['t'] + prev_state['v'] * dt_interval + prev_state['R'].dot(self.preint.delta_p)


            # VG-PnP solve for pose
            known_ids = [id_val for id_val in self.landmarks.keys()]
            
            pnp_obj_pts = []
            pnp_img_pts = []
            if tracked_ids_cur is not None and undistorted_tracked_pts_cur is not None:
                for i, id_val in enumerate(tracked_ids_cur):
                    if id_val in known_ids:
                        pnp_obj_pts.append(self.landmarks[id_val])
                        pnp_img_pts.append(undistorted_tracked_pts_cur[i])
            
            pnp_obj_pts = np.array(pnp_obj_pts)
            pnp_img_pts = np.array(pnp_img_pts)

            if len(pnp_obj_pts) >= 4:
                R_opt, t_opt_raw = self.vg_pnp_solve(pnp_obj_pts, pnp_img_pts, R_pred, self.K, self.dist)
                if t_opt_raw is None: # PnP failed to get translation
                    t_opt = t_pred # Fallback translation
                else:
                    t_opt = t_opt_raw
            else:
                R_opt, t_opt = R_pred, t_pred
                print("Not enough correspondences for PnP, using IMU prediction.")

            # Update Velocity (in world frame)
            # v_k = v_k-1 + R_k-1 * delta_v_k-1_k (+ g_world * dt, if g is known and compensated in preint or here)
            v_opt = prev_state['v'] + prev_state['R'].dot(self.preint.delta_v)
            # If gravity is estimated (e.g. self.gravity), it should be applied:
            # v_opt += self.gravity * dt_interval (if self.gravity is in world frame)


            new_state = {
                'R': R_opt, 't': t_opt, 'v': v_opt,
                'delta_R': self.preint.delta_R.copy(), 
                'delta_v': self.preint.delta_v.copy(), 
                'delta_p': self.preint.delta_p.copy(),
                'timestamp': current_timestamp
            }
            self.states.append(new_state)

            # SLIDING WINDOW: Prune old states if window is full
            if len(self.states) > self.window:
                self.states.pop(0) # Remove the oldest state
                # Note: self.landmarks are not pruned here for simplicity.
                # Pruning landmarks would require checking which landmarks are still observed
                # by poses within the current window.

            # Triangulate new landmarks if any new features were matched by track()
            # This part needs more robust handling of new features from tracker vs existing landmarks
            # For now, we assume triangulation mainly happens at init.
            # If track() was modified to detect and return *new* features, they could be triangulated here.


            # Run Optimizations if enough states in window
            if len(self.states) >= 3: # Need at least 3 poses for some BA/Align operations
                # print("VG-BA START")
                # self.vg_ba_g2o()
                # print("VG-BA END")

                if len(self.states) >= 3: # va_align needs at least 2 intervals (3 states)
                    # print("VA-Align START")
                    # self.va_align()
                    # print("VA-Align END")
                    pass # VA Align can make things unstable without good initial scale

                print("VI-BA START")
                self.vi_ba_g2o() # Now with IMU factors
                print("VI-BA END")
        
        return self.states[-1] if self.states else None


    def two_point_ransac(self, pts0, pts1, R_imu, thresh=0.005, iters=100):
        # pts0, pts1 are already undistorted and normalized by K
        if len(pts0) < 2 or len(pts1) < 2:
            return np.zeros(3)
        
        # Convert to normalized homogeneous coordinates (z=1)
        p0_norm = np.hstack([pts0, np.ones((len(pts0), 1))])
        p1_norm = np.hstack([pts1, np.ones((len(pts1), 1))])

        best_t_dir, best_inliers_count = np.zeros(3), 0
        
        for _i in range(iters):
            idx = np.random.choice(len(p0_norm), 2, replace=False)
            
            # Epipolar constraint: p1_norm[k].T @ E @ p0_norm[k] = 0
            # E = [t_x]R_imu. We are solving for t.
            # p1_norm[k].T @ skew(t) @ R_imu @ p0_norm[k] = 0
            # This can be rewritten as: (R_imu @ p0_norm[k]) x p1_norm[k]) • t = 0
            # Or: dot(cross(p1_norm[k], R_imu.dot(p0_norm[k])), t_normalized) = 0 for essential matrix
            # For fundamental matrix approach with known K:
            # A_k = [ (p1_x * R20 - R00), (p1_x * R21 - R01), (p1_x * R22 - R02) ]
            #       [ (p1_y * R20 - R10), (p1_y * R21 - R11), (p1_y * R22 - R12) ]
            # where R is R_imu, and p0, p1 are normalized coords. This is for solving E.
            # For 2-point algorithm with known R, we solve for t direction.
            # (p1_j x R p0_j) . t = 0
            
            A_ransac = []
            for k_idx in idx:
                # v1 = p1_norm[k_idx, :2] # using normalized image coordinates directly
                # v0 = p0_norm[k_idx, :2]
                # A_k_row = np.array([ v1[0]*R_imu[2,0] - R_imu[0,0], v1[0]*R_imu[2,1] - R_imu[0,1], v1[0]*R_imu[2,2] - R_imu[0,2],
                #                      v1[1]*R_imu[2,0] - R_imu[1,0], v1[1]*R_imu[2,1] - R_imu[1,1], v1[1]*R_imu[2,2] - R_imu[1,2] ])
                # This is for solving Essential matrix elements, not directly t.

                # Simpler: use the geometric interpretation: t is orthogonal to (p1_j x R p0_j)
                # These are vectors from camera center to points in normalized coords.
                rp0 = R_imu @ p0_norm[idx[0]]
                rp1 = R_imu @ p0_norm[idx[1]]

                # Direction vectors in camera 0 frame
                d0_c0 = p0_norm[idx[0]] 
                d1_c0 = p0_norm[idx[1]]

                # Direction vectors in camera 1 frame, but expressed in camera 0's orientation
                d0_c1_in_c0_orientation = R_imu.T @ p1_norm[idx[0]]
                d1_c1_in_c0_orientation = R_imu.T @ p1_norm[idx[1]]
                
                # This formulation was problematic. Using the standard approach:
                # n_k = cross(p1_k, R @ p0_k)
                # t must be orthogonal to n_k.
                # So, n_k^T @ t = 0
                n0 = np.cross(p1_norm[idx[0]], R_imu @ p0_norm[idx[0]])
                n1 = np.cross(p1_norm[idx[1]], R_imu @ p0_norm[idx[1]])
                A_ransac.extend([n0, n1])


            A_ransac_mat = np.array(A_ransac)
            if A_ransac_mat.shape[0] < 2 : continue # Need at least 2 constraints for SVD on 3 unknowns

            _u, _s, vt = np.linalg.svd(A_ransac_mat)
            t_candidate = vt[-1]
            t_candidate /= np.linalg.norm(t_candidate) # Normalize

            # Count inliers
            inliers_count = 0
            all_errors = []
            for k_pt in range(len(p0_norm)):
                # error = abs(np.dot(np.cross(p1_norm[k_pt, :2], R_imu.dot(p0_norm[k_pt, :2])), t_candidate[:2])) # Error in 2D projection
                error = abs(np.dot(np.cross(p1_norm[k_pt], R_imu @ p0_norm[k_pt]), t_candidate))
                all_errors.append(error)
                if error < thresh:
                    inliers_count += 1
            
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_t_dir = t_candidate
                # Log reprojection residual for the best set of inliers
                # current_inlier_errors = [e for e in all_errors if e < thresh]
                # if current_inlier_errors: self.logs['reproj'].append(np.mean(current_inlier_errors))


        print(f"XRVIO two_point_ransac: {best_inliers_count} inliers found.")
        if best_inliers_count > 0 and self.logs['reproj']: # Check if reproj has entries
             pass # Logging was problematic, simplifying
        return best_t_dir


    def triangulate_landmarks(self, pts0, pts1, R_C0_to_C1, t_C1_in_C0, point_ids):
        # pts0, pts1 are undistorted (but not necessarily normalized by K for cv2.triangulatePoints)
        # R_C0_to_C1, t_C1_in_C0 define the pose of Camera 1 relative to Camera 0
        # P0 = K @ [I | 0]
        # P1 = K @ [R_C0_to_C1 | t_C1_in_C0]
        
        if pts0 is None or pts1 is None or len(pts0) == 0 or len(pts1) == 0:
            return

        P0 = self.K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P1 = self.K @ np.hstack((R_C0_to_C1, t_C1_in_C0.reshape(3,1)))

        # Ensure pts0 and pts1 are (N, 2)
        pts0_reshaped = pts0.reshape(-1, 2).T # Needs to be 2xN for cv2.triangulatePoints
        pts1_reshaped = pts1.reshape(-1, 2).T # Needs to be 2xN

        if pts0_reshaped.shape[1] != pts1_reshaped.shape[1]:
            print(f"Triangulation point mismatch: pts0 {pts0_reshaped.shape[1]}, pts1 {pts1_reshaped.shape[1]}")
            return

        if pts0_reshaped.shape[1] == 0:
            return

        points_4d_hom = cv2.triangulatePoints(P0, P1, pts0_reshaped, pts1_reshaped)
        points_3d = points_4d_hom[:3] / points_4d_hom[3]

        for i in range(points_3d.shape[1]):
            pt_id = point_ids[i] # Get the corresponding ID
            # Landmarks are stored in the world frame.
            # At initialization, Cam0 is the world frame.
            # So, the triangulated points are already in the world frame.
            self.landmarks[pt_id] = points_3d[:, i].flatten()
            # print(f"Triangulated landmark {pt_id}: {self.landmarks[pt_id]}")
    
    def va_align(self):
        """
        Jointly estimate scale s, per-frame velocities v_k, and gravity g
        by aligning the up-to-scale visual trajectory to IMU preintegrations.
        Assumes states[0] is the start of the alignment window.
        """
        # Number of intervals in the current window
        # If self.states has W poses, there are W-1 intervals.
        num_states_in_window = len(self.states)
        if num_states_in_window <= 1: # Need at least one interval
            return

        N = num_states_in_window - 1 # Number of intervals
        
        # total unknowns: 3*N velocities (v_0 to v_N-1) + 1 scale + 3 gravity = 3N + 4
        # Velocities are v_world_Bi for i = 0 to N-1 (i.e. for states[0] to states[N-1])
        n_unknowns = 3 * N + 1 + 3 # Scale, N velocities, Gravity
        # total equations: 3 per interval (from delta_p) + 3 per interval (from delta_v)
        # Using only delta_p for now as per original XR-VIO paper structure for VA-Align
        # Or VINS-Mono structure: 6 equations per interval (pos and vel)
        # The provided code uses only position constraint from preint.delta_p
        # Let's stick to the provided structure: 3 equations per interval for delta_p
        # p_k+1 = p_k + v_k*dt + R_k * delta_p_k,k+1_body - 0.5*g*dt^2 (if delta_p is IMU measurement)
        # If delta_p is from visual SfM (up to scale): s*delta_p_sfm = v_k*dt + R_k*delta_p_imu - 0.5*g*dt^2
        # The current code uses: s * (p_k+1_vis - p_k_vis) = v_k*dt + delta_p_imu_body - 0.5*g*dt^2
        # This seems to mix visual positions (scaled) with IMU delta_p.
        # Let's follow a more standard formulation like VINS-Mono for VA-Align:
        # R_w_bk * (s * p_ck+1 - s * p_ck) = R_w_bk * (p_bk+1 - p_bk)  -- position constraint
        # R_w_bk * (v_bk+1) = R_w_bk * (v_bk - g_bk * dt_k + beta_k,k+1) -- velocity constraint
        # where p_ck are camera positions from visual SfM (up-to-scale), p_bk are body positions.
        # And alpha, beta are IMU preintegration terms.
        # The current code's va_align uses:
        # A[row:row+3, 3*N] = delta_v_vis.reshape(3,1)  <- This is s * (p_k1 - p_k)
        # A[row:row+3, 3*k:3*k+3] = -dt_k * np.eye(3)   <- This is -v_k * dt
        # A[row:row+3, 3*N+1:3*N+4] = -0.5 * (dt_k**2) * np.eye(3) <- This is -0.5 * g * dt^2
        # b[row:row+3] = dp_k (IMU preintegrated delta_p)
        # So: s*(p_k1-p_k) - v_k*dt - 0.5*g*dt^2 = dp_k_imu
        # This means p_k1 and p_k are visual (up-to-scale) positions.
        # And dp_k_imu is the IMU preintegrated delta position.
        # This formulation is standard for scale, gravity, and velocity initialization.

        n_eqs = 3 * N # Using only position constraints

        if n_eqs == 0: return

        A = lil_matrix((n_eqs, n_unknowns), dtype=float)
        b_vec = np.zeros(n_eqs, dtype=float)

        # fill row‐blocks for k = 0…N–1 (interval k is between state[k] and state[k+1])
        for k in range(N):
            state_k = self.states[k]
            state_k1 = self.states[k+1]

            # visual up-to-scale displacement (in world frame)
            # These are already in world frame from self.states
            delta_p_visual = state_k1['t'] - state_k['t'] # shape (3,)

            # IMU preintegration for interval k (from state_k to state_k1)
            # This is stored in state_k1 as it's the preintegration *to* state_k1
            # dp_k_imu = state_k1['delta_p'] # This is delta_p in body frame of state_k
            # We need R_k to transform it to world frame: R_k @ dp_k_imu
            # The equation seems to be: s*delta_p_visual_world = v_k_world*dt + R_k_world*delta_p_imu_body_k - 0.5*g_world*dt^2
            
            # Let's use the formulation from the code to ensure consistency with its logic:
            # s * (p_k1_vis - p_k_vis) - v_k*dt - 0.5*g*dt^2 = dp_k_imu_body_rotated_by_R_k ???
            # The original code has: b[row:row+3] = dp_k (IMU preintegrated delta_p in body frame of k)
            # This means the equation is:
            # s*delta_p_vis_world - R_k_world * v_k_body * dt - 0.5 * R_k_world * g_body * dt^2 = delta_p_imu_body_k
            # This is not standard. Let's assume v_k and g are in WORLD frame.
            # s*delta_p_vis_world - v_k_world*dt - 0.5*g_world*dt^2 = R_k_world @ delta_p_imu_body_k

            # IMU preintegration delta_p for interval k is stored in state_k1
            # This delta_p is in the body frame of state_k.
            # We need to rotate it to world frame using R_k.
            dp_k_imu_world = state_k['R'] @ state_k1['delta_p']


            dt_k = state_k1['timestamp'] - state_k['timestamp']
            if dt_k <= 0: dt_k = self.dt # Avoid division by zero or negative dt

            row = 3 * k

            # Column for scale 's' (index 0)
            A[row:row+3, 0] = delta_p_visual 

            # Velocity block for v_k (cols 1 + 3k .. 1 + 3k+2)
            # v_k is the velocity of state_k
            A[row:row+3, 1 + 3*k : 1 + 3*k+3] = -dt_k * np.eye(3)

            # Gravity columns g_world (cols 1 + 3N .. 1 + 3N + 2)
            A[row:row+3, 1 + 3*N : 1 + 3*N + 3] = -0.5 * (dt_k**2) * np.eye(3)

            # right‐hand side
            b_vec[row:row+3] = dp_k_imu_world


        # solve A x ≃ b in the least-squares sense
        # x = [s, v0, v1, ..., vN-1, gx, gy, gz]
        try:
            sol = lsqr(A.tocsr(), b_vec, atol=1e-6, btol=1e-6, iter_lim=200)[0]
        except Exception as e:
            print(f"VA Align lsqr failed: {e}")
            return

        # unpack solution
        s_est  = sol[0]                              # scale
        v_list = [ sol[1+3*k : 1+3*k+3] for k in range(N) ]  # velocities v0 to vN-1
        g_est  = sol[1+3*N : 1+3*N+3]                      # gravity_world

        # ——— CLAMP THE SCALE ———
        min_s, max_s = 0.01, 10.0  # Adjusted bounds
        s_clamped = float(np.clip(s_est, min_s, max_s))

        if self.logs['imu'] and 'scale' in self.logs['imu'][-1]:
            prev_s = self.logs['imu'][-1].get('scale', 1.0)
            rel_min = prev_s * 0.5
            rel_max = prev_s * 2.0
            s_clamped = float(np.clip(s_clamped, rel_min, rel_max))
        
        final_s = s_clamped
        print(f"VA Align raw_s={s_est:.3f} → clamped to {final_s:.3f}")

        # Apply scale to all visual positions *in the window*
        # And update velocities *in the window*
        for i in range(num_states_in_window): # Iterate through all states in the current window
            # Scale translations relative to the first pose in the window
            if i > 0:
                 t_relative_to_window_start = self.states[i]['t'] - self.states[0]['t']
                 self.states[i]['t'] = self.states[0]['t'] + final_s * t_relative_to_window_start
            elif i == 0 and num_states_in_window == len(self.states): # If window start is absolute start
                 # If this is the very first state ever, its 't' is likely (0,0,0)
                 # Scaling its 't' might be problematic if it's not (0,0,0) and we scale it.
                 # For simplicity, only scale relative translations.
                 # If you want to scale all absolute positions:
                 # self.states[i]['t'] *= final_s # This assumes origin was (0,0,0) or scales everything
                 pass


            if i < N: # Velocities v_0 to v_N-1 correspond to states[0] to states[N-1]
                self.states[i]['v'] = v_list[i]
            elif i == N: # For the last state (states[N]), velocity needs to be propagated or estimated
                      # v_N = v_N-1 + R_N-1 @ delta_v_N-1,N_body + g_world * dt_N-1
                if N > 0 : # if there is a state N-1
                    prev_v = v_list[N-1]
                    R_prev = self.states[N-1]['R']
                    delta_v_body = self.states[N]['delta_v'] # preint from N-1 to N
                    dt_prev = self.states[N]['timestamp'] - self.states[N-1]['timestamp']
                    self.states[N]['v'] = prev_v + R_prev @ delta_v_body + g_est * dt_prev


        # Store gravity (if you need it later)
        self.gravity = g_est # Store estimated gravity in world frame

        self.logs.setdefault('imu', []).append({
            'scale': final_s, 
            'gravity': g_est.copy()
        })
        # print(f"VA Align → scale={final_s:.4f}, gravity={g_est}")

    
    def vg_ba_g2o(self):
        if len(self.states) < 2: return # Need at least two poses

        opt = g2o.SparseOptimizer()
        
        # Choose solver: For SE3, Dense or Cholesky are common
        # For larger problems, sparse solvers like CSparse or CHOLMOD are better
        # block_solver_type = g2o.BlockSolverSE3
        # linear_solver_type = g2o.LinearSolverDenseSE3() # For small problems
        linear_solver_type = g2o.LinearSolverCholmodSE3() # For potentially larger
        linear_solver_type.set_block_ordering(False)
        
        block_solver = BlockSolverSE3(linear_solver_type)
        algo = OptimizationAlgorithmLevenberg(block_solver)
        opt.set_algorithm(algo)
        opt.set_verbose(False)

        # Camera parameters
        focal_length = (self.K[0,0] + self.K[1,1]) / 2.0 # Average fx, fy
        principal_point = (self.K[0,2], self.K[1,2])
        cam_params = CameraParameters(focal_length, principal_point, 0) # baseline=0 for mono
        cam_params.set_id(0)
        opt.add_parameter(cam_params)

        # Add pose vertices (for all states currently in the window)
        pose_vertex_map = {} # Maps state index in self.states to g2o vertex id
        for i, state_data in enumerate(self.states):
            R_i = state_data['R']
            t_i = state_data['t'].reshape(3,1) # Ensure t_i is (3,1)
            se3 = SE3Quat(R_i, t_i)
            
            v_se3 = VertexSE3Expmap()
            v_se3.set_id(i) # Use local index in window as ID
            pose_vertex_map[i] = i # Store mapping

            v_se3.set_estimate(se3)
            v_se3.set_fixed(i == 0) # Fix the first pose in the window
            opt.add_vertex(v_se3)

        # Add landmark vertices
        # For simplicity, iterate through all global landmarks.
        # A more advanced version would only add landmarks observed by poses in the current window.
        landmark_g2o_id_offset = len(self.states) # Ensure landmark IDs don't clash with pose IDs
        landmark_vertex_map = {} # Maps landmark_id (from self.landmarks) to g2o vertex id
        
        # This part of landmark and observation handling is kept from original for now
        # It assumes self.tracker.prev_pts are observations for all landmarks for all poses in window
        # which is generally not true. This needs a proper observation model.
        # For now, we proceed with this structure to minimize changes.
        
        # Simplified: Add all landmarks for now.
        # In a real sliding window, you'd only add landmarks visible in the current window.
        temp_landmark_observations = {} # lm_id -> {pose_idx_in_window: uv}
                                        # This should be populated from a better observation model.
                                        # For now, this will be empty, and the original loop structure is used.

        # The original code's landmark addition was:
        # for lm_id, P3 in self.landmarks.items():
        #    vp = g2o.VertexPointXYZ(); vp.set_id(lm_id*2+1) ...
        # This assumes lm_id is an integer. If prev_ids are large, lm_id*2+1 can be very large.
        # Let's use a map for landmark g2o IDs.
        
        current_g2o_landmark_id = landmark_g2o_id_offset
        for lm_id_global, P3_world in self.landmarks.items():
            vp = VertexPointXYZ()
            vp.set_id(current_g2o_landmark_id)
            landmark_vertex_map[lm_id_global] = current_g2o_landmark_id
            vp.set_estimate(P3_world) 
            vp.set_marginalized(True) # Marginalize landmarks for efficiency
            opt.add_vertex(vp)
            current_g2o_landmark_id += 1

        # Add reprojection edges (USING ORIGINAL LOGIC - NEEDS REFACTORING FOR CORRECT OBSERVATIONS)
        # This loop assumes self.tracker.prev_pts are observations for ALL landmarks
        # for ALL poses in the window, which is incorrect.
        # This should iterate over actual stored observations.
        # For now, to make it "work" with minimal changes to this part:
        if self.tracker.prev_pts is not None and self.tracker.prev_ids is not None:
            # Assume prev_pts/prev_ids are for the LATEST frame in the window (self.states[-1])
            latest_pose_idx_in_window = len(self.states) - 1
            if latest_pose_idx_in_window >=0: # If there are states
                g2o_pose_id = pose_vertex_map[latest_pose_idx_in_window]
                for i in range(len(self.tracker.prev_ids)):
                    lm_id_global = self.tracker.prev_ids[i]
                    uv = self.tracker.prev_pts[i]
                    if lm_id_global in landmark_vertex_map:
                        g2o_landmark_id = landmark_vertex_map[lm_id_global]
                        
                        edge = EdgeProjectXYZ2UV()
                        edge.set_vertex(0, opt.vertex(g2o_landmark_id)) # Landmark
                        edge.set_vertex(1, opt.vertex(g2o_pose_id))   # Pose
                        edge.set_measurement(uv)
                        edge.set_information(np.eye(2)) # Simple information matrix
                        edge.set_parameter_id(0,0) # Use camera parameters with id 0
                        opt.add_edge(edge)


        # Add gyro rotation edges (IMU preintegration constraints for rotation)
        for i in range(len(self.states) - 1):
            pose0_idx_in_window = i
            pose1_idx_in_window = i + 1

            g2o_pose0_id = pose_vertex_map[pose0_idx_in_window]
            g2o_pose1_id = pose_vertex_map[pose1_idx_in_window]

            # Preintegration from state i to state i+1 is stored in self.states[i+1]
            preint_data = self.states[pose1_idx_in_window]
            R_meas = preint_data['delta_R'] # Delta R from body_i to body_i+1

            edge = EdgeSE3()
            edge.set_vertex(0, opt.vertex(g2o_pose0_id))
            edge.set_vertex(1, opt.vertex(g2o_pose1_id))
            
            # Measurement is T_B_i_to_B_i+1 (rotation only for VG-BA)
            # For VG-BA, translation part of measurement is zero, info matrix reflects this.
            meas_isometry = Isometry3d(R_meas, np.zeros(3)) # Zero translation for pure gyro edge
            edge.set_measurement(meas_isometry)
            
            info = np.eye(6)
            info[:3,:3] *= 100.0 # High weight for rotation
            info[3:,3:] *= 0.01   # Low weight for translation (ideally zero if not constrained)
            edge.set_information(info)
            opt.add_edge(edge)
        
        try:
            opt.initialize_optimization()
            opt.optimize(5) # Number of iterations

            # Update poses from optimizer
            for i, state_data in enumerate(self.states):
                g2o_pose_id = pose_vertex_map[i]
                est_se3 = opt.vertex(g2o_pose_id).estimate()
                self.states[i]['R'] = est_se3.rotation().matrix() # g2o quat to matrix
                self.states[i]['t'] = est_se3.translation().flatten()
            
            # Update landmarks (optional, if not marginalized or if update desired)
            # If marginalized, their estimates are implicitly updated.
            # If not marginalized and want to update:
            # for lm_id_global, g2o_lm_id in landmark_vertex_map.items():
            #    if opt.vertex(g2o_lm_id): # Check if vertex exists
            #        self.landmarks[lm_id_global] = opt.vertex(g2o_lm_id).estimate()

        except Exception as e:
            print(f"VG-BA optimization failed: {e}")
        # print("VG-BA complete")


    def vi_ba_g2o(self):
        if len(self.states) < 2: return

        optimizer = SparseOptimizer()
        # linear_solver_type = g2o.LinearSolverDenseSE3()
        linear_solver_type = g2o.LinearSolverCholmodSE3()
        linear_solver_type.set_block_ordering(False)
        
        block_solver = BlockSolverSE3(linear_solver_type)
        algo = OptimizationAlgorithmLevenberg(block_solver)
        optimizer.set_algorithm(algo)
        optimizer.set_verbose(False)

        focal_length = (self.K[0,0] + self.K[1,1]) / 2.0
        principal_point = (self.K[0,2], self.K[1,2])
        cam_params = CameraParameters(focal_length, principal_point, 0)
        cam_params.set_id(0)
        optimizer.add_parameter(cam_params)

        pose_vertex_map = {}
        for i, state_data in enumerate(self.states):
            se3 = SE3Quat(state_data['R'], state_data['t'].reshape(3,1))
            v_se3 = VertexSE3Expmap()
            v_se3.set_id(i) # Local window index as ID
            pose_vertex_map[i] = i
            v_se3.set_estimate(se3)
            v_se3.set_fixed(i == 0)
            optimizer.add_vertex(v_se3)

        landmark_g2o_id_offset = len(self.states)
        landmark_vertex_map = {}
        current_g2o_landmark_id = landmark_g2o_id_offset
        for lm_id_global, P3_world in self.landmarks.items():
            vp = VertexPointXYZ()
            vp.set_id(current_g2o_landmark_id)
            landmark_vertex_map[lm_id_global] = current_g2o_landmark_id
            vp.set_estimate(P3_world)
            vp.set_marginalized(True)
            optimizer.add_vertex(vp)
            current_g2o_landmark_id += 1
        
        # Add reprojection edges (KEPT ORIGINAL LOGIC - NEEDS REFACTORING)
        if self.tracker.prev_pts is not None and self.tracker.prev_ids is not None:
            latest_pose_idx_in_window = len(self.states) - 1
            if latest_pose_idx_in_window >=0:
                g2o_pose_id = pose_vertex_map[latest_pose_idx_in_window]
                for i in range(len(self.tracker.prev_ids)):
                    lm_id_global = self.tracker.prev_ids[i]
                    uv = self.tracker.prev_pts[i]
                    if lm_id_global in landmark_vertex_map:
                        g2o_landmark_id = landmark_vertex_map[lm_id_global]
                        edge = EdgeProjectXYZ2UV()
                        edge.set_vertex(0, optimizer.vertex(g2o_landmark_id))
                        edge.set_vertex(1, optimizer.vertex(g2o_pose_id))
                        edge.set_measurement(uv)
                        edge.set_information(np.eye(2))
                        edge.set_parameter_id(0,0)
                        optimizer.add_edge(edge)

        # ***** ADDED IMU FACTORS (Simplified SE3 Edge) *****
        for i in range(len(self.states) - 1):
            pose0_idx_in_window = i
            pose1_idx_in_window = i + 1

            g2o_pose0_id = pose_vertex_map[pose0_idx_in_window]
            g2o_pose1_id = pose_vertex_map[pose1_idx_in_window]

            # Preintegration from state i to state i+1 is stored in self.states[i+1]
            preint_data = self.states[pose1_idx_in_window]
            delta_R_meas = preint_data['delta_R'] 
            delta_p_meas = preint_data['delta_p'] # delta_p is in body frame of pose0

            edge = EdgeSE3()
            edge.set_vertex(0, optimizer.vertex(g2o_pose0_id)) # From pose
            edge.set_vertex(1, optimizer.vertex(g2o_pose1_id)) # To pose
            
            # Measurement is T_B_i_to_B_i+1 = (delta_R_meas, delta_p_meas)
            meas_isometry = Isometry3d(delta_R_meas, delta_p_meas)
            edge.set_measurement(meas_isometry)
            
            # Information matrix: reflects uncertainty of preintegration.
            # Higher values = more confidence.
            # This is a placeholder. Proper covariance propagation is needed for accurate info.
            info = np.eye(6)
            info[:3,:3] *= 100.0 # Rotation part (gyro) - higher confidence
            info[3:,3:] *= 50.0  # Translation part (accel) - moderate confidence
            edge.set_information(info)
            optimizer.add_edge(edge)
        # *****************************************************

        try:
            optimizer.initialize_optimization()
            optimizer.optimize(10) # Number of iterations

            for i, state_data in enumerate(self.states):
                g2o_pose_id = pose_vertex_map[i]
                se3_opt = optimizer.vertex(g2o_pose_id).estimate()
                
                # g2o SE3Quat.rotation() is a g2o.Quaternion
                # Need to convert g2o.Quaternion to scipy.Rotation or numpy matrix
                q_g2o = se3_opt.rotation()
                # Scipy expects (x,y,z,w)
                q_scipy = np.array([q_g2o.x(), q_g2o.y(), q_g2o.z(), q_g2o.w()])
                
                self.states[i]['R'] = SRot.from_quat(q_scipy).as_matrix()
                self.states[i]['t'] = se3_opt.translation().flatten()
            
            # Update landmarks
            for lm_id_global, g2o_lm_id in landmark_vertex_map.items():
                 if optimizer.vertex(g2o_lm_id): # Check if vertex exists
                    self.landmarks[lm_id_global] = optimizer.vertex(g2o_lm_id).estimate()
            
            if self.logs['reproj'] is not None : self.logs['reproj'].append(optimizer.chi2())

        except Exception as e:
            print(f"VI-BA optimization failed: {e}")
        # print("VI-BA complete")


    def reset(self):
        self.preint.reset()
        self.tracker = FeatureTracker() # Re-initialize tracker
        self.states.clear()
        self.landmarks.clear()
        # self.imu_slices.clear() # Removed, preint stored in states
        self.initialized = False
        self.logs = {'reproj': [], 'imu': []} # Reset logs
        self.tlist = []


    def __call__(self, tstamp, input_tensor, intrinsics, 
                 curr_imu_data = None, save_slam_steps_path = None):
        
        # self.tlist.append(tstamp) # Moved to process_frame for more accurate timing

        if self.K is None or np.all(self.K == np.eye(3)): # If K not set or is default identity
            fx, fy, cx, cy = intrinsics.cpu().numpy()
            self.K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

        if len(input_tensor) < 3:
            __events, images = input_tensor 
        elif len(input_tensor) == 3:
            __events, images, _mask = input_tensor
        else:
            raise Exception("Wrong input tensor")
        
        img_squeezed = images.squeeze(0).squeeze(0).permute(1,2,0)
        # assert img_squeezed.shape == (480,640,3), f"Image not correct: Shape{img_squeezed.shape}"
        if img_squeezed.shape[2] == 3: # RGB
            img = img_squeezed.cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img_squeezed.shape[2] == 1: # Grayscale
            img = img_squeezed.cpu().numpy()
            img = (img * 255).astype(np.uint8)
        else:
            raise Exception(f"Unsupported image channel count: {img_squeezed.shape[2]}")


        imu_t, imu_acc, imu_gyro = curr_imu_data
        _state = self.process_frame(img, imu_t, imu_acc, imu_gyro)
    
    def terminate(self):
        """interpolate missing poses"""
        if not self.tlist or not self.states:
            return np.array([]), np.array([])

        poses = np.zeros((len(self.states),7)) # Output poses based on states available

        for i, state in enumerate(self.states):
            if i < len(poses): # Ensure we don't go out of bounds if tlist/states mismatch
                rot_q =  SRot.from_matrix(state['R']).as_quat() # x,y,z,w
                poses[i] = np.concatenate((state['t'],rot_q))

        # Timestamps should correspond to the states for which poses are generated
        # If tlist has more entries than states, trim tlist or handle appropriately.
        # Assuming tlist corresponds to each call of process_frame, and states are appended there.
        # The number of poses should match the number of states.
        
        # Use timestamps stored with states for better alignment
        tstamps_from_states = np.array([s['timestamp'] for s in self.states if 'timestamp' in s], dtype=float)
        
        # If number of poses derived from states is less than tlist, adjust.
        # This can happen if some process_frame calls didn't result in a new state.
        # For simplicity, we return poses and timestamps strictly from self.states.
        if len(tstamps_from_states) != poses.shape[0]:
             # Fallback to tlist if timestamps in states are inconsistent
             if len(self.tlist) >= poses.shape[0]:
                 tstamps = np.array(self.tlist[:poses.shape[0]], dtype=float)
             else: # Not enough timestamps in tlist either, this is problematic
                 tstamps = np.arange(poses.shape[0]) * self.dt # Approximate timestamps
        else:
            tstamps = tstamps_from_states


        return poses[:len(tstamps)], tstamps # Ensure poses and tstamps have same length


if __name__=='__main__':
    # Create a dummy config.yaml for testing if it doesn't exist
    try:
        with open('config.yaml', 'r') as f:
            yaml.safe_load(f)
    except FileNotFoundError:
        print("config.yaml not found, creating a dummy one for testing.")
        dummy_config_content = {
            'camera': {
                'intrinsics': [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
                'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
            }
        }
        with open('config.yaml', 'w') as f:
            yaml.dump(dummy_config_content, f)
    
    cam_cfg = 'config.yaml'
    # Ensure K is loaded in __init__ if cam_cfg is provided
    vio = XRVIO(dt=1/30.0, window=7, cam_config=cam_cfg) # Example: 30Hz, window of 7 frames

    # Create dummy images and IMU data for testing
    num_frames = 20
    img_h, img_w = 480, 640
    
    # Dummy image files (not actually read, XRVIO takes numpy array)
    # img_files = [f'dummy_img_{i}.png' for i in range(num_frames)] 

    # Dummy IMU data: (imu_t, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
    # Simulate some motion: forward motion with some rotation
    imu_measurements_per_frame = 10 # 10 IMU readings between frames
    
    print("Starting dummy VIO run...")
    for frame_idx in range(num_frames):
        # Create a dummy grayscale image
        img = np.random.randint(0, 255, (img_h, img_w), dtype=np.uint8)
        
        # Create dummy IMU data for the interval leading to this image
        # Timestamps for IMU data
        current_time = frame_idx * vio.dt
        imu_ts_interval = np.linspace(current_time - vio.dt + vio.dt/imu_measurements_per_frame, 
                                      current_time, 
                                      imu_measurements_per_frame, endpoint=True)
        
        imu_acc_data = np.random.randn(imu_measurements_per_frame, 3) * 0.1 # Small random accelerations
        imu_acc_data[:, 2] += 9.81 # Gravity component if IMU is upright
        
        imu_gyro_data = np.random.randn(imu_measurements_per_frame, 3) * 0.05 # Small random angular velocities
        if frame_idx > 5: # Simulate some rotation
            imu_gyro_data[:, 1] += 0.1 # Yaw rate

        # Call VIO processing
        # The __call__ method expects a different input tensor format, adapting for direct test
        # For direct call to process_frame:
        state = vio.process_frame(img, imu_ts_interval, imu_acc_data, imu_gyro_data)
        
        if state:
            print(f"Frame {frame_idx}: t={state['t']}, R_det={np.linalg.det(state['R']):.2f}")
            if vio.logs['reproj'] and vio.logs['reproj'][-1] is not None:
                 print(f"  Reproj Chi2: {vio.logs['reproj'][-1]:.4f}")
            if vio.logs['imu'] and vio.logs['imu'][-1] is not None and 'scale' in vio.logs['imu'][-1]:
                 print(f"  IMU Scale: {vio.logs['imu'][-1]['scale']:.4f}")
        else:
            print(f"Frame {frame_idx}: No state returned (e.g. during initialization).")

    final_poses, final_tstamps = vio.terminate()
    print(f"\nGenerated {len(final_poses)} poses.")
    # for i in range(len(final_poses)):
    #     print(f"Pose {i} @t={final_tstamps[i]:.3f}: p={final_poses[i][:3]}, q={final_poses[i][3:]}")

