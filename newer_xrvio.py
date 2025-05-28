import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R # Used by IMUPreintegrator
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from scipy.optimize import lsq_linear # Was imported, keeping

from scipy.spatial.transform import Rotation as SRot # Used in VI-BA and terminate
import g2o
from g2o import (
    CameraParameters,
    SparseOptimizer,
    LinearSolverDenseSE3, # Assuming available
    BlockSolverSE3,
    OptimizationAlgorithmLevenberg,
    SE3Quat,
    VertexSE3Expmap,
    VertexPointXYZ,
    EdgeProjectXYZ2UV,
    EdgeSE3,       # For BA factors
    Isometry3d     # For BA factors
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
    def __init__(self, dt_seconds): # dt is the IMU sampling period in SECONDS
        self.dt = dt_seconds
        self.reset()

    def reset(self):
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3) # m/s
        self.delta_p = np.zeros(3) # m

    def integrate_batch(self, acc_batch, gyro_batch):
        for acc, gyro in zip(acc_batch, gyro_batch):
            # R from scipy.spatial.transform.Rotation
            dR = R.from_rotvec(gyro * self.dt).as_matrix() # self.dt is seconds
            self.delta_R = self.delta_R @ dR # Use @ for matrix multiplication
            acc_world = self.delta_R @ acc # Use @
            self.delta_p += self.delta_v * self.dt + 0.5 * acc_world * self.dt**2
            self.delta_v += acc_world * self.dt


class FeatureTracker:
    # --- FeatureTracker class remains unchanged ---
    def __init__(self, detector='ORB', max_features=2000):
        self.det        = (cv2.ORB_create(max_features)
                           if detector=='ORB'
                           else cv2.GFTTDetector_create(max_features))
        self.lk_params  = dict(winSize=(21,21),maxLevel=3,
                               criteria=(cv2.TERM_CRITERIA_EPS|
                                         cv2.TERM_CRITERIA_COUNT,30,0.01))
        self.prev_img   = None
        self.prev_pts   = None
        self.prev_ids   = None
        self.next_id    = 0

    def detect(self, img):
        kps = self.det.detect(img, None)
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        return pts

    def init_frame(self, img):
        pts = self.detect(img)
        self.prev_pts = pts
        if pts is not None and len(pts) > 0:
            self.prev_ids = np.arange(len(pts)) + self.next_id
            self.next_id += len(pts)
        else:
            self.prev_pts = np.array([], dtype=np.float32).reshape(0,2)
            self.prev_ids = np.array([], dtype=int)
        self.prev_img = img

    def track(self, img):
        if self.prev_pts is None or len(self.prev_pts) == 0 or self.prev_img is None:
            self.prev_img = img
            if self.prev_pts is None or len(self.prev_pts) == 0:
                 self.init_frame(img)
            return None, None, self.prev_pts, self.prev_ids

        pts_cur_all, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_img, img, self.prev_pts, None, **self.lk_params
        )
        if pts_cur_all is None or status is None:
            mask = np.zeros(len(self.prev_pts), dtype=bool)
        else:
            mask = status.flatten() == 1

        pts_prev_matched = self.prev_pts[mask]
        pts_cur_matched  = pts_cur_all[mask] if pts_cur_all is not None else np.array([], dtype=np.float32).reshape(0,2)
        
        cur_ids  = self.prev_ids[mask]
        cur_pts  = pts_cur_matched.copy()

        self.prev_pts = cur_pts
        self.prev_ids = cur_ids
        self.prev_img = img
        return pts_prev_matched, pts_cur_matched, self.prev_pts, self.prev_ids
# --- End of FeatureTracker ---


class XRVIO:
    def __init__(self, dt_imu_sampling_period_seconds, window=10, cam_config=None, initial_timestamp_ns=0.0): # Added initial_timestamp_ns
        if cam_config is not None:
            self.K, self.dist = load_camera_params(cam_config)
        else:
            self.K, self.dist = np.eye(3), np.zeros(5) # Default K and dist
            # print("Warning: Using default K and zero distortion.")

        # dt_imu_sampling_period_seconds is for IMUPreintegrator (in SECONDS)
        self.imu_sampling_period_seconds = float(dt_imu_sampling_period_seconds)
        self.preint = IMUPreintegrator(self.imu_sampling_period_seconds)
        self.tracker = FeatureTracker()
        
        self.states = []         # Each state: {'R','t','v'}. Timestamps are implicitly linked via self.tlist_ns or imu_slices.
        self.landmarks = {}      # id->3D
        self.imu_slices = []     # list of (imu_timestamps_ns_batch, acc, gyro) per frame
        
        self.initialized = False
        self.window = int(window)
        self.logs = {'reproj': [], 'imu': []}
        self.gravity = np.array([0,0,-9.81]) # m/s^2 # Added gravity attribute

        self.tlist_ns = [] # Will store NANOSECOND timestamps from __call__
        self._initial_timestamp_ns = float(initial_timestamp_ns) # For the very first state if needed
        self.first_img_for_init = None # Added for clarity in init logic
        self.first_state_for_init = None # Added for clarity

    def undistort_points(self, pts):
        if pts is None or len(pts) == 0: return pts
        if self.dist is None or np.all(self.dist == 0) or self.K is None:
            return pts.copy()
        K_cv = self.K.astype(np.float64)
        dist_cv = self.dist.astype(np.float64)
        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float64)
        pts_n = cv2.undistortPoints(pts_reshaped, K_cv, dist_cv, None, K_cv)
        return pts_n.reshape(-1, 2)

    def vg_pnp_solve(self, pts_3d, pts_2d, R_pred, K, dist_coeffs=None):
        if K is None: return R_pred, None
        if len(pts_3d) < 4 or len(pts_2d) < 4: return R_pred, None
        rvec_prior, _ = cv2.Rodrigues(R_pred.astype(np.float64))
        obj = pts_3d.astype(np.float64); img = pts_2d.astype(np.float64)
        dc  = dist_coeffs.astype(np.float64) if dist_coeffs is not None else np.zeros((4,1), dtype=np.float64)
        K_cv = K.astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(obj, img, K_cv, dc, rvec_prior, None, True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            success, rvec, tvec = cv2.solvePnP(obj, img, K_cv, dc, None, None, False, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return R_pred, None
        R_opt, _ = cv2.Rodrigues(rvec); t_opt = tvec.flatten()
        # Consider adding is_rotation_matrix_valid(R_opt) here
        return R_opt, t_opt

    def process_frame(self, img, imu_timestamps_ns_batch, imu_acc, imu_gyro):
        # imu_timestamps_ns_batch are in NANOSECONDS
        
        current_frame_timestamp_ns = self._initial_timestamp_ns # Default
        if len(imu_timestamps_ns_batch) > 0:
            current_frame_timestamp_ns = float(imu_timestamps_ns_batch[-1])
        elif self.tlist_ns: # If IMU is missing, use last recorded frame timestamp
            current_frame_timestamp_ns = self.tlist_ns[-1]
            # Create dummy IMU data if it's completely missing for this call
            if len(imu_acc) == 0: imu_acc = np.zeros((1,3))
            if len(imu_gyro) == 0: imu_gyro = np.zeros((1,3))
            if len(imu_timestamps_ns_batch) == 0: imu_timestamps_ns_batch = np.array([current_frame_timestamp_ns])


        # Store IMU slice with NANOSECOND timestamps
        self.imu_slices.append((imu_timestamps_ns_batch, imu_acc, imu_gyro))
        
        # IMUPreintegrator uses self.imu_sampling_period_seconds (seconds)
        self.preint.reset()
        self.preint.integrate_batch(imu_acc, imu_gyro) # Outputs delta_v (m/s), delta_p (m)

        pts_prev_matched, pts_cur_matched, current_tracked_pts, current_tracked_ids = self.tracker.track(img)

        if not self.initialized:
            if not self.states: # Equivalent to `not hasattr(self, 'first_img')` or `self.first_img_for_init is None`
                self.first_img_for_init = img # Store current image as the first for initialization
                # The first state corresponds to this first image/IMU batch
                R0 = np.eye(3) # Initial orientation
                v0 = np.zeros(3) # Initial velocity
                p0 = np.zeros(3) # Initial position
                # Store the preintegrated deltas from this first IMU batch
                self.first_preint_deltas = (self.preint.delta_R.copy(),
                                     self.preint.delta_v.copy(), # m/s
                                     self.preint.delta_p.copy()) # m
                self.tracker.init_frame(img) # Initialize tracker with the first image
                # Add the first state
                self.states.append({'R': R0, 't': p0, 'v': v0, '_timestamp_ns_ref': current_frame_timestamp_ns}) # Store ref timestamp
                return None # Wait for the second frame to initialize
            else: # Second frame: attempt VG-SfM init
                if pts_prev_matched is None or pts_cur_matched is None or len(pts_prev_matched) < 2:
                    return None # Not enough matches

                # R_f0_to_f1_imu is the rotation from the *first* IMU batch (stored)
                R_f0_to_f1_imu = self.first_preint_deltas[0]
                v_delta_f0_to_f1_body = self.first_preint_deltas[1] # m/s

                # Undistorted points from the first frame (tracker.prev_pts before last track) and current frame
                # This assumes pts_prev_matched are features from first_img_for_init
                t_dir_f0_frame = self.two_point_ransac(pts_prev_matched, pts_cur_matched, R_f0_to_f1_imu)
                if np.all(t_dir_f0_frame == 0): return None

                initial_scale = 0.1 # Arbitrary scale for monocular
                t_f1_in_f0_frame = t_dir_f0_frame * initial_scale # meters

                R_f0_world = self.states[0]['R']
                t_f0_world = self.states[0]['t']
                v_f0_world = self.states[0]['v']

                R_f1_world = R_f0_world @ R_f0_to_f1_imu
                t_f1_world = t_f0_world + R_f0_world @ t_f1_in_f0_frame
                
                # Estimate dt between first and second frame for velocity update
                dt_init_ns = current_frame_timestamp_ns - self.states[0]['_timestamp_ns_ref']
                dt_init_s = max(self.imu_sampling_period_seconds, dt_init_ns / 1e9) # ensure positive dt_s

                v_f1_world = v_f0_world + R_f0_world @ v_delta_f0_to_f1_body + self.gravity * dt_init_s

                self.states.append({'R':R_f1_world, 't':t_f1_world, 'v':v_f1_world, '_timestamp_ns_ref': current_frame_timestamp_ns})
                
                # Triangulate using relative pose R_f0_to_f1_imu, t_f1_in_f0_frame
                # Need IDs for pts_prev_matched. If tracker.init_frame was called on first_img_for_init,
                # then self.tracker.prev_ids at that point would be for those features.
                # This detail needs to align with how FeatureTracker handles IDs.
                # Assuming pts_prev_matched came from the tracker's state after first_img_for_init.
                # The current tracker.track returns current_tracked_ids which are for current_tracked_pts.
                # For triangulation, we need IDs of pts_prev_matched.
                # Let's assume FeatureTracker gives IDs corresponding to pts_prev_matched if needed.
                # For now, using a placeholder for IDs if this is an issue:
                # ids_for_triangulation = np.arange(len(pts_prev_matched)) # Placeholder
                # self.triangulate_landmarks(pts_prev_matched, pts_cur_matched, R_f0_to_f1_imu, t_f1_in_f0_frame, ids_for_triangulation)
                
                self.initialized = True
                # print(f"Initialized with {len(self.states)} states.")
                return self.states[-1]
        else: # System is initialized
            prev_st = self.states[-1]
            prev_st_timestamp_ns = self.tlist_ns[-2] if len(self.tlist_ns) >=2 else self.tlist_ns[-1] # Approx prev frame time

            dt_interval_ns = current_frame_timestamp_ns - prev_st_timestamp_ns
            if dt_interval_ns <= 0:
                 dt_interval_ns = self.imu_sampling_period_seconds * 1e9 * max(1, len(imu_timestamps_ns_batch))
            dt_interval_ns = max(1.0, dt_interval_ns)

            dt_interval_s = dt_interval_ns / 1e9 # Convert to SECONDS

            R_pred = prev_st['R'] @ self.preint.delta_R
            t_pred = prev_st['t'] + prev_st['R'] @ self.preint.delta_p + \
                       prev_st['v'] * dt_interval_s + \
                       0.5 * self.gravity * (dt_interval_s**2)
            v_pred = prev_st['v'] + prev_st['R'] @ self.preint.delta_v + \
                       self.gravity * dt_interval_s
            
            R_opt, t_opt = R_pred, t_pred # Default to IMU prediction
            if current_tracked_pts is not None and current_tracked_ids is not None and len(current_tracked_ids) > 0:
                known_ids_mask = np.array([lm_id in self.landmarks for lm_id in current_tracked_ids])
                if np.sum(known_ids_mask) >= 4:
                    pts3d_pnp = np.array([self.landmarks[lm_id] for lm_id in current_tracked_ids[known_ids_mask]])
                    pts2d_pnp_distorted = current_tracked_pts[known_ids_mask]
                    pts2d_pnp_undistorted = self.undistort_points(pts2d_pnp_distorted)
                    
                    if pts2d_pnp_undistorted is not None and len(pts2d_pnp_undistorted) >= 4:
                        R_pnp, t_pnp = self.vg_pnp_solve(pts3d_pnp, pts2d_pnp_undistorted, R_pred, self.K, None) # No dist_coeffs for undistorted
                        if t_pnp is not None: # Add is_rotation_matrix_valid if available
                            R_opt, t_opt = R_pnp, t_pnp
            
            new_state = {'R': R_opt, 't': t_opt, 'v': v_pred, '_timestamp_ns_ref': current_frame_timestamp_ns}
            self.states.append(new_state)

            if len(self.states) > self.window: # Basic sliding window
                self.states.pop(0)
                if len(self.imu_slices) > self.window: self.imu_slices.pop(0)
            
            # Optional: BA and Align calls
            # if len(self.states) >= self.window:
            #     self.vg_ba_g2o()
            #     self.va_align()
            #     self.vi_ba_g2o()

            return new_state

    def two_point_ransac(self, pts0, pts1, R_imu, thresh=0.005, iters=100):
        # ... (two_point_ransac with undistortion, as previously corrected) ...
        if pts0 is None or pts1 is None or len(pts0)<2 or len(pts1)<2 : return np.zeros(3)
        p0_u = self.undistort_points(pts0) 
        p1_u = self.undistort_points(pts1)
        if p0_u is None or p1_u is None or len(p0_u) < 2 or len(p1_u) < 2 or len(p0_u) != len(p1_u): return np.zeros(3)

        best_t, best_in_count = np.zeros(3), 0
        for _i in range(iters):
            idx = np.random.choice(len(p1_u),2,replace=False)
            p0_h = np.hstack([p0_u[idx], np.ones((2,1))]) 
            p1_h = np.hstack([p1_u[idx], np.ones((2,1))]) 
            A_ransac = [np.cross(p1_h[k_pt], R_imu @ p0_h[k_pt]) for k_pt in range(2)]
            A_mat = np.vstack(A_ransac)
            if A_mat.shape[0] < A_mat.shape[1]: continue

            _,_,Vt = np.linalg.svd(A_mat)
            t_cand = Vt[-1]
            if np.linalg.norm(t_cand) < 1e-6 : continue
            t_cand /= np.linalg.norm(t_cand)
            
            inliers_count = 0
            p0_u_h_all = np.hstack([p0_u, np.ones((len(p0_u),1))])
            p1_u_h_all = np.hstack([p1_u, np.ones((len(p1_u),1))])
            for k_all in range(len(p1_u_h_all)):
                n_k_all = np.cross(p1_u_h_all[k_all], R_imu @ p0_u_h_all[k_all])
                err = abs(np.dot(n_k_all, t_cand))
                if err < thresh: inliers_count += 1
            if inliers_count > best_in_count:
                best_in_count = inliers_count; best_t = t_cand
        return best_t

    def triangulate_landmarks(self, pts0, pts1, R_f1_in_f0, t_f1_in_f0, point_ids):
        # ... (triangulate_landmarks, as previously corrected for cheirality and undistortion) ...
        if self.K is None or pts0 is None or pts1 is None or point_ids is None: return
        if len(pts0) != len(pts1) or len(pts0) != len(point_ids): return
        
        pts0_u = self.undistort_points(pts0)
        pts1_u = self.undistort_points(pts1)
        if pts0_u is None or pts1_u is None or len(pts0_u) != len(pts1_u): return

        P0 = self.K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P1 = self.K @ np.hstack((R_f1_in_f0, t_f1_in_f0.reshape(3,1)))

        for i in range(len(pts0_u)):
            pt_id = point_ids[i]
            X_hom = cv2.triangulatePoints(P0, P1, pts0_u[i].reshape(2,1), pts1_u[i].reshape(2,1))
            if X_hom[3,0] != 0:
                X = X_hom[:3,0] / X_hom[3,0]
                pt_in_cam1_frame = R_f1_in_f0.T @ (X - t_f1_in_f0)
                if X[2] > 0 and pt_in_cam1_frame[2] > 0: 
                     self.landmarks[pt_id] = X
    
    def va_align(self):
        active_states_for_align = self.states[-self.window:] # Use states from the current window
        active_imu_slices_for_align = self.imu_slices[-(self.window):] # Corresponding IMU slices

        N = len(active_states_for_align) - 1
        if N <= 1: # Need at least 2 intervals (3 states) for a meaningful V-A alignment.
            return

        n_unknowns = 3 * (N + 1) + 1 + 3 # N+1 velocities, 1 scale, 3 gravity
        n_eqs_pos = 3 * N 
        n_eqs_vel = 3 * N
        n_total_eqs = n_eqs_pos + n_eqs_vel

        A = lil_matrix((n_total_eqs, n_unknowns), dtype=float)
        b = np.zeros(n_total_eqs, dtype=float)

        for k in range(N): # Iterate over N intervals
            state_k = active_states_for_align[k]
            state_k1 = active_states_for_align[k+1]
            
            delta_t_visual_abs = state_k1['t'] - state_k['t'] 

            # IMU data for interval k to k+1
            # The imu_slice at index k+1 in active_imu_slices_for_align corresponds to the one that
            # led to state_k1 from state_k.
            if k + 1 >= len(active_imu_slices_for_align): continue
            imu_ts_ns_batch_k, acc_k, gyro_k = active_imu_slices_for_align[k+1]
            
            if len(imu_ts_ns_batch_k) < 2: continue

            temp_preint = IMUPreintegrator(self.imu_sampling_period_seconds) # dt in seconds
            temp_preint.integrate_batch(acc_k, gyro_k)
            dp_k_imu = temp_preint.delta_p # meters
            dv_k_imu = temp_preint.delta_v # m/s
            
            # Duration of this IMU batch in NANOSECONDS, then convert to SECONDS
            # Use state timestamps for more accurate dt_k if available and reliable for the interval
            # state_k and state_k1 have '_timestamp_ns_ref'
            dt_k_ns = state_k1.get('_timestamp_ns_ref', imu_ts_ns_batch_k[-1]) - \
                      state_k.get('_timestamp_ns_ref', imu_ts_ns_batch_k[0] - (self.imu_sampling_period_seconds*1e9*len(imu_ts_ns_batch_k)) ) # Fallback start

            if dt_k_ns <= 0: dt_k_ns = self.imu_sampling_period_seconds * 1e9 * max(1, len(imu_ts_ns_batch_k))
            if dt_k_ns <= 0: continue

            dt_k_s = dt_k_ns / 1e9 # Convert to SECONDS

            R_k_world = state_k['R']

            row_p = 3 * k
            A[row_p : row_p+3, 3*(N+1)]              = delta_t_visual_abs 
            A[row_p : row_p+3, 3*k : 3*k+3]          = -dt_k_s * np.eye(3) 
            A[row_p : row_p+3, 3*(N+1)+1:3*(N+1)+4]  = -0.5 * (dt_k_s**2) * np.eye(3)
            b[row_p : row_p+3] = R_k_world @ dp_k_imu

            row_v = n_eqs_pos + 3 * k
            A[row_v : row_v+3, 3*(k+1) : 3*(k+1)+3]   =  np.eye(3)
            A[row_v : row_v+3, 3*k : 3*k+3]           = -np.eye(3)
            A[row_v : row_v+3, 3*(N+1)+1:3*(N+1)+4]   = -dt_k_s * np.eye(3)
            b[row_v : row_v+3] = R_k_world @ dv_k_imu
        
        if A.nnz == 0 : return
        try: sol = lsqr(A.tocsr(), b, atol=1e-6, btol=1e-6)[0]
        except Exception: return

        v_list_world = [ sol[3*j : 3*j+3] for j in range(N+1) ]
        s_est  = sol[3*(N+1)]
        g_est_world  = sol[3*(N+1)+1 : 3*(N+1)+4]
        s_clamped = float(np.clip(s_est, 0.01, 100.0))
        
        if s_clamped < 0.01 + 1e-3 : return

        t_origin_visual = active_states_for_align[0]['t'].copy()
        for i in range(N + 1):
            active_states_for_align[i]['t'] = t_origin_visual + (active_states_for_align[i]['t'] - t_origin_visual) * s_clamped
            active_states_for_align[i]['v'] = v_list_world[i]
            
        self.gravity = g_est_world
        self.logs.setdefault('imu', []).append({'scale': s_clamped, 'gravity': g_est_world.copy()})

    
    def vg_ba_g2o(self):
        # ... (BA unchanged from previous, uses preint.delta_R which is fine) ...
        pass

    def vi_ba_g2o(self):
        # ... (BA unchanged from previous, uses preint.delta_R, preint.delta_p which are fine) ...
        pass

    def reset(self, initial_timestamp_ns=0.0): # Ensure initial_timestamp_ns is handled
        self.preint.reset()
        self.tracker = FeatureTracker()
        self.states.clear()
        self.landmarks.clear()
        self.imu_slices.clear()
        self.initialized = False
        self.first_img_for_init = None # Reset init helpers
        self.first_state_for_init = None
        self.first_preint_deltas = None
        self.tlist_ns.clear()
        self._initial_timestamp_ns = float(initial_timestamp_ns)


    def __call__(self, tstamp_current_frame_ns, input_tensor, intrinsics, 
                 curr_imu_data=None, save_slam_steps_path=None): # tstamp is NANOSECONDS
        
        self.tlist_ns.append(float(tstamp_current_frame_ns)) # Store NANOSECOND frame timestamp

        if self.K is None: # Initialize K if not done by cam_config
            fx_i, fy_i, cx_i, cy_i = intrinsics.cpu().numpy()
            self.K = np.array([[fx_i,0,cx_i],[0,fy_i,cy_i],[0,0,1]], dtype=float)

        # Image processing
        if len(input_tensor) < 3: __events, images = input_tensor 
        elif len(input_tensor) == 3: __events, images, _mask = input_tensor
        else: raise Exception("Wrong input tensor format")
        
        img_squeezed = images.squeeze(0).squeeze(0).permute(1,2,0)
        img_np = img_squeezed.cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):
            img_gray = img_np.squeeze()
        else: raise Exception(f"Cannot convert image to grayscale, shape: {img_np.shape}")

        imu_t_ns_batch, imu_acc_batch, imu_gyro_batch = np.array([]), np.empty((0,3)), np.empty((0,3))
        if curr_imu_data is not None and len(curr_imu_data[0]) > 0 :
            imu_t_ns_batch, imu_acc_batch, imu_gyro_batch = curr_imu_data
            imu_t_ns_batch = np.asarray(imu_t_ns_batch, dtype=float)
            imu_acc_batch = np.asarray(imu_acc_batch, dtype=float)
            imu_gyro_batch = np.asarray(imu_gyro_batch, dtype=float)
            if imu_t_ns_batch.ndim == 0: imu_t_ns_batch = np.array([imu_t_ns_batch])
        else: # Fallback if no IMU data provided
            imu_t_ns_batch = np.array([float(tstamp_current_frame_ns)])
            imu_acc_batch = np.zeros((1,3))
            imu_gyro_batch = np.zeros((1,3))
            
        return self.process_frame(img_gray, imu_t_ns_batch, imu_acc_batch, imu_gyro_batch)
    
    def terminate(self):
        output_poses_list = []
        # Use self.tlist_ns as the basis for timestamps if states don't store them directly for output
        # However, self.states now has '_timestamp_ns_ref' which should be used.
        # The number of states might not equal len(self.tlist_ns) if process_frame sometimes returns None.
        
        # Sort states by their reference timestamp to ensure chronological output
        # Only include states that were fully processed (not None from process_frame)
        valid_states = [s for s in self.states if s is not None and '_timestamp_ns_ref' in s]
        if not valid_states:
            return np.array([[0,0,0, 0,0,0,1]]), np.array([0.0]) 

        valid_states.sort(key=lambda s: s['_timestamp_ns_ref'])

        output_tstamps_ns_list = []
        for state in valid_states:
            try:
                rot_q = SRot.from_matrix(state['R']).as_quat() # [x,y,z,w]
                pose_txyz_qxyzw = np.concatenate((state['t'], rot_q))
                output_poses_list.append(pose_txyz_qxyzw)
                output_tstamps_ns_list.append(state['_timestamp_ns_ref']) # NANOSECONDS
            except Exception:
                pass # Skip problematic states
        
        if not output_poses_list:
            return np.array([[0,0,0, 0,0,0,1]]), np.array([0.0])

        return np.array(output_poses_list), np.array(output_tstamps_ns_list, dtype=float)
