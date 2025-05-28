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
    # LinearSolverDenseSE3, 
    # LinearSolverCholmodSE3, # Make sure this is available or handle fallback
    BlockSolverSE3,
    OptimizationAlgorithmLevenberg,
    SE3Quat,
    VertexSE3Expmap,
    VertexPointXYZ,
    EdgeProjectXYZ2UV,
    EdgeSE3, 
    Isometry3d 
)

NANOSECONDS_TO_SECONDS = 1e-9

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

def is_rotation_matrix_valid(R_mat):
    """Checks if a matrix is a valid rotation matrix."""
    if not isinstance(R_mat, np.ndarray) or R_mat.shape != (3,3):
        return False
    if not np.isfinite(R_mat).all():
        return False
    should_be_identity = R_mat.T @ R_mat
    identity = np.identity(3)
    if not np.allclose(should_be_identity, identity, atol=1e-3): # Looser tolerance for after BA
        # print(f"Rotation matrix orthogonality check failed. R.T @ R:\n{should_be_identity}")
        return False
    if not np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-3):
        # print(f"Rotation matrix determinant check failed. det(R): {np.linalg.det(R_mat)}")
        return False
    return True


class IMUPreintegrator:
    def __init__(self, dt_seconds): # dt is the IMU sampling period in seconds
        self.dt = dt_seconds 
        self.delta_R_raw = np.eye(3)
        self.delta_v_raw = np.zeros(3)
        self.delta_p_raw = np.zeros(3)
        self.reset()

    def reset(self):
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        
        self.delta_R_raw = np.eye(3)
        self.delta_v_raw = np.zeros(3)
        self.delta_p_raw = np.zeros(3)


    def integrate_batch(self, acc_batch, gyro_batch):
        # Assumes self.dt is the time delta for EACH IMU measurement in the batch
        for acc, gyro in zip(acc_batch, gyro_batch):
            if not np.isfinite(gyro).all():
                continue
            try:
                dR = R.from_rotvec(gyro * self.dt).as_matrix()
            except Exception as e:
                dR = np.eye(3) 

            self.delta_R_raw = self.delta_R_raw @ dR 
            
            if not np.isfinite(acc).all():
                continue

            acc_world = self.delta_R_raw @ acc 
            self.delta_p_raw += self.delta_v_raw * self.dt + 0.5 * acc_world * self.dt**2
            self.delta_v_raw += acc_world * self.dt
        
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
        matched_ids  = self.prev_ids[mask]
        
        self.prev_pts = pts_cur_matched 
        self.prev_ids = matched_ids
        self.prev_img = img

        return pts_prev_matched, pts_cur_matched, self.prev_pts, self.prev_ids


class XRVIO:
    def __init__(self, imu_sample_period_sec, window=10, cam_config = None): # Renamed dt to imu_sample_period_sec
        if cam_config is not None:
            self.K, self.dist = load_camera_params(cam_config)
        else:
            self.K = np.eye(3) 
            self.dist = np.zeros(5)
            print("Warning: Camera config not provided. Using default identity K and zero distortion.")

        self.imu_dt = imu_sample_period_sec # IMU sampling period in seconds
        self.preint = IMUPreintegrator(self.imu_dt) 
        self.tracker = FeatureTracker()
        
        self.window = window 
        self.states = []  
        self.finalized_trajectory = [] 
        
        self.landmarks = {}      
        
        self.initialized = False
        self.logs = {'reproj': [], 'imu': []}
        self.gravity = np.array([0,0,-9.81]) 

    def undistort_points(self, pts):
        if pts is None or len(pts) == 0:
            return np.array([])
        if self.dist is None or np.all(self.dist == 0):
            return pts.copy()
        K_cv = self.K.astype(np.float64)
        dist_cv = self.dist.astype(np.float64)
        pts_n = cv2.undistortPoints(pts.reshape(-1,1,2), K_cv, dist_cv, None, K_cv) 
        return pts_n.reshape(-1,2)

    
    def vg_pnp_solve(self, pts_3d, pts_2d, R_pred, K, dist_coeffs=None):
        if len(pts_3d) < 4 or len(pts_2d) < 4:
            return R_pred, None 

        rvec_prior, _ = cv2.Rodrigues(R_pred.astype(np.float64))
        obj = pts_3d.astype(np.float64)
        img = pts_2d.astype(np.float64) 
        dc  = dist_coeffs if dist_coeffs is not None else np.zeros((4,1))
        K_cv = K.astype(np.float64) 

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K_cv, dc,
            useExtrinsicGuess=True, rvec=rvec_prior, tvec=None, 
            iterationsCount=100, reprojectionError=8.0, confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE 
        )
        
        if not success:
            initial_tvec_guess = np.zeros((3, 1), dtype=np.float64) 
            success, rvec, tvec = cv2.solvePnP(
                obj, img, K_cv, dc,
                rvec_prior, initial_tvec_guess, True, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        
        if not success: 
             success, rvec, tvec = cv2.solvePnP(
                obj, img, K_cv, dc,
                None, None, False, 
                flags=cv2.SOLVEPNP_ITERATIVE 
            )

        if not success: 
            return R_pred, None

        R_opt, _ = cv2.Rodrigues(rvec)
        t_opt = tvec.flatten()

        if not is_rotation_matrix_valid(R_opt):
            return R_pred, None 

        return R_opt, t_opt

    def process_frame(self, img, imu_timestamps_ns, imu_acc, imu_gyro):
        if len(imu_timestamps_ns) > 0:
            current_timestamp_sec = imu_timestamps_ns[-1] * NANOSECONDS_TO_SECONDS
        else: 
            last_known_time_sec = 0
            if self.states: last_known_time_sec = self.states[-1]['timestamp']
            elif self.finalized_trajectory: last_known_time_sec = self.finalized_trajectory[-1]['timestamp']
            current_timestamp_sec = last_known_time_sec + (1.0/30.0) 

        self.preint.reset() 
        self.preint.integrate_batch(imu_acc, imu_gyro) 
        
        if not is_rotation_matrix_valid(self.preint.delta_R):
            self.preint.delta_R = np.eye(3)

        pts_prev_matched, pts_cur_matched, current_tracked_pts, current_tracked_ids = self.tracker.track(img)
        undistorted_current_tracked_pts = self.undistort_points(current_tracked_pts)

        if not self.initialized:
            if not self.states: 
                self.tracker.init_frame(img) 
                initial_R = np.eye(3)
                if not is_rotation_matrix_valid(initial_R): 
                    initial_R = SRot.identity().as_matrix() 

                initial_state_data = {
                    'R': initial_R, 't': np.zeros(3), 'v': np.zeros(3),
                    'delta_R': np.eye(3), 'delta_v': np.zeros(3), 'delta_p': np.zeros(3), 
                    'timestamp': current_timestamp_sec 
                }
                self.states.append(initial_state_data)
                return None 
            else: 
                if pts_prev_matched is None or len(pts_prev_matched) < 5 or pts_cur_matched is None or len(pts_cur_matched) < 5: 
                    self.tracker.prev_img = img 
                    return None

                R0_to_R1_imu = self.preint.delta_R 
                if not is_rotation_matrix_valid(R0_to_R1_imu):
                    R0_to_R1_imu = np.eye(3) 
                
                undistorted_pts0_matched = self.undistort_points(pts_prev_matched)
                undistorted_pts1_matched = self.undistort_points(pts_cur_matched)

                if len(undistorted_pts0_matched) < 5 or len(undistorted_pts1_matched) < 5:
                    self.tracker.prev_img = img
                    return None
                
                ids_for_matched_points = current_tracked_ids 

                t_dir_normalized = self.two_point_ransac(undistorted_pts0_matched, undistorted_pts1_matched, R0_to_R1_imu)

                if np.all(t_dir_normalized == 0): 
                    self.tracker.prev_img = img
                    return None
                
                initial_scale = 0.1 
                t1_in_R0_frame = t_dir_normalized * initial_scale

                R0_world = self.states[0]['R']
                R1_world = R0_world @ R0_to_R1_imu 
                if not is_rotation_matrix_valid(R1_world):
                    R1_world = R0_world 
                
                t1_world = self.states[0]['t'] + R0_world @ t1_in_R0_frame 
                v1_world = self.states[0]['v'] + R0_world @ self.preint.delta_v

                new_state_data = {
                    'R': R1_world, 't': t1_world, 'v': v1_world,
                    'delta_R': self.preint.delta_R.copy(), 
                    'delta_v': self.preint.delta_v.copy(), 
                    'delta_p': self.preint.delta_p.copy(),
                    'timestamp': current_timestamp_sec 
                }
                self.states.append(new_state_data)
                
                self.triangulate_landmarks(undistorted_pts0_matched, undistorted_pts1_matched, 
                                           R0_to_R1_imu, t1_in_R0_frame, 
                                           ids_for_matched_points) 
                
                self.initialized = True
                print(f"System Initialized at frame {len(self.finalized_trajectory) + len(self.states)-1}. Performing initial BA and Align.")
                if len(self.states) >= 2: 
                    pass
                if len(self.states) >= 4: 
                    print("Initial VA-Align...")
                    self.va_align() 
                print("Initialization complete.")

        else: 
            prev_state = self.states[-1]
            if not is_rotation_matrix_valid(prev_state['R']):
                return None

            dt_interval_sec = current_timestamp_sec - prev_state['timestamp'] 
            if dt_interval_sec <= 1e-9 : dt_interval_sec = self.imu_dt 

            R_pred_world = prev_state['R'] @ self.preint.delta_R
            if not is_rotation_matrix_valid(R_pred_world):
                R_pred_world = prev_state['R'] 

            t_pred_world = prev_state['t'] + prev_state['v'] * dt_interval_sec + \
                       0.5 * self.gravity * dt_interval_sec**2 + \
                       prev_state['R'] @ self.preint.delta_p
            v_pred_world = prev_state['v'] + self.gravity * dt_interval_sec + \
                       prev_state['R'] @ self.preint.delta_v

            known_ids_global = list(self.landmarks.keys())
            
            pnp_obj_pts = []
            pnp_img_pts = []
            if current_tracked_ids is not None and undistorted_current_tracked_pts is not None:
                for i, id_val in enumerate(current_tracked_ids):
                    if id_val in known_ids_global: 
                        pnp_obj_pts.append(self.landmarks[id_val])
                        pnp_img_pts.append(undistorted_current_tracked_pts[i])
            
            pnp_obj_pts = np.array(pnp_obj_pts)
            pnp_img_pts = np.array(pnp_img_pts)

            R_opt_world, t_opt_world_raw = R_pred_world, None 
            if len(pnp_obj_pts) >= 4:
                R_pnp, t_pnp = self.vg_pnp_solve(pnp_obj_pts, pnp_img_pts, R_pred_world, self.K, self.dist)
                if t_pnp is not None and is_rotation_matrix_valid(R_pnp): 
                    R_opt_world = R_pnp
                    t_opt_world_raw = t_pnp
            
            if t_opt_world_raw is None:
                t_opt_world = t_pred_world
            else:
                t_opt_world = t_opt_world_raw
            
            v_opt_world = v_pred_world 

            new_state_data = { 
                'R': R_opt_world, 't': t_opt_world, 'v': v_opt_world,
                'delta_R': self.preint.delta_R.copy(), 
                'delta_v': self.preint.delta_v.copy(), 
                'delta_p': self.preint.delta_p.copy(),
                'timestamp': current_timestamp_sec 
            }
            self.states.append(new_state_data)

            if len(self.states) > self.window:
                finalized_state_for_history = self.states.pop(0) 
                self.finalized_trajectory.append(finalized_state_for_history) 

            if self.initialized and len(self.states) == self.window: 
                # print("VG-BA START (Tracking)")
                # self.vg_ba_g2o() 
                
                print("VA-Align START (Tracking)")
                self.va_align() 
                
                print("VI-BA START (Tracking)")
                self.vi_ba_g2o() 
        
        return self.states[-1] if self.states else None


    def two_point_ransac(self, pts0_norm, pts1_norm, R_imu, thresh=0.01, iters=100): 
        if len(pts0_norm) < 2 or len(pts1_norm) < 2:
            return np.zeros(3)
        
        p0_unit_vecs = np.hstack((pts0_norm, np.ones((len(pts0_norm), 1))))
        p1_unit_vecs = np.hstack((pts1_norm, np.ones((len(pts1_norm), 1))))
        for i in range(len(p0_unit_vecs)): p0_unit_vecs[i] /= np.linalg.norm(p0_unit_vecs[i])
        for i in range(len(p1_unit_vecs)): p1_unit_vecs[i] /= np.linalg.norm(p1_unit_vecs[i])

        best_t_dir, best_inliers_count = np.zeros(3), 0
        
        for _i in range(iters):
            idx = np.random.choice(len(p0_unit_vecs), 2, replace=False)
            
            n0 = np.cross(p1_unit_vecs[idx[0]], R_imu @ p0_unit_vecs[idx[0]])
            n1 = np.cross(p1_unit_vecs[idx[1]], R_imu @ p0_unit_vecs[idx[1]])
            
            A_ransac_mat = np.array([n0, n1])
            if A_ransac_mat.shape[0] < 2 : continue

            _u, _s, vt = np.linalg.svd(A_ransac_mat)
            t_candidate = vt[-1]
            if np.linalg.norm(t_candidate) < 1e-6: continue 
            t_candidate /= np.linalg.norm(t_candidate) 

            inliers_count = 0
            for k_pt in range(len(p0_unit_vecs)):
                error = abs(np.dot(np.cross(p1_unit_vecs[k_pt], R_imu @ p0_unit_vecs[k_pt]), t_candidate))
                if error < thresh:
                    inliers_count += 1
            
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_t_dir = t_candidate
        
        return best_t_dir


    def triangulate_landmarks(self, pts0, pts1, R_C0_to_C1, t_C1_in_C0, point_ids):
        if pts0 is None or pts1 is None or len(pts0) == 0 or len(pts1) == 0 or point_ids is None:
            return
        if len(pts0) != len(pts1) or len(pts0) != len(point_ids):
            return

        P0 = self.K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P1 = self.K @ np.hstack((R_C0_to_C1, t_C1_in_C0.reshape(3,1)))

        pts0_reshaped = pts0.reshape(-1, 2).T 
        pts1_reshaped = pts1.reshape(-1, 2).T 

        if pts0_reshaped.shape[1] == 0:
            return

        points_4d_hom = cv2.triangulatePoints(P0, P1, pts0_reshaped, pts1_reshaped)
        valid_mask = points_4d_hom[3] > 1e-5 
        
        points_4d_hom_valid = points_4d_hom[:, valid_mask]
        point_ids_valid = point_ids[valid_mask]

        if points_4d_hom_valid.shape[1] == 0:
            return

        points_3d_valid = points_4d_hom_valid[:3] / points_4d_hom_valid[3]

        for i in range(points_3d_valid.shape[1]):
            pt_id = point_ids_valid[i] 
            self.landmarks[pt_id] = points_3d_valid[:, i].flatten()
    
    def va_align(self):
        num_states_in_window = len(self.states)
        if num_states_in_window < 4: 
            print(f"VA-Align: Not enough states ({num_states_in_window}). Need at least 4. Skipping.")
            return

        N = num_states_in_window - 1 
        n_unknowns = 1 + 3 * (N + 1) + 3 
        n_eqs = 6 * N 

        if n_eqs == 0: return 

        A = lil_matrix((n_eqs, n_unknowns), dtype=float)
        b_vec = np.zeros(n_eqs, dtype=float)
        
        # print_va_align_debug = True 
        # va_align_debug_count = 0
        # max_va_align_debug_prints = 3

        all_inputs_finite = True
        for k in range(N): 
            state_k = self.states[k]
            state_k1 = self.states[k+1]

            dt_k_sec = state_k1['timestamp'] - state_k['timestamp'] 
            if dt_k_sec <= 1e-9: 
                dt_k_sec = self.imu_dt 
            if not np.isfinite(dt_k_sec):
                print(f"VA-Align Error: dt_k_sec for interval {k} is not finite ({dt_k_sec}). Using self.imu_dt.")
                dt_k_sec = self.imu_dt
                all_inputs_finite = False


            R_k_world = state_k['R']
            delta_p_bk_bk1 = state_k1['delta_p'] 
            delta_v_bk_bk1 = state_k1['delta_v'] 
            
            visual_t_diff = state_k1['t'] - state_k['t']

            # Check for non-finite inputs before building A and b
            if not (np.isfinite(visual_t_diff).all() and \
                    is_rotation_matrix_valid(R_k_world) and \
                    np.isfinite(delta_p_bk_bk1).all() and \
                    np.isfinite(delta_v_bk_bk1).all()):
                print(f"VA-Align Error: Non-finite input detected for interval k={k}. Skipping this interval's equations.")
                all_inputs_finite = False
                # One option is to skip adding rows for this k, making the system smaller.
                # Another is to zero out the rows, which might still lead to issues if other parts are huge.
                # For now, we'll mark that inputs are not all finite and proceed, lsqr might still fail.
                # A more robust solution would be to not add these rows to A and b.
                # However, lil_matrix makes selective row skipping tricky without rebuilding.
                # Let's just flag and proceed for now, relying on lsqr's behavior with potential NaNs from bad rows.
                # If an input is NaN, the corresponding A or b element will be NaN.

            # --- Start VA-Align Debug Prints ---
            # if print_va_align_debug and va_align_debug_count < max_va_align_debug_prints:
            #     print(f"--- VA-Align Debug: Interval k={k} ---")
            #     print(f"  dt_k_sec: {dt_k_sec:.4f}")
            #     print(f"  Visual t_diff: {visual_t_diff}")
            #     print(f"  R_k_world (det: {np.linalg.det(R_k_world):.2f})") # Removed matrix print
            #     print(f"  delta_p_bk_bk1: {delta_p_bk_bk1}")
            #     print(f"  delta_v_bk_bk1: {delta_v_bk_bk1}")
            # --- End VA-Align Debug Prints ---

            row_p_start = 6 * k
            A[row_p_start:row_p_start+3, 0] = visual_t_diff
            A[row_p_start:row_p_start+3, 1 + 3*k : 1 + 3*k+3] = -dt_k_sec * np.eye(3)
            A[row_p_start:row_p_start+3, 1 + 3*(N+1) : 1 + 3*(N+1)+3] = -0.5 * (dt_k_sec**2) * np.eye(3) 
            b_vec[row_p_start:row_p_start+3] = R_k_world @ delta_p_bk_bk1
            
            row_v_start = 6 * k + 3
            A[row_v_start:row_v_start+3, 1 + 3*(k+1) : 1 + 3*(k+1)+3] = np.eye(3)
            A[row_v_start:row_v_start+3, 1 + 3*k : 1 + 3*k+3] = -np.eye(3)
            A[row_v_start:row_v_start+3, 1 + 3*(N+1) : 1 + 3*(N+1)+3] = -dt_k_sec * np.eye(3) 
            b_vec[row_v_start:row_v_start+3] = R_k_world @ delta_v_bk_bk1
        
        # if print_va_align_debug and va_align_debug_count < max_va_align_debug_prints:
        #     va_align_debug_count +=1

        # Check A and b_vec for NaNs before calling lsqr
        A_csr = A.tocsr()
        if not np.isfinite(A_csr.data).all() or not np.isfinite(A_csr.indices).all() or not np.isfinite(A_csr.indptr).all():
            print("VA-Align FATAL: Matrix A contains non-finite values before lsqr! Skipping va_align.")
            return
        if not np.isfinite(b_vec).all():
            print("VA-Align FATAL: Vector b_vec contains non-finite values before lsqr! Skipping va_align.")
            return

        try:
            # Added damp parameter to lsqr for regularization
            sol = lsqr(A_csr, b_vec, damp=1e-4, atol=1e-7, btol=1e-7, iter_lim=max(500, A_csr.shape[1]*3), show=False)[0] 
        except Exception as e:
            print(f"VA Align lsqr failed: {e}")
            return

        s_est  = sol[0]                              
        v_list_world = [ sol[1+3*k : 1+3*k+3] for k in range(N+1) ] 
        g_est_world  = sol[1+3*(N+1) : 1+3*(N+1)+3]                     

        print(f"VA-Align estimated: s_raw={s_est:.4f}, g_norm={np.linalg.norm(g_est_world):.4f}, g_vec={g_est_world}")

        if not np.isfinite(s_est) or not np.isfinite(g_est_world).all():
            print("VA-Align: Solution contains NaN/Inf. Skipping update.")
            self.logs.setdefault('imu', []).append({'scale': None, 'gravity': None, 's_raw': s_est})
            return

        if not (8.0 < np.linalg.norm(g_est_world) < 12.0):
            print(f"VA-Align: Estimated gravity norm {np.linalg.norm(g_est_world):.2f} is unusual. Skipping update.")
            self.logs.setdefault('imu', []).append({'scale': None, 'gravity': g_est_world.copy(), 's_raw': s_est})
            return

        min_s, max_s = 0.1, 10.0  
        s_clamped = float(np.clip(s_est, min_s, max_s))
        
        final_s = s_clamped
        print(f"VA-Align: s_est={s_est:.4f} -> final_s={final_s:.4f}")


        if final_s < min_s + 1e-3 : 
            print(f"VA-Align: Estimated scale {final_s:.4f} is at lower bound. Solution might be unreliable.")

        t_origin_window = self.states[0]['t'].copy() 
        for i in range(num_states_in_window): 
            self.states[i]['t'] = t_origin_window + (self.states[i]['t'] - t_origin_window) * final_s
            self.states[i]['v'] = v_list_world[i] 
            
        self.gravity = g_est_world 

        self.logs.setdefault('imu', []).append({
            'scale': final_s, 
            'gravity': g_est_world.copy(),
            's_raw': s_est
        })
    
    def get_g2o_solver(self):
        try:
            linear_solver_type = g2o.LinearSolverCholmodSE3()
            linear_solver_type.set_block_ordering(False)
        except AttributeError:
            try:
                linear_solver_type = g2o.LinearSolverCSparseSE3()
                linear_solver_type.set_block_ordering(False)
            except AttributeError:
                linear_solver_type = g2o.LinearSolverDenseSE3()
        return BlockSolverSE3(linear_solver_type)


    def vg_ba_g2o(self): 
        if len(self.states) < 2: return 

        opt = g2o.SparseOptimizer()
        block_solver = self.get_g2o_solver()
        algo = OptimizationAlgorithmLevenberg(block_solver)
        opt.set_algorithm(algo)
        opt.set_verbose(False)

        focal_length = (self.K[0,0] + self.K[1,1]) / 2.0 
        principal_point = (self.K[0,2], self.K[1,2])
        cam_params = CameraParameters(focal_length, principal_point, 0) 
        cam_params.set_id(0)
        opt.add_parameter(cam_params)

        pose_vertex_map = {} 
        for i, state_data in enumerate(self.states):
            R_i = state_data['R']
            t_i = state_data['t'].reshape(3,1) 
            se3 = SE3Quat(R_i, t_i)
            
            v_se3 = VertexSE3Expmap()
            v_se3.set_id(i) 
            pose_vertex_map[i] = i 

            v_se3.set_estimate(se3)
            v_se3.set_fixed(i == 0) 
            opt.add_vertex(v_se3)

        landmark_g2o_id_offset = len(self.states) 
        landmark_vertex_map = {} 
        
        current_g2o_landmark_id = landmark_g2o_id_offset
        active_landmarks_in_window = {} 

        if self.tracker.prev_pts is not None and self.tracker.prev_ids is not None and len(self.states)>0:
            latest_pose_idx_in_window = len(self.states) - 1
            g2o_pose_id_latest = pose_vertex_map[latest_pose_idx_in_window]
            
            for idx_obs, lm_id_global in enumerate(self.tracker.prev_ids):
                if lm_id_global not in self.landmarks: continue 

                P3_world = self.landmarks[lm_id_global]
                uv_obs = self.tracker.prev_pts[idx_obs]

                g2o_landmark_id = landmark_vertex_map.get(lm_id_global)
                if g2o_landmark_id is None:
                    vp = VertexPointXYZ()
                    g2o_landmark_id = current_g2o_landmark_id
                    vp.set_id(g2o_landmark_id)
                    vp.set_estimate(P3_world) 
                    vp.set_marginalized(True) 
                    opt.add_vertex(vp)
                    landmark_vertex_map[lm_id_global] = g2o_landmark_id
                    current_g2o_landmark_id += 1
                
                active_landmarks_in_window[lm_id_global] = g2o_landmark_id
                edge = EdgeProjectXYZ2UV()
                edge.set_vertex(0, opt.vertex(g2o_landmark_id)) 
                edge.set_vertex(1, opt.vertex(g2o_pose_id_latest))   
                edge.set_measurement(uv_obs)
                edge.set_information(np.eye(2)) 
                edge.set_parameter_id(0,0) 
                opt.add_edge(edge)

        for i in range(len(self.states) - 1):
            pose0_idx_in_window = i
            pose1_idx_in_window = i + 1

            g2o_pose0_id = pose_vertex_map[pose0_idx_in_window]
            g2o_pose1_id = pose_vertex_map[pose1_idx_in_window]

            preint_data = self.states[pose1_idx_in_window]
            R_meas = preint_data['delta_R'] 

            edge = EdgeSE3()
            edge.set_vertex(0, opt.vertex(g2o_pose0_id))
            edge.set_vertex(1, opt.vertex(g2o_pose1_id))
            
            meas_isometry = Isometry3d(R_meas, np.zeros(3)) 
            edge.set_measurement(meas_isometry)
            
            info = np.eye(6)
            info[:3,:3] *= 1000.0 
            info[3:,3:] *= 1.0   
            edge.set_information(info)
            opt.add_edge(edge)
        
        if not opt.edges(): 
            return

        try:
            opt.initialize_optimization()
            opt.optimize(5) 

            for i, state_data in enumerate(self.states):
                g2o_pose_id = pose_vertex_map[i]
                vertex = opt.vertex(g2o_pose_id)
                if vertex is None: continue 
                
                est_se3 = vertex.estimate()
                R_opt_ba = est_se3.rotation().matrix()
                t_opt_ba = est_se3.translation().flatten()

                if is_rotation_matrix_valid(R_opt_ba):
                    self.states[i]['R'] = R_opt_ba
                    self.states[i]['t'] = t_opt_ba
            
            for lm_id_global, g2o_lm_id in active_landmarks_in_window.items():
               vertex = opt.vertex(g2o_lm_id)
               if vertex: 
                   self.landmarks[lm_id_global] = vertex.estimate()

        except Exception as e:
            pass


    def vi_ba_g2o(self): 
        if len(self.states) < 2: return

        optimizer = SparseOptimizer()
        block_solver = self.get_g2o_solver()
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
            v_se3.set_id(i) 
            pose_vertex_map[i] = i
            v_se3.set_estimate(se3)
            v_se3.set_fixed(i == 0)
            optimizer.add_vertex(v_se3)

        landmark_g2o_id_offset = len(self.states)
        landmark_vertex_map = {}
        current_g2o_landmark_id = landmark_g2o_id_offset
        active_landmarks_in_window = {}

        if self.tracker.prev_pts is not None and self.tracker.prev_ids is not None and len(self.states)>0:
            latest_pose_idx_in_window = len(self.states) - 1
            g2o_pose_id_latest = pose_vertex_map[latest_pose_idx_in_window]
            
            for idx_obs, lm_id_global in enumerate(self.tracker.prev_ids):
                if lm_id_global not in self.landmarks: continue

                P3_world = self.landmarks[lm_id_global]
                uv_obs = self.tracker.prev_pts[idx_obs]

                g2o_landmark_id = landmark_vertex_map.get(lm_id_global)
                if g2o_landmark_id is None:
                    vp = VertexPointXYZ()
                    g2o_landmark_id = current_g2o_landmark_id
                    vp.set_id(g2o_landmark_id)
                    vp.set_estimate(P3_world)
                    vp.set_marginalized(True)
                    optimizer.add_vertex(vp)
                    landmark_vertex_map[lm_id_global] = g2o_landmark_id
                    current_g2o_landmark_id += 1
                
                active_landmarks_in_window[lm_id_global] = g2o_landmark_id
                edge = EdgeProjectXYZ2UV()
                edge.set_vertex(0, optimizer.vertex(g2o_landmark_id))
                edge.set_vertex(1, optimizer.vertex(g2o_pose_id_latest))
                edge.set_measurement(uv_obs)
                edge.set_information(np.eye(2))
                edge.set_parameter_id(0,0)
                optimizer.add_edge(edge)

        for i in range(len(self.states) - 1):
            pose0_idx_in_window = i
            pose1_idx_in_window = i + 1

            g2o_pose0_id = pose_vertex_map[pose0_idx_in_window]
            g2o_pose1_id = pose_vertex_map[pose1_idx_in_window]

            preint_data = self.states[pose1_idx_in_window]
            delta_R_meas = preint_data['delta_R'] 
            delta_p_meas = preint_data['delta_p'] 

            edge = EdgeSE3()
            edge.set_vertex(0, optimizer.vertex(g2o_pose0_id)) 
            edge.set_vertex(1, optimizer.vertex(g2o_pose1_id)) 
            
            meas_isometry = Isometry3d(delta_R_meas, delta_p_meas)
            edge.set_measurement(meas_isometry)
            
            info = np.eye(6) 
            info[:3,:3] *= 500.0 
            info[3:,3:] *= 200.0  
            edge.set_information(info)
            optimizer.add_edge(edge)
        
        if not optimizer.edges():
            return

        try:
            optimizer.initialize_optimization()
            optimizer.optimize(10) 

            for i, state_data in enumerate(self.states):
                g2o_pose_id = pose_vertex_map[i]
                vertex = optimizer.vertex(g2o_pose_id)
                if vertex is None: continue

                se3_opt = vertex.estimate()
                
                q_g2o = se3_opt.rotation()
                q_scipy = np.array([q_g2o.x(), q_g2o.y(), q_g2o.z(), q_g2o.w()])
                
                R_opt_ba = SRot.from_quat(q_scipy).as_matrix()
                t_opt_ba = se3_opt.translation().flatten()

                if is_rotation_matrix_valid(R_opt_ba):
                    self.states[i]['R'] = R_opt_ba
                    self.states[i]['t'] = t_opt_ba

            for lm_id_global, g2o_lm_id in active_landmarks_in_window.items():
                 vertex = optimizer.vertex(g2o_lm_id)
                 if vertex: 
                    self.landmarks[lm_id_global] = vertex.estimate()
            
            if self.logs['reproj'] is not None and optimizer.chi2() is not None: 
                self.logs['reproj'].append(optimizer.chi2())

        except Exception as e:
            pass


    def reset(self):
        self.preint.reset()
        self.tracker = FeatureTracker() 
        self.states.clear()
        self.finalized_trajectory.clear() 
        self.landmarks.clear()
        self.initialized = False
        self.logs = {'reproj': [], 'imu': []} 


    def __call__(self, tstamp_sec, input_tensor, intrinsics, # tstamp is now expected in seconds
                 curr_imu_data = None, save_slam_steps_path = None):
        
        # curr_imu_data is (imu_timestamps_ns, imu_acc, imu_gyro)
        # tstamp_sec is the image timestamp in seconds

        if self.K is None or np.all(self.K == np.eye(3)): 
            fx, fy, cx, cy = intrinsics.cpu().numpy()
            self.K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

        if len(input_tensor) < 3:
            __events, images = input_tensor 
        elif len(input_tensor) == 3:
            __events, images, _mask = input_tensor
        else:
            raise Exception("Wrong input tensor")
        
        img_squeezed = images.squeeze(0).squeeze(0).permute(1,2,0)
        if img_squeezed.shape[2] == 3: 
            img_np = img_squeezed.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif img_squeezed.shape[2] == 1: 
            img_np = img_squeezed.cpu().numpy()
            img_gray = (img_np * 255).astype(np.uint8)
        else:
            raise Exception(f"Unsupported image channel count: {img_squeezed.shape[2]}")

        imu_timestamps_ns, imu_acc, imu_gyro = curr_imu_data
        # print()
        # print(imu_timestamps_ns[0], "->", imu_timestamps_ns[-1])
        # print(tstamp_sec)
        # print("TIMES")
        # Pass imu_timestamps_ns directly, process_frame will handle conversion for its internal current_timestamp_sec
        _state = self.process_frame(img_gray, imu_timestamps_ns, imu_acc, imu_gyro)  

    
    def terminate(self):
        all_history_for_output = []
        all_history_for_output.extend(self.finalized_trajectory)
        
        for state_in_window in self.states:
            all_history_for_output.append({
                'R': state_in_window['R'],
                't': state_in_window['t'],
                'timestamp': state_in_window['timestamp'] # Already in seconds
            })

        if not all_history_for_output:
            dummy_pose = np.array([[0,0,0, 0,0,0,1]], dtype=float) 
            dummy_tstamp = np.array([0.0], dtype=float)
            return dummy_pose, dummy_tstamp
        
        valid_poses_list = []
        valid_tstamps_list = []

        for state_data in all_history_for_output:
            R_val = state_data['R']
            t_val = state_data['t']
            ts_val = state_data['timestamp'] # In seconds

            if is_rotation_matrix_valid(R_val) and \
               isinstance(t_val, np.ndarray) and np.isfinite(t_val).all():
                try:
                    rot_q = SRot.from_matrix(R_val).as_quat() 
                    current_pose = np.concatenate((t_val, rot_q))
                    valid_poses_list.append(current_pose)
                    valid_tstamps_list.append(ts_val)
                except np.linalg.LinAlgError:
                    pass 
                except ValueError as ve: 
                    pass
            pass

        if not valid_poses_list:
            dummy_pose = np.array([[0,0,0, 0,0,0,1]], dtype=float) 
            dummy_tstamp = np.array([0.0], dtype=float)
            return dummy_pose, dummy_tstamp

        poses_out = np.array(valid_poses_list)
        tstamps_out = np.array(valid_tstamps_list, dtype=float)*1e9 #To output in nanoseconds
        
        if len(tstamps_out) > 1:
            sorted_indices = np.argsort(tstamps_out)
            tstamps_out = tstamps_out[sorted_indices]
            poses_out = poses_out[sorted_indices]
            
        return poses_out, tstamps_out