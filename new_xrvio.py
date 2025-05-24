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
    LinearSolverDenseSE3,
    BlockSolverSE3,
    OptimizationAlgorithmLevenberg,
    SE3Quat,
    VertexSE3Expmap,
    VertexPointXYZ,
    EdgeProjectXYZ2UV,
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
        self.reset()

    def reset(self):
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)

    def integrate_batch(self, acc_batch, gyro_batch):
        for acc, gyro in zip(acc_batch, gyro_batch):
            dR = R.from_rotvec(gyro * self.dt).as_matrix()
            self.delta_R = self.delta_R.dot(dR)
            acc_world = self.delta_R.dot(acc)
            self.delta_p += self.delta_v * self.dt + 0.5 * acc_world * self.dt**2
            self.delta_v += acc_world * self.dt


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
        self.prev_ids = np.arange(len(pts)) + self.next_id
        self.next_id += len(pts)
        self.prev_img = img

    def track(self, img):
        """
        Tracks self.prev_pts→current points, maintains IDs.
        Returns:
          - pts_prev (K×2), pts_cur (K×2) for two-view init
          - cur_pts (M×2), cur_ids (M,)       for PnP in sliding-window
        """
        if self.prev_pts is None or self.prev_img is None:
            # nothing to track
            self.prev_img = img
            return None, None, None, None

        # calc optical flow
        pts_cur_all, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_img, img, self.prev_pts, None, **self.lk_params
        )
        mask = status.flatten() == 1

        # matched prev→cur for two-view
        pts_prev = self.prev_pts[mask]
        pts_cur  = pts_cur_all[mask]

        # filter cur for next PnP
        cur_ids  = self.prev_ids[mask]
        cur_pts  = pts_cur.copy()

        # now detect brand-new features if you want (optional)

        # update history
        self.prev_pts = cur_pts
        self.prev_ids = cur_ids
        self.prev_img = img

        return pts_prev, pts_cur, cur_pts, cur_ids



class XRVIO:
    def __init__(self, dt, window=10, cam_config = None):
        if cam_config is not None:
            self.K, self.dist = load_camera_params(cam_config)
        else:
            self.K, self.dist = None, None
        self.dt = dt
        self.preint = IMUPreintegrator(dt)
        self.tracker = FeatureTracker()
        self.states = []         # each: {'R','t','v'}
        self.landmarks = {}      # id->3D
        self.imu_slices = []     # list of (imu_t, acc, gyro) per frame
        self.initialized = False
        self.window = window
        self.logs = {'reproj': [], 'imu': []}

        self.tlist = []

    def undistort_points(self, pts):
        # If no distortion, return pts unchanged
        if self.dist is None:
            return pts.copy()
        pts_n = cv2.undistortPoints(pts.reshape(-1,1,2), self.K, self.dist)
        return pts_n.reshape(-1,2)
    
    def __vg_pnp_solve(self, pts_3d, pts_2d, R_pred, K, dist_coeffs=None):
        """
        Visual-Gyro PnP: solve for pose given 3D-2D correspondences and gyro-based prior rotation.
        Inputs:
        - pts_3d: (N,3) array of 3D landmark positions in visual frame
        - pts_2d: (N,2) array of corresponding image points
        - R_pred: (3,3) numpy array, predicted rotation from IMU preintegration
        - K:    (3,3) camera intrinsic matrix
        - dist_coeffs: (k,) distortion coefficients or None
        Returns:
        - R_opt: (3,3) optimized rotation matrix
        - t_opt: (3,) optimized translation vector
        """
        # 1. Convert predicted rotation to Rodrigues vector
        rvec_prior, _ = cv2.Rodrigues(R_pred)
        # 2. Prepare inputs for solvePnP
        object_points = pts_3d.astype(np.float64)
        image_points  = pts_2d.astype(np.float64)
        # 3. Run solvePnP with prior as initial guess
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist_coeffs if dist_coeffs is not None else np.zeros((4,1)),
            rvec_prior,
            None,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            # fallback: try without prior
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist_coeffs if dist_coeffs is not None else np.zeros((4,1)),
                None,
                None,
                False,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        # 4. Convert back to rotation matrix
        R_opt, _ = cv2.Rodrigues(rvec)
        t_opt = tvec.flatten()
        return R_opt, t_opt
    
    def vg_pnp_solve(self, pts_3d, pts_2d, R_pred, K, dist_coeffs=None):
        """
        Visual-Gyro PnP: solve for pose given 3D-2D correspondences and gyro-based prior rotation.
        """
        # 1) Rodrigues init from R_pred
        rvec_prior, _ = cv2.Rodrigues(R_pred.astype(np.float64))
        # 2) Prepare inputs
        obj = pts_3d.astype(np.float64)
        img = pts_2d.astype(np.float64)
        dc  = dist_coeffs if dist_coeffs is not None else np.zeros((4,1))
        # 3) solvePnP
        success, rvec, tvec = cv2.solvePnP(
            obj, img, K, dc,
            rvec_prior, None, True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                obj, img, K, dc,
                None, None, False,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        # 4) back to matrix
        R_opt, _ = cv2.Rodrigues(rvec)
        t_opt = tvec.flatten()
        return R_opt, t_opt

    def process_frame(self, img, imu_t, imu_acc, imu_gyro):
        # Store IMU slice for BA
        self.imu_slices.append((imu_t, imu_acc, imu_gyro))
        self.preint.reset()
        self.preint.integrate_batch(imu_acc, imu_gyro)

        pts_prev, pts_cur, pts2d, ids2d = self.tracker.track(img)

        if not self.initialized:
            if not hasattr(self, 'first_img'):
                # First frame
                self.first_img = img
                self.first_preint = (self.preint.delta_R.copy(),
                                     self.preint.delta_v.copy(),
                                     self.preint.delta_p.copy())
                self.tracker.init_frame(img)
                return None
            else:
                # Second frame: VG-SfM init
                R0, v0, p0 = self.first_preint

                # pts0, _, _, _ = self.tracker.track(self.first_img)
                # _, pts1, _, _ = self.tracker.track(img)

                pts0, pts1  = pts_prev, pts_cur

                t_dir = self.two_point_ransac(pts0, pts1, R0)
                self.states.append({'R':R0, 't':t_dir, 'v':v0})
                self.triangulate_landmarks(pts0, pts1, R0, t_dir)
                self.tracker.init_frame(img)
                self.initialized = True

                return self.states[-1]
        else:
            prev = self.states[-1]
            R_prev = prev['R']      
            R_pred = R_prev.dot(self.preint.delta_R)

            # 1) Prepare time-based prediction
            dt = imu_t[-1] - self.imu_slices[-2][0][-1]
            t_pred = prev['t'] + prev['v'] * dt + self.preint.delta_p

            # 2) VG-PnP solve for pose if enough correspondences, else fallback
            # Grab only those landmarks that we have 3D for:
            known_ids = [id for id in self.landmarks.keys()]
            valid = [i for i in ids2d if i in known_ids]
            if len(valid) >= 4:
                pts3d = np.stack([self.landmarks[i] for i in valid])
                idxs = [np.where(ids2d == i)[0][0] for i in valid]
                pts2d = pts_prev[idxs]
                R_opt, t_opt = self.vg_pnp_solve(pts3d, pts2d, R_pred, self.K, self.dist)
            else:
                R_opt, t_opt = R_pred, t_pred

            # 3) Form new state
            new_state = {
                'R': R_opt,
                't': t_opt,
                'v': prev['v'] + self.preint.delta_v
            }
            self.states.append(new_state)
            self.tracker.init_frame(img)

            # 4) VA-Align and VI-BA
            self.va_align()
            print("BA START")
            self.vi_ba_g2o()
            print("BA END")

            return new_state

    def two_point_ransac(self, pts0, pts1, R_imu, thresh=0.005, iters=100):
        if len(pts0)<2:
            return np.zeros(3)
        p0 = self.undistort_points(pts0) #cv2.undistortPoints(pts0.reshape(-1,1,2), self.K, self.dist).reshape(-1,2)
        p1 = self.undistort_points(pts1) #cv2.undistortPoints(pts1.reshape(-1,1,2), self.K, self.dist).reshape(-1,2)
        best_t, best_in = np.zeros(3), []
        for _i in range(iters):
            # setting len(p1)
            p_len = len(p1)
            # p_len = len(p0) #out fo bounds
            idx = np.random.choice(p_len,2,replace=False)
            A = [np.cross(np.hstack([p1[k],1]), R_imu.dot(np.hstack([p0[k],1]))) for k in idx]
            _,_,Vt = np.linalg.svd(np.vstack(A))
            t = Vt[-1]/np.linalg.norm(Vt[-1])
            errs = [abs(np.dot(np.cross(np.hstack([p1[k],1]),
                                       R_imu.dot(np.hstack([p0[k],1]))), t))
                    for k in range(p_len)]
            inliers = np.where(np.array(errs)<thresh)[0]
            if len(inliers)>len(best_in):
                best_in, best_t = inliers, t
        print("xrvio_two_point: DONE", )
        # Log reprojection residual
        self.logs['reproj'].append(np.mean([errs[i] for i in best_in]))
        return best_t

    def triangulate_landmarks(self, pts0, pts1, R0, t0):
        P0 = self.K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
        P1 = self.K.dot(np.hstack((R0, t0.reshape(3,1))))
        for i,(u0,u1) in enumerate(zip(pts0, pts1)):
            A = np.vstack([u0[0]*P0[2]-P0[0], u0[1]*P0[2]-P0[1],
                           u1[0]*P1[2]-P1[0], u1[1]*P1[2]-P1[1]])
            _,_,Vt = np.linalg.svd(A)
            X = Vt[-1]
            X/=X[3]
            self.landmarks[i] = X[:3]

    def __degenerate_va_align(self):
        print("VA Align START")
        # Loosely-coupled scale estimation using imu_slices
        N = min(self.window, len(self.states)-1)
        num, den = 0.0, 0.0
        for i in range(-N-1, -1):
            si = self.states[i]
            sj = self.states[i+1]
            imu_t, acc, gyro = self.imu_slices[i+1]
            pre = IMUPreintegrator(self.dt)
            pre.integrate_batch(acc, gyro)
            dp = pre.delta_p
            dt = imu_t[-1]-imu_t[0]
            num += dp.dot(sj['t']-si['t'])
            den += dp.dot(dp)
        scale = num/den if den>0 else 1.0
        for st in self.states:
            st['t'] *= scale
        self.logs['imu'].append(scale)
        print("VA Align END")

    def __va_align(self):
        """
        Loosely-coupled scale estimation using IMU preintegrations.
        Improved robustness via per-segment scale and median filtering.
        """
        N = min(self.window, len(self.states)-1)
        scales = []
        # collect individual scale candidates
        for i in range(-N-1, -1):
            si = self.states[i]
            sj = self.states[i+1]
            imu_t, acc, gyro = self.imu_slices[i+1]
            pre = IMUPreintegrator(self.dt)
            pre.integrate_batch(acc, gyro)
            dp = pre.delta_p
            dx = sj['t'] - si['t']
            denom = float(dp.dot(dp))
            # skip degenerate segments
            if denom > 1e-6 and np.isfinite(denom):
                num = float(dp.dot(dx))
                scales.append(num / denom)
        # choose robust scale (median) or default to 1
        if scales:
            scale = float(np.median(scales))
            # clamp scale to reasonable range
            scale = np.clip(scale, 0.1, 10.0)
        else:
            scale = 1.0
        # apply scale to all states' positions
        for st in self.states:
            if np.all(np.isfinite(st['t'])):
                st['t'] = st['t'] * scale
        # log the robust scale
        self.logs['imu'].append(scale)
        print(f"VA Align scale: {scale:.4f}")
    
    def va_align(self):
        """
        Jointly estimate scale s, per-frame velocities v_k, and gravity g
        by aligning the up-to-scale visual trajectory to IMU preintegrations.
        """
        # Number of intervals in the current window
        N = min(self.window, len(self.states) - 1)
        if N <= 0:
            return

        # total unknowns: 3*N velocities + 1 scale + 3 gravity = 3N + 4
        n_unknowns = 3 * N + 4
        # total equations: 3 per interval
        n_eqs = 3 * N

        # build sparse A and dense b
        A = lil_matrix((n_eqs, n_unknowns), dtype=float)
        b = np.zeros(n_eqs, dtype=float)

        # fill row‐blocks for k = 0…N–1
        for k in range(N):
            # visual up-to-scale displacement
            p_k  = self.states[k]['t']
            p_k1 = self.states[k+1]['t']
            delta_v_vis = p_k1 - p_k    # shape (3,)

            # IMU preintegration for interval k
            t_k, acc_k, gyro_k = self.imu_slices[k+1]
            pre = IMUPreintegrator(self.dt)
            pre.integrate_batch(acc_k, gyro_k)
            dp_k = pre.delta_p          # shape (3,)
            dt_k = float(t_k[-1] - t_k[0])

            # which rows in A/b
            row = 3 * k

            # scale column at index 3N
            A[row:row+3, 3*N] = delta_v_vis.reshape(3,1)

            # velocity block for v_k (cols 3k..3k+3)
            A[row:row+3, 3*k:3*k+3] = -dt_k * np.eye(3)

            # gravity columns (3N+1 .. 3N+3)
            A[row:row+3, 3*N+1:3*N+4] = -0.5 * (dt_k**2) * np.eye(3)

            # right‐hand side
            b[row:row+3] = dp_k

        # solve A x ≃ b in the least-squares sense
        sol = lsqr(A.tocsr(), b, atol=1e-6, btol=1e-6)[0]

        # unpack solution
        v_list = [ sol[3*k:3*k+3] for k in range(N) ]  # velocities
        s_est  = sol[3*N]                              # scale
        g_est  = sol[3*N+1:3*N+4]                      # gravity

        # apply scale to all visual positions
        for k in range(N+1):
            self.states[k]['t'] = s_est * self.states[k]['t']

        # overwrite velocities
        for k in range(N):
            self.states[k]['v'] = v_list[k]

        # store gravity (if you need it later)
        self.gravity = g_est

        # log results
        self.logs.setdefault('imu', []).append({
            'scale': s_est, 
            'gravity': g_est.copy()
        })
        print(f"VA Align → scale={s_est:.4f}, gravity={g_est}")

    def vi_ba_g2o(self):
        """
        Perform visual-inertial BA using g2o.
        Expects on self:
        - self.states: list of dicts with 'R' (3x3) and 't' (3,)
        - self.landmarks: dict {lm_id: np.array([x,y,z])}
        - self.K: 3×3 intrinsics
        - self.window: int
        - self.tracker.prev_pts: list of np.array([u,v]) same order as landmarks
        - self.logs: dict for logging
        """
        # 1) build optimizer
        optimizer = SparseOptimizer()
        lin = LinearSolverDenseSE3()
        blk = BlockSolverSE3(lin)
        algo = OptimizationAlgorithmLevenberg(blk)
        optimizer.set_algorithm(algo)
        optimizer.set_verbose(False)

        # 2) camera intrinsics
        f  = float(self.K[0,0])
        pp = np.array([[self.K[0,2]], [self.K[1,2]]])  # (2×1)
        cam = CameraParameters(f, pp, 0.0)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        # 3) sliding window
        start = max(0, len(self.states) - self.window)
        idxs  = list(range(start, len(self.states)))

        # 4) add pose vertices
        for i in idxs:
            R_i = self.states[i]['R']
            t_i = np.array(self.states[i]['t']).reshape(3,1)
            se3 = SE3Quat(R_i, t_i)
            v   = VertexSE3Expmap()
            v.set_id(i*2)
            v.set_estimate(se3)
            v.set_fixed(i == start)
            optimizer.add_vertex(v)

        # 5) add landmark vertices
        for lm_id, P3 in self.landmarks.items():
            vp = VertexPointXYZ()
            vp.set_id(lm_id*2 + 1)
            vp.set_estimate(np.array(P3))
            vp.set_marginalized(True)
            optimizer.add_vertex(vp)

        # 6) add reprojection edges
        for i in idxs:
            pose_id = i*2
            for lm_id, uv in zip(self.landmarks.keys(), self.tracker.prev_pts):
                pt_id = lm_id*2 + 1
                e = EdgeProjectXYZ2UV()
                e.set_vertex(0, optimizer.vertex(pt_id))   # landmark
                e.set_vertex(1, optimizer.vertex(pose_id)) # pose
                e.set_measurement(np.array(uv).reshape(2,))
                e.set_information(np.eye(2))
                e.set_parameter_id(0, 0)  # use cam param #0
                optimizer.add_edge(e)

        # 7) (optional) IMU edges …

        # 8) optimize
        optimizer.initialize_optimization()
        optimizer.optimize(10)

        # 9) unpack poses
        for i in idxs:
            se3_opt = optimizer.vertex(i*2).estimate()
            # extract quaternion
            q = se3_opt.rotation()
            # build numpy rotation matrix via SciPy
            R_opt = SRot.from_quat([q.x(), q.y(), q.z(), q.w()]).as_matrix()
            t_opt = se3_opt.translation().flatten()
            self.states[i]['R'] = R_opt
            self.states[i]['t'] = t_opt

        # 10) unpack landmarks
        for lm_id in self.landmarks.keys():
            xyz_opt = optimizer.vertex(lm_id*2 + 1).estimate()
            self.landmarks[lm_id] = np.array(xyz_opt).flatten()

        # 11) log reprojection (chi2)
        self.logs.setdefault('reproj', []).append(optimizer.chi2())

    def reset(self):
        self.preint.reset()
        self.tracker = FeatureTracker()
        self.states.clear()
        self.landmarks.clear()
        self.imu_slices.clear()
        self.initialized = False

    def __call__(self, tstamp, input_tensor, intrinsics, 
                 curr_imu_data = None, save_slam_steps_path = None):
        
        self.tlist.append(tstamp)

        if self.K is None:
            fx, fy, cx, cy = intrinsics.cpu().numpy()
            self.K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

        """track new frame"""
        if len(input_tensor) < 3:
            __events, images = input_tensor 
        elif len(input_tensor) == 3:
            __events, images, _mask = input_tensor
        else:
            raise Exception("Wrong input tendor")
        

        img_squeezed = images.squeeze(0).squeeze(0).permute(1,2,0)
        assert img_squeezed.shape == (480,640,3), "Image not correct: Shape{img_squeezed.shape}"
        img = img_squeezed.cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        imu_t, imu_acc, imu_gyro = curr_imu_data

        _state = self.process_frame(img, imu_t, imu_acc, imu_gyro)
    
    def terminate(self):
        """interpolate missing poses"""

        poses = np.zeros((len(self.tlist),7))

        for i, state in enumerate(self.states):
            rot_q =  SRot.from_matrix(state['R']).as_quat()
            poses[i] = np.concatenate((state['t'],rot_q))

        tstamps = np.array(self.tlist, dtype=float)

        return poses, tstamps



if __name__=='__main__':
    import glob
    cam_cfg = 'config.yaml'
    vio = XRVIO(cam_cfg, dt=0.005, window=10)
    img_files = sorted(glob.glob('images/*.png'))
    imu_data = np.loadtxt('imu.csv', delimiter=',', skiprows=1)
    imu_t, imu_acc, imu_gyro = imu_data[:,0], imu_data[:,1:4], imu_data[:,4:7]
    frame_times = [0] + [i/30.0 for i in range(1,len(img_files)+1)]
    for idx,fp in enumerate(img_files):
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        mask = (imu_t>=frame_times[idx])&(imu_t<frame_times[idx+1])
        state = vio.process_frame(img, imu_t[mask], imu_acc[mask], imu_gyro[mask])
        if state:
            print(f"Frame{idx}: t={state['t']}, reproj_err={vio.logs['reproj'][-1]:.4f}, imu_scale={vio.logs['imu'][-1]:.4f}")
