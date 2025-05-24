import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares


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
        self.det = cv2.ORB_create(max_features) if detector=='ORB' else cv2.GFTTDetector_create(max_features)
        self.lk_params = dict(winSize=(21,21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
        self.prev_img = None
        self.prev_pts = None

    def detect(self, img):
        kps = self.det.detect(img, None)
        return np.array([kp.pt for kp in kps], dtype=np.float32)

    def init_frame(self, img):
        self.prev_img = img
        self.prev_pts = self.detect(img)

    def track(self, img):
        pts_prev, pts_cur = [], []
        if self.prev_pts is not None and self.prev_img is not None:
            pts_cur_all, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_img, img, self.prev_pts, None, **self.lk_params)
            mask = status.flatten()==1
            pts_prev = self.prev_pts[mask]
            pts_cur = pts_cur_all[mask]
        self.prev_img = img
        self.prev_pts = pts_cur
        return pts_prev, pts_cur


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

    def undistort_points(self, pts):
        # If no distortion, return pts unchanged
        if self.dist is None:
            return pts.copy()
        pts_n = cv2.undistortPoints(pts.reshape(-1,1,2), self.K, self.dist)
        return pts_n.reshape(-1,2)

    def process_frame(self, img, imu_t, imu_acc, imu_gyro):
        # Store IMU slice for BA
        self.imu_slices.append((imu_t, imu_acc, imu_gyro))
        self.preint.reset()
        self.preint.integrate_batch(imu_acc, imu_gyro)

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
                pts0, _ = self.tracker.track(self.first_img)
                _, pts1 = self.tracker.track(img)
                t_dir = self.two_point_ransac(pts0, pts1, R0)
                self.states.append({'R':R0, 't':t_dir, 'v':v0})
                self.triangulate_landmarks(pts0, pts1, R0, t_dir)
                self.tracker.init_frame(img)
                self.initialized = True
                return self.states[-1]
        else:
            prev = self.states[-1]
            # IMU prediction
            R_pred = prev['R'].dot(self.preint.delta_R)
            dt = imu_t[-1] - self.imu_slices[-2][0][-1]
            t_pred = prev['t'] + prev['v'] * dt + self.preint.delta_p
            # Visual correction via 2-point
            pts_prev, pts_cur = self.tracker.track(img)
            t_dir = self.two_point_ransac(pts_prev, pts_cur, R_pred)
            new_state = {'R':R_pred,
                         't':t_pred + t_dir,
                         'v':prev['v'] + self.preint.delta_v}
            self.states.append(new_state)
            self.tracker.init_frame(img)
            # VA-Align and VI-BA
            self.va_align()
            self.vi_ba()
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

    def va_align(self):
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

    def vi_ba(self):
        print("VI_BA START")
        # Joint optimization over sliding window of frames
        start = max(0, len(self.states)-self.window)
        idxs = list(range(start, len(self.states)))
        # Pack parameters
        x0 = []
        for i in idxs:
            R_i = self.states[i]['R']
            x0.extend(R.from_matrix(R_i).as_rotvec())
            x0.extend(self.states[i]['t'])
            x0.extend(self.states[i]['v'])
        for lm in self.landmarks.values():
            x0.extend(lm)
        # Residual function
        def residuals(x):
            res = []
            ptr = 0
            states_opt = {}
            for i in idxs:
                rv = x[ptr:ptr+3]; ptr+=3
                Rm = R.from_rotvec(rv).as_matrix()
                ti = x[ptr:ptr+3]; ptr+=3
                vi = x[ptr:ptr+3]; ptr+=3
                states_opt[i] = {'R':Rm, 't':ti, 'v':vi}
            lms_opt = {}
            for k in self.landmarks.keys():
                lms_opt[k] = x[ptr:ptr+3]; ptr+=3
            # reprojection residuals
            for i in idxs:
                Pts = self.landmarks.keys()
                for k,u in zip(Pts, self.tracker.prev_pts):
                    P3 = lms_opt[k]
                    cam = states_opt[i]['R'].T.dot(P3 - states_opt[i]['t'])
                    proj = self.K.dot(cam/cam[2])[:2]
                    res.extend(proj - u)
            # IMU preintegration residuals
            for j in range(len(idxs)-1):
                i0, i1 = idxs[j], idxs[j+1]
                imu_t, acc, gyro = self.imu_slices[i1]
                pre = IMUPreintegrator(self.dt)
                pre.integrate_batch(acc, gyro)
                si = states_opt[i0]; sj = states_opt[i1]
                # rotation error
                R_err = pre.delta_R.T.dot(sj['R'].dot(si['R'].T))
                res.extend(R.from_matrix(R_err).as_rotvec())
                # velocity error
                dv = sj['v'] - (si['v'] + si['R'].dot(pre.delta_v))
                res.extend(dv)
                # position error
                dp = sj['t'] - (si['t'] + si['R'].dot(pre.delta_p) + si['v']*(imu_t[-1]-imu_t[0]))
                res.extend(dp)
            return np.array(res)
        sol = least_squares(residuals, np.array(x0), verbose=0)
        # Optionally unpack back; omitted for clarity
        self.logs['reproj'].append(np.linalg.norm(sol.fun))
        print("VI_BA END")

    def reset(self):
        self.preint.reset()
        self.tracker = FeatureTracker()
        self.states.clear()
        self.landmarks.clear()
        self.imu_slices.clear()
        self.initialized = False

    def __call__(self, tstamp, input_tensor, intrinsics, 
                 curr_imu_data = None, save_slam_steps_path = None):
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
