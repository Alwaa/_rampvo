import numpy as np
from scipy.spatial.transform import Rotation

class DeadReckoning:
    """
    A class to perform dead reckoning pose estimation from IMU data.
    The IMU data file is expected to have columns: idx, t, wx, wy, wz, ax, ay, az
    """

    def __init__(self, initial_state=None, g_vector=None):
        """
        Initializes the DeadReckoning class.

        Args:
            initial_state (dict, optional): A dictionary with initial state.
                Expected keys:
                'ts': initial timestamp (float)
                'pos': initial position (3x1 numpy array) [x, y, z]
                'ori': initial orientation (scipy.spatial.transform.Rotation object or 3x3 rotation matrix)
                'vel': initial velocity (3x1 numpy array) [vx, vy, vz]
                If None, a default initial state (zeros, identity orientation) at t=0 is used.
            g_vector (np.array, optional): Gravity vector. Defaults to [0, 0, -9.81].
        """
        self.imu_data_raw = None
        self.imu_data = None
        self.trajectory = None

        if initial_state is None:
            self.initial_state = {
                "ts": 0.0,
                "pos": np.zeros(3),
                "ori": Rotation.identity(), # Stored as Rotation object internally
                "vel": np.zeros(3),
            }
        else:
            self.initial_state = initial_state
            # Ensure orientation is a Rotation object
            if isinstance(initial_state.get("ori"), np.ndarray) and initial_state["ori"].shape == (3,3):
                self.initial_state["ori"] = Rotation.from_matrix(initial_state["ori"])
            elif not isinstance(initial_state.get("ori"), Rotation):
                print("Warning: Initial orientation should be a scipy Rotation object or a 3x3 matrix. Defaulting to identity.")
                self.initial_state["ori"] = Rotation.identity()


        self.g = np.array([0, 0, -9.81*0]) if g_vector is None else np.array(g_vector)

    def _hat(self, v_vec):
        """
        Computes the skew-symmetric matrix for a 3x1 vector.
        """
        v_vec = v_vec.flatten()
        return np.array([[0, -v_vec[2], v_vec[1]],
                         [v_vec[2], 0, -v_vec[0]],
                         [-v_vec[1], v_vec[0], 0]])

    def _mat_exp(self, omega_dt):
        """
        Computes the matrix exponential for SO(3) (Rodrigues' formula).
        omega_dt is the axis-angle vector (rotation vector).
        """
        if len(omega_dt) != 3:
            raise ValueError("Tangent vector (omega_dt) must have length 3")
        
        angle = np.linalg.norm(omega_dt)

        if angle < 1e-10: # Near zero, use first-order Taylor expansion
            return np.identity(3) + self._hat(omega_dt)

        axis = omega_dt / angle
        s = np.sin(angle)
        c = np.cos(angle)

        # Rodrigues' formula
        return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * self._hat(axis)

    def load_imu_data(self, filepath):
        """
        Loads IMU data from a text file.
        Expected format: idx t wx wy wz ax ay az (space separated)
        Lines starting with '#' are skipped.

        Args:
            filepath (str): The path to the IMU data file.
        """
        try:
            # Try to determine the number of columns to decide if 'idx' is present.
            # This is a bit fragile; a more robust parser might be needed for varying formats.
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                while first_line.startswith('#'): # skip comments
                    first_line = f.readline().strip()
            
            num_cols = len(first_line.split())
            
            if num_cols == 8: # idx t wx wy wz ax ay az
                data = np.loadtxt(filepath, comments='#')
                self.imu_data_raw = data
                self.imu_data = {
                    "ts": data[:, 1],        # Timestamps
                    "gyros": data[:, 2:5],   # Angular velocities (wx, wy, wz)
                    "accels": data[:, 5:8]   # Linear accelerations (ax, ay, az)
                }
            elif num_cols == 7: # t wx wy wz ax ay az (assuming idx is missing)
                print("Note: Assuming 7 columns (idx column missing): t wx wy wz ax ay az")
                data = np.loadtxt(filepath, comments='#')
                self.imu_data_raw = data
                self.imu_data = {
                    "ts": data[:, 0],        # Timestamps
                    "gyros": data[:, 1:4],   # Angular velocities (wx, wy, wz)
                    "accels": data[:, 4:7]   # Linear accelerations (ax, ay, az)
                }
            else:
                raise ValueError(f"Unexpected number of columns ({num_cols}) in IMU data. Expected 7 or 8.")


            print(f"Successfully loaded IMU data from {filepath}")
            if self.imu_data["ts"].shape[0] > 0 :
                if self.initial_state.get("ts", 0.0) == 0.0 or not self.initial_state.get("ts_is_set", False):
                    self.initial_state["ts"] = self.imu_data["ts"][0]
                    self.initial_state["ts_is_set"] = True # Mark that ts is now intentionally set
                    print(f"Updated initial state timestamp to the first IMU timestamp: {self.initial_state['ts']:.4f}s")


        except Exception as e:
            print(f"Error loading IMU data from {filepath}: {e}")
            self.imu_data = None
            self.imu_data_raw = None

    def estimate_pose(self):
        """
        Estimates the pose (trajectory) using the loaded IMU data and initial state.
        The integration method is based on the vectorized propagation found in
        the provided `imu_integration.py`.

        Returns:
            dict: A dictionary containing the trajectory:
                  'ts': timestamps (N x 1)
                  'pos': positions (N x 3)
                  'ori_quat': orientations as quaternions (qx, qy, qz, qw) (N x 4)
                  'ori_rotm': orientations as rotation matrices (N x 3 x 3)
                  'vel': velocities (N x 3)
            Returns None if IMU data is not loaded.
        """
        if self.imu_data is None:
            print("Error: IMU data not loaded. Call load_imu_data(filepath) first.")
            return None

        ts_all = self.imu_data["ts"]
        acc_all = self.imu_data["accels"]
        gyr_all = self.imu_data["gyros"]

        # Initialize from initial_state
        t0 = self.initial_state["ts"]
        p0 = self.initial_state["pos"].copy()
        R0_rot_obj = self.initial_state["ori"]
        R0 = R0_rot_obj.as_matrix()
        v0 = self.initial_state["vel"].copy()
        
        start_idx = np.searchsorted(ts_all, t0, side='left')
        
        # Handle case where t0 is before the first IMU timestamp or exactly at it
        if start_idx == 0 and t0 <= ts_all[0]:
            ts_proc = ts_all
            gyr_proc = gyr_all
            acc_proc = acc_all
            # Update t0 to the actual first timestamp if it was set slightly before
            if t0 < ts_all[0]:
                print(f"Warning: Initial timestamp {t0} is before first IMU data point {ts_all[0]}. Starting from first IMU data point.")
            t0 = ts_all[0] 
            self.initial_state["ts"] = t0 # Align t0 with the first processed timestamp

        # Handle case where t0 is after the last IMU timestamp
        elif start_idx >= len(ts_all):
            print(f"Warning: Initial timestamp {t0} is after all IMU data points. No data to process.")
            self.trajectory = {
                "ts": np.array([t0]), "pos": np.array([p0]),
                "ori_quat": np.array([R0_rot_obj.as_quat()]), "ori_rotm": np.array([R0]),
                "vel": np.array([v0])}
            return self.trajectory
            
        # Handle case where t0 is between two IMU samples (interpolation) or exactly matches a sample
        else:
            # If t0 is exactly at ts_all[start_idx], no interpolation needed for data selection
            if np.isclose(t0, ts_all[start_idx]):
                ts_proc = ts_all[start_idx:]
                gyr_proc = gyr_all[start_idx:]
                acc_proc = acc_all[start_idx:]
            # If t0 is between ts_all[start_idx-1] and ts_all[start_idx], interpolate
            elif start_idx > 0 and t0 > ts_all[start_idx-1] and t0 < ts_all[start_idx]:
                dt_segment = ts_all[start_idx] - ts_all[start_idx-1]
                w_interp = (t0 - ts_all[start_idx-1]) / dt_segment
                
                initial_gyr = gyr_all[start_idx-1] * (1-w_interp) + gyr_all[start_idx] * w_interp
                initial_acc = acc_all[start_idx-1] * (1-w_interp) + acc_all[start_idx] * w_interp
                
                ts_proc = np.hstack((t0, ts_all[start_idx:]))
                gyr_proc = np.vstack((initial_gyr, gyr_all[start_idx:]))
                acc_proc = np.vstack((initial_acc, acc_all[start_idx:]))
            else: # Should ideally not happen if start_idx logic is correct
                 ts_proc = ts_all[start_idx:]
                 gyr_proc = gyr_all[start_idx:]
                 acc_proc = acc_all[start_idx:]
                 if ts_proc.shape[0] > 0:
                    t0 = ts_proc[0]
                    self.initial_state["ts"] = t0
                 else: # no data after start_idx
                    print(f"Warning: No IMU data at or after initial timestamp {t0}. No data to process.")
                    self.trajectory = {
                        "ts": np.array([t0]), "pos": np.array([p0]),
                        "ori_quat": np.array([R0_rot_obj.as_quat()]), "ori_rotm": np.array([R0]),
                        "vel": np.array([v0])}
                    return self.trajectory


        num_samples = len(ts_proc)
        if num_samples <= 1:
            print("Not enough IMU samples to process after initial timestamp.")
            self.trajectory = {
                "ts": np.array([t0]),
                "pos": np.array([p0]),
                "ori_quat": np.array([R0_rot_obj.as_quat()]),
                "ori_rotm": np.array([R0]),
                "vel": np.array([v0])
            }
            return self.trajectory

        p_wb = np.zeros((num_samples, 3))
        R_wb = np.zeros((num_samples, 3, 3))
        v_wb = np.zeros((num_samples, 3))

        p_wb[0], R_wb[0], v_wb[0] = p0, R0, v0
        
        dt_arr = np.diff(ts_proc)
        
        for i in range(num_samples - 1):
            dt = dt_arr[i]
            if dt <= 0: # Ensure dt is positive
                print(f"Warning: Non-positive dt ({dt}) at index {i}. Skipping step.")
                # Copy previous state
                R_wb[i+1] = R_wb[i]
                v_wb[i+1] = v_wb[i]
                p_wb[i+1] = p_wb[i]
                continue

            w_avg = 0.5 * (gyr_proc[i] + gyr_proc[i+1])
            R_wb[i+1] = R_wb[i] @ self._mat_exp(w_avg * dt)

            a_body_avg = 0.5 * (acc_proc[i] + acc_proc[i+1])
            
            # Use orientation at start of interval for transforming avg specific force
            # R_mid_approx = R_wb[i]
            # Or use average orientation (more complex for rotation matrices, easier with quaternions via SLERP)
            # For simplicity, using R_wb[i] (SO(3) average is non-trivial)
            # Or better, use the average of R_wb[i] and R_wb[i+1] if available before this step
            # As in imu_integration.py, use 0.5 * (R_wb[i] + R_wb[i+1]) - this is not a valid rotation matrix
            # A common approach: R_k * exp(0.5 * omega_k * dt)
            R_mid_step = R_wb[i] @ self._mat_exp(0.5 * w_avg * dt)


            dv_from_specific_force = R_mid_step @ (a_body_avg * dt)
            
            v_wb[i+1] = v_wb[i] + dv_from_specific_force + self.g * dt
            # Trapezoidal integration for position
            p_wb[i+1] = p_wb[i] + 0.5 * (v_wb[i] + v_wb[i+1]) * dt


        orientations_quat = Rotation.from_matrix(R_wb).as_quat()

        self.trajectory = {
            "ts": ts_proc,
            "pos": p_wb,
            "ori_quat": orientations_quat, 
            "ori_rotm": R_wb,
            "vel": v_wb
        }
        print("Pose estimation complete.")
        return self.trajectory

    def get_trajectory_as_numpy(self):
        """
        Returns the estimated trajectory as a single NumPy array.
        Format: [t, x, y, z, qx, qy, qz, qw, vx, vy, vz]

        Returns:
            np.array: Trajectory data or None if not estimated.
        """
        if self.trajectory is None:
            print("Trajectory not estimated yet. Call estimate_pose() first.")
            return None

        return np.hstack((
            self.trajectory["ts"][:, None],
            self.trajectory["pos"],
            self.trajectory["ori_quat"],
            self.trajectory["vel"]
        ))


# import numpy as np
# from scipy.spatial.transform import Rotation

# class DeadReckoning:
#     """
#     A class to perform dead reckoning pose estimation from IMU data.
#     The IMU data file is expected to have columns: idx, t, wx, wy, wz, ax, ay, az
#     """

#     def __init__(self, initial_state=None, g_vector=None):
#         """
#         Initializes the DeadReckoning class.

#         Args:
#             initial_state (dict, optional): A dictionary with initial state.
#                 Expected keys:
#                 'ts': initial timestamp (float)
#                 'pos': initial position (3x1 numpy array) [x, y, z]
#                 'ori': initial orientation (scipy.spatial.transform.Rotation object or 3x3 rotation matrix)
#                 'vel': initial velocity (3x1 numpy array) [vx, vy, vz]
#                 If None, a default initial state (zeros, identity orientation) at t=0 is used.
#             g_vector (np.array, optional): Gravity vector. Defaults to [0, 0, -9.81].
#         """
#         self.imu_data_raw = None
#         self.imu_data = None
#         self.trajectory = None

#         if initial_state is None:
#             self.initial_state = {
#                 "ts": 0.0,
#                 "pos": np.zeros(3),
#                 "ori": Rotation.identity(), # Stored as Rotation object internally
#                 "vel": np.zeros(3),
#             }
#         else:
#             self.initial_state = initial_state
#             # Ensure orientation is a Rotation object
#             if isinstance(initial_state.get("ori"), np.ndarray) and initial_state["ori"].shape == (3,3):
#                 self.initial_state["ori"] = Rotation.from_matrix(initial_state["ori"])
#             elif not isinstance(initial_state.get("ori"), Rotation):
#                 print("Warning: Initial orientation should be a scipy Rotation object or a 3x3 matrix. Defaulting to identity.")
#                 self.initial_state["ori"] = Rotation.identity()


#         self.g = np.array([0, 0, -9.81]) if g_vector is None else np.array(g_vector)

#     def _hat(self, v_vec):
#         """
#         Computes the skew-symmetric matrix for a 3x1 vector.
#         """
#         v_vec = v_vec.flatten()
#         return np.array([[0, -v_vec[2], v_vec[1]],
#                          [v_vec[2], 0, -v_vec[0]],
#                          [-v_vec[1], v_vec[0], 0]])

#     def _mat_exp(self, omega_dt):
#         """
#         Computes the matrix exponential for SO(3) (Rodrigues' formula).
#         omega_dt is the axis-angle vector (rotation vector).
#         """
#         if len(omega_dt) != 3:
#             raise ValueError("Tangent vector (omega_dt) must have length 3")
        
#         angle = np.linalg.norm(omega_dt)

#         if angle < 1e-10: # Near zero, use first-order Taylor expansion
#             return np.identity(3) + self._hat(omega_dt)

#         axis = omega_dt / angle
#         s = np.sin(angle)
#         c = np.cos(angle)

#         # Rodrigues' formula
#         return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * self._hat(axis)

#     def load_imu_data(self, filepath):
#         """
#         Loads IMU data from a text file.
#         Expected format: idx t wx wy wz ax ay az (space separated)
#         Lines starting with '#' are skipped.

#         Args:
#             filepath (str): The path to the IMU data file.
#         """
#         try:
#             data = np.loadtxt(filepath, comments='#')
#             self.imu_data_raw = data
#             self.imu_data = {
#                 "ts": data[:, 1],        # Timestamps
#                 "gyros": data[:, 2:5],   # Angular velocities (wx, wy, wz)
#                 "accels": data[:, 5:8]   # Linear accelerations (ax, ay, az)
#             }
#             print(f"Successfully loaded IMU data from {filepath}")
#             # Set initial timestamp from data if not explicitly set or if it's default 0
#             if self.initial_state["ts"] == 0.0 and self.imu_data["ts"].shape[0] > 0 :
#                  self.initial_state["ts"] = self.imu_data["ts"][0]
#                  print(f"Updated initial state timestamp to the first IMU timestamp: {self.initial_state['ts']:.4f}s")

#         except Exception as e:
#             print(f"Error loading IMU data from {filepath}: {e}")
#             self.imu_data = None
#             self.imu_data_raw = None

#     def estimate_pose(self):
#         """
#         Estimates the pose (trajectory) using the loaded IMU data and initial state.
#         The integration method is based on the vectorized propagation found in
#         the provided `imu_integration.py`.

#         Returns:
#             dict: A dictionary containing the trajectory:
#                   'ts': timestamps (N x 1)
#                   'pos': positions (N x 3)
#                   'ori_quat': orientations as quaternions (qx, qy, qz, qw) (N x 4)
#                   'ori_rotm': orientations as rotation matrices (N x 3 x 3)
#                   'vel': velocities (N x 3)
#             Returns None if IMU data is not loaded.
#         """
#         if self.imu_data is None:
#             print("Error: IMU data not loaded. Call load_imu_data(filepath) first.")
#             return None

#         ts = self.imu_data["ts"]
#         acc = self.imu_data["accels"]
#         gyr = self.imu_data["gyros"]

#         # Initialize from initial_state
#         t0 = self.initial_state["ts"]
#         p0 = self.initial_state["pos"].copy()
#         # Ensure R0 is a 3x3 matrix for calculations
#         R0_rot_obj = self.initial_state["ori"]
#         R0 = R0_rot_obj.as_matrix()
#         v0 = self.initial_state["vel"].copy()
        
#         # Find the starting index in IMU data corresponding to t0 or later
#         start_idx = np.searchsorted(ts, t0, side='left')
#         if start_idx > 0 and ts[start_idx] > t0 and ts[start_idx -1] < t0 :
#              # Interpolate initial gyro and accel if t0 is between two IMU samples
#             dt_segment = ts[start_idx] - ts[start_idx-1]
#             w_interp = (t0 - ts[start_idx-1]) / dt_segment
            
#             initial_gyr = gyr[start_idx-1] * (1-w_interp) + gyr[start_idx] * w_interp
#             initial_acc = acc[start_idx-1] * (1-w_interp) + acc[start_idx] * w_interp
            
#             # Prepend interpolated measurement at t0
#             ts_proc = np.hstack((t0, ts[start_idx:]))
#             gyr_proc = np.vstack((initial_gyr, gyr[start_idx:]))
#             acc_proc = np.vstack((initial_acc, acc[start_idx:]))
#         else: # t0 aligns with an IMU sample or is before the first one
#             ts_proc = ts[start_idx:]
#             gyr_proc = gyr[start_idx:]
#             acc_proc = acc[start_idx:]
#             if ts_proc.shape[0] > 0: # If we have data to process
#                  self.initial_state["ts"] = ts_proc[0] # Align t0 with the first processed timestamp
#                  t0 = ts_proc[0]


#         num_samples = len(ts_proc)
#         if num_samples <= 1:
#             print("Not enough IMU samples to process after initial timestamp.")
#             # Return trajectory with only initial state
#             self.trajectory = {
#                 "ts": np.array([t0]),
#                 "pos": np.array([p0]),
#                 "ori_quat": np.array([R0_rot_obj.as_quat()]),
#                 "ori_rotm": np.array([R0]),
#                 "vel": np.array([v0])
#             }
#             return self.trajectory


#         # Prepare arrays for results
#         p_wb = np.zeros((num_samples, 3))
#         R_wb = np.zeros((num_samples, 3, 3))
#         v_wb = np.zeros((num_samples, 3))

#         p_wb[0], R_wb[0], v_wb[0] = p0, R0, v0
        
#         # IMU integration loop (vectorized mean for w and a)
#         # dt values are between consecutive timestamps
#         dt_arr = np.diff(ts_proc)

#         # Midpoint integration for angular rates
#         # w_mid = 0.5 * (gyr_proc[:-1] + gyr_proc[1:]) # (N-1) x 3
#         # d_theta = w_mid * dt_arr[:, None] # (N-1) x 3, element-wise
        
#         # For orientation, iterate using matrix exponential
#         for i in range(num_samples - 1):
#             dt = dt_arr[i]
#             w_i = gyr_proc[i] # omega at t_i
#             # R_wb[i+1] = R_wb[i] @ self._mat_exp(w_i * dt) # Simplified: using w_i over interval
#             # More accurate: use average angular velocity
#             w_avg = 0.5 * (gyr_proc[i] + gyr_proc[i+1])
#             R_wb[i+1] = R_wb[i] @ self._mat_exp(w_avg * dt)


#         # Acceleration integration (midpoint rule style)
#         # Transform accelerations to world frame
#         # acc_world = np.einsum('ijk,ik->ij', R_wb, acc_proc) # R_wb[k] * acc_proc[k]
#         # More robustly apply rotations one by one
#         acc_body_avg = 0.5 * (acc_proc[:-1] + acc_proc[1:]) # (N-1) x 3
        
#         for i in range(num_samples - 1):
#             dt = dt_arr[i]
#             # Orientation at the start and end of the interval
#             R_i = R_wb[i]
#             R_i_plus_1 = R_wb[i+1]

#             # Average acceleration in body frame
#             a_body_avg_i = acc_body_avg[i]

#             # Transform average acceleration to world frame using average orientation
#             # (or orientation at the start/midpoint of interval)
#             # Using R_i (orientation at the start of the interval for a_body_avg_i)
#             # dv_world = R_i @ (a_body_avg_i * dt) - self.g * dt # If acc_body already has g removed
#             # Assuming acc is specific force (measures g pointing upwards if static)
#             # a_world = R_i @ a_body_avg_i + self.g # specific force to world acc
            
#             # As per imu_integration.py, gravity is added separately.
#             # Accelerometer measures specific force: a_meas = a_true - g_body
#             # So, a_true_body = a_meas + R_transpose * g_world
#             # Then a_true_world = R * a_true_body = R * a_meas + g_world

#             # The `imu_integration.py` does:
#             # dv = R_mid @ (a_body_corrected * dt) + g_world * dt
#             # where a_body_corrected is a_raw - bias.
#             # Let's assume a_body_avg_i is already bias corrected.

#             # Effective acceleration in world frame (from specific force)
#             # R_mid can be approximated by R_i or an average
#             R_mid_approx = R_i # Or slerp R_i and R_i_plus_1 if highly dynamic
            
#             dv_from_specific_force = R_mid_approx @ (a_body_avg_i * dt)
            
#             v_wb[i+1] = v_wb[i] + dv_from_specific_force + self.g * dt
#             p_wb[i+1] = p_wb[i] + v_wb[i] * dt + 0.5 * (dv_from_specific_force + self.g * dt) * dt


#         orientations_quat = Rotation.from_matrix(R_wb).as_quat()

#         self.trajectory = {
#             "ts": ts_proc,
#             "pos": p_wb,
#             "ori_quat": orientations_quat, # qx, qy, qz, qw
#             "ori_rotm": R_wb,
#             "vel": v_wb
#         }
#         print("Pose estimation complete.")
#         return self.trajectory

#     def get_trajectory_as_numpy(self):
#         """
#         Returns the estimated trajectory as a single NumPy array.
#         Format: [t, x, y, z, qx, qy, qz, qw, vx, vy, vz]

#         Returns:
#             np.array: Trajectory data or None if not estimated.
#         """
#         if self.trajectory is None:
#             print("Trajectory not estimated yet. Call estimate_pose() first.")
#             return None

#         return np.hstack((
#             self.trajectory["ts"][:, None],
#             self.trajectory["pos"],
#             self.trajectory["ori_quat"],
#             self.trajectory["vel"]
#         ))
