import numpy as np
# cv2 import is not strictly necessary for this shell if image data is truly ignored
# import cv2 
from scipy.spatial.transform import Rotation as SRot # Used for rotation conversions

# Helper function (copied from your original code)
def is_rotation_matrix_valid(R_mat):
    """Checks if a matrix is a valid rotation matrix."""
    if not isinstance(R_mat, np.ndarray) or R_mat.shape != (3,3):
        return False
    if not np.isfinite(R_mat).all():
        return False
    should_be_identity = R_mat.T @ R_mat
    identity = np.identity(3)
    if not np.allclose(should_be_identity, identity, atol=1e-5): # Adjusted tolerance
        # print(f"Rotation matrix orthogonality check failed. R.T @ R:\n{should_be_identity}")
        return False
    if not np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-5): # Adjusted tolerance
        # print(f"Rotation matrix determinant check failed. det(R): {np.linalg.det(R_mat)}")
        return False
    return True

# IMUPreintegrator class (copied from your original code)
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
            # Ensure gyro values are finite
            if not np.isfinite(gyro).all():
                # print("Warning: Non-finite gyro values in preintegration. Skipping this measurement.")
                continue
            try:
                dR = SRot.from_rotvec(gyro * self.dt).as_matrix() # Using SRot for consistency
            except Exception as e:
                # print(f"Warning: Could not create rotation from gyro: {gyro}. Error: {e}. Skipping dR update.")
                dR = np.eye(3) # No rotation update if gyro is problematic

            self.delta_R_raw = self.delta_R_raw @ dR 
            
            # Ensure acc values are finite
            if not np.isfinite(acc).all():
                # print("Warning: Non-finite acc values in preintegration. Skipping this measurement.")
                continue

            # acc_world is acc in the local frame of the start of this preintegration step
            acc_world_local_frame = self.delta_R_raw @ acc 
            self.delta_p_raw += self.delta_v_raw * self.dt + 0.5 * acc_world_local_frame * self.dt**2
            self.delta_v_raw += acc_world_local_frame * self.dt
        
        # At the end of the batch copy the integrated values.
        self.delta_R = self.delta_R_raw.copy()
        self.delta_v = self.delta_v_raw.copy()
        self.delta_p = self.delta_p_raw.copy()

class XRVIOShell:
    def __init__(self, dt, window_size=1000, # Set a large default window if not finalizing often
                 initial_R=None, initial_t=None, initial_v=None, 
                 initial_timestamp=0.0, gravity=None, K_cam=None):
        """
        Initializes the IMU integration shell.
        Args:
            dt (float): IMU sampling period for the preintegrator.
            window_size (int): Number of states to keep in the active window.
            initial_R (np.ndarray, optional): Initial 3x3 rotation matrix. Defaults to identity.
            initial_t (np.ndarray, optional): Initial 3x1 translation vector. Defaults to zeros.
            initial_v (np.ndarray, optional): Initial 3x1 velocity vector. Defaults to zeros.
            initial_timestamp (float, optional): Timestamp for the initial state. Defaults to 0.0.
            gravity (np.ndarray, optional): 3x1 gravity vector in the world frame. Defaults to [0,0,-9.81].
            K_cam (np.ndarray, optional): 3x3 camera intrinsic matrix. Placeholder. Defaults to identity.
        """
        self.dt = dt
        self.preint = IMUPreintegrator(dt)
        
        self.gravity = np.array([0, 0, -9.81]) if gravity is None else np.asarray(gravity, dtype=float).copy()
        self.K = np.eye(3) if K_cam is None else np.asarray(K_cam, dtype=float).copy()
        
        self.window = window_size

        self.states = []  # Stores dictionaries: {'R', 't', 'v', 'timestamp', 'delta_R_imu', ...}
        self.finalized_trajectory = [] # Stores states popped from the window

        # Set initial state
        _initial_R = np.eye(3) if initial_R is None else np.asarray(initial_R, dtype=float).copy()
        _initial_t = np.zeros(3) if initial_t is None else np.asarray(initial_t, dtype=float).copy()
        _initial_v = np.zeros(3) if initial_v is None else np.asarray(initial_v, dtype=float).copy()

        if not is_rotation_matrix_valid(_initial_R):
            # print("Warning: Provided initial_R is invalid. Using identity.")
            _initial_R = np.eye(3)

        self.states.append({
            'R': _initial_R, 't': _initial_t, 'v': _initial_v,
            'timestamp': float(initial_timestamp),
            'delta_R_imu': np.eye(3), # IMU deltas that led to this state (none for initial)
            'delta_v_imu': np.zeros(3),
            'delta_p_imu': np.zeros(3)
        })

    def process_frame(self, img_placeholder, imu_timestamps, imu_acc_measurements, imu_gyro_measurements):
        """
        Processes a batch of IMU data to update the pose.
        Args:
            img_placeholder: Not used in this shell, for signature compatibility.
            imu_timestamps (np.ndarray): Timestamps for each IMU measurement.
            imu_acc_measurements (np.ndarray): Accelerometer data (N,3).
            imu_gyro_measurements (np.ndarray): Gyroscope data (N,3).
        Returns:
            dict: The newly computed state dictionary, or the last state if no update.
        """
        if not (isinstance(imu_timestamps, np.ndarray) and imu_timestamps.size > 0 and
                isinstance(imu_acc_measurements, np.ndarray) and imu_acc_measurements.ndim == 2 and imu_acc_measurements.shape[0] > 0 and
                isinstance(imu_gyro_measurements, np.ndarray) and imu_gyro_measurements.ndim == 2 and imu_gyro_measurements.shape[0] > 0):
            # print("Warning: Empty or invalid IMU data format provided to process_frame.")
            return self.states[-1] if self.states else None

        if not (imu_acc_measurements.shape[0] == imu_gyro_measurements.shape[0] == imu_timestamps.size):
            # print("Warning: IMU data batch length mismatch.")
            return self.states[-1] if self.states else None

        current_batch_timestamp = float(imu_timestamps[-1])
        prev_state = self.states[-1]

        dt_interval = current_batch_timestamp - prev_state['timestamp']
        
        if dt_interval < 0: # Timestamps should be monotonic
            # print(f"Warning: Negative dt_interval ({dt_interval}). Timestamps may be out of order.")
            # Option: return prev_state or try to use self.dt * num_samples
            dt_interval = len(imu_acc_measurements) * self.dt # Fallback if timestamps are problematic
        elif dt_interval <= 1e-9: # If interval is zero or extremely small
            if prev_state['timestamp'] == current_batch_timestamp: # No time has passed
                 # Return previous state as no actual update is expected
                return prev_state
            # If small but positive, use num_samples * self.dt if it's larger (more stable)
            estimated_dt = len(imu_acc_measurements) * self.dt
            if estimated_dt > dt_interval : dt_interval = estimated_dt


        self.preint.reset()
        self.preint.integrate_batch(imu_acc_measurements, imu_gyro_measurements)

        delta_R_from_imu = self.preint.delta_R
        delta_v_from_imu = self.preint.delta_v
        delta_p_from_imu = self.preint.delta_p

        if not is_rotation_matrix_valid(delta_R_from_imu):
            # print("Warning: Invalid delta_R from preintegration. Using identity for this step's rotation delta.")
            delta_R_from_imu = np.eye(3)

        R_prev = prev_state['R']
        t_prev = prev_state['t']
        v_prev = prev_state['v']
        
        current_R_world = R_prev @ delta_R_from_imu
        if not is_rotation_matrix_valid(current_R_world):
            # print("Warning: Resulting current_R_world is invalid. Reverting to R_prev for safety.")
            current_R_world = R_prev.copy()

        current_v_world = v_prev + R_prev @ delta_v_from_imu + self.gravity * dt_interval
        current_t_world = t_prev + v_prev * dt_interval + R_prev @ delta_p_from_imu + 0.5 * self.gravity * (dt_interval**2)
        
        new_state_data = {
            'R': current_R_world, 't': current_t_world, 'v': current_v_world,
            'timestamp': current_batch_timestamp,
            'delta_R_imu': delta_R_from_imu.copy(), 
            'delta_v_imu': delta_v_from_imu.copy(),
            'delta_p_imu': delta_p_from_imu.copy()
        }
        self.states.append(new_state_data)
        
        if len(self.states) > self.window:
            self.finalized_trajectory.append(self.states.pop(0))

        return new_state_data

    def __call__(self, tstamp_event_frame, input_tensor, intrinsics, curr_imu_data=None, save_slam_steps_path = None):
        """
        Call interface matching the original XRVIO structure.
        Args:
            tstamp_event_frame: Timestamp for the visual frame (can be ignored if IMU timestamps are absolute).
            input_tensor_visual: Visual data (ignored by this shell).
            intrinsics_tensor: Tensor [fx, fy, cx, cy] for camera intrinsics.
            curr_imu_data (tuple): (imu_timestamps, imu_acc_data, imu_gyro_data).
                                   imu_timestamps should be absolute.
        Returns:
            dict: The latest computed state dictionary.
        """
        if intrinsics is not None:
            # Assuming intrinsics_tensor is on CPU and is a 1D tensor/list [fx, fy, cx, cy]
            try:
                fx, fy, cx, cy = intrinsics.cpu().numpy().flatten()
                new_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
                if not np.array_equal(self.K, new_K):
                    self.K = new_K
                    # print("XRVIOShell: Camera intrinsics K updated.")
            except Exception as e:
                # print(f"XRVIOShell: Failed to update K from intrinsics_tensor. Error: {e}")
                pass
        
        img_gray_placeholder = np.array([[]], dtype=np.uint8) # Visual data is not used

        if curr_imu_data is None or len(curr_imu_data) != 3:
            # print("Warning: curr_imu_data is missing or malformed in __call__.")
            return self.states[-1] if self.states else None
            
        imu_t_batch, imu_acc_batch, imu_gyro_batch = curr_imu_data
        
        imu_t_np = np.asarray(imu_t_batch, dtype=float)
        imu_acc_np = np.asarray(imu_acc_batch, dtype=float)
        imu_gyro_np = np.asarray(imu_gyro_batch, dtype=float)

        # Ensure data is 2D (N,3) for acc/gyro and 1D (N,) for timestamps
        if imu_acc_np.ndim == 1: imu_acc_np = imu_acc_np.reshape(-1, 3)
        if imu_gyro_np.ndim == 1: imu_gyro_np = imu_gyro_np.reshape(-1, 3)
        if imu_t_np.ndim == 0: imu_t_np = np.array([imu_t_np]) # Handle single timestamp

        return self.process_frame(img_gray_placeholder, imu_t_np, imu_acc_np, imu_gyro_np)

    def reset(self, initial_R=None, initial_t=None, initial_v=None, initial_timestamp=0.0):
        """Resets the integrator to an initial state."""
        self.preint.reset()
        
        _initial_R = np.eye(3) if initial_R is None else np.asarray(initial_R, dtype=float).copy()
        _initial_t = np.zeros(3) if initial_t is None else np.asarray(initial_t, dtype=float).copy()
        _initial_v = np.zeros(3) if initial_v is None else np.asarray(initial_v, dtype=float).copy()

        if not is_rotation_matrix_valid(_initial_R):
            # print("Warning: Provided initial_R for reset is invalid. Using identity.")
            _initial_R = np.eye(3)
            
        self.states = [{
            'R': _initial_R, 't': _initial_t, 'v': _initial_v,
            'timestamp': float(initial_timestamp),
            'delta_R_imu': np.eye(3), 
            'delta_v_imu': np.zeros(3),
            'delta_p_imu': np.zeros(3)
        }]
        self.finalized_trajectory.clear()
        # print("XRVIOShell has been reset.")

    def terminate(self):
        """
        Returns the computed trajectory as poses (translation + quaternion) and timestamps.
        Returns:
            tuple: (poses_array, timestamps_array)
                   poses_array is (M,7) [tx,ty,tz, qx,qy,qz,qw]
                   timestamps_array is (M,)
        """
        all_history_for_output = self.finalized_trajectory + self.states

        if not all_history_for_output:
            # print("Terminate: No trajectory data available.")
            return np.array([[0,0,0, 0,0,0,1]], dtype=float), np.array([0.0], dtype=float)
        
        # Sort by timestamp to ensure chronological order
        try:
            all_history_for_output.sort(key=lambda s: s['timestamp'])
        except (TypeError, KeyError) as e: # Catch if timestamp is missing or not comparable
            # print(f"Warning: Could not sort history by timestamp due to error: {e}. Filtering problematic entries.")
            all_history_for_output = [s for s in all_history_for_output if isinstance(s.get('timestamp'), (int, float))]
            if all_history_for_output:
                 all_history_for_output.sort(key=lambda s: s['timestamp'])
            else: # If filtering removed everything
                return np.array([[0,0,0, 0,0,0,1]], dtype=float), np.array([0.0], dtype=float)


        poses_out_list = []
        tstamps_out_list = []
        last_ts = -np.inf

        for state_data in all_history_for_output:
            ts_val = state_data.get('timestamp')
            # Skip if timestamp is missing, not a number, or duplicated (after sorting)
            if not isinstance(ts_val, (int, float)) or ts_val <= last_ts and not np.isclose(ts_val, last_ts):
                if ts_val <= last_ts and not np.isclose(ts_val, last_ts):
                    print(f"Skipping state at ts {ts_val} due to non-monotonic timestamp (last was {last_ts}).")
                continue
            
            R_val = state_data.get('R')
            t_val = state_data.get('t')

            if not (isinstance(R_val, np.ndarray) and R_val.shape == (3,3) and \
                    isinstance(t_val, np.ndarray) and t_val.shape == (3,)):
                # print(f"Skipping state at ts {ts_val} due to invalid R/t type or shape.")
                continue

            if not is_rotation_matrix_valid(R_val):
                # print(f"Skipping state at ts {ts_val} due to invalid rotation matrix.")
                continue
            if not np.isfinite(t_val).all():
                # print(f"Skipping state at ts {ts_val} due to non-finite translation.")
                continue
            
            try:
                rot_q = SRot.from_matrix(R_val).as_quat() # [x,y,z,w]
                current_pose = np.concatenate((t_val, rot_q)) # [tx,ty,tz, qx,qy,qz,qw]
                poses_out_list.append(current_pose)
                tstamps_out_list.append(ts_val)
                last_ts = ts_val
            except Exception as e:
                # print(f"Error converting R to quat or concatenating for state at ts {ts_val}: {e}")
                pass

        if not poses_out_list:
            # print("Terminate: No valid poses to return after filtering.")
            return np.array([[0,0,0, 0,0,0,1]], dtype=float), np.array([0.0], dtype=float)

        return np.array(poses_out_list, dtype=float), np.array(tstamps_out_list, dtype=float)