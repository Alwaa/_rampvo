import numpy as np
from scipy.spatial.transform import Rotation as SRot # Used for rotation conversions

# Helper function
def is_rotation_matrix_valid(R_mat):
    # ... (implementation as provided previously) ...
    if not isinstance(R_mat, np.ndarray) or R_mat.shape != (3,3): return False
    if not np.isfinite(R_mat).all(): return False
    if not np.allclose(R_mat.T @ R_mat, np.eye(3), atol=1e-5): return False
    if not np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-5): return False
    return True

def hat(v):
    v = v.flatten()
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def mat_exp(omega): # From your _python_tools/imu_integration.py
    if not isinstance(omega, np.ndarray) or omega.shape != (3,):
        if np.allclose(omega, 0): return np.eye(3)
        # print(f"Warning: mat_exp omega is not a 3-vector: {omega}. Using Identity.")
        return np.eye(3) # Or raise error
        
    angle = np.linalg.norm(omega)
    if angle < 1e-10:
        return np.identity(3) + hat(omega)
    axis = omega / angle
    s = np.sin(angle); c = np.cos(angle)
    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)

class XRVIOShell:
    def __init__(self, dt_imu_sampling_period_seconds, # Nominal IMU sampling period in SECONDS
                 window_size=1000,
                 initial_R=None, initial_t=None, initial_v=None, 
                 initial_timestamp_ns=0.0, # Initial timestamp in NANOSECONDS
                 gravity=None, K_cam=None): # K_cam is a placeholder
        
        self.nominal_imu_sampling_period_seconds = float(dt_imu_sampling_period_seconds)
        self.gravity = np.array([0, 0, -9.81]) if gravity is None else np.asarray(gravity, dtype=float).copy()
        self.K = np.eye(3) if K_cam is None else np.asarray(K_cam, dtype=float).copy()
        
        self.window = int(window_size)
        self.states = [] 
        self.finalized_trajectory = []
        self._initial_timestamp_ns = float(initial_timestamp_ns)

        _initial_R = np.eye(3) if initial_R is None else np.asarray(initial_R, dtype=float).copy()
        _initial_t = np.zeros(3) if initial_t is None else np.asarray(initial_t, dtype=float).copy()
        _initial_v = np.zeros(3) if initial_v is None else np.asarray(initial_v, dtype=float).copy()

        if not is_rotation_matrix_valid(_initial_R): _initial_R = np.eye(3)

        self.states.append({
            'R': _initial_R, 't': _initial_t, 'v': _initial_v,
            'timestamp': float(initial_timestamp_ns) # NANOSECONDS
        })

    def process_frame(self, img_placeholder, imu_timestamps_ns_batch, 
                      imu_acc_measurements, imu_gyro_measurements):
        
        if not self.states: # Should have been initialized
            # print("CRITICAL: XRVIOShell.states is empty in process_frame.")
            # Re-initialize a default first state if it's missing
            self.reset(initial_timestamp_ns=self._initial_timestamp_ns)


        # Start propagation from the previous state
        prev_state = self.states[-1]
        current_R_w_b = prev_state['R'].copy()
        current_t_w = prev_state['t'].copy()
        current_v_w = prev_state['v'].copy()
        # prev_timestamp_ns = prev_state['timestamp'] # Not directly used in loop below, ts from batch are used

        num_imu_measurements = len(imu_timestamps_ns_batch)
        final_timestamp_ns_for_state = prev_state['timestamp'] # Default if no IMU data

        if num_imu_measurements == 0:
            # No IMU data in this batch. State R,t,v effectively don't change due to IMU.
            # The timestamp of the new state will be the target timestamp of the current frame.
            # This is handled by how __call__ sets current_frame_end_ts_ns if needed.
            # For now, just means no loop iterations. final_timestamp_ns_for_state needs to be set by caller via __call__.
            pass

        elif num_imu_measurements == 1:
            # Single IMU measurement: propagate over nominal_dt_s
            if self.nominal_imu_sampling_period_seconds > 1e-9:
                dt_s = self.nominal_imu_sampling_period_seconds
                
                w = imu_gyro_measurements[0] # Biases assumed zero
                a = imu_acc_measurements[0]  # Biases assumed zero

                dtheta_vec = w * dt_s
                dR_step = mat_exp(dtheta_vec)
                R_next_step = current_R_w_b @ dR_step

                # For Rmid, use R_k @ Exp(0.5 * omega * dt) as a robust approximation
                R_mid_step = current_R_w_b @ mat_exp(w * (0.5 * dt_s))
                
                dv_acc_world = R_mid_step @ (a * dt_s) # dv = R_mid * a_body * dt
                dp_acc_world = 0.5 * R_mid_step @ (a * dt_s**2) # dp = 0.5 * R_mid * a_body * dt^2
                                                              # (Matches 0.5 * dv_acc_world * dt_s)
                
                current_v_w += dv_acc_world + self.gravity * dt_s
                current_t_w += prev_state['v'] * dt_s + dp_acc_world + 0.5 * self.gravity * (dt_s**2)
                current_R_w_b = R_next_step
            final_timestamp_ns_for_state = float(imu_timestamps_ns_batch[0])

        else: # num_imu_measurements >= 2, perform step-by-step integration
            for i in range(num_imu_measurements - 1):
                # Timestamps for the current small interval
                ts0_ns = imu_timestamps_ns_batch[i]
                ts1_ns = imu_timestamps_ns_batch[i+1]

                dt_ns_step = ts1_ns - ts0_ns
                dt_s_step = 0.0
                if dt_ns_step <= 1e-3: # If dt is less than 1 nanosecond (effectively zero or negative)
                    dt_s_step = self.nominal_imu_sampling_period_seconds
                    if dt_s_step <= 1e-9: dt_s_step = 1e-3 # Ultimate fallback for safety
                else:
                    dt_s_step = dt_ns_step / 1e9

                # Midpoint IMU values for this step
                gyro0, gyro1 = imu_gyro_measurements[i], imu_gyro_measurements[i+1]
                acc0, acc1 = imu_acc_measurements[i], imu_acc_measurements[i+1]
                
                w_mid = 0.5 * (gyro0 + gyro1) 
                a_mid = 0.5 * (acc0 + acc1)   

                dtheta_vec_step = w_mid * dt_s_step
                dR_step = mat_exp(dtheta_vec_step)
                R_next_step = current_R_w_b @ dR_step

                # More robust Rmid: R_k @ Exp(0.5 * omega_mid * dt)
                R_mid_integration_step = current_R_w_b @ mat_exp(w_mid * (0.5 * dt_s_step))
                
                # dv_acc_world = R_mid @ a_body @ dt (from reference structure)
                dv_component_from_accel_world = R_mid_integration_step @ (a_mid * dt_s_step)
                # dp_acc_world = 0.5 * R_mid @ a_body @ dt^2 (from reference structure: 0.5 * dv_acc_world * dt)
                dp_component_from_accel_world = 0.5 * dv_component_from_accel_world * dt_s_step 
                
                # Update velocity: v_k+1 = v_k + dv_world_from_accel + g*dt
                v_next_step = current_v_w + dv_component_from_accel_world + self.gravity * dt_s_step
                
                # Update position: p_k+1 = p_k + v_k*dt + dp_world_from_accel + 0.5*g*dt^2
                current_t_w = current_t_w + current_v_w * dt_s_step + \
                              dp_component_from_accel_world + \
                              0.5 * self.gravity * (dt_s_step**2)
                
                current_R_w_b = R_next_step
                current_v_w = v_next_step
            
            final_timestamp_ns_for_state = float(imu_timestamps_ns_batch[-1])

        new_state_data = {
            'R': current_R_w_b, 't': current_t_w, 'v': current_v_w,
            'timestamp': final_timestamp_ns_for_state, # NANOSECONDS
        }
        self.states.append(new_state_data)
        
        if len(self.states) > self.window:
            self.finalized_trajectory.append(self.states.pop(0))

        return new_state_data

    def __call__(self, tstamp_current_frame_ns, input_tensor, 
                 intrinsics, curr_imu_data=None, save_slam_steps_path=None):
        
        img_gray_placeholder = np.array([[]], dtype=np.uint8) 

        imu_timestamps_ns_batch_np = np.array([float(tstamp_current_frame_ns)]) # Default if no IMU data
        imu_acc_batch_np = np.zeros((1,3))
        imu_gyro_batch_np = np.zeros((1,3))

        if curr_imu_data is not None and len(curr_imu_data) == 3 and \
           curr_imu_data[0] is not None and len(curr_imu_data[0]) > 0:
            imu_timestamps_ns_b, imu_acc_b, imu_gyro_b = curr_imu_data
            
            imu_timestamps_ns_batch_np = np.asarray(imu_timestamps_ns_b, dtype=float)
            imu_acc_batch_np = np.asarray(imu_acc_b, dtype=float)
            imu_gyro_batch_np = np.asarray(imu_gyro_b, dtype=float)

            if imu_timestamps_ns_batch_np.ndim == 0: 
                imu_timestamps_ns_batch_np = np.array([imu_timestamps_ns_batch_np.item()])
            if imu_acc_batch_np.ndim == 1 and imu_acc_batch_np.size > 0 : 
                imu_acc_batch_np = imu_acc_batch_np.reshape(-1, 3)
            if imu_gyro_batch_np.ndim == 1 and imu_gyro_batch_np.size > 0 : 
                imu_gyro_batch_np = imu_gyro_batch_np.reshape(-1, 3)
            
            # If, after processing, any array is empty while others are not, or shapes are wrong, handle fallback
            if imu_timestamps_ns_batch_np.size == 0 or \
               imu_acc_batch_np.shape[0] != imu_timestamps_ns_batch_np.size or \
               imu_gyro_batch_np.shape[0] != imu_timestamps_ns_batch_np.size:
                # Fallback to using frame timestamp and zero IMU readings for this call
                # print(f"Warning: Mismatched or empty IMU data in __call__ for ts {tstamp_current_frame_ns}. Using fallback.")
                imu_timestamps_ns_batch_np = np.array([float(tstamp_current_frame_ns)])
                imu_acc_batch_np = np.zeros((1,3))
                imu_gyro_batch_np = np.zeros((1,3))
        
        return self.process_frame(img_gray_placeholder, 
                                  imu_timestamps_ns_batch_np, 
                                  imu_acc_batch_np, 
                                  imu_gyro_batch_np)

    def reset(self, initial_R=None, initial_t=None, initial_v=None, initial_timestamp_ns=0.0):
        _initial_R = np.eye(3) if initial_R is None else np.asarray(initial_R, dtype=float).copy()
        _initial_t = np.zeros(3) if initial_t is None else np.asarray(initial_t, dtype=float).copy()
        _initial_v = np.zeros(3) if initial_v is None else np.asarray(initial_v, dtype=float).copy()

        if not is_rotation_matrix_valid(_initial_R): _initial_R = np.eye(3)
            
        self.states = [{
            'R': _initial_R, 't': _initial_t, 'v': _initial_v,
            'timestamp': float(initial_timestamp_ns), # NANOSECONDS
        }]
        self.finalized_trajectory.clear()
        self._initial_timestamp_ns = float(initial_timestamp_ns)

    def terminate(self): # Returns timestamps in NANOSECONDS
        # ... (terminate method largely unchanged, ensure it uses 'timestamp' field from states) ...
        all_history_for_output = self.finalized_trajectory + self.states
        if not all_history_for_output:
            return np.array([[0,0,0, 0,0,0,1]]), np.array([0.0]) 
        
        try:
            valid_states = [s for s in all_history_for_output if isinstance(s,dict) and 'timestamp' in s and 'R' in s and 't' in s]
            if not valid_states: return np.array([[0,0,0, 0,0,0,1]]), np.array([0.0])
            unique_states_by_timestamp_ns = {s['timestamp']: s for s in valid_states} # Keep last if duplicate ts
            sorted_unique_states = sorted(unique_states_by_timestamp_ns.values(), key=lambda s: s['timestamp'])
        except (TypeError, KeyError) :
             return np.array([[0,0,0, 0,0,0,1]]), np.array([0.0])

        poses_out_list = []
        tstamps_out_list_ns = [] 
        last_ts_ns = -np.inf # To ensure monotonicity for output if somehow states were not perfectly sorted

        for state_data in sorted_unique_states:
            ts_val_ns = state_data['timestamp']
            # Monotonicity check (optional if sorted_unique_states guarantees it)
            # if ts_val_ns < last_ts_ns and not np.isclose(ts_val_ns, last_ts_ns): continue 
            
            R_val, t_val = state_data['R'], state_data['t']
            if not (is_rotation_matrix_valid(R_val) and np.isfinite(t_val).all()): continue
            
            try:
                rot_q = SRot.from_matrix(R_val).as_quat() 
                current_pose = np.concatenate((t_val, rot_q))
                poses_out_list.append(current_pose)
                tstamps_out_list_ns.append(ts_val_ns) 
                last_ts_ns = ts_val_ns
            except Exception: pass

        if not poses_out_list:
            return np.array([[0,0,0, 0,0,0,1]]), np.array([0.0])

        return np.array(poses_out_list, dtype=float), np.array(tstamps_out_list_ns, dtype=float)