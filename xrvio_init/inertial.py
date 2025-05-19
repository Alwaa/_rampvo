import numpy as np
from scipy.optimize import lsq_linear

def rodrigues_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Compute a rotation matrix using Rodrigues' formula.

    Args:
        axis: 3-vector, axis of rotation (need not be unit length).
        angle: rotation angle in radians.
    Returns:
        3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def integrate_gyro_per_frame(gyro_ts, gyro_meas, frame_ts):
    # returns R_frames[k] = rotation from frame 0 to frame k
    R_frames = []
    for k in range(len(frame_ts)-1):
        # select imu samples in [frame_ts[k], frame_ts[k+1]]
        mask = (gyro_ts >= frame_ts[k]) & (gyro_ts <= frame_ts[k+1])
        R_seg = integrate_gyro(gyro_ts[mask], gyro_meas[mask])[-1]
        R_frames.append(R_seg)
    # prepend identity for frame 0
    return [np.eye(3)] + R_frames

def integrate_gyro(gyro_ts: np.ndarray, gyro_meas: np.ndarray) -> list:
    """
    Integrate gyroscope measurements to obtain relative rotations over time.

    Uses midpoint rule on angular rates to compute incremental rotations.

    Args:
        gyro_ts: (N,) array of timestamps (seconds) for gyro measurements.
        gyro_meas: (N,3) array of angular rates [rad/s] in body frame.

    Returns:
        R_list: list of (3x3) rotation matrices; R_list[0] = identity,
                R_list[i] rotates vectors from frame 0 to frame i.
    """
    assert gyro_ts.ndim == 1 and gyro_meas.ndim == 2
    N = gyro_ts.shape[0]
    R = np.eye(3)
    R_list = [R.copy()]
    for i in range(N - 1):
        dt = gyro_ts[i + 1] - gyro_ts[i]
        # midpoint angular velocity
        w_mid = 0.5 * (gyro_meas[i] + gyro_meas[i + 1])
        angle = np.linalg.norm(w_mid * dt)
        if angle > 1e-12:
            dR = rodrigues_matrix(w_mid, angle)
        else:
            dR = np.eye(3)
        R = R @ dR
        R_list.append(R.copy())
    return R_list


def preintegrate_accel(acc_ts: np.ndarray,
                        acc_meas: np.ndarray,
                        bias_a: np.ndarray) -> tuple:
    """
    Pre-integrate accelerometer measurements to compute delta velocities and positions.

    Args:
        acc_ts: (N,) array of timestamps (seconds) for accel measurements.
        acc_meas: (N,3) array of accelerations [m/s^2] in body frame.
        bias_a: (3,) accelerometer bias to subtract.

    Returns:
        delta_p: list of (3,) delta positions from frame 0 to i.
        delta_v: list of (3,) delta velocities from frame 0 to i.
    """
    assert acc_ts.ndim == 1 and acc_meas.ndim == 2
    N = acc_ts.shape[0]
    v = np.zeros(3)
    p = np.zeros(3)
    delta_v = [v.copy()]
    delta_p = [p.copy()]
    for i in range(N - 1):
        dt = acc_ts[i + 1] - acc_ts[i]
        a_i = acc_meas[i] - bias_a
        a_i1 = acc_meas[i + 1] - bias_a
        a_mid = 0.5 * (a_i + a_i1)
        # velocity update
        v = v + a_mid * dt
        # position update
        p = p + v * dt + 0.5 * a_mid * (dt**2)
        delta_v.append(v.copy())
        delta_p.append(p.copy())
    return delta_p, delta_v


def _old_va_align(R_list: list, acc_mean: np.ndarray) -> tuple:
    """
    Estimate accelerometer bias and gravity vector by visual-accelerometer alignment.

    Solves for bias b and gravity g (|g| = 9.81) from static measurements:
        R_i * b + g = R_i * acc_mean
    for all rotations R_i in R_list.

    Args:
        R_list: list of (3x3) rotation matrices from camera frame i to ref frame.
        acc_mean: (3,) mean accelerometer measurement in camera (body) frame.

    Returns:
        bias_a: (3,) estimated accelerometer bias.
        g: (3,) gravity vector in reference frame (magnitude 9.81 m/s^2).
    """
    m = len(R_list)
    # Build linear system: for each i, R_i b + g = R_i acc_mean
    M = np.zeros((3*m, 6))
    v = np.zeros(3*m)
    for i, R in enumerate(R_list):
        M[3*i:3*i+3, 0:3] = R
        M[3*i:3*i+3, 3:6] = np.eye(3)
        v[3*i:3*i+3] = R.dot(acc_mean)
    # Solve least-squares for [b; g]
    sol, *_ = np.linalg.lstsq(M, v, rcond=None)
    b = sol[0:3]
    g_est = sol[3:6]
    # Enforce gravity magnitude
    g = g_est / np.linalg.norm(g_est) * 9.81
    return b, g


def va_align(R_list: list, acc_mean: np.ndarray) -> tuple:
    """
    Estimate accelerometer bias and gravity vector by visual-accelerometer alignment.

    Assumes the first rotation R_list[0] aligns the camera (body) frame to a world
    reference where gravity is [0,0,-9.81]. Computes:
        g_world = [0,0,-9.81]
        g_body0 = R_list[0].T @ g_world
        bias = acc_mean - g_body0

    Args:
        R_list: list of (3x3) rotation matrices from camera frame i to ref frame
        acc_mean: (3,) mean accelerometer measurement in camera (body) frame

    Returns:
        bias: (3,) estimated accelerometer bias.
        g_world: (3,) gravity vector in reference frame (magnitude 9.81 m/s^2).
    """
    # Define gravity in world frame
    g_world = np.array([0.0, 0.0, -9.81])
    # Transform gravity into body (camera) frame at time 0
    R0 = R_list[0]
    g_body0 = R0.T @ g_world
    # Bias is the difference between measured mean accel and true gravity in body0
    bias = acc_mean - g_body0
    return bias, g_world


def va_align_with_scale(
    R_list: list[np.ndarray],
    delta_p: list[np.ndarray],
    t_dict: dict[tuple[int,int], np.ndarray]
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Solve for scale s, gravity vector g, and accel bias b in camera frame.

    We set up the linear system arising from:
      s * t_ij  ≈ R_j @ (delta_p[j] - delta_p[i])  - 0.5 * g * dt^2_ij
    and also incorporate the bias term via:
      R_list[i] @ (acc_mean - b) ≈ g

    Args:
        R_list:  list of (3×3) rotations from time-0 to time-i
        delta_p: list of (3,) numpy arrays of pre-integrated position from time-0 to i
        t_dict:  {(i,j) : unit-translation t_ij from solve_translation}

    Returns:
        s:   scalar metric scale
        g:   (3,) gravity vector in camera frame
        b:   (3,) estimated accelerometer bias
    """
    # 1) Collect all pairs and build least-squares for scale & gravity
    # We'll solve: s * t_ij + 0.5*g*dt2 = R_j (dp_j - dp_i)
    # => [t_ij, 0.5*dt2*I] [s; g] = rhs_ij
    A_rows = []
    b_rows = []
    # For bias, we also solve R0 (b) + g = acc_mean, but we can add as a single constraint
    # after we compute mean accel.

    # Here we assume constant frame-rate dt between consecutive frames:
    #   dt_ij = 1 (or known). If actual timestamps are known, replace dt=ts[j]-ts[i].
    for (i, j), t_unit in t_dict.items():
        dp_i = delta_p[i]
        dp_j = delta_p[j]
        dt = 1.0  # or your actual dt_list[j]-dt_list[i]
        rhs = R_list[j] @ (dp_j - dp_i)
        # build row-block: [ t_ij , 0.5*dt^2 * I3 ]
        A_rows.append(np.hstack([ t_unit.reshape(3,1), 0.5*(dt**2)*np.eye(3) ]))  # shape (3,4)
        b_rows.append(rhs.reshape(3,1))

    # Stack into big least-squares: A @ [s; g] = b
    A = np.vstack(A_rows)  # shape (3M, 4)
    b = np.vstack(b_rows)  # shape (3M, 1)


    # Solve [s; g] by normal equations or lstsq
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    s = float(x[0])
    g = x[1:4, 0]

    # # bounds: s ∈ [0, +∞), g_x/g_y/g_z ∈ (−∞, +∞)
    # lower = np.array([0.0, -np.inf, -np.inf, -np.inf])
    # upper = np.array([np.inf, np.inf, np.inf, np.inf])

    # res = lsq_linear(A, b.ravel(), bounds=(lower, upper), lsmr_tol='auto')
    # x = res.x.reshape(4,1)
    # s = float(x[0,0])
    # g = x[1:4,0]



    # 2) Now recover accelerometer bias b via mean-acc alignment:
    #    R0 @ (acc_mean - b) = g  =>  R0 b = acc_mean - R0^T g
    # We can compute acc_mean from delta_p double derivative or directly from measurements.
    # For simplicity, let's assume acc_mean known externally or zero-mean gravity.
    # If you have raw acc_meas list:
    #     acc_mean = np.mean(acc_meas, axis=0)
    # else we approximate with g direction:
    acc_mean = None  # you must supply this from your data
    if acc_mean is None:
        # fallback: assume bias=0
        b = np.zeros(3)
    else:
        R0 = R_list[0]
        # R0 @ (acc_mean - b) = g  =>  b = acc_mean - R0.T @ g
        b = acc_mean - R0.T @ g

    return s, g, b