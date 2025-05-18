import numpy as np


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