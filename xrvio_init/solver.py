import numpy as np
from typing import List, Dict, Tuple

from .inertial import integrate_gyro, integrate_gyro_per_frame, preintegrate_accel, va_align, va_align_with_scale
from .visual import extract_and_match, estimate_rel_rot, solve_translation

def initialize(imu_ts: Dict[str, np.ndarray],
               frame_ts,
               gyro_meas: np.ndarray,
               acc_meas: np.ndarray,
               K: np.ndarray,
               frames: List[np.ndarray] = None,
               matches: Dict[Tuple[int,int], Tuple[np.ndarray, np.ndarray]] = None,
               weights=None
               ) -> Tuple[
                   Dict[Tuple[int,int], np.ndarray],  # refined rotations
                   Dict[Tuple[int,int], np.ndarray],  # translations
                   np.ndarray,                        # accel bias
                   np.ndarray                         # gravity vector #TODO: Add weight in signature
               ]:
    """
    Full decoupled rotation-translation init:
      1) Integrate gyro for initial rotations
      2) Extract & match features
      3) Refine rotations per-pair
      4) Solve translations per-pair
      5) Align accel for bias & gravity

    Args:
        frames: list of N images (H,W,3)
        imu_ts: dict with keys 'gyro' and 'acc', each (M,) timestamps
        gyro_meas: (M,3) rad/s
        acc_meas: (M,3) m/s^2
        K: (3x3) camera intrinsics

    Returns:
        R_dict: {(i,j): R_ij}
        t_dict: {(i,j): t_ij}
        bias_a: (3,) accelerometer bias
        gravity: (3,) gravity vector in ref frame
    """
    imu_times = imu_ts['gyro'] #TODO: fix format
    # 1) Gyro integration
    R_list = integrate_gyro(imu_times, gyro_meas)
    # R_list = integrate_gyro_per_frame(imu_ts['gyro'], gyro_meas, frame_ts)
        

    # 2) Visual matching
    # matches = extract_and_match(frames) if matches is None else matches
    assert matches is not None, "Must have matches dict"

    # 3) Refine rotations
    R_dict = {}
    for (i, j), (pts1, pts2) in matches.items():
        # prior rotation from i to j
        R_prior = R_list[j] @ R_list[i].T
        R_ref = estimate_rel_rot(R_prior, pts1, pts2, K)
        R_dict[(i, j)] = R_ref

    # 4) Solve translations
    t_dict = {}
    for key, R in R_dict.items():
        pts1, pts2 = matches[key]
        t = solve_translation(R, pts1, pts2, K)
        t_dict[key] = t

    # 5) VA-align for accel bias & gravity
    acc_mean = np.mean(acc_meas, axis=0)
    bias_guess, _g = va_align(list(R_list), acc_mean)
    delta_p_list, delta_v_list = preintegrate_accel(imu_times, acc_meas, bias_guess)

    scale, gravity, bias_a = va_align_with_scale(R_list, delta_p_list, t_dict)



    return R_dict, t_dict, bias_a, gravity, weights, scale
