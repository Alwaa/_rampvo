#!/usr/bin/env python3
"""
Smoke-test the xrvio_init package after installation
"""
import numpy as np
import cv2

# Import modules to test
from xrvio_init.inertial import integrate_gyro, preintegrate_accel, va_align
from xrvio_init.visual import extract_and_match, estimate_rel_rot, solve_translation
from xrvio_init.solver import initialize
from xrvio_init.bundle_adjustment import bundle_adjustment

# Synthetic data setup for all tests
# 1. Inertial data
gyro_ts = np.linspace(0, 0.1, 5)
gyro_meas = np.zeros((5, 3))
acc_ts = gyro_ts.copy()
acc_meas = np.tile(np.array([0, 0, -9.81]), (5, 1))

# 2. Simple synthetic image pair
img1 = np.zeros((200, 200), dtype=np.uint8)
img2 = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(img1, (100, 100), 5, 255, -1)
cv2.circle(img2, (105, 100), 5, 255, -1)  # shifted dot
f1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
f2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
frames = [f1, f2]
K = np.eye(3)


# 1. Inertial tests
def test_inertial():
    R_list = integrate_gyro(gyro_ts, gyro_meas)
    for R in R_list:
        assert np.allclose(R, np.eye(3), atol=1e-8)
    print("❏ integrate_gyro OK")

    dp, dv = preintegrate_accel(acc_ts, acc_meas, bias_a=np.zeros(3))
    assert dv[-1][2] < -0.9
    print("❏ preintegrate_accel OK")

    bias, g = va_align(R_list, acc_meas.mean(axis=0))
    assert np.allclose(bias, 0, atol=1e-6)
    assert np.isclose(np.linalg.norm(g), 9.81, atol=1e-3)
    print("❏ va_align OK")

# 2. Visual tests
def test_visual():
    matches = extract_and_match(frames, max_features=500, ratio=0.8)
    assert (0, 1) in matches
    pts1, pts2 = matches[(0, 1)]
    assert pts1.shape[1] == 2 and pts2.shape[1] == 2
    print("❏ extract_and_match OK")

    # rotation refinement
    R_ref = estimate_rel_rot(np.eye(3), pts1, pts1, K)
    assert np.allclose(R_ref, np.eye(3), atol=1e-6)
    print("❏ estimate_rel_rot OK")

    # translation solving
    t = solve_translation(np.eye(3), pts1, pts2, K)
    assert t.shape == (3,)
    print("❏ solve_translation OK")

# 3. Solver test
def test_solver():
    imu_ts = {"gyro": gyro_ts, "acc": acc_ts}
    R_dict, t_dict, bias_a, g = initialize(frames, imu_ts, gyro_meas, acc_meas, K)
    assert (0, 1) in R_dict and (0, 1) in t_dict
    print("❏ initialize() OK")
    return R_dict, t_dict

# 4. Bundle adjustment test
def test_ba():
    R_dict, t_dict = test_solver()
    abs_poses = bundle_adjustment(R_dict, t_dict, max_iterations=5)
    assert 0 in abs_poses and 1 in abs_poses
    R0, t0 = abs_poses[0]
    assert R0.shape == (3, 3) and t0.shape == (3,)
    print("❏ bundle_adjustment OK")

if __name__ == "__main__":
    test_inertial()
    test_visual()
    test_solver()
    test_ba()
    print("\n✅ All xrvio_init tests passed!")
