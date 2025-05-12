"""Propagate imu measurements to get 6-DOF trajectory."""

import argparse

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import curve_fit
import os
import tqdm
import time

#import transformations as tf
from scipy.spatial.transform import Rotation

def fit_func(x, a, b, c):
    return a * x * x + b * x + c

def fixRotationMatrix(R):
    u, _, vt = np.linalg.svd(R)
    R_new = np.dot(u, vt)
    if np.linalg.det(R_new) < 0:
        R_new = -R_new
    return R_new

def getRotationAngle(R):
    return np.arccos((np.trace(R) - 1 - 1e-6) / 2)


def hat(v):
    v = v.flatten()
    R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return R


def mat_exp(omega):
    if len(omega) != 3:
        raise ValueError("tangent vector must have length 3")
    angle = np.linalg.norm(omega)

    # Near phi==0, use first order Taylor expansion
    if angle < 1e-10:
        return np.identity(3) + hat(omega)

    axis = omega / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)


def propagate(init_state_dic, imu_dic, ba=None, bg=None):
    g = np.array([0, 0, -9.81])

    ts = imu_dic["ts"]
    acc = imu_dic["accels"]
    gyr = imu_dic["gyros"]

    t0 = init_state_dic["ts"]
    p0 = init_state_dic["pos"]
    R0 = init_state_dic["ori"]
    v0 = init_state_dic["vel"]

    if ba == None:
        ba = np.zeros((3,))
    if bg == None:
        bg = np.zeros((3,))

    ts_b, p_wb, R_wb, v_wb = [], [], [], []
    # init. values
    ts_b.append(t0)
    p_wb.append(p0)
    R_wb.append(R0)
    v_wb.append(v0)

    t_prev = t0
    dts = range(1, len(ts))
    for i in tqdm.tqdm(dts):
        dt = ts[i] - t_prev

        w0 = gyr[i - 1, :]
        a0 = acc[i - 1, :]

        w1 = gyr[i, :]
        a1 = acc[i, :]

        w = 0.5 * (w0 + w1) - bg
        a = 0.5 * (a0 + a1) - ba

        # propagate rot
        dtheta = w * dt
        dR = mat_exp(dtheta)
        R_wbi = R_wb[-1] @ dR

        # propagate vel and pos
        Rmid = 0.5 * (R_wb[-1] + R_wbi)
        dv = Rmid @ (a * dt)
        dp = 0.5 * dv * dt
        gdt = g * dt
        gdt2 = gdt * dt
        v_wbi = v_wb[-1] + dv + gdt
        p_wbi = p_wb[-1] + v_wb[-1] * dt + dp + 0.5 * gdt2

        ts_b.append(ts[i])
        p_wb.append(p_wbi)
        R_wb.append(R_wbi)
        v_wb.append(v_wbi)

        t_prev = ts[i]

    traj = {}
    traj["ts"] = ts_b
    traj["pos"] = p_wb
    traj["ori"] = R_wb
    traj["vel"] = v_wb

    return traj

def propagate_vectorized(init_state_dic, imu_dic, ba=None, bg=None):
    g = np.array([0, 0, -9.81])

    ts = imu_dic["ts"]
    acc = imu_dic["accels"]
    gyr = imu_dic["gyros"]

    t0 = init_state_dic["ts"]
    p0 = init_state_dic["pos"]
    R0 = init_state_dic["ori"]
    v0 = init_state_dic["vel"]

    if ba is None:
        ba = np.zeros((3,))
    if bg is None:
        bg = np.zeros((3,))

    dt = np.diff(ts)
    w = 0.5 * (gyr[:-1] + gyr[1:]) - bg
    a = 0.5 * (acc[:-1] + acc[1:]) - ba

    dtheta = w * dt[:, None]
    dR = np.array([mat_exp(dtheta_i) for dtheta_i in dtheta])
    R_wb = np.zeros((len(ts), 3, 3))
    R_wb[0] = R0
    for i in range(1, len(ts)):
        R_wb[i] = R_wb[i-1] @ dR[i-1]

    Rmid = 0.5 * (R_wb[:-1] + R_wb[1:])
    dv = np.einsum('ijk,ik->ij', Rmid, a * dt[:, None])
    dp = 0.5 * dv * dt[:, None]
    gdt = g * dt[:, None]
    gdt2 = gdt * dt[:, None]

    v_wb = np.zeros((len(ts), 3))
    v_wb[0] = v0
    v_wb[1:] = np.cumsum(dv + gdt, axis=0) + v0

    p_wb = np.zeros((len(ts), 3))
    p_wb[0] = p0
    p_wb[1:] = np.cumsum(v_wb[:-1] * dt[:, None] + dp + 0.5 * gdt2, axis=0) + p0

    traj = {}
    traj["ts"] = ts
    traj["pos"] = p_wb
    traj["ori"] = R_wb
    traj["vel"] = v_wb

    return traj

def compare_propagation_methods(init_state_dic, imu_dic, ba=None, bg=None):
    start = time.time()
    traj1 = propagate(init_state_dic, imu_dic, ba, bg)
    print(f"Time for non-vectorized method: {time.time() - start}")
    start = time.time()
    traj2 = propagate_vectorized(init_state_dic, imu_dic, ba, bg)
    print(f"Time for vectorized method: {time.time() - start}")
    for key in traj1:
        if not np.allclose(traj1[key], traj2[key], rtol=1e-5, atol=1e-6):
            print(f"Difference found in {key}")
            return False
    print("Both methods produce the same results.")
    return True

def xyPlot(title, labelx, labely,
    vec1, label1,
    vec2 = None, label2 = None,
    vec3 = None, label3 = None,
    vec4 = None, label4 = None):
    plt.plot(vec1[:, 0], vec1[:, 1], label=label1)
    if vec2 is not None:
        plt.plot(vec2[:, 0], vec2[:, 1], label=label2)
    if vec3 is not None:
        plt.plot(vec3[:, 0], vec3[:, 1], label=label3)
    if vec4 is not None:
        plt.plot(vec4[:, 0], vec4[:, 1], label=label4)
    plt.grid()
    plt.legend()
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_fn", type=str, default=None,
                        help=".txt file containing IMU")
    parser.add_argument("--out_fn", type=str, default=None,
                        help="save trajectory to this file")
    parser.add_argument("--gt_fn", type=str, default=None,
                        help="Trajectory ground truth as .txt")
    parser.add_argument('--no_plots', action='store_true', help="Plot graphs")
    parser.add_argument('--suffix', type=str, default="",)
    
    args = parser.parse_args()

    print("Reading from .txt file: %s" % args.in_fn)
    imu_meas = np.loadtxt(args.in_fn)  # [ts,wx,wy,wz,ax,ay,az]

    gt_meas = None
    if args.gt_fn is not None:
        gt_meas = np.loadtxt(args.gt_fn)  # [ts,wx,wy,wz,ax,ay,az]

    # propagate
    initial_state_dic = {}
    initial_state_dic["ts"] = imu_meas[0, 0]
    initial_state_dic["pos"] = np.zeros((3,))
    initial_state_dic["ori"] = np.identity(3)
    initial_state_dic["vel"] = np.zeros((3,))
    if args.gt_fn is not None:
        initial_state_dic["pos"] = gt_meas[0, 1:4]
        initial_state_q = np.quaternion(gt_meas[0, 4:][3], gt_meas[0, 4:][0], gt_meas[0, 4:][1], gt_meas[0, 4:][2])
        initial_state_dic["ori"] = quaternion.as_rotation_matrix(initial_state_q)
        initial_state_dic["vel"] = (gt_meas[1, 1:4] - gt_meas[0, 1:4]) / (gt_meas[1, 0] - gt_meas[0, 0])

    meas_dic = {}
    meas_dic["ts"] = imu_meas[:, 0]
    meas_dic["accels"] = imu_meas[:, 4:]
    meas_dic["gyros"] = imu_meas[:, 1:4]

    #print("Comparing propagation methods ...")
    #if compare_propagation_methods(initial_state_dic, meas_dic):
    #    print("Propagation methods are consistent.")
    #else:
    #    print("Propagation methods are not consistent.")

    print("Propagating IMU ...")
    traj_dic = propagate(initial_state_dic, meas_dic)

    ts = np.asarray(traj_dic["ts"])
    pos = np.asarray(traj_dic["pos"])
    #ori = np.array([xyzwQuatFromMat(R) for R in traj_dic["ori"]])
    ori = Rotation.from_matrix(traj_dic["ori"]).as_quat()
    traj = np.concatenate((ts[:, None], pos, ori), axis=1)

    log_folder = None
    if args.out_fn is not None:
        log_folder = args.out_fn + "/imu_integration/"
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)
        
        np.savetxt(log_folder + "/stamped_traj_" + args.suffix + ".txt", traj, fmt="%1.12f", header="ts x y z qx qy qz qw")
        print("Integrated trajectory saved to %s" % args.out_fn)
        
    print("Plotting ...")
    # Plot position
    fig = plt.figure('2D views')
    gs = gridspec.GridSpec(2, 2)

    fig.add_subplot(gs[:, 0])
    xyPlot('XY plot', 'x', 'y',
           traj[:, 1:3], 'propagated imu', gt_meas[:, 1:3], 'ground truth')
    fig.add_subplot(gs[0, 1])
    xyPlot('XY plot', 'x', 'z',
           traj[:, [1, 3]], 'propagated imu', gt_meas[:, [1, 3]], 'ground truth')
    fig.add_subplot(gs[1, 1])
    xyPlot('XY plot', 'y', 'z',
           traj[:, [2, 3]], 'propagated imu', gt_meas[:, [2, 3]], 'ground truth')
    
    if args.out_fn is not None:
        plt.savefig(args.out_fn + "/imu_integration/2d_views_" + args.suffix + ".png")
        
    zyx_eulers = [] 
    for v in traj:
        zyx_eulers.append(Rotation.from_quat(v[4:]).as_euler('zyx', degrees=True))
    zyx_eulers = np.asarray(zyx_eulers)

    fig = plt.figure('Rotations')
    plt.plot(ts, zyx_eulers[:, 0], label="yaw", marker='o', alpha=0.5)
    plt.plot(ts, zyx_eulers[:, 1], label="pitch", marker='o', alpha=0.5)
    plt.plot(ts, zyx_eulers[:, 2], label="roll", marker='o', alpha=0.5)
    
    pos_error = []
    ts_error = []
    start_ts = None
    if args.gt_fn is not None:
        zyx_eulers_gt = []
        orient_error = []
        for i, v in enumerate(gt_meas):
            zyx_eulers_gt.append(Rotation.from_quat(v[4:]).as_euler('zyx', degrees=True))
            try:
                pos_error.append(np.linalg.norm(v[1:4] - traj[i, 1:4]))
                
                if start_ts is None:
                    start_ts = v[0]
                ts_error.append(v[0] - start_ts)
            except:
                pass
            
        zyx_eulers_gt = np.asarray(zyx_eulers_gt)
        pos_error = np.asarray(pos_error)
        ts_error = np.asarray(ts_error)
        
        plt.plot(gt_meas[:, 0], zyx_eulers_gt[:, 0], label="gt yaw"  )
        plt.plot(gt_meas[:, 0], zyx_eulers_gt[:, 1], label="gt pitch")
        plt.plot(gt_meas[:, 0], zyx_eulers_gt[:, 2], label="gt roll" )
        
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('rot. angle [deg]')
    
    if log_folder is not None and not args.no_plots:
        plt.savefig(log_folder + "/eulers_" + args.suffix + ".png")
        
    fig = plt.figure('Quaternion')
    sign = 1.0
    if args.gt_fn is not None:
        if np.sign(gt_meas[0, 7]) != np.sign(traj[0,7]):
            sign = -1.0
    for i, v in enumerate(traj):
        traj[i, 4:] *= sign
    
    plt.plot(traj[:, 0], traj[:, 4], label="x", marker='o', alpha=0.5)
    plt.plot(traj[:, 0], traj[:, 5], label="y", marker='o', alpha=0.5)
    plt.plot(traj[:, 0], traj[:, 6], label="z", marker='o', alpha=0.5)
    plt.plot(traj[:, 0], traj[:, 7], label="w", marker='o', alpha=0.5)
    
    if args.gt_fn is not None:
        plt.plot(gt_meas[:, 0], gt_meas[:, 4], label="gt x")
        plt.plot(gt_meas[:, 0], gt_meas[:, 5], label="gt y")
        plt.plot(gt_meas[:, 0], gt_meas[:, 6], label="gt z")
        plt.plot(gt_meas[:, 0], gt_meas[:, 7], label="gt w")
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('quaternion value')

    if log_folder is not None and not args.no_plots:
        plt.savefig(log_folder + "/quaternions_" + args.suffix + ".png")
        
    idx = np.array([i for i in range(len(pos_error))])
    
    fig = plt.figure('Position error')
    plt.plot(ts_error, pos_error, label="error")
    
    popt, pcov = curve_fit(fit_func, ts_error, pos_error)
    plt.plot(ts_error, fit_func(ts_error, *popt), 'g--',
         label='fit: a=%1.4e, b=%1.4e, c=%1.4e' % tuple(popt))
    
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('error [m]')
    if log_folder is not None and not args.no_plots:
        plt.savefig(log_folder + "/pose_error_" + args.suffix + ".png")
        
    if not args.no_plots:
        plt.show()