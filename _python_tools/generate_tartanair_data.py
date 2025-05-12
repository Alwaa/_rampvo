import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation
import sys

from imu_integration import propagate_vectorized, xyPlot
from traj_smoother import correct_quaternion, smoother
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import multiprocessing

import tqdm


np.set_printoptions(precision=4, suppress=True)

def generate_poses_accel_vel(poses):    
    t_ = np.array(poses[:, 0:1], dtype=np.float64)
    idx = np.arange(0, t_.shape[0])[:, None]

    data = np.hstack((t_, idx, t_, idx, poses[:, 1:]))

    opt = {
        "INTERPOLATE": True,
        "SMOOTHPLOTS": False,
        "SPLINE": True
    }
    # Sample poses, velocities and accelerations at one tenth of the IMU rate,
    # equal to the image rate
    dt = 1.0 / np.round(1.0/np.mean(np.diff(data[:,2])))
    imu_dt = dt / 10.0
    
    data = correct_quaternion(data)
    trajectory = smoother(data, t_, opt, imu_dt)
    
    gt_imu = trajectory
    gt_poses = trajectory[0::10]

    return gt_poses, gt_imu

def generate_files(gt_poses, gt_imu, output_path, suffix=""):
    RIb = Rotation.from_quat(gt_imu[:, 7:11])
    accel_body_frame = RIb.apply(gt_imu[:, 11:14], inverse=True)

    t_poses = gt_poses[:, 0, None]
    t_imu = gt_imu[:, 0, None]
    np.savetxt(os.path.join(output_path, "stamped_groundtruth" + suffix + ".txt"), np.hstack((t_poses, gt_poses[:, 17:], gt_poses[:, 7:11])), header='t tx ty tz qx qy qz qw', fmt="%1.16f %1.16f %1.16f %1.16f %1.16f %1.16f %1.16f %1.16f")
    np.savetxt(os.path.join(output_path, "velocities" + suffix + ".txt"), np.hstack((t_poses, gt_poses[:, 14:17])), header='t vx vy vz', fmt="%1.16f %1.16f %1.16f %1.16f")
    np.savetxt(os.path.join(output_path, "accel" + suffix + ".txt"), np.hstack((t_poses, gt_poses[:, 4:7], gt_poses[:, 11:14])), header='t wx wy wz ax ay az', fmt="%1.16f %1.16f %1.16f %1.16f %1.16f %1.16f %1.16f")
    np.savetxt(os.path.join(output_path, "imu" + suffix + ".txt"), np.hstack((np.arange(0, t_imu.shape[0])[:, None], t_imu, gt_imu[:, 4:7], accel_body_frame)), header='idx t wx wy wz ax ay az', fmt="%d %1.16f %1.16f %1.16f %1.16f %1.16f %1.16f %1.16f")
    
def validate_imu(output_path, gen_plots=False, suffix=""):
    imu_data = np.loadtxt(os.path.join(output_path, "imu" + suffix + ".txt"), dtype=np.float64)
    vel_data = np.loadtxt(os.path.join(output_path, "velocities" + suffix + ".txt"), dtype=np.float64)
    pose_data = np.loadtxt(os.path.join(output_path, "stamped_groundtruth" + suffix + ".txt"), dtype=np.float64)
    assert vel_data.shape[0] == pose_data.shape[0]
    
    imu_dict = {
                    "ts": imu_data[:, 1],
                    "accels" : imu_data[:, 5:],
                    "gyros" : imu_data[:, 2:5]
    }
    init_state_dic = {
                    "ts" : pose_data[0, 0],
                    "pos" : pose_data[0, 1:4],
                    "ori" :  Rotation.from_quat(pose_data[0, 4:]).as_matrix(),
                    "vel" : vel_data[0, 1:]
    }
    traj_dic = propagate_vectorized(init_state_dic, imu_dict)

    ts = np.asarray(traj_dic["ts"])
    pos = np.asarray(traj_dic["pos"])
    ori = Rotation.from_matrix(traj_dic["ori"]).as_quat()
    traj = np.concatenate((ts[:, None], pos, ori), axis=1)

    if gen_plots:
        fig = plt.figure('2D views')
        gs = gridspec.GridSpec(2, 2)

        fig.add_subplot(gs[:, 0])
        xyPlot('XY plot', 'x', 'y',
            traj[:, 1:3], 'propagated imu', pose_data[:, 1:3], 'ground truth')
        fig.add_subplot(gs[0, 1])
        xyPlot('XY plot', 'x', 'z',
            traj[:, [1, 3]], 'propagated imu', pose_data[:, [1, 3]], 'ground truth')
        fig.add_subplot(gs[1, 1])
        xyPlot('XY plot', 'y', 'z',
            traj[:, [2, 3]], 'propagated imu', pose_data[:, [2, 3]], 'ground truth')
        
        plt.savefig(os.path.join(output_path, "2d_views" + suffix + ".png")         )
    
    error = np.linalg.norm(traj[-1, 1:4] - pose_data[-1, 1:4])
    
    return error

def generate_tartanair(sequence):
    output = "."
    
    for camera in ["_left", "_right"]:
        poses = np.loadtxt(os.path.join(output, sequence, "pose"+ camera + ".txt"), dtype=np.float64, comments='#')
        timestamps = np.loadtxt(os.path.join(output, sequence, "timestamps.txt"), dtype=np.float64, comments='#')[:, None]
        
        if timestamps.shape[0] != poses.shape[0]:
            timestamps = np.arange(0.0, poses.shape[0]*(1.0/30.0) * 1e9, (1.0/30.0) * 1e9)[:poses.shape[0], None]
            np.savetxt(os.path.join(output, sequence, "timestamps.txt"), timestamps)
        
        timestamps /= 1e9
        assert timestamps.shape[0] == poses.shape[0], ("Error on sequence: ", sequence)
        poses = np.hstack((timestamps, poses))
        ###################
        # 2. Generate GT poses, accel, velocities
        ###################
        gt_poses, gt_imu = generate_poses_accel_vel(poses)

        ###################
        # 3. Generate required files
        ###################
        generate_files(gt_poses, gt_imu, os.path.join(output, sequence), suffix=camera)
        ###################
        # 4. Verify integration error
        ###################   
        error = validate_imu(os.path.join(output, sequence), True, suffix=camera) 
        if error > 1.0:
            print("Integrated error is too high on sequence: ", sequence, error)
        
        print("Processed sequence: ", sequence, " on camera: ", camera)

#WHITELIST = ["ocean", "seasidetown"]
WHITELIST = []
BLACKLIST = ["westerndesert","neighborhood","office"]
if __name__ == "__main__":

    sequences = []
    for root, _dirs, _files in os.walk("."):
        if len(root.split("/")) == 4  and "P0" in root:
            print(root)
            if len(WHITELIST) > 0 and not root.split("/")[1] in WHITELIST:
                print("Skipped:0")
                continue

            if root.split("/")[1] in BLACKLIST:
                print("Skipped:0")
                continue
            sequences.append(root)


    with multiprocessing.Pool(8) as p, tqdm.tqdm(total=len(sequences)) as pbar:
        for result in p.imap(generate_tartanair, sequences):
            pbar.update()
            pbar.refresh()
    #workers_pool = multiprocessing.Pool(10)
    #workers_pool.map(generate_tartanair, sequences)
    #r = p_map(generate_tartanair, sequences)