import numpy as np
import os
import os.path as osp


def save_results(
    traj_ref, traj_est, scene, j=0, eval_type="None"
):
    # save poses for finer evaluations
    save_dir = osp.join(
        os.getcwd(),
        "trajectory_evaluation",
        f"{eval_type}",
        "trial_" + str(j),
        scene,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    time_ref = (traj_ref.timestamps * 10 ** -9)[..., np.newaxis]
    time_est = (traj_est.timestamps * 10 ** -9)[..., np.newaxis]
    np.savetxt(
        osp.join(save_dir, "stamped_groundtruth.txt"),
        np.concatenate((time_ref, traj_ref.positions_xyz, traj_ref.orientations_quat_wxyz), axis=1),
    )
    np.savetxt(
        osp.join(save_dir, "stamped_traj_estimate.txt"),
        np.concatenate((time_est, traj_est.positions_xyz, traj_est.orientations_quat_wxyz), axis=1),
    )