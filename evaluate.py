import asyncio
import os

import datetime
from queue import Queue
import sys
import glob
import threading
import time
import cv2
import yaml
import json
import torch
import argparse
import torchvision
import numpy as np
import polars as pl
import os.path as osp
from tqdm.asyncio import tqdm as atqdm
from pathlib import Path
from evo.core import sync
from functools import partial
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D

import matplotlib.pyplot as plt
from evo.tools import plot

import warnings
import matplotlib

# filtering faulty deprication warnings TODO: Remove when updated/fixed
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


from ramp.lietorch.groups import SE3
from utils.seed_everything import seed_everything
from ramp.data_readers.TartanEvent import TartanEvent
from utils.rotation_error_with_euler import rot_error_with_alignment_from_pose3d
from utils.eval_utils import (
    read_eds_format_poses,
    read_stereodavis_format_poses,
    read_tartan_format_poses,
    read_moonlanding_format_poses
)
from data import H5EventHandle
from ramp.utils import (
    Timer,
    input_resize,
    normalize_image,
    print_timing_summary,
    save_output_for_COLMAP
)
from ramp.config import cfg as VO_cfg
from ramp.Ramp_vo import Ramp_vo

from utils.new_eval_utils import save_results, Visualizer


from config import (
    QUEUE_BUFFER_SIZE,
    QUEUE_ASYNC_MIN_SIZE,
    QUEUE_ASYNC_STALL_TIMEOUT,
    QUEUE_ASYNC_SLEEP_BETWEEN_STARTUP_CHECKS, 
    LOADING_THREAD_TORCH_INTRA_OP_THREAD_NUM,
    IMU_TESTING,
    TARTAN_PATH_PREFIX,
    VISUALIZATION,
)

seed_everything(seed=1234)
sys.setrecursionlimit(100000)

from constans import TARTAN_2_XYZQ4, TARTAN_2_XYZW3


# TODO: Investigate 'standard_pose_format' variable that was unused, but set for EDS and SteroDavis
def set_global_params(K_path=None, resize_to=None):
    global fx, fy, cx, cy

    if K_path is None or not os.path.exists(K_path):
        fx, fy, cx, cy = [320, 320, 320, 240]
        print("Using default intrinsics", [fx, fy, cx, cy])
        return (fx, fy, cx, cy)
    else:
        # Load the YAML file
        with open(K_path, "r") as file:
            data = yaml.safe_load(file)

        # Extract the intrinsics
        intrinsics = data["cam0"]["intrinsics"]

        # Extract the individual components
        fx, fy, cx, cy = intrinsics

    if resize_to is not None:
        resolution = data["cam0"]["resolution"]
        slack = np.array(resize_to) - np.array(resolution)
        d_cx, d_cy = slack[0] / 2, slack[1] / 2
        cx = cx + d_cx
        cy = cy + d_cy

    print("Using intrinsics from {}".format(K_path), (fx, fy, cx, cy))
    return (fx, fy, cx, cy)


def async_data_loader_all_events(
    queue: Queue, config, full_scene, downsample_fact=1, norm_to=None, extension=".png"
):
    """
    (Image, EventVox, Intrinsics, Mask, FrameIndex, )
    """
    suffix = "_left"

    torch.set_num_threads(LOADING_THREAD_TORCH_INTRA_OP_THREAD_NUM)

    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    images_paths = osp.join(full_scene, "image_left", "*{}".format(extension))
    imfiles = sorted(glob.glob(images_paths))
    evfile = osp.join(full_scene, "events.h5")
    print("EventFile: ", evfile)
    TartanEvent_loader = TartanEvent(config=config, path=full_scene, just_validation = True)
    timestamps = np.loadtxt(osp.join(full_scene, "timestamps.txt"))

    # idx t wx wy wz ax ay az
    imu_data = np.loadtxt(osp.join(full_scene, "imu" + suffix + ".txt"), dtype=np.float64)
    # t vx vy vz
    # vel_data = np.loadtxt(osp.join(full_scene, "velocities" + suffix + ".txt"), dtype=np.float64)


    imu_t_s = imu_data[:, 1] #seconds
    imu_t_ns = imu_t_s * 1e9 # Convert to NANOSECONDS

    imu_measurements_raw = imu_data[:, 2:8]
    imu_measurements = imu_measurements_raw[:, TARTAN_2_XYZW3]

    imu_gyro     = imu_measurements[:, 0:3]
    imu_accel    = imu_measurements[:, 3:6]
    def imu_slice(start_t_ns, end_t_ns):
        _i0 = np.searchsorted(imu_t_ns, start_t_ns, side="left")
        _i1 = np.searchsorted(imu_t_ns, end_t_ns,   side="right")
        return imu_t_ns[_i0:_i1], imu_gyro[_i0:_i1], imu_accel[_i0:_i1]
    
    

    # skip first element (no events for it)
    image_files = imfiles[1 :: downsample_fact]
    corresponding_timestamps = timestamps[1 :: downsample_fact]

    # load events and compute how many are they
    event = H5EventHandle.from_path(Path(evfile))
    n_events = len(event.t)
    n_events_selected = TartanEvent_loader.num_events_selected
    n_events_voxels = n_events // n_events_selected
    corr_events_timestamps = event.t[0:n_events:n_events_selected][1::]

    time_vicinity = (
        np.subtract.outer(corr_events_timestamps, corresponding_timestamps) ** 2
    )
    corresponding_frame_indices = np.argmin(time_vicinity, axis=1)
    corresponding_events_indices = np.argmin(time_vicinity, axis=0)
    

    loading_bar = atqdm(range(n_events_voxels))
    loading_bar.set_description("Async Importing Images+Events")
 
    i1 = 0
    for i in loading_bar:
        i0, i1 = i1, i1 + n_events_selected

        event_voxel = TartanEvent_loader.events_from_indices(
            event=event, i_start=i0, i_stop=i1
        )

        frame_ind = corresponding_frame_indices[i]

        imfile = image_files[frame_ind]
        image = torchvision.io.read_image(imfile)
        image = normalize_image(images=image, norm_img_to=norm_to)

        tstamp = corresponding_timestamps[frame_ind]

        # plot_events(event, image, i0, i1, i) #TODO: Fix absolute paths
        # the index of the smallest error between the event voxel timestamp and the image timestamp is event index
        event_ind = corresponding_events_indices[frame_ind]

        mask = bool(event_ind == i) #Explicit bool as to not turn an np.bool_ into tensor (depricated)

        if IMU_TESTING:
            ts_start_ns, ts_end_ns = event.t[i0], event.t[i1 - 1] # first and last event in voxel
            ts_start, ts_end = ts_start_ns, ts_end_ns

            imu_tuple = imu_slice(ts_start, ts_end) #imu_ts, imu_gyro, imu_accel
            
            tup = (image, event_voxel, intrinsics, torch.tensor([mask]), frame_ind, tstamp, imu_tuple)
        else:    
            tup = (image, event_voxel, intrinsics, torch.tensor([mask]), frame_ind, tstamp)
            
        queue.put(tup)

    queue.put(None) #DONE:=0

    return "Done!:0"

def base_unpacker(item_tuple: tuple) -> tuple:
    image, events, intrinsics, mask, f_i, tstamp = item_tuple
    im = image[None, None, ...].cuda()
    ev = events[None, None, ...].float().cuda()
    intr = intrinsics.cuda()
    mask.cuda()

    return (im, ev, intr, mask, f_i, tstamp)

#TODO: Should be actual imu data in the end
def imu_unpacker(item_tuple:tuple) -> tuple:
    image, events, intrinsics, mask, f_i, tstamp, imu_tuple = item_tuple
    im = image[None, None, ...].cuda()
    ev = events[None, None, ...].float().cuda()
    intr = intrinsics.cuda()
    mask.cuda()

    # torch.from_numpy(imu_g).float()

    return (im, ev, intr, mask, f_i, tstamp, imu_tuple)

async def _queue_iterator(data_queue: Queue):
    last_size, last_growth  = data_queue.qsize(),time.monotonic()

    while data_queue.qsize() < QUEUE_ASYNC_MIN_SIZE:
        size = data_queue.qsize()
        atqdm.write(f"[Evaluator] waiting for buffer (size={size}); need {QUEUE_ASYNC_MIN_SIZE}")

        if size > 1 and size != last_size:
            last_growth = time.monotonic()
        elif time.monotonic() - last_growth > QUEUE_ASYNC_STALL_TIMEOUT:
            if size < 2:
                print("No Event Loaded after {QUEUE_ASYNC_STALL_TIMEOUT}s\n\tCheck Setup!\n")
                break
            # Otherwise, if there are events, break out of checking loop
            atqdm.write(f"[Evaluator] queue size={size} stalled for {QUEUE_ASYNC_STALL_TIMEOUT}s, proceeding")
            break

        await asyncio.sleep(QUEUE_ASYNC_SLEEP_BETWEEN_STARTUP_CHECKS)
        last_size = size
    
    unpacker = imu_unpacker if IMU_TESTING else base_unpacker

    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, data_queue.get)
        if item is None:
            break #No more yield

        yield unpacker(item)

def resize_input(image, events):
    default_shape = torch.tensor([480, 640])
    data_shape = image.shape[-2:]
    if data_shape != default_shape:
        image, events = input_resize(
            image, events, desired_ht=data_shape[0] + 1, desired_wh=data_shape[1] + 1
        )

    image = (
        torch.stack((image, image, image), dim=3)[0, ...]
        if image.shape[-3] == 1
        else image
    )
    image.squeeze(0).squeeze(0)
    return image, events


@torch.no_grad()
async def async_run(cfg_VO, network, eval_cfg, data_queue: Queue, enable_timing = False, save_slam_steps_path= None):
    """Run the slam on the given data_list and return the trajectory and timestamps

    Args:
        cfg_VO: config for the slam
        network: the network to use for the slam
        eval_cfg: config for the evaluation
        data_queue: list of tuples (image, events, intrinsics)

    Returns:
        traj_est: the estimated trajectory
        tstamps: the timestamps of the estimated trajectory
    """

    img_timestamps = []
    train_cfg = eval_cfg["data_loader"]["train"]["args"]
    slam = Ramp_vo(cfg=cfg_VO, network=network, train_cfg=train_cfg, enable_timing=enable_timing)
    
    if VISUALIZATION:
        assert False, "Test to make sure server doesnt break"
        visualizer = Visualizer() 

    evaluation_iter_bar = atqdm(_queue_iterator(data_queue))
    evaluation_iter_bar.set_description("Async Evaluating")

    #TODO: Shouldn't duplicate the whole thing + Queue enumerate
    t = 0
    async for iter_tuple in evaluation_iter_bar:
        
        if IMU_TESTING:
            (image, events, intrinsics, mask, f_i, tstamp, imu_tuple) = iter_tuple
        else:
            (image, events, intrinsics, mask, f_i, tstamp) = iter_tuple
            imu_tuple = None

        image, events = resize_input(image, events)
        with Timer("SLAM", enabled=enable_timing):
            slam(t, 
                input_tensor=(events, image, mask),
                timestamp=tstamp,
                intrinsics=intrinsics, 
                curr_imu_data=imu_tuple)
        t += 1
    
        if mask:
            img_timestamps.append(f_i)
        else:
            continue
            
        if slam.current_patch_coords_feat is not None and VISUALIZATION:
            # Coords are in feature map scale (H/RES, W/RES) of image_resized_for_slam
            patch_centers_on_img = slam.current_patch_coords_feat.cpu().numpy() * slam.RES
            patch_display_size_pixels = slam.P_feat * slam.RES
            display_img = _image_to_cv_fmt(image)                
            
            visualizer.draw_patch_locations(
                    display_img,
                    patch_centers_on_img,
                    patch_display_size_pixels
                )
            
            #time.sleep(0.05)
            visualizer.overrite_est_traj_(slam.get_current_trajectory()[0])
            visualizer.plot_trajectory_2d()
            try:

                # visualizer.update_trajectories(
                # gt_pose=None,
                # pre_update_buffer=slam.pre_update_poses_for_viz,
                # post_update_buffer=slam.poses_[:slam.n], # slam.poses points to the updated poses_ buffer
                # num_valid_poses=slam.n
                # )
                #print(f_i, slam.poses[0][slam.n])
                #visualizer.add_pose(pose_est=slam.stabilized_pose)
                #visualizer.add_pose(pose_est=slam.poses_[slam.n - 1])

                # current_est_pose = slam.poses.get(tstamp.item())

                # # Add the current ground truth and estimated poses to the visualizer
                # if current_est_pose is not None:
                #     visualizer.add_pose(pose_gt=gt_pose, pose_est=current_est_pose)

                pass
            except:
                print("NO TRAJ VIZ")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    
    if t == 0:
        return None

    poses, tstamps = slam.terminate()

    try:
        #poses_est = poses.copy()
        #poses_est[:,:3] = poses_est[:,:3]*5.3

        print(slam.delta_poses[:3])
        print(slam.test_imu_poses[:3])

        global g_traj_ref 
        poses_gt = g_traj_ref.positions_xyz
        poses_gt = poses_gt - poses_gt[0]

        poses_delta_int = torch.cat(slam.delta_poses).numpy()

        poses_delta_key = torch.cat(slam.key_delta_poses).numpy()

        poses_imu = torch.cat(slam.test_imu_poses).numpy()
        covs_imu = torch.stack(slam.test_imu_covs, dim = 0).numpy()

        plot_2d_3d_comps([(poses_imu, covs_imu), (poses_delta_int, None), (poses_delta_key, None)], 
                         ["PyPose", "Deltas Preint.", "Deltas KeyFrames Tensor"])

        # plt.figure(figsize=(5, 5))
        # plt.plot(np.diff(slam.test_imu_times))
        # plt.title("PyPose IMU times")
        # plt.savefig("figs/try/imu_times_test.png")
        # print("\nIMU TIME CHECK: ",np.sum(np.diff(slam.test_imu_times)), "---", slam.test_imu_times[-1],"\n")

    except Exception as e:
        print("Failed to plot IMU", e)
        raise e
        
    return poses, tstamps, img_timestamps

def plot_2d_3d_comps(poses_cov, labels):
    save_folder = "figs/Debug/"
    os.makedirs(save_folder, exist_ok=True)
    colors = ["g", "r", "b-.", "y--"]
    assert len(poses_cov) == len(labels), "Not matching label and traj/pose lists"

    plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    for num, (poses,_) in enumerate(poses_cov):
        ax.plot3D(poses[:,0], poses[:,1], poses[:,2], colors[num])
    plt.title("Pose Comparisson")
    plt.legend(labels)
    plt.savefig(save_folder + "3Dpypose_comp.png")

    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    for num, (poses,covs) in enumerate(poses_cov):
        ax.plot(poses[:,0], poses[:,1], colors[num])
        if covs is not None:
            plot_gaussian(ax, poses[:, 0:2], covs[:, 6:8,6:8], color=colors[num])
    plt.title("Pose Comparisson")
    plt.legend(labels)
    plt.savefig(save_folder + "2Dpypose_test.png")
    

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

def plot_gaussian(ax, means, covs, color=None, sigma=3):
    ''' Set specific color to show edges, otherwise same with facecolor.'''
    ellipses = []
    for i in range(len(means)):
        eigvals, eigvecs = np.linalg.eig(covs[i])
        axis = np.sqrt(eigvals) * sigma
        slope = eigvecs[1][0] / eigvecs[1][1]
        angle = 180.0 * np.arctan(slope) / np.pi
        ellipses.append(Ellipse(means[i, 0:2], axis[0], axis[1], angle=angle))
    ax.add_collection(PatchCollection(ellipses, edgecolors=color, color=color, alpha= 0.005, linewidth=1))

def _image_to_cv_fmt(image):
    # Convert image_resized_for_slam (network input) to BGR for display
    img_to_show_tensor = image.squeeze(0).squeeze(0) # CHW tensor

    # Denormalization # Default normalization: 2 * (img/255) - 0.5
    img_denorm = torch.clamp((img_to_show_tensor + 0.5) * (255.0 / 2.0), 0, 255)

    display_bgr_np = img_denorm.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    if display_bgr_np.shape[2] == 1: # Grayscale
        display_bgr_np = cv2.cvtColor(display_bgr_np, cv2.COLOR_GRAY2BGR)
    elif display_bgr_np.shape[2] == 3: # Check if it's RGB and convert to BGR
        display_bgr_np = cv2.cvtColor(display_bgr_np, cv2.COLOR_RGB2BGR)

    return display_bgr_np


def async_evaluate_sequence(
    config_VO, net, eval_cfg, data_queue: Queue, traj_ref, use_pose_pred, img_timestamps_all, enable_timing = False, save_slam_steps_path = None
):
    

    if use_pose_pred:
        raise NotImplementedError("Removed Pose Prediction as it didn't increase performance enough")
    else:
        res = asyncio.run(async_run(
            cfg_VO=config_VO, network=net, eval_cfg=eval_cfg, data_queue=data_queue, enable_timing=enable_timing, save_slam_steps_path= save_slam_steps_path
        ))
        if res is None:
            return None
        
        traj_est, _tstamps, frame_indecies = res

    n = len(traj_est)  #cuttoff for testing  
    timestamps = _tstamps if IMU_TESTING else img_timestamps_all[frame_indecies]
    timestamps = img_timestamps_all[frame_indecies]
    time_traj = timestamps[-1]

    print("\nLEN OF EST: ", n,"\n\n")      
    print(img_timestamps_all[frame_indecies[-1]], time_traj,traj_ref.timestamps[n],"\n\n")

    n = np.array(traj_ref.timestamps).searchsorted(time_traj,side="left")
    print("new N", n)

    traj_ref = PoseTrajectory3D( 
        positions_xyz            = traj_ref.positions_xyz[:n],
        orientations_quat_wxyz   = traj_ref.orientations_quat_wxyz[:n],
        timestamps               = traj_ref.timestamps[:n]
    )

    raw_traj_est_ = PoseTrajectory3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, 3:][:, (1, 2, 3, 0)],
        timestamps= timestamps
    )

    # save_output_for_COLMAP("colmap_saving", traj_est_, points, colors, fx, fy, cx, cy)

    try:
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, raw_traj_est_)

        result = main_ape.ape(
            traj_ref=traj_ref,
            traj_est=traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
        )
        
        ate_score = result.stats["rmse"]
        rot_score = rot_error_with_alignment_from_pose3d(
            ref=traj_ref, est=traj_est, correct_scale=True
        )



    except Exception as e:
        ate_score = 1000
        rot_score = [1000, 1000, 1000]
        print(f"\nWARNING: Result not computed correctly for sequence beacase fo exception: {e}")

    try:
        if traj_ref is not None and traj_est is not None:
            np.savez_compressed(
                "aligned_trajs.npz",
                ref_t   = traj_ref.timestamps,
                ref_pos = traj_ref.positions_xyz,
                ref_q   = traj_ref.orientations_quat_wxyz,
                est_t   = traj_est.timestamps,
                est_pos = traj_est.positions_xyz,
                est_q   = traj_est.orientations_quat_wxyz,
            )
            print("→ dumped aligned_trajs.npz")
    except Exception as e:
        print("Failed to save trajectories with exception: ", e)

    
    # --- Quick matplotlib plot -----
    # get N×3 arrays of [x,y,z]
    ref_xyz = traj_ref.positions_xyz  
    est_xyz = traj_est.positions_xyz

    plt.figure(figsize=(6,6))
    plt.plot(ref_xyz[:,0], ref_xyz[:,1], '--', label='reference')
    plt.plot(est_xyz[:,0], est_xyz[:,1], '-',  label='estimate')

    # mark start/end
    plt.scatter(ref_xyz[0,0], ref_xyz[0,1], marker='o', s=60, label='start')
    plt.scatter(ref_xyz[-1,0], ref_xyz[-1,1], marker='x', s=60, label='end')

    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory Comparison (XY)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/traj_compare_mat.png")
    plt.close()

    plt.figure(figsize=(6,6))
    plt.plot(ref_xyz[:,0], ref_xyz[:,2], '--', label='reference')
    plt.plot(est_xyz[:,0], est_xyz[:,2], '-',  label='estimate')

    # mark start/end
    plt.scatter(ref_xyz[0,0], ref_xyz[0,2], marker='o', s=60, label='start')
    plt.scatter(ref_xyz[-1,0], ref_xyz[-1,2], marker='x', s=60, label='end')

    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory Comparison (Xz)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/traj_side_compare_mat.png")
    plt.close()


    print(result)
    T = result.np_arrays["alignment_transformation_sim3"]
    sR = T[:3, :3]                    
    scale = np.cbrt(np.linalg.det(sR)) 
    print("scale =", scale)
    print("==================")

    return ate_score, rot_score, traj_est, traj_ref, raw_traj_est_

@torch.no_grad()
def evaluate(
    net, 
    trials=1, 
    downsample_fact=1, 
    config_VO=None, 
    eval_cfg=None, 
    results_path=None, 
    enable_timing = False,
    save_encoder_path = None
):
    test_ = eval_cfg["data_loader"]["test"]    
    test_split = test_["test_split"]
    dataset_name = test_["dataset_name"]
    use_pose_pred = test_["use_pose_pred"]

    train_ = eval_cfg["data_loader"]["train"]["args"]
    norm_to = train_.get("norm_to", None)

    if config_VO is None:
        config_VO = VO_cfg
        config_VO.merge_from_file("config/default.yaml")

    results = {}
    skipped_scenes = []

    for scene in test_split:
        print(f"loading training data from scene:{scene}")
        print(f"Dataset: {dataset_name}")

        scene_location = scene
        if "Tartan" in dataset_name:
            scene_location = osp.join(TARTAN_PATH_PREFIX,scene)
        
        print("SCENE LOCATION: ", scene_location)
            
        if not os.path.exists(scene_location):
            raise FileNotFoundError(f"scene {scene_location} not found")
        traj_ref_path = osp.join(scene_location, "pose_left.txt")
        scene_name = os.path.basename(scene_location) if os.path.isdir(scene_location) else scene_location
        timestamps_path = osp.join(scene_location, "timestamps.txt")
        img_timestamps = np.loadtxt(timestamps_path)

        set_global_params(K_path=osp.join(scene_location, "K.yaml")) #standard pose format for EDS and Stereo Davis did nothing

        if "Tartan" in dataset_name:
            traj_ref = read_tartan_format_poses(
                traj_path=traj_ref_path, timestamps_path=timestamps_path
            )
        elif "MoonLanding" in dataset_name:
            traj_ref = read_moonlanding_format_poses(
                traj_path=traj_ref_path, timestamps_path=timestamps_path
            )
        else:
            raise NotImplementedError("dataset not supported")
        
        
        #TODO: Fix for multiple trails
        async_q = Queue(maxsize=QUEUE_BUFFER_SIZE)
        loader_kwargs = {"queue":async_q,
                         "config":eval_cfg, 
                         "full_scene":scene_location, 
                         "downsample_fact": downsample_fact, 
                         "norm_to":norm_to}
        loader_prod_thread = threading.Thread(
            target=async_data_loader_all_events, kwargs=loader_kwargs, daemon=True
        )

        loader_prod_thread.start()
        global g_traj_ref 
        g_traj_ref = traj_ref

        eval_subtraj = partial(
            async_evaluate_sequence,
            config_VO=config_VO,
            net=net,
            eval_cfg=eval_cfg,
            data_queue=async_q,
            traj_ref=traj_ref,
            use_pose_pred=use_pose_pred,
            img_timestamps_all=img_timestamps,
            enable_timing = enable_timing,
            save_slam_steps_path =  save_encoder_path,
        )
        


        results[scene] = {}
        for j in range(trials):
            res = eval_subtraj()

            if res is None:
                print(f"SKIPPING: {scene_name}")
                skipped_scenes.append(scene_name)
                continue

            ate_error, rot_error, traj_est, traj_ref, raw_traj_est = res
            
            print("\n full_data ate ------->", ate_error, "\nfull_data rot ------->", rot_error)

            save_results(traj_est=traj_est, traj_ref=traj_ref,scene=scene_name, j=j, eval_type="aligned")
            save_results(traj_est=raw_traj_est, traj_ref=traj_ref,scene=scene_name, j=j, eval_type="unaligned")

            results[scene][f"trial_{j}"] = {
                "ate": ate_error,
                "rot_err": list(rot_error),
            }


        if results_path is not None:
            with open(results_path, "w") as json_file:
                json.dump(results, json_file, indent=4)

    print("SKIPPED THE FOLLOGING: \n")
    for s in skipped_scenes:
        print("- ", s)
        

    if results_path is not None:
        with open(results_path, "w") as json_file:
            results["test_info"] = [
                {"config_VO": dict(config_VO)},
                train_,
                test_,
            ]
            json.dump(results, json_file, indent=4)

    return results

def pare_dataset_name(name: str) -> str:
    path_parts = name.split("/")
    while path_parts[0] != "datasets" and len(path_parts) > 3:
        path_parts = path_parts[1:]
    if path_parts[0] == "datasets":
        path_parts = path_parts[1:]
    else:
        print(f"\nFormatting! Check dataset name(s): {name}\nShould include 'datasets'")
    if path_parts[0] == "TartanEvent":
        path_parts[0] = "TartanE"
    # ASSUMES 2 level down

    return "-".join(path_parts), "-".join(path_parts[:2]),"-".join(path_parts[2:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="dpvo.pth") #TODO: Fic to singlescae/Multiscale from config eval
    parser.add_argument("--config_VO", default="config_vo/default.yaml")
    parser.add_argument("--config_eval", type=str, default="config/TartanEvent.json")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--downsample_fact", type=int, default=1)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--timeit", action='store_true')
    parser.add_argument("--save_slam_steps_path", type=str, default=None)

    args = parser.parse_args()

    VO_cfg.merge_from_file(args.config_VO)
    eval_cfg = json.load(open(args.config_eval))

    print("Running evaluation...")
    print(args)

    results = evaluate(
        config_VO=VO_cfg,
        eval_cfg=eval_cfg,
        net=args.weights,
        trials=args.trials,
        downsample_fact=args.downsample_fact,
        results_path=args.results_path,
        enable_timing=args.timeit,
        save_encoder_path = args.save_slam_steps_path
    )

    rows = []
    for dataset_path, trials in results.items():
        full, outer_name, subset = pare_dataset_name(dataset_path)
        for trial_name, metrics in trials.items():
            rows.append({
                # "dataset": full,
                "dataset": outer_name,
                "subset" : subset,
                "trial":    trial_name,
                "ate":      metrics["ate"],
                "rot_err":  metrics["rot_err"],
            })

    df = pl.DataFrame(rows)

    df = df.with_columns([
        pl.col("rot_err").list.get(0).alias("x_rot_err"),
        pl.col("rot_err").list.get(1).alias("y_rot_err"),
        pl.col("rot_err").list.get(2).alias("z_rot_err"),
    ]).drop("rot_err")

    _year, week_num, week_day = tuple(datetime.date.today().isocalendar())
    time_min = datetime.datetime.now().strftime("%H%M")

    save_dir = osp.join(
        os.getcwd(),
        "results_summary",
        f"week_{week_num:02d}",
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(df)
    df.write_csv(osp.join(save_dir, f"result_{week_day:02d}_{time_min}.csv"))
    
    if args.timeit:
        print_timing_summary()
