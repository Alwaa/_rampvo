import asyncio
import os

from queue import Queue
import sys
import glob
import threading
import time
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

from config import (
    QUEUE_BUFFER_SIZE,
    QUEUE_ASYNC_MIN_SIZE,
    QUEUE_ASYNC_STALL_TIMEOUT,
    QUEUE_ASYNC_SLEEP_BETWEEN_STARTUP_CHECKS, 
    LOADING_THREAD_TORCH_INTRA_OP_THREAD_NUM,
    IMU_TESTING,
    TARTAN_PATH_PREFIX
)

seed_everything(seed=1234)
sys.setrecursionlimit(100000)


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
    vel_data = np.loadtxt(osp.join(full_scene, "velocities" + suffix + ".txt"), dtype=np.float64)
    # t tx ty tz qx qy qz qw
    pose_data = np.loadtxt(osp.join(full_scene, "stamped_groundtruth" + suffix + ".txt"), dtype=np.float64)
    assert vel_data.shape[0] == pose_data.shape[0]

    imu_t        = imu_data[:, 0] #seconds
    imu_gyro     = imu_data[:, 2:5]
    imu_accel    = imu_data[:, 5:8]
    def imu_slice(start_t, end_t):
        _i0 = np.searchsorted(imu_t, start_t, side="left")
        _i1 = np.searchsorted(imu_t, end_t,   side="right")
        return imu_t[_i0:_i1], imu_gyro[_i0:_i1], imu_accel[_i0:_i1]
    
 
    # ---------------------------------------------
    pose_t = pose_data[:, 0]
    poses_raw = pose_data[:, 1:]
    poses_raw = torch.as_tensor(poses_raw, dtype=torch.float32, device='cpu').contiguous()
    P_all = SE3(poses_raw)
    P0_inv   = P_all[0:1].inv()  
    P_rel    = P0_inv * P_all

    poses_rel = P_rel.data.detach().cpu().numpy()  
    def pose_slice(start_t, end_t):
        _i0 = np.searchsorted(pose_t, start_t, side="left")
        _i1 = np.searchsorted(pose_t, end_t,   side="right")
        return poses_rel[_i0:_i1, :]
    
    def pose_last(start_t, end_t):
        _i1 = np.searchsorted(pose_t, end_t,   side="right")

    # ---------------------------------------------            

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

        # plot_events(event, image, i0, i1, i) #TODO: Fix absolute paths
        # the index of the smallest error between the event voxel timestamp and the image timestamp is event index
        event_ind = corresponding_events_indices[frame_ind]

        mask = bool(event_ind == i) #Explicit bool as to not turn an np.bool_ into tensor (depricated)

        if IMU_TESTING:
            ts_start_ns, ts_end_ns = event.t[i0], event.t[i1 - 1] # first and last event in voxel
            ts_start, ts_end = ts_start_ns/1e9, ts_end_ns/1e9

            pose = pose_last(ts_start,ts_end)
            pose = torch.from_numpy(pose).float()
            tup = (image, event_voxel, intrinsics, torch.tensor([mask]), frame_ind, pose)
        else:    
            tup = (image, event_voxel, intrinsics, torch.tensor([mask]), frame_ind)
            
        queue.put(tup)

    queue.put(None) #DONE:=0

    return "Done!:0"

def base_unpacker(item_tuple: tuple) -> tuple:
    image, events, intrinsics, mask, f_i = item_tuple
    im = image[None, None, ...].cuda()
    ev = events[None, None, ...].float().cuda()
    intr = intrinsics.cuda()
    mask.cuda()

    return (im, ev, intr, mask, f_i)

#TODO: Should be actual imu data in the end
def imu_unpacker(item_tuple:tuple) -> tuple:
    image, events, intrinsics, mask, f_i, imu_pose = item_tuple
    im = image[None, None, ...].cuda()
    ev = events[None, None, ...].float().cuda()
    intr = intrinsics.cuda()
    mask.cuda()

    return (im, ev, intr, mask, f_i, imu_pose)

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
async def async_run(cfg_VO, network, eval_cfg, data_queue: Queue, enable_timing = False):
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

    evaluation_iter_bar = atqdm(_queue_iterator(data_queue))
    evaluation_iter_bar.set_description("Async Evaluating")

    #TODO: Shouldn't duplicate the whole thing + Queue enumerate
    t = 0
    if IMU_TESTING:
        async for (image, events, intrinsics, mask, f_i, imu_pose) in evaluation_iter_bar:
            image, events = resize_input(image, events)
            with Timer("SLAM", enabled=enable_timing):
                slam(t, input_tensor=(events, image, mask), intrinsics=intrinsics, curr_imu_pose=imu_pose)
            t += 1
        
            if mask:
                img_timestamps.append(f_i)
    else:
        async for (image, events, intrinsics, mask, f_i) in evaluation_iter_bar:
            image, events = resize_input(image, events)
            with Timer("SLAM", enabled=enable_timing):
                slam(t, input_tensor=(events, image, mask), intrinsics=intrinsics)
            t += 1
        
            if mask:
                img_timestamps.append(f_i)
    
    # with Timer("FinalUpdates", enabled=enable_timing):
    #     for _ in range(12):
    #             slam.update()
    
    if t == 0:
        return None

    points = slam.points_.cpu().numpy()[:slam.m]
    colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]
    poses, tstamps = slam.terminate()
    return poses, tstamps, points, colors, img_timestamps


def async_evaluate_sequence(
    config_VO, net, eval_cfg, data_queue: Queue, traj_ref, use_pose_pred, img_timestamps_all, enable_timing = False
):
    if use_pose_pred:
        raise NotImplementedError("Removed Pose Prediction as it didn't increase performance enough")
    else:
        res = asyncio.run(async_run(
            cfg_VO=config_VO, network=net, eval_cfg=eval_cfg, data_queue=data_queue, enable_timing=enable_timing
        ))
        if res is None:
            return None
        
        traj_est, _tstamps, points, colors, frame_indecies = res

    traj_est_ = PoseTrajectory3D(
        positions_xyz=traj_est[:, :3],
        orientations_quat_wxyz=traj_est[:, 3:][:, (1, 2, 3, 0)],
        timestamps=img_timestamps_all[frame_indecies],
    )

    save_output_for_COLMAP("colmap_saving", traj_est_, points, colors, fx, fy, cx, cy)

    try:
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est_)

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
    
    print(result)
    T = result.np_arrays["alignment_transformation_sim3"]
    sR = T[:3, :3]                    
    scale = np.cbrt(np.linalg.det(sR)) 
    print("scale =", scale)
    print("==================")

    return ate_score, rot_score, traj_est, traj_ref

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
        elif "StereoDavis" in dataset_name:
            img_timestamps = img_timestamps / 1e6
            traj_ref = read_stereodavis_format_poses(
                traj_path=osp.join(scene_location, "poses.txt"),
                timestamps_path=osp.join(scene_location, "timestamps_poses.txt"),
            )
        elif "EDS" in dataset_name:
            img_timestamps = img_timestamps / 1e6
            traj_ref = read_eds_format_poses(traj_ref_path)
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
        )
        save_res = partial(save_results, scene=scene_name, eval_type="full_data")

        results[scene] = {}
        skipped_scenes = []
        for j in range(trials):
            res = eval_subtraj()
            if res is None:
                print(f"SKIPPING: {scene_name}")
                skipped_scenes.append(scene_name)
                continue
            ate_error, rot_error, traj_est, traj_ref = res
            print("\n full_data ate ------->", ate_error)
            print("\n full_data rot ------->", rot_error)
            save_res(traj_est=traj_est, traj_ref=traj_ref, j=j)
            results[scene][f"trial_{j}"] = {
                "ate": ate_error,
                "rot_err": list(rot_error),
            }

        print("SKIPPED THE FOLLOGING: \n")
        for s in skipped_scenes:
            print("- ", s)

        if results_path is not None:
            with open(results_path, "w") as json_file:
                json.dump(results, json_file, indent=4)
        

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

    print(df)
    
    if args.timeit:
        print_timing_summary()
