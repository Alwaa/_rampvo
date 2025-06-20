import os
import torch
import numpy as np
import torch.nn.functional as F

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .net import VONet
from .utils import Timer, flatmeshgrid
from .utils import preprocess_input, filter_features
from . import projective_ops as pops
from collections import OrderedDict

import pypose as pp

autocast = torch.amp.autocast("cuda", enabled=True)
Id = SE3.Identity(1, device="cuda")


from constans import TARTAN_2_XYZ_P, TARTAN_2_XYZ_Q
from ramp.ba import BA as pyBA

#BundleAdjustment = fastba.BA
BundleAdjustment = pyBA
USE_IMU_IN_BA = True

GRAVITY_BASE = torch.tensor([0, 0, -9.81])

def IMU_prop(current_dict, delta_dict, GRAVITY=GRAVITY_BASE):
    new_dict = {}
    new_dict["p"] = current_dict["p"] + current_dict["v"] * delta_dict["Dt"] + 0.5 * GRAVITY * delta_dict["Dt"]**2 + current_dict["R"] * delta_dict["Dp"]
    new_dict["v"] = current_dict["v"] + GRAVITY * delta_dict["Dt"] + current_dict["R"] * delta_dict["Dv"]
    new_dict["R"] = current_dict["R"] * delta_dict["Dr"]

    return new_dict

def commbine_deltas(delta_0: dict, delta_1: dict) -> dict:
    """ Combine Deltas from (0->1) and (1->2) to get (0->2) """
    comb_delta = {}
   
    R_01 = delta_0['Dr']
    R_12 = delta_1['Dr']
    R_02 =  R_01 * R_12

    dt_02 = delta_0['Dt'] + delta_1['Dt']
    dv_02 = delta_0['Dv'] + R_01 * delta_1['Dv']
    dp_02 = delta_0['Dp'] + delta_0['Dv'] * delta_1['Dt'] + R_01 * delta_1['Dp']

    comb_delta["Dr"] = R_02
    comb_delta["Dt"] = dt_02
    comb_delta["Dv"] = dv_02
    comb_delta["Dp"] = dp_02

    return comb_delta

def tensor_commbine_deltas(delta_0, delta_1):
    """ Combine Deltas from (0->1) and (1->2) to get (0->2) """
    delta_p_R_0, delta_v_0, delta_t_0 = delta_0
    delta_p_R_1, delta_v_1, delta_t_1 = delta_1
    
   
    R_01 = pp.SO3(delta_p_R_0[3:])
    R_12 = pp.SO3(delta_p_R_1[3:])
    R_02 =  R_01 * R_12

    Dp_0, Dp_1 = delta_p_R_0[:3],delta_p_R_1[:3]
    Dv_0, Dv_1 = delta_v_0, delta_v_1
    Dt_0, Dt_1 = delta_t_0, delta_t_1

    dt_02 = Dt_0 + Dt_1
    dv_02 = Dv_0 + R_01 * Dv_1
    dp_02 = Dp_0 + Dv_0 *Dt_1 + R_01 * Dp_1

    D_r_R_02 = torch.zeros(7, dtype=torch.float, device="cuda")
    D_r_R_02[:3] = dp_02
    D_r_R_02[3:] = R_02
    
    #torch.cat((dp_02, R_02))

    return D_r_R_02, dv_02, dt_02

class Ramp_vo:
    def __init__(self, cfg, network, train_cfg, ht=480, wd=640, enable_timing=False):
        self.cfg = cfg
        self.event_bias = train_cfg["event_bias"]
        self.train_cfg = train_cfg

        # attributes for pose prediction (REMOVED for now)
        self.patch_dict_ = None
        self.patches_models = None
        self.lmbda = torch.as_tensor([1e-4], device="cuda")

        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = enable_timing

        self.n = 0  # number of frames
        self.m = 0  # number of patches
        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht  # image height
        self.wd = wd  # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0


        self.frame_indxs_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")

        self.velocities_ = torch.zeros(self.N, 3, dtype=torch.float, device="cuda") #Merge with poses_ later?

        self.patches_ = torch.zeros(
            self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda"
        )   # Patch index, Buff num, (px,py,depth), PATCH DIM, PATCH DIM
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        ### Ading imu ###

        self.imu_delta_p_R_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda") # Tranlation + Rot
        self.imu_delta_v_ = torch.zeros(self.N, 3, dtype=torch.float, device="cuda")
        self.imu_delta_t_ = torch.zeros(self.N, 1, dtype=torch.float, device="cuda")
        self.imu_biases_ = torch.zeros(self.N, 9, dtype=torch.float, device="cuda") # TODO: Incorporate

        ### --------- ###

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        self.imap_ = torch.zeros(self.mem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P, **kwargs)

        ht = ht // RES
        wd = wd // RES

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")

        # initialize poses to identity matrix
        self.poses_[:, 6] = 1.0
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.pose_delta = {}


        # For Viz        
        self.current_patch_coords_feat = None # Current frame's patch coords (feature map scale)
        self.current_step_flow_map = None     # Current frame's flow (t-1 -> t) (feature map scale)
        self.current_patch_flow_vectors_feat = None
        self.current_dense_flow_map_feat = None
        self.current_active_patch_coords_feat = None
        self.P_feat = self.network.P  # Patch size in feature map grid


        q_xyzw_tartan = [0.1074549338237731, 0.1516151244003950, 0.9557535343733162, 0.2280586720601046]
        q_tensor = torch.tensor(q_xyzw_tartan)
        v_init_tartan = torch.tensor([-5.4288209852760119, 3.2772077143020484, -1.6340889856881029])

        so3_init_cheat = pp.SO3(q_tensor[TARTAN_2_XYZ_Q])
        cheat_vel_init =  v_init_tartan[TARTAN_2_XYZ_P]

        #cheat_vel_init = torch.tensor([0.0, 0.0, 0.0])
        self.test_imu_preintegrator = pp.module.IMUPreintegrator(vel=cheat_vel_init,
                                                            #gravity=0.0,
                                                            rot=so3_init_cheat)
        
        # IMU preintegrator for the deltas
        self.integrator = pp.module.IMUPreintegrator(
            gravity=0.0
        ) #.to(device)

        self.currect_pred_dict = {"p": torch.zeros(3),
                                  "v": cheat_vel_init.clone(),
                                  "R": so3_init_cheat.clone()}

        
        self.dt = torch.tensor([0.0033333333333333], dtype=torch.float32)
        self.dt = torch.tensor([(3.0 + (1/3))*1e-3 ], dtype=torch.float32)

        self.test_imu_poses = []
        self.test_imu_covs = []
        self.test_imu_times = []

        self.delta_poses = []
        self.delta_covs = []

        self.key_delta_poses = []

        self.imu_deltas = {} #Dict for now while prototyping

        init_delta_dict = self.integrator.integrate(torch.zeros((1,1,1)), torch.zeros((1,1,3)),torch.zeros((1,1,3)))
        delta_p = init_delta_dict["Dp"].clone()[...,-1,:]
        delta_v = init_delta_dict["Dv"].clone()[...,-1,:]
        delta_r = init_delta_dict["Dr"].clone()[...,-1,:]
        delta_t = init_delta_dict["Dt"].clone()[...,-1,:]
        
        self.imu_deltas_buffer = {"Dp": delta_p, "Dv": delta_v,"Dr": delta_r,"Dt": delta_t}
        self.imu_deltas[0] = {"Dp": delta_p.clone(), "Dv": delta_v.clone(),"Dr": delta_r.clone(),"Dt": delta_t.clone()}

        self.imu_deltas_zero = {"p": torch.zeros(3),
                                  "v": cheat_vel_init.clone(),
                                  "R": so3_init_cheat.clone()}

        print("Startbuffer:")
        print(self.imu_deltas_buffer, "\n")

        self.all_poses = []

        self.all_deltas = []
        self.test_deltas = []
        self.imu_key_deltas = []


        self.imu_preintegrations = {} # Store preintegration between keyframes

        self.straight_deltas_t_cum = 0
        self.key_added_t = 0

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            checkpoint = torch.load(network, weights_only=False)

            if checkpoint.get("model_state_dict"):
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace("module.", "")] = v

            self.network = VONet(cfg=self.train_cfg)
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N * self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)
    
    @property
    def stabilized_pose(self):
        """
        Returns a pose that is a few frames behind the most recent one.
        This pose is more 'settled' as it has been through the BA optimizer
        more times than the absolute latest pose.
        """
        # Define how many frames of lag you want for a more stable visualization.
        # A value of 5-15 is usually good.
        lag_frames = 30

        if self.n > lag_frames:
            # Get the pose from 'lag_frames' ago from the main buffer.
            # self.n is the current number of active frames.
            # The index is self.n - 1 - lag_frames to get the desired lag.
            stabilized_index = self.n - 1 - lag_frames
            return self.poses[0][stabilized_index]
        elif self.n > 0:
            # If we don't have enough frames for the full lag, return the oldest pose.
            return self.poses[0][0]
        else:
            # If no frames have been processed, return None.
            return None

    def get_pose(self, i):
        if i in self.traj:
            return SE3(self.traj[i])

        t0, dP = self.pose_delta[i]
        return dP * self.get_pose(t0)

    def terminate(self):
        """interpolate missing poses"""

        self.traj = {}
        for i in range(self.n):
            self.traj[self.frame_indxs_[i].item()] = self.poses_[i]
        
        imu_deltas = self.imu_key_deltas #self.test_deltas

        print("\nNN\n", self.n, len(imu_deltas), "\n\n")
        _state = self.imu_deltas_zero
        t = 0
        
        for i, delta in enumerate(imu_deltas):
            t += delta["Dt"]
            _state = IMU_prop(_state, delta)
            self.key_delta_poses.append(_state["p"].clone())
        
        print(_state)

        poses = [self.get_pose(count_i) for count_i in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=float)

        print(self.straight_deltas_t_cum, t, self.curr_timestamp)

        _state = {"p": self.imu_deltas_zero["p"].cuda(),
                  "v": self.imu_deltas_zero["v"].cuda(),
                  "R": self.imu_deltas_zero["R"].cuda(),
                  }
        
        self.key_delta_poses = []
        GRAVITY_GPU = torch.tensor([0, 0, -9.81], device="cuda")
        for i in range(self.n):
            _delta = {"Dr": pp.SO3(self.imu_delta_p_R_[i,3:]),
                      "Dp": self.imu_delta_p_R_[i,:3],
                      "Dt": self.imu_delta_t_[i],
                      "Dv": self.imu_delta_v_[i,:],}
            _state = IMU_prop(_state, _delta, GRAVITY=GRAVITY_GPU)
            
            self.key_delta_poses.append(_state["p"].cpu().unsqueeze(0))
        print(_state)


        # print("\n\nLEN OF DELTAS:\n", len(self.all_deltas))
        # deltas_to_marginalize = [100]*100 + [300]*300 #+ list(range(500,850))
        # for di in deltas_to_marginalize:
        #     self.all_deltas[di - 1] = commbine_deltas(self.all_deltas[di - 1],self.all_deltas[di])

        #     for _i in range(di, len(self.all_deltas)-1):
        #         self.all_deltas[_i] = self.all_deltas[_i +1 ]
        #     self.all_deltas = self.all_deltas[:-1]
        

        # print("\n\nNEW LEN OF DELTAS:\n", len(self.all_deltas))

        # _state = self.imu_deltas_zero
        # _delta_running = self.imu_deltas[0]
        # self.key_delta_poses = []
        # for new_delta in self.all_deltas:
        #     _state = IMU_prop(_state, new_delta)
        #     self.key_delta_poses.append(_state["p"])

        #     # _delta_running = commbine_deltas(_delta_running, new_delta)
        #     # self.key_delta_poses.append(IMU_prop(self.imu_deltas_zero, _delta_running)["p"])

        return poses, tstamps
    

    def get_current_trajectory(self):
        """
        Reconstructs and returns the full estimated trajectory up to the current frame.
        This is useful for live visualization.
        """
        # Can't create a trajectory if no frames have been processed
        if self.counter == 0:
            return None, None

        # 1. Create a temporary dictionary of all known poses (active keyframes)
        # This mirrors the logic in the terminate() function.
        traj_lookup = {}
        for i in range(self.n):
            # The key is the absolute frame counter timestamp, value is the pose
            traj_lookup[self.frame_indxs_[i].item()] = self.poses_[i]

        # 2. Define a recursive helper function to resolve poses
        # This is the same as self.get_pose, included here for clarity
        def _resolve_pose(i):
            if i in traj_lookup:
                return SE3(traj_lookup[i])
            
            # If the pose is not in the active keyframes, it must be in `self.delta`
            # which stores the relative pose of a removed keyframe.
            t0, dP = self.pose_delta[i]
            # Recursively find the pose of the frame it's relative to and apply the delta.
            return dP * _resolve_pose(t0)

        # 3. Reconstruct the pose for every frame processed so far
        poses_list = [_resolve_pose(i) for i in range(self.counter)]

        # 4. Stack, invert (for visualization), and convert to NumPy
        if not poses_list:
            return None, None
            
        poses = lietorch.stack(poses_list, dim=0)
        poses = poses.inv().data.cpu().numpy() # .inv() is crucial for correct visualization
        
        # 5. Get the corresponding timestamps
        tstamps = np.array(self.tlist, dtype=float)

        return poses, tstamps

    def corr(self, coords, indicies=None):
        """local correlation volume"""
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None, poses=None, patches=None, intrinsics=None):
        """reproject patch k from i -> j"""
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        poses = poses if poses is not None else self.poses
        patches = patches if patches is not None else self.patches
        intrinsics = intrinsics if intrinsics is not None else self.intrinsics

        coords = pops.transform(SE3(poses), patches, intrinsics, ii, jj, kk)
        return coords.permute(
            0, 1, 4, 2, 3
        ).contiguous()  # torch.Size([1, 96 * self.n * num_frame_connected, 2, 3, 3])

    def append_factors(self, ii, jj):
        """add factors to the graph"""
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        """remove factors from the graph"""
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:, ~m]

    def motion_probe(self):
        """kinda hacky way to ensure enough motion for initialization"""
        with Timer("MotionProbe", enabled=self.enable_timing):
            kk = torch.arange(self.m - self.M, self.m, device="cuda")
            jj = self.n * torch.ones_like(kk)
            ii = self.ix[kk]

            net = torch.zeros(
                1, len(ii), self.DIM, **self.kwargs
            )  # torch.Size([1, 96, 384])
            coords = self.reproject(
                indicies=(ii, jj, kk)
            )  # torch.Size([1, 96, 2, 3, 3])

            with autocast:
                with Timer("MotionProbe.Corr", enabled=self.enable_timing):
                    corr = self.corr(coords, indicies=(kk, jj))

                with Timer("MotionProbe.Ctx", enabled=self.enable_timing):
                    ctx = self.imap[:, kk % (self.M * self.mem)]
                with Timer("MotionProbe.NetUpdate", enabled=self.enable_timing):
                    net, (delta, weight, _) = self.network.update(
                        net, ctx, corr, None, ii, jj, kk
                    )

            return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        """compute motion magnitude (mean flow of patches) between frames i and j"""
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5
        )
        return flow.mean().item()

    def keyframe(self):
        """remove keyframe if motion is small"""
        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.frame_indxs_[k - 1].item()
            t1 = self.frame_indxs_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k - 1]).inv()
            self.pose_delta[t1] = (t0, dP)

            #print("\n\nDELTA:", t1, self.curr_timestamp, "\n", k, self.n, "\n")
            self.imu_deltas[k-1] = commbine_deltas(self.imu_deltas[k-1], self.imu_deltas[k]) #TODO: Check off by one
            self.imu_key_deltas[k-1] = commbine_deltas(self.imu_key_deltas[k-1], self.imu_key_deltas[k])
            # print("\n\n",max(self.imu_deltas.keys()), self.n, k, "\n\n")

            self.imu_delta_p_R_[k-1],self.imu_delta_v_[k-1],self.imu_delta_t_[k-1] = tensor_commbine_deltas(
                (self.imu_delta_p_R_[k-1],self.imu_delta_v_[k-1],self.imu_delta_t_[k-1]),
                (self.imu_delta_p_R_[k],self.imu_delta_v_[k],self.imu_delta_t_[k])
            )

            # self.imu_delta_p_R_
            # self.imu_delta_v_
            # self.imu_delta_t_

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n - 1):
                self.imu_deltas[i] = self.imu_deltas[i + 1]

                self.imu_delta_p_R_[i],self.imu_delta_v_[i],self.imu_delta_t_[i] = self.imu_delta_p_R_[i+1],self.imu_delta_v_[i+1],self.imu_delta_t_[i+1]

                self.frame_indxs_[i] = self.frame_indxs_[i + 1]
                self.colors_[i] = self.colors_[i + 1]
                self.poses_[i] = self.poses_[i + 1]
                self.patches_[i] = self.patches_[i + 1]
                self.intrinsics_[i] = self.intrinsics_[i + 1]

                self.imap_[i % self.mem] = self.imap_[(i + 1) % self.mem]
                self.gmap_[i % self.mem] = self.gmap_[(i + 1) % self.mem]
                self.fmap1_[0, i % self.mem] = self.fmap1_[0, (i + 1) % self.mem]
                self.fmap2_[0, i % self.mem] = self.fmap2_[0, (i + 1) % self.mem]

            #print("\n\n", len(self.imu_key_deltas), self.n)
            #assert len(self.imu_key_deltas) == self.n, "Not N in imu key deltas"
            for i in range(k, self.n - 1):
                self.imu_key_deltas[i] = self.imu_key_deltas[i + 1]
            self.imu_key_deltas.pop()

            for _i in range(self.n, max(self.imu_deltas.keys()) + 1):
                self.imu_deltas.pop(_i, None)

            self.n -= 1
            self.m -= self.M

        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    @Timer("Update")  # TODO: how can it be turned off?
    def update(self):
        with Timer("Update.ReProj", enabled=self.enable_timing):
            coords = self.reproject()

        with autocast:
            with Timer("Update.Corr", enabled=self.enable_timing):
                corr = self.corr(coords)

            with Timer("Update.Ctx", enabled=self.enable_timing):
                ctx = self.imap[:, self.kk % (self.M * self.mem)]
            with Timer("Update.NetUpdate", enabled=self.enable_timing):
                self.net, (delta, weight, _) = self.network.update(
                    self.net, ctx, corr, None, self.ii, self.jj, self.kk
                )

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[..., self.P // 2, self.P // 2] + delta.float()
            curr_patch_centers = coords[..., self.P // 2, self.P // 2]

            with Timer("Update.FilterFeat", enabled=self.enable_timing):
                weight = filter_features(
                    confidences=weight,
                    target=target,
                    data_shape=(self.ht // 4, self.wd // 4),
                )

                self.last_weight = weight.clone()

            # New Viz:
            self._update_store_new_viz(delta, weight, curr_patch_centers)
        
        with Timer("Update.BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            try:
                if USE_IMU_IN_BA:
                    imu_pre = self.imu_key_deltas
                    vel = self.velocities_
                else:
                    imu_pre, vel = None, None

                BundleAdjustment(
                    self.poses,
                    self.patches,
                    self.intrinsics,
                    target,
                    weight,
                    lmbda,
                    self.ii,
                    self.jj,
                    self.kk,
                    t0,
                    self.n,
                    M=self.M,
                    iterations=2,
                    eff_impl=False,
                    imu_preintegrations=imu_pre,
                    velocities=vel
                )
            except Exception as e:
                print(f"WARNING: BA failed...{e}")
                raise e #TODO: REMOVE

        # Old Viz
        self._update_store_old_viz()


    def _update_store_new_viz(self, delta, weight, current_patch_centers_feat):
        """ Call Before BA"""
        # Store patch-specific flow vectors and confidences
        if delta is not None and delta.ndim == 3 and delta.shape[0] == 1 and delta.shape[2] == 2:
            # delta shape is (1, #edges, 2)
            self.current_graph_patch_flow_vectors = delta.clone().squeeze(0) # Shape: (#edges, 2)
            self.current_graph_patch_start_coords = current_patch_centers_feat.clone() .squeeze(0)# Shape: (#edges, 2)

            if weight is not None and weight.shape == delta.shape:
                self.current_graph_patch_confidences = weight.clone().squeeze(0) # Shape: (#edges, 2)
                # Optionally, convert confidences to uncertainties:
                # self.current_graph_patch_uncertainties = 1.0 - self.current_graph_patch_confidences
            else:
                print(f"Warning: Weight from network.update has unexpected shape or is None. Weight shape: {weight.shape if weight is not None else 'None'}")
                self.current_graph_patch_confidences = None
        else:
            print(f"Warning: Delta from network.update has unexpected shape: {delta.shape if delta is not None else 'None'}. Expected (1, #edges, 2).")
            self.current_graph_patch_flow_vectors = None
            self.current_graph_patch_start_coords = None
            self.current_graph_patch_confidences = None

    def _update_store_old_viz(self):
        """ Call AFTER BA"""   
        points = pops.point_cloud(
            SE3(self.poses),
            self.patches[:, : self.m],
            self.intrinsics,
            self.ix[: self.m],
        )
        points = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
        self.points_[: len(points)] = points[:]

    def __edges_forw(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n - 1, self.n, device="cuda"),
            indexing="ij",
        )

    def __edges_back(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n - r, 0), self.n, device="cuda"),
            indexing="ij",
        )

    def __call__(self, frame_indx, input_tensor, intrinsics, 
                 curr_imu_data= None, timestamp = None, save_slam_steps_path = None):
        """track new frame"""
        self.curr_timestamp = timestamp

        # store intrinsics once
        if self.n == 0:
            fx, fy, cx, cy = intrinsics.cpu().numpy()
            self._init_K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

        with Timer("SLAM.PreProcess", enabled=self.enable_timing):
            input_ = preprocess_input(input_tensor=input_tensor)

        with Timer("SLAM.Patchify", enabled=self.enable_timing):
            with autocast:
                fmap, gmap, imap, patches, _index, clr, coords_patch_feat = self.network.patchify(
                    input_=input_,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    event_bias=self.event_bias,
                    reinit_hidden=True if frame_indx == 0 else False,
                )
            
            if coords_patch_feat is not None:
                self.current_patch_coords_feat = coords_patch_feat.squeeze(0) # Shape (M, 2) assuming batch size 1
            else:
                self.current_patch_coords_feat = None
    
        if curr_imu_data is not None:
            imu_t, gyro, acc = curr_imu_data
            #print("\n\n",imu_t[-1], timestamp,"\n")
            dt = self.dt
            self.test_imu_times.extend(imu_t)
            for gyro_single, acc_single in zip(gyro, acc):

                _gy = torch.tensor(gyro_single, dtype=torch.float32)
                _ac = torch.tensor(acc_single, dtype=torch.float32)

                imu_state = self.test_imu_preintegrator(dt=dt, 
                                                   gyro=_gy, 
                                                   acc=_ac)
                
            self.test_imu_poses.append(imu_state['pos'][..., -1, :].cpu())
            self.test_imu_covs.append(imu_state['cov'][..., -1, :, :].cpu())



            frame_gy = torch.tensor(gyro, dtype=torch.float32)
            frame_ac = torch.tensor(acc, dtype=torch.float32)

            frame_dt = torch.tensor(np.concatenate((np.diff(imu_t)*1e-9, self.dt)).reshape(-1,1), dtype=torch.float32)

            out_dict = self.integrator.integrate(frame_dt.unsqueeze(0), 
                                                frame_gy.unsqueeze(0), 
                                                frame_ac.unsqueeze(0))
                
            # No need to reset when calling .integrate

            # if self.n > 0:
            delta_p = out_dict["Dp"].clone()[...,-1,:]
            delta_v = out_dict["Dv"].clone()[...,-1,:]
            delta_r = out_dict["Dr"].clone()[...,-1,:]
            delta_t = out_dict["Dt"].clone()[...,-1,:]

            frame_delta = {"Dp": delta_p, "Dv": delta_v,"Dr": delta_r,"Dt": delta_t,}
            self.straight_deltas_t_cum += delta_t


            test = commbine_deltas(self.imu_deltas[0], frame_delta)
            
            self.currect_pred_dict = IMU_prop(self.currect_pred_dict, test)
            self.delta_poses.append(self.currect_pred_dict["p"].clone())
            self.delta_covs.append(torch.ones((9,9), dtype=torch.float32)*1e-1)

            self.imu_deltas_buffer = commbine_deltas(self.imu_deltas_buffer, frame_delta)
            self.all_deltas.append(frame_delta)


        if len(input_) > 2:
            _, _, mask = input_
            if not mask and mask is not None:
                # if only events only update the super state but not the VO

                return
        

        ### update state attributes ###

        with Timer("SLAM.UpdateStateAttr", enabled=self.enable_timing):
            self.tlist.append(frame_indx)
            self.frame_indxs_[self.n] = self.counter
            self.intrinsics_[self.n] = intrinsics / self.RES

            self.index_[self.n + 1] = self.n + 1
            self.index_map_[self.n + 1] = self.m + self.M

            # color info for visualization
            clr = (clr[0, :, [2, 1, 0]] + 0.5) * (255.0 / 2)
            self.colors_[self.n] = clr.to(torch.uint8)


            if self.n > 1:
                if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                    with Timer("SLAM.UpdateStateAttr.DampedLin", enabled=self.enable_timing):
                        P1 = SE3(self.poses_[self.n-1])
                        P2 = SE3(self.poses_[self.n-2])
                        
                        xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                        tvec_qvec = (SE3.exp(xi) * P1).data
                        self.poses_[self.n] = tvec_qvec

                else:
                    with Timer(
                        "SLAM.UpdateStateAttr.NotDampedLin", enabled=self.enable_timing
                    ):
                        tvec_qvec = self.poses[self.n - 1] #TODO: Check if poses or poses_
                        self.poses_[self.n] = tvec_qvec


        # TODO better depth initialization
        with Timer("SLAM.DepthInit", enabled=self.enable_timing):
            patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
            if self.is_initialized:
                s = torch.median(self.patches_[self.n - 3 : self.n, :, 2])
                patches[:, :, 2] = s

            self.patches_[self.n] = patches

        ### update network attributes ###

        with Timer("SLAM.NetworkAttrUpdate", enabled=self.enable_timing):
            # every self.mem=32 times update imap memory with the new imap
            self.imap_[self.n % self.mem] = imap.squeeze()
            self.gmap_[self.n % self.mem] = gmap.squeeze()
            self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
            self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

            self.counter += 1
            if self.n > 0 and not self.is_initialized:
                if self.motion_probe() < 2.0:

                    self.pose_delta[self.counter - 1] = (self.counter - 2, Id[0])

                    return
        

        #print("\n\n", self.imu_delta_p_R_[self.n])
        self.imu_delta_p_R_[self.n, :3] = self.imu_deltas_buffer["Dp"]
        self.imu_delta_p_R_[self.n, 3:] = self.imu_deltas_buffer["Dr"]
        self.imu_delta_v_[self.n]       = self.imu_deltas_buffer["Dv"]
        self.imu_delta_t_[self.n]       = self.imu_deltas_buffer["Dt"]
        #print("\n\n", self.imu_delta_p_R_[self.n])

        # update number of keyframes and number of total patches
        self.n += 1
        self.m += self.M

        with Timer("SLAM.AddEdges", enabled=self.enable_timing):
            # add edges to the graph
            self.append_factors(*self.__edges_forw())
            self.append_factors(*self.__edges_back())
        
        self.test_deltas.append(self.imu_deltas_buffer)
        self.imu_key_deltas.append(self.imu_deltas_buffer)

        self.imu_deltas[self.n] = self.imu_deltas_buffer
        self.imu_deltas_buffer = self.imu_deltas[0]
        self.key_added_t += self.imu_deltas[self.n]["Dt"]

        # initialize with 8 valid frames and do 12 slam updates
        if self.n == 8 and not self.is_initialized:
            with Timer("SLAM.NotInitializedUpdate", enabled=self.enable_timing):
                self.is_initialized = True

                for itr in range(12):
                    self.update()

        elif self.is_initialized:

            # snapshot of active poses right before the BA optimization
            #self.pre_update_poses_for_viz = self.poses_[:self.n].clone()

            with Timer("SLAM.InitializedUpdate", enabled=self.enable_timing):
                self.update()
            with Timer("SLAM.InitializedKeyframe", enabled=self.enable_timing):
                self.keyframe()

        # if not time=8 and not SLAM initialized do nothing
        else:
            pass