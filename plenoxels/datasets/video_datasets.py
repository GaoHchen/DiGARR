import glob
import json
import logging as log
import math
import os
import time
from collections import defaultdict
from typing import Optional, List, Tuple, Any, Dict

import numpy as np
import torch

from .base_dataset import BaseDataset
from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .llff_dataset import load_llff_poses_helper
from .ray_utils import (
    generate_spherical_poses, create_meshgrid, stack_camera_dirs, get_rays, generate_spiral_path
)
from .synthetic_nerf_dataset import (
    load_360_images, load_360_intrinsics,
)

# from lightglue import SuperPoint
# from lightglue.utils import rbd


class Video360Dataset(BaseDataset):
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 keyframes: bool = False,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 isg: bool = False,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 near_scaling: float = 0.9,
                 ndc_far: float = 2.6,
                 hold_every: int = 8,
                 n_frame_per_cam = 300,):
        self.keyframes = keyframes
        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.downsample = downsample
        self.isg = isg
        self.ist = False
        # self.lookup_time = False
        self.per_cam_near_fars = None
        self.global_translation = torch.tensor([0, 0, 0])
        self.global_scale = torch.tensor([1, 1, 1])
        self.near_scaling = near_scaling
        self.ndc_far = ndc_far
        self.hold_every = hold_every
        self.n_frame_per_cam = n_frame_per_cam
        self.median_imgs = None
        # self.interest_list = None
        # self.num_rays_roi = None
        # self.num_rays_rand = None
        if contraction and ndc:
            raise ValueError("Options 'contraction' and 'ndc' are exclusive.")
        if "lego" in datadir or "dnerf" in datadir:
            dset_type = "synthetic"
        else:
            dset_type = "llff"

        # Note: timestamps are stored normalized between -1, 1.
        if dset_type == "llff":
            if split == "render":
                assert ndc, "Unable to generate render poses without ndc: don't know near-far."
                per_cam_poses, per_cam_near_fars, intrinsics, _, num_cams_train = load_llffvideo_poses(
                    datadir, downsample=self.downsample, split='all', near_scaling=self.near_scaling)
                render_poses = generate_spiral_path(
                    per_cam_poses.numpy(), per_cam_near_fars.numpy(), n_frames=300,
                    n_rots=2, zrate=0.5, dt=self.near_scaling, percentile=60)
                self.poses = torch.from_numpy(render_poses).float()
                self.per_cam_near_fars = torch.tensor([[0.4, self.ndc_far]])
                timestamps = torch.linspace(0, 299, len(self.poses))
                imgs = None
            else:  # train/test
                per_cam_poses, per_cam_near_fars, intrinsics, videopaths, num_cams_train = load_llffvideo_poses_holdevery(
                    datadir, downsample=self.downsample, split=split, near_scaling=self.near_scaling, hold_every=self.hold_every)
                if split == 'test':
                    keyframes = False
                poses, imgs, timestamps, self.median_imgs = load_llffvideo_data(
                    videopaths=videopaths, cam_poses=per_cam_poses, intrinsics=intrinsics,
                    split=split, keyframes=keyframes, keyframes_take_each=30, n_frame_per_cam=n_frame_per_cam)
                self.per_cam_poses = per_cam_poses.float()
                self.poses = poses.float()  # [N, 3, 4]
                if contraction:
                    self.per_cam_near_fars = per_cam_near_fars.float()
                else:
                    self.per_cam_near_fars = torch.tensor(
                        [[0.0, self.ndc_far]]).repeat(per_cam_near_fars.shape[0], 1)
            # These values are tuned for the salmon video
            self.global_translation = torch.tensor([0, 0, 2.])
            self.global_scale = torch.tensor([0.5, 0.6, 1])
            # Normalize timestamps between -1, 1
            if n_frame_per_cam > 1:
                timestamps = (timestamps.float() / (n_frame_per_cam-1)) * 2 - 1  # [N]
                
            # use superpoint method to get point matches
            # #! Superpoint 关键点采样
            # interest_list = []
            # print('using superpoint to extract interest points')
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
            # for idx in range(self.imgs.shape[0]):  # N = n_cams*n_frame_per_cam
            #     img_ = imgs[idx]
            #     img_ = torch.from_numpy(img_).unsqueeze(0).to(device)
            #     img_ = torch.tensor(img_, dtype=torch.float)
            #     feats1 = extractor.extract(img_.to(device))
            #     feats1 = rbd(feats1)
            #     keypoints = feats1['keypoints'] # 2048*2
            #     interest_indices = keypoints.cpu().numpy().astype(int)
            #     interest_indices = torch.from_numpy(interest_indices)
            #     interest_list.append(interest_indices)
            # self.interest_list = torch.cat(interest_list)  # (n_cam*n_frame_per_cam, 2048, 2)
        elif dset_type == "synthetic":
            assert not contraction, "Synthetic video dataset does not work with contraction."
            assert not ndc, "Synthetic video dataset does not work with NDC."
            if split == 'render':
                num_tsteps = 120
                dnerf_durations = {'hellwarrior': 100, 'mutant': 150, 'hook': 100, 'bouncingballs': 150, 'lego': 50, 'trex': 200, 'standup': 150, 'jumpingjacks': 200}
                for scene in dnerf_durations.keys():
                    if 'dnerf' in datadir and scene in datadir:
                        num_tsteps = dnerf_durations[scene]
                render_poses = torch.stack([
                    generate_spherical_poses(angle, -30.0, 4.0)
                    for angle in np.linspace(-180, 180, num_tsteps + 1)[:-1]
                ], 0)
                imgs = None
                self.poses = render_poses
                timestamps = torch.linspace(0.0, 1.0, render_poses.shape[0])
                _, transform = load_360video_frames(
                    datadir, 'train', max_cameras=self.max_cameras, max_tsteps=self.max_tsteps)
                img_h, img_w = 800, 800
            else:
                frames, transform = load_360video_frames(
                    datadir, split, max_cameras=self.max_cameras, max_tsteps=self.max_tsteps)
                imgs, self.poses = load_360_images(frames, datadir, split, self.downsample)
                timestamps = torch.tensor(
                    [fetch_360vid_info(f)[0] for f in frames], dtype=torch.float32)
                img_h, img_w = imgs[0].shape[:2]
            if ndc:
                self.per_cam_near_fars = torch.tensor([[0.0, self.ndc_far]])
            else:
                self.per_cam_near_fars = torch.tensor([[2.0, 6.0]])
            if "dnerf" in datadir:
                # dnerf time is between 0, 1. Normalize to -1, 1
                timestamps = timestamps * 2 - 1
            else:
                # lego (our vid) time is like dynerf: between 0, 30.
                timestamps = (timestamps.float() / torch.amax(timestamps)) * 2 - 1
            intrinsics = load_360_intrinsics(
                transform, img_h=img_h, img_w=img_w, downsample=self.downsample)
        else:
            raise ValueError(datadir)
        
        assert dset_type == "llff", "别急, 动态合成数据集还没写呢....."
        self.num_cams_train = num_cams_train
        self.num_images = len(videopaths)  # N_cams @todo 
        self.num_cams = len(videopaths)
        # self.num_images = len(self.poses)  # N_cams * N_frames_per_cam, train:19*300, test:1*300

        self.timestamps = timestamps
        if split == 'train':
            self.timestamps = self.timestamps[:, None, None].repeat(
                1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]
        assert self.timestamps.min() >= -1.0 and self.timestamps.max() <= 1.0, "timestamps out of range."
        if imgs is not None and imgs.dtype != torch.uint8:
            imgs = (imgs * 255).to(torch.uint8)
        if self.median_imgs is not None and self.median_imgs.dtype != torch.uint8:
            self.median_imgs = (self.median_imgs * 255).to(torch.uint8)
        if split == 'train':
            imgs = imgs.view(-1, imgs.shape[-1])
        elif imgs is not None:
            imgs = imgs.view(-1, intrinsics.height * intrinsics.width, imgs.shape[-1])

        # ISG/IST weights are computed on 4x subsampled data.
        weights_subsampled = max(int(4 / downsample), 1)  #@note
        if scene_bbox is not None:
            scene_bbox = torch.tensor(scene_bbox)
        else:
            scene_bbox = get_bbox(datadir, is_contracted=contraction, dset_type=dset_type)
        super().__init__(
            datadir=datadir,
            split=split,
            batch_size=batch_size,
            is_ndc=ndc,
            is_contracted=contraction,
            scene_bbox=scene_bbox,
            rays_o=None,
            rays_d=None,
            intrinsics=intrinsics,
            imgs=imgs,
            sampling_weights=None,  # Start without importance sampling, by default
            weights_subsampled=weights_subsampled,
        )

        self.isg_weights = None
        self.ist_weights = None
        if split == "train" and dset_type == 'llff':  # Only use importance sampling with DyNeRF videos
            if os.path.exists(os.path.join(datadir, f"isg_weights.pt")):
                self.isg_weights = torch.load(os.path.join(datadir, f"isg_weights.pt"))
                log.info(f"Reloaded {self.isg_weights.shape[0]} ISG weights from file.")
            else:
                # Precompute ISG weights
                t_s = time.time()
                gamma = 1e-3 if self.keyframes else 2e-2
                self.isg_weights = dynerf_isg_weight(
                    imgs.view(-1, intrinsics.height, intrinsics.width, imgs.shape[-1]),
                    median_imgs=self.median_imgs, gamma=gamma)
                # Normalize into a probability distribution, to speed up sampling
                self.isg_weights = (self.isg_weights.reshape(-1) / torch.sum(self.isg_weights))
                torch.save(self.isg_weights, os.path.join(datadir, f"isg_weights.pt"))
                t_e = time.time()
                log.info(f"Computed {self.isg_weights.shape[0]} ISG weights in {t_e - t_s:.2f}s.")

            if os.path.exists(os.path.join(datadir, f"ist_weights.pt")):
                self.ist_weights = torch.load(os.path.join(datadir, f"ist_weights.pt"))
                log.info(f"Reloaded {self.ist_weights.shape[0]} IST weights from file.")
            else:
                # Precompute IST weights
                t_s = time.time()
                self.ist_weights = dynerf_ist_weight(
                    imgs.view(-1, self.img_h, self.img_w, imgs.shape[-1]),
                    num_cameras=self.median_imgs.shape[0])  # (num_cameras, num_timestamps, h, w)
                # Normalize into a probability distribution, to speed up sampling
                self.ist_weights = (self.ist_weights.reshape(-1) / torch.sum(self.ist_weights))
                torch.save(self.ist_weights, os.path.join(datadir, f"ist_weights.pt"))
                t_e = time.time()
                log.info(f"Computed {self.ist_weights.shape[0]} IST weights in {t_e - t_s:.2f}s.")

        if self.isg:
            self.enable_isg()

        log.info(f"VideoDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                 f"Loaded {self.split} set from {self.datadir}: "
                 f"{len(self.poses)} images of size {self.img_h}x{self.img_w}. "
                 f"Images loaded: {self.imgs is not None}. "
                 f"{len(torch.unique(timestamps))} timestamps. Near-far: {self.per_cam_near_fars}. "
                 f"ISG={self.isg}, IST={self.ist}, weights_subsampled={self.weights_subsampled}. "
                 f"Sampling without replacement={self.use_permutation}. {intrinsics}")

    def enable_isg(self):
        self.isg = True
        self.ist = False
        self.sampling_weights = self.isg_weights
        log.info(f"Enabled ISG weights.")

    def switch_isg2ist(self):
        self.isg = False
        self.ist = True
        self.sampling_weights = self.ist_weights
        log.info(f"Switched from ISG to IST weights.")

    def get_superpoint_nums(self, alpha):
        self.num_rays_roi = int(alpha * self.batch_size)
        self.num_rays_rand = self.batch_size - self.num_rays_roi
    
    def del_sp(self):
        self.num_rays_roi = None
        self.num_rays_rand = None

    # def get_superpoint_ids(self, h, w, index):
    #     # self.interest_list: (n_cam*frame, 2048, 2)
    #     # self.interest_list.reshape(n_cam, n_frame_per_cam*2048, 2)
    #     interest_idx = 
    #     rand_idx = torch.randperm(self.interest_list.shape[1])  # n_frame_per_cam*2048
    #     self.interest_perm = self.interest_list[:,rand_idx,:]  # (n_cam, n_frame_per_cam*2048, 2)
    #     # ROI = self.interest_list[0]
    #     ROI = ROI[... ,0] * h +  ROI[... ,1]
    #     # ROI_idx = ROI[..., 0] * w + ROI[..., 1]  # index = y * w + x
    #     roi_ids = ROI[torch.randperm(ROI.shape[0], device=device)[:num_rays_roi]]    # (num_rays_roi)
    #     id_rand = torch.randperm(self.intrinsics.height*self.intrinsics.width, device=device)[:num_rays_rand]    # (num_rays_rand)
    #     # remove same points in roi_ids and id_rand, and finally get n_points
    #     # roi_ids = torch.unique(torch.cat([roi_ids, id_rand], dim=0))  # (num_rays_roi + num_rays_rand)
    #     ray_idx = torch.cat([roi_ids, id_rand], dim=0)  # (n_points)
    #     while roi_ids.shape[0] < n_points:
    #         id_rand_res = torch.randperm(h*w, device=device)[:roi_ids.shape[0]-n_points]
    #         roi_ids = torch.unique(torch.cat([roi_ids, id_rand_res], dim=0))
    #     self.sp_ray_idx = ray_idx

    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        if self.split == 'train':
            index = self.get_rand_ids(index)  # [batch_size // (weights_subsampled**2)]
            # index = self.get_sp_rayid(index)
            if self.weights_subsampled == 1 or self.sampling_weights is None:
                # Nothing special to do, eit1her weights_subsampled = 1, or not using weights.
                image_id = torch.div(index, h * w, rounding_mode='floor')
                y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
                x = torch.remainder(index, h * w).remainder(w)
            else:
                # We must deal with the fact that ISG/IST weights are computed on a dataset with
                # different 'downsampling' factor. E.g. if the weights were computed on 4x
                # downsampled data and the current dataset is 2x downsampled, `weights_subsampled`
                # will be 4 / 2 = 2.
                # Split each subsampled index into its 16 components in 2D.
                hsub, wsub = h // self.weights_subsampled, w // self.weights_subsampled
                image_id = torch.div(index, hsub * wsub, rounding_mode='floor')
                ysub = torch.remainder(index, hsub * wsub).div(wsub, rounding_mode='floor')
                xsub = torch.remainder(index, hsub * wsub).remainder(wsub)
                # xsub, ysub is the first point in the 4x4 square of finely sampled points
                x, y = [], []
                for ah in range(self.weights_subsampled):
                    for aw in range(self.weights_subsampled):
                        x.append(xsub * self.weights_subsampled + aw)
                        y.append(ysub * self.weights_subsampled + ah)
                x = torch.cat(x)
                y = torch.cat(y)
                image_id = image_id.repeat(self.weights_subsampled ** 2)
                # Inverse of the process to get x, y from index. image_id stays the same.
                index = x + y * w + image_id * h * w
            x, y = x + 0.5, y + 0.5
        else:  # 'test'对整张图采样
            # image_id = [index]
            image_id = torch.tensor([index])
            x, y = create_meshgrid(height=h, width=w, dev=dev, add_half=True, flat=True)  # [h,w] --(flatten)--> [h*w]

        out = {
            "timestamps": self.timestamps[index],      # (num_rays or 1, )
            "imgs": None,
        }
        
        # @remind
        if self.split == 'train':
            num_frames_per_camera = len(self.imgs) // (len(self.per_cam_near_fars) * h * w)
        else:
            num_frames_per_camera = len(self.imgs) // (len(self.per_cam_near_fars)) 
        assert num_frames_per_camera == self.n_frame_per_cam, f"n_frame_per_cam({self.n_frame_per_cam}) calculated wrong to {num_frames_per_camera}!!"
        camera_id = torch.div(image_id, num_frames_per_camera, rounding_mode='floor')  # (num_rays)
        out['near_fars'] = self.per_cam_near_fars[camera_id, :]

        if self.imgs is not None:
            out['imgs'] = (self.imgs[index] / 255.0).view(-1, self.imgs.shape[-1])

        # c2w = self.poses[image_id]                                    # [num_rays or 1, 3, 4]
        camera_dirs = stack_camera_dirs(x, y, self.intrinsics, True)  # [num_rays or h*w, 3]
        # out['rays_o'], out['rays_d'] = get_rays(
        #     camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=self.intrinsics,
        #     normalize_rd=True)                                        # [num_rays, 3]
        out["intrinsics"] = self.intrinsics.get_matrix()
        out["camera_dirs"] = camera_dirs
        if not isinstance(camera_id, torch.Tensor):
            out["camera_id"] = torch.from_numpy(np.array(camera_id))  # [n_rays or 1]
        else:
            out["camera_id"] = camera_id

        imgs = out['imgs']
        # Decide BG color
        bg_color = torch.ones((1, 3), dtype=torch.float32, device=dev)
        if self.split == 'train' and imgs.shape[-1] == 4:
            bg_color = torch.rand((1, 3), dtype=torch.float32, device=dev)
        out['bg_color'] = bg_color
        # Alpha compositing
        if imgs is not None and imgs.shape[-1] == 4:
            imgs = imgs[:, :3] * imgs[:, 3:] + bg_color * (1.0 - imgs[:, 3:])
        out['imgs'] = imgs

        return out


def get_bbox(datadir: str, dset_type: str, is_contracted=False) -> torch.Tensor:
    """Returns a default bounding box based on the dataset type, and contraction state.

    Args:
        datadir (str): Directory where data is stored
        dset_type (str): A string defining dataset type (e.g. synthetic, llff)
        is_contracted (bool): Whether the dataset will use contraction

    Returns:
        Tensor: 3x2 bounding box tensor
    """
    if is_contracted:
        radius = 2
    elif dset_type == 'synthetic':
        radius = 1.5
    elif dset_type == 'llff':
        return torch.tensor([[-3.0, -1.67, -1.2], [3.0, 1.67, 1.2]])
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


def fetch_360vid_info(frame: Dict[str, Any]):
    timestamp = None
    fp = frame['file_path']
    if '_r' in fp:
        timestamp = int(fp.split('t')[-1].split('_')[0])
    if 'r_' in fp:
        pose_id = int(fp.split('r_')[-1])
    else:
        pose_id = int(fp.split('r')[-1])
    if timestamp is None:  # will be None for dnerf
        timestamp = frame['time']
    return timestamp, pose_id


def load_360video_frames(datadir, split, max_cameras: int, max_tsteps: Optional[int]) -> Tuple[Any, Any]:
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']

    timestamps = set()
    pose_ids = set()
    fpath2poseid = defaultdict(list)
    for frame in frames:
        timestamp, pose_id = fetch_360vid_info(frame)
        timestamps.add(timestamp)
        pose_ids.add(pose_id)
        fpath2poseid[frame['file_path']].append(pose_id)
    timestamps = sorted(timestamps)
    pose_ids = sorted(pose_ids)

    if max_cameras is not None:
        num_poses = min(len(pose_ids), max_cameras or len(pose_ids))
        subsample_poses = int(round(len(pose_ids) / num_poses))
        pose_ids = set(pose_ids[::subsample_poses])
        log.info(f"Selected subset of {len(pose_ids)} camera poses: {pose_ids}.")

    if max_tsteps is not None:
        num_timestamps = min(len(timestamps), max_tsteps or len(timestamps))
        subsample_time = int(math.floor(len(timestamps) / (num_timestamps - 1)))
        timestamps = set(timestamps[::subsample_time])
        log.info(f"Selected subset of timestamps: {sorted(timestamps)} of length {len(timestamps)}")

    sub_frames = []
    for frame in frames:
        timestamp, pose_id = fetch_360vid_info(frame)
        if timestamp in timestamps and pose_id in pose_ids:
            sub_frames.append(frame)
    # We need frames to be sorted by pose_id
    sub_frames = sorted(sub_frames, key=lambda f: fpath2poseid[f['file_path']])
    return sub_frames, meta


def load_llffvideo_poses(datadir: str,
                         downsample: float,
                         split: str,
                         near_scaling: float) -> Tuple[
                            torch.Tensor, torch.Tensor, Intrinsics, List[str]]:
    """Load poses and metadata for LLFF video.

    Args:
        datadir (str): Directory containing the videos and pose information
        downsample (float): How much to downsample videos. The default for LLFF videos is 2.0
        split (str): 'train' or 'test'.
        near_scaling (float): How much to scale the near bound of poses.

    Returns:
        Tensor: A tensor of size [N, 4, 4] containing c2w poses for each camera.
        Tensor: A tensor of size [N, 2] containing near, far bounds for each camera.
        Intrinsics: The camera intrinsics. These are the same for every camera.
        List[str]: List of length N containing the path to each camera's data.
    """
    poses, near_fars, intrinsics = load_llff_poses_helper(datadir, downsample, near_scaling)

    videopaths = np.array(glob.glob(os.path.join(datadir, '*.mp4')))  # [n_cameras]
    assert poses.shape[0] == len(videopaths), \
        'Mismatch between number of cameras and number of poses!'
    videopaths.sort()

    # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0 @remind
    # i_test = np.arange(0, poses.shape[0], hold_every)
    if split == 'train':
        split_ids = np.arange(1, poses.shape[0])
    elif split == 'test':
        split_ids = np.array([0])
    else:
        split_ids = np.arange(poses.shape[0])
    if 'coffee_martini' in datadir:
        # https://github.com/fengres/mixvoxels/blob/0013e4ad63c80e5f14eb70383e2b073052d07fba/dataLoader/llff_video.py#L323
        log.info(f"Deleting unsynchronized camera from coffee-martini video.")
        split_ids = np.setdiff1d(split_ids, 12)

    # num_cams_train = poses.shape[0] - i_test.shape[0]
    num_cams_train = poses.shape[0] - 1

    poses = torch.from_numpy(poses[split_ids])
    near_fars = torch.from_numpy(near_fars[split_ids])
    videopaths = videopaths[split_ids].tolist()

    poses[:,0,:] = poses[:,0,:] * -1
    return poses, near_fars, intrinsics, videopaths, num_cams_train


def load_llffvideo_poses_holdevery(datadir: str,
                                   downsample: float,
                                   split: str,
                                   near_scaling: float,
                                   hold_every: int,) -> Tuple[
                                       torch.Tensor, torch.Tensor, Intrinsics, List[str]]:
    """Load poses and metadata for LLFF video.

    Args:
        datadir (str): Directory containing the videos and pose information
        downsample (float): How much to downsample videos. The default for LLFF videos is 2.0
        split (str): 'train' or 'test'.
        near_scaling (float): How much to scale the near bound of poses.

    Returns:
        Tensor: A tensor of size [N, 4, 4] containing c2w poses for each camera.
        Tensor: A tensor of size [N, 2] containing near, far bounds for each camera.
        Intrinsics: The camera intrinsics. These are the same for every camera.
        List[str]: List of length N containing the path to each camera's data.
    """
    poses, near_fars, intrinsics = load_llff_poses_helper(datadir, downsample, near_scaling)

    videopaths = np.array(glob.glob(os.path.join(datadir, '*.mp4')))  # [n_cameras]
    assert poses.shape[0] == len(videopaths), \
        'Mismatch between number of cameras and number of poses!'
    videopaths.sort()

    # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
    # i_test = np.arange(0, poses.shape[0], hold_every)[1:]
    i_test = np.array([0])  #TODO:
    # i_test = np.array([0,15])
    if split == 'test':
        split_ids = i_test
    elif split == 'train':
        split_ids = list(set(np.arange(len(poses))) - set(i_test))
    else:
        split_ids = np.arange(poses.shape[0])
    if 'coffee_martini' in datadir:
        # https://github.com/fengres/mixvoxels/blob/0013e4ad63c80e5f14eb70383e2b073052d07fba/dataLoader/llff_video.py#L323
        log.info(f"Deleting unsynchronized camera from coffee-martini video.")
        split_ids = np.setdiff1d(split_ids, 12)
        num_cams_train = poses.shape[0] - i_test.shape[0] - 1
    else:
        num_cams_train = poses.shape[0] - i_test.shape[0]

    poses = torch.from_numpy(poses[split_ids])
    near_fars = torch.from_numpy(near_fars[split_ids])
    videopaths = videopaths[split_ids].tolist()
    poses[:,0,:] = poses[:,0,:] * -1

    return poses, near_fars, intrinsics, videopaths, num_cams_train


def load_llffvideo_data(videopaths: List[str],
                        cam_poses: torch.Tensor,
                        intrinsics: Intrinsics,
                        split: str,
                        keyframes: bool,
                        keyframes_take_each: Optional[int] = None,
                        n_frame_per_cam: int=300,
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if keyframes and (keyframes_take_each is None or keyframes_take_each < 1):
        raise ValueError(f"'keyframes_take_each' must be a positive number, "
                         f"but is {keyframes_take_each}.")

    loaded = parallel_load_images(
        dset_type="video",
        tqdm_title=f"Loading {split} data",
        num_images=len(videopaths),
        paths=videopaths,
        poses=cam_poses,
        out_h=intrinsics.height,
        out_w=intrinsics.width,
        load_every=keyframes_take_each if keyframes else 1,
        n_frame_per_cam=n_frame_per_cam
    )
    imgs, poses, median_imgs, timestamps = zip(*loaded)
    # Stack everything together
    # N = n_cams * num_frames_per_cam(300)
    timestamps = torch.cat(timestamps, 0)  # [N]
    poses = torch.cat(poses, 0)            # [N, 3, 4]
    imgs = torch.cat(imgs, 0)              # [N, h, w, 3]
    median_imgs = torch.stack(median_imgs, 0)  # [num_cameras, h, w, 3]
    return poses, imgs, timestamps, median_imgs


def load_llffvideo_data_permute(videopaths: List[str],
                        cam_poses: torch.Tensor,
                        intrinsics: Intrinsics,
                        split: str,
                        keyframes: bool,
                        keyframes_take_each: Optional[int] = None,
                        n_frame_per_cam: int=300,
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if keyframes and (keyframes_take_each is None or keyframes_take_each < 1):
        raise ValueError(f"'keyframes_take_each' must be a positive number, "
                         f"but is {keyframes_take_each}.")

    loaded = parallel_load_images(
        dset_type="video",
        tqdm_title=f"Loading {split} data",
        num_images=len(videopaths),
        paths=videopaths,
        poses=cam_poses,
        out_h=intrinsics.height,
        out_w=intrinsics.width,
        load_every=keyframes_take_each if keyframes else 1,
        n_frame_per_cam=n_frame_per_cam
    )
    imgs, poses, median_imgs, timestamps = zip(*loaded)  # imgs:tuple(len:18, each:[n_frame_per_cam, h, w, c])
    # Stack everything together
    # N = n_cams * num_frames_per_cam(300)
    # @remind
    timestamps = torch.stack(timestamps).permute(1,0).reshape(-1)  # [n_cam, n_frame_per_cam] -> [n_frame_per_cam, n_cam] -> [N]
    poses = torch.stack(poses).permute(1,0).reshape(-1)            # -> [n_frame_per_cam, n_cam, 3, 4] -> [N, 3, 4]
    imgs = torch.stack(imgs).permute(1,0).reshape(-1)              # -> [n_frame_per_cam, n_cam, ,h , w, 3] -> [N, h, w, 3]
    median_imgs = torch.stack(median_imgs, 0)  # [num_cameras, h, w, 3]
    return poses, imgs, timestamps, median_imgs


@torch.no_grad()
def dynerf_isg_weight(imgs, median_imgs, gamma):
    # imgs is [num_cameras * num_frames, h, w, 3]
    # median_imgs is [num_cameras, h, w, 3]
    assert imgs.dtype == torch.uint8
    assert median_imgs.dtype == torch.uint8
    num_cameras, h, w, c = median_imgs.shape
    squarediff = (
        imgs.view(num_cameras, -1, h, w, c)
            .float()  # creates new tensor, so later operations can be in-place
            .div_(255.0)
            .sub_(
                median_imgs[:, None, ...].float().div_(255.0)
            )
            .square_()  # noqa
    )  # [num_cameras, num_frames, h, w, 3]
    # differences = median_imgs[:, None, ...] - imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_frames, h, w, 3]
    # squarediff = torch.square_(differences)
    psidiff = squarediff.div_(squarediff + gamma**2)
    psidiff = (1./3) * torch.sum(psidiff, dim=-1)  # [num_cameras, num_frames, h, w]
    return psidiff  # valid probabilities, each in [0, 1]


@torch.no_grad()
def dynerf_ist_weight(imgs, num_cameras, alpha=0.1, frame_shift=25):  # DyNerf uses alpha=0.1
    assert imgs.dtype == torch.uint8
    N, h, w, c = imgs.shape
    frames = imgs.view(num_cameras, -1, h, w, c).float()  # [num_cameras, num_timesteps, h, w, 3]
    max_diff = None
    shifts = list(range(frame_shift + 1))[1:]
    for shift in shifts:
        shift_left = torch.cat([frames[:, shift:, ...], torch.zeros(num_cameras, shift, h, w, c)], dim=1)
        shift_right = torch.cat([torch.zeros(num_cameras, shift, h, w, c), frames[:, :-shift, ...]], dim=1)
        mymax = torch.maximum(torch.abs_(shift_left - frames), torch.abs_(shift_right - frames))
        if max_diff is None:
            max_diff = mymax
        else:
            max_diff = torch.maximum(max_diff, mymax)  # [num_timesteps, h, w, 3] lu:?, 感觉应该是(num_cameras, num_timestamps, h, w, 3)
    max_diff = torch.mean(max_diff, dim=-1)  # [num_timesteps, h, w] lu:?, 感觉应该是(num_cameras, num_timestamps, h, w)
    max_diff = max_diff.clamp_(min=alpha)
    return max_diff
