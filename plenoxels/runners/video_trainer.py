import logging as log
import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Any, List

import pandas as pd
import torch
import torch.utils.data
import numpy as np

from plenoxels.datasets.video_datasets import Video360Dataset
from plenoxels.utils.ema import EMA
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.models_all.lowrank_model import LowrankModel  #@remind
from models_all.poses import LearnPose
from datasets.ray_utils import get_rays
from models_all.cal_pose import make_c2w
from .base_trainer import BaseTrainer, init_dloader_random, initialize_model, initialize_posenet
from .regularization import (
    PlaneTV, TimeSmoothness, HistogramLoss, L1TimePlanes, DistortionLoss
)
import time
from utils_poses.comp_ate import compute_ate

class VideoTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 tr_dset: torch.utils.data.TensorDataset,
                 ts_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 isg_step: int,
                 ist_step: int,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.train_dataset = tr_dset
        self.test_dataset = ts_dset
        self.is_ndc = self.test_dataset.is_ndc
        self.is_contracted = self.test_dataset.is_contracted
        self.ist_step = ist_step
        self.isg_step = isg_step
        self.validrender_every = kwargs.get('validrender_every', -1)
        self.n_frame_per_cam = kwargs.get('n_frame_per_cam', 300)
        self.n_imgs_eval_per_step = kwargs.get('n_imgs_eval_per_step', 10)
        self.save_video = self.n_imgs_eval_per_step > 1
        # Switch to compute extra video metrics (FLIP, JOD)
        self.compute_video_metrics = False
        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=False,  # False since we're saving video
            device=device,
            num_cams=self.test_dataset.num_cams_train,
            **kwargs)
    
    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)  # self.model.eval()
        batch_size = self.eval_batch_size   
        channels = {"rgb", "depth", "proposal_depth"}
        with torch.cuda.amp.autocast(enabled=self.train_fp16) and torch.no_grad():
            # near_far, camera_id, intrinsics and bg_color are constant over mini-batches
            near_far = data["near_fars"].to(self.device)
            camera_id = data['camera_id'].to(self.device)
            timestamps = data['timestamps'].to(self.device)
            intrinsics = data['intrinsics'].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)

            # try: 
            c2w = self.eval_posenet(camera_id)
            # except RuntimeError:
            #     breakpoint()
            camera_dirs = data['camera_dirs'].to(self.device)
            rays_o, rays_d = get_rays(
                camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=intrinsics, normalize_rd=True
            )  # h*w, 3
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):  # rays_o.shape[0]:h*w
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamps.expand(rays_o_b.shape[0]).to(self.device)
                outputs = self.model(rays_o=rays_o_b, rays_d=rays_d_b, near_far=near_far,
                                    bg_color=bg_color, timestamps=timestamps_d_b)
                for k, v in outputs.items():
                    if k in channels or "depth" in k:
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}
    
    @torch.no_grad()
    def eval(self):
        dataset = self.test_dataset

        per_scene_metrics: Dict[str, Union[float, List]] = defaultdict(list)
        log.info(f"Start eval........")
        self.logger_py.info(f"Start eval........")
        pred_frames, out_depths = [], []
        self.eval_posenet.eval()
        pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
        for img_idx, data in enumerate(dataset):
            preds = self.eval_step(data)
            out_metrics, out_img, out_depth = self.evaluate_metrics(
                data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            pred_frames.append(out_img)
            if out_depth is not None:
                out_depths.append(out_depth)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            self.logger_py.info(f"PSNR={out_metrics['psnr']:.2f}")
            pb.update(1)
        pb.close()
        if self.save_video:
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}.mp4"),
                pred_frames
            )
            if len(out_depths) > 0:
                write_video_to_file(
                    os.path.join(self.log_dir, f"step{self.global_step}-depth.mp4"),
                    out_depths
                )
        # Calculate JOD (on whole video)
        if self.compute_video_metrics:
            per_scene_metrics["JOD"] = metrics.jod(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )
            per_scene_metrics["FLIP"] = metrics.flip(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )

        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name=None),
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))
    
    @torch.no_grad()
    def eval_holdevery(self):
        dataset = self.test_dataset

        log.info(f"Start eval........")
        self.eval_posenet.eval()

        for cam_idx in range(dataset.num_cams):
            per_scene_metrics: Dict[str, Union[float, List]] = defaultdict(list)
            pred_frames, out_depths = [], []
            # pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
            # pb = tqdm(total=300, desc=f"Test scene ({dataset.name})")
            n_frame_per_cam = self.n_frame_per_cam  # @todo
            # pb = tqdm(total=10, desc=f"Test scene ({dataset.name})")
            pb = tqdm(total=self.n_frame_per_cam, desc=f"Test scene ({dataset.name}) cam {cam_idx}")
            # img_list = torch.arange(cam_idx*300, (cam_idx+1)*300)
            # breakpoint()
            img_list = torch.arange(cam_idx * self.n_frame_per_cam, (cam_idx+1) * self.n_frame_per_cam)
            # for img_idx, data in enumerate(dataset):
            for i,img_idx in enumerate(img_list):
                data = dataset[img_idx]
                preds = self.eval_step(data)
                out_metrics, out_img, out_depth = self.evaluate_metrics(
                    data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None,
                    # save_outputs=self.save_outputs)
                    save_outputs=(i==0))
                pred_frames.append(out_img)
                if out_depth is not None:
                    out_depths.append(out_depth)
                for k, v in out_metrics.items():
                    per_scene_metrics[k].append(v)
                pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
                self.logger_py.info(f"PSNR={out_metrics['psnr']:.2f}")
                pb.update(1)
            pb.close()
            if self.save_video:
                write_video_to_file(
                    os.path.join(self.log_dir, f"step{self.global_step}_cam{cam_idx}.mp4"),
                    pred_frames
                )
                if len(out_depths) > 0:
                    write_video_to_file(
                        os.path.join(self.log_dir, f"step{self.global_step}_cam{cam_idx}-depth.mp4"),
                        out_depths
                    )
            # Calculate JOD (on whole video)
            if self.compute_video_metrics:
                per_scene_metrics["JOD"] = metrics.jod(
                    [f[:dataset.img_h, :, :] for f in pred_frames],
                    [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
                )
                per_scene_metrics["FLIP"] = metrics.flip(
                    [f[:dataset.img_h, :, :] for f in pred_frames],
                    [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
                )

            val_metrics = [
                self.report_test_metrics(per_scene_metrics, extra_name=None),
            ]
            df = pd.DataFrame.from_records(val_metrics)
            df.to_csv(os.path.join(self.log_dir, f"{np.mean(np.asarray(per_scene_metrics['psnr'])).item():.2f}_test_metrics_step{self.global_step}_cam{cam_idx}.csv"))  #@remind
    
    def eval_step_trainpose(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)  # self.model.eval()
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            # near_far, camera_id, intrinsics and bg_color are constant over mini-batches
            near_far = data["near_fars"].to(self.device)
            camera_id = data['camera_id'].to(self.device)
            timestamps = data['timestamps'].to(self.device)
            intrinsics = data['intrinsics'].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)

            # try: 
            c2w = self.eval_posenet(camera_id)
            # breakpoint()
            # breakpoint()
            # except RuntimeError:
            #     breakpoint()
            camera_dirs = data['camera_dirs'].to(self.device)
            rays_o, rays_d = get_rays(
                camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=intrinsics, normalize_rd=True
            )  # h*w, 3
            eval_pixels = torch.randperm(rays_o.shape[0])[:batch_size]
            rays_o = rays_o[eval_pixels]
            rays_d = rays_d[eval_pixels]
            timestamps_d = timestamps.expand(rays_o.shape[0]).to(self.device)
            outputs = self.model(rays_o=rays_o, rays_d=rays_d, near_far=near_far,
                                    bg_color=bg_color, timestamps=timestamps_d)
        
        gt = data['imgs'][eval_pixels].to(self.device)
        recon_loss = self.criterion(outputs['rgb'], gt)
        return recon_loss

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        scale_ok = super().train_step(data, **kwargs)

        if self.global_step == self.isg_step:
            self.train_dataset.enable_isg()
            raise StopIteration  # Whenever we change the dataset
        if self.global_step == self.ist_step:
            self.train_dataset.switch_isg2ist()
            raise StopIteration  # Whenever we change the dataset

        return scale_ok

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in train-dataset
        self.train_dataset.reset_iter()

    @torch.no_grad()
    def validate_origin(self):
        dataset = self.test_dataset
        per_scene_metrics: Dict[str, Union[float, List]] = defaultdict(list)
        pred_frames, out_depths = [], []
        pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
        for img_idx, data in enumerate(dataset):
            preds = self.eval_step(data)
            out_metrics, out_img, out_depth = self.evaluate_metrics(
                data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            pred_frames.append(out_img)
            if out_depth is not None:
                out_depths.append(out_depth)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
        pb.close()
        if self.save_video:
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}.mp4"),
                pred_frames
            )
            if len(out_depths) > 0:
                write_video_to_file(
                    os.path.join(self.log_dir, f"step{self.global_step}-depth.mp4"),
                    out_depths
                )
        # Calculate JOD (on whole video)
        if self.compute_video_metrics:
            per_scene_metrics["JOD"] = metrics.jod(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )
            per_scene_metrics["FLIP"] = metrics.flip(
                [f[:dataset.img_h, :, :] for f in pred_frames],
                [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            )

        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name=None),
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def validate(self):
        self.model.eval()
        self.model_pose.eval()
        dataset = self.test_dataset
        self.logger_py.info("\n")
        self.logger_py.info(f"Begin validate at step{self.global_step} ---------\n")

        with torch.no_grad():
            try:
                learn_train_poses = torch.stack([self.model_pose(torch.tensor([i])) for i in range(self.num_cams)])[[3,13]].squeeze(1).detach()

            except RuntimeError:
                breakpoint()
        print(f"device for eval pose net: {self.device}")
        self.eval_posenet = self.init_posenet(
            num_cams=dataset.num_cams, 
            for_eval=True,
            # init_c2ws=learn_train_poses,
            # camera_noise = 0.0,
            init_c2ws=dataset.per_cam_poses,
            camera_noise = 0.1,  # FIXME:
            **self.extra_args).to(self.device)
        self.eval_optimizer = self.init_pose_optim_eval(**self.extra_args)
        self.eval_scheduler = self.init_pose_lr_scheduler_eval(**self.extra_args)
        self.eval_gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)

        eval_steps = self.extra_args["eval_steps"]
        assert self.n_imgs_eval_per_step <= self.n_frame_per_cam, "n_imgs_eval_per_step out of range!"
        pb_learnpose = [tqdm(total=eval_steps, desc=f"Train eval cam{idx} poses") for idx in range(dataset.num_cams)]
        # pb_learnpose = tqdm(total=eval_steps, desc=f"Train eval cam0 poses")
        self.eval_posenet.train()
        for eval_step in range(eval_steps):
            now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
            message = now + " | step" + str(eval_step)+ ":"
            with torch.cuda.amp.autocast(enabled=self.train_fp16):  # 混合精度训练
                # loss_vals = [[] for cam_idx in range(dataset.num_cams)]
                # psnrs = [[] for cam_idx in range(dataset.num_cams)]
                loss_vals = torch.zeros((dataset.num_cams,)).to(self.device)
                psnrs = torch.zeros((dataset.num_cams,)).to(self.device)
                # rand_imgs_per_cam = torch.randint(0, self.n_frame_per_cam, size=(self.n_imgs_eval_per_step, ))
                rand_imgs_per_cam = torch.randperm(self.n_frame_per_cam)[:self.n_imgs_eval_per_step]
                rand_imgs = torch.arange(dataset.num_cams).view(-1,1).repeat(1,self.n_imgs_eval_per_step) * self.n_frame_per_cam + rand_imgs_per_cam
                rand_imgs = rand_imgs.view(-1)
                rand_imgs = rand_imgs[torch.randperm(rand_imgs.shape[0])]
                for img_idx in rand_imgs:
                    # print(img_idx)
                    data = dataset[img_idx]
                # for img_idx, data in enumerate(dataset):  # 300张cam0拍摄的图
                    loss = self.eval_step_trainpose(data)
                    with torch.no_grad():
                        psnr = -10 * math.log10(loss.item())
                        # psnrs[img_idx//self.n_frame_per_cam].append(psnr)
                        # loss_vals[img_idx//self.n_frame_per_cam].append(loss.item())
                        psnrs[img_idx//self.n_frame_per_cam] = psnrs[img_idx//self.n_frame_per_cam] + psnr
                        loss_vals[img_idx//self.n_frame_per_cam] = loss_vals[img_idx//self.n_frame_per_cam] + loss.item()
                    # pb_learnpose[img_idx//300].set_postfix_str(f"mse={loss} PSNR={psnr:.2f}", refresh=False)
                    # pb_learnpose[img_idx//300].update(1)

                    # Update weights
                    self.eval_optimizer.zero_grad(set_to_none=True)
                    # with torch.autograd.set_detect_anomaly(True):
                    try:
                        self.eval_gscaler.scale(loss).backward()
                    except RuntimeError:
                        breakpoint()

                    self.eval_gscaler.step(self.eval_optimizer)
                    scale = self.eval_gscaler.get_scale()
                    self.eval_gscaler.update()

                for cam_idx in range(dataset.num_cams):
                    with torch.no_grad():
                        # breakpoint()
                        
                        learned_poses = torch.tensor(self.eval_posenet([cam_idx]))[:,:3,:].detach()
                        # breakpoint()
                        gtpose = dataset.per_cam_poses[cam_idx]
                        # breakpoint()
                        stats_tran, stats_rot, _ = compute_ate(learned_poses, gtpose.unsqueeze(0))  #@remind

                        # pb_learnpose[cam_idx].set_postfix_str(f"mse={np.mean(loss_vals[cam_idx])} PSNR={np.mean(psnrs[cam_idx]):.2f}", refresh=False)
                        pb_learnpose[cam_idx].set_postfix_str(f"t={(stats_tran['mean']):.4f} r={(stats_rot['mean']):.3f} mse={loss_vals[cam_idx] / self.n_imgs_eval_per_step} PSNR={(psnrs[cam_idx] / self.n_imgs_eval_per_step):.2f}", refresh=False)
                        pb_learnpose[cam_idx].update(1)

                        message += f" | cam {cam_idx}: | t={stats_tran['mean']} | r={stats_rot['mean']} | mse={loss_vals[cam_idx] / self.n_imgs_eval_per_step} | PSNR={psnrs[cam_idx] / self.n_imgs_eval_per_step}"
            
            with open(self.logfile, "a") as file:
                file.write(message+"\n")

            step_successful = scale <= self.eval_gscaler.get_scale()
            if step_successful and self.eval_scheduler is not None:
                self.eval_scheduler.step()
        for cam_idx in range(dataset.num_cams):
            pb_learnpose[cam_idx].close()
        # pb_learnpose.close()
        print()
        self.logger_py.info(f"Train eval cam poses end!\n")

        self.save_evalposenet()
        # if self.validrender_every > -1 and self.global_step % self.validrender_every == 0:
        self.eval_holdevery()


    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    
    def save_evalposenet(self):
        if self.global_step >= self.num_steps - 1:
            model_fname = os.path.join(self.log_dir, f'evalposenet.pth')
        else:
            model_fname = os.path.join(self.log_dir, f'evalposenet_step{self.global_step}.pth')
        # log.info(f'Saving model checkpoint to: {model_fname}')
        self.logger_py.info(f'Saving evalposemodel checkpoint to: {model_fname}')
        torch.save(self.get_save_dict_evalpose(), model_fname)
    

    def load_model(self, checkpoint_data, ckpt_path: str = None, training_needed: bool = True):
        super().load_model(checkpoint_data, ckpt_path, training_needed)
        if self.train_dataset is not None:
            if -1 < self.isg_step < self.global_step < self.ist_step:
                self.train_dataset.enable_isg()
            elif -1 < self.ist_step < self.global_step:
                self.train_dataset.switch_isg2ist()

    def init_epoch_info(self):
        ema_weight = 0.9
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankModel:
        return initialize_model(self, **kwargs)
    
    def init_posenet(self, **kwargs) -> LearnPose:
        return initialize_posenet(self, **kwargs)

    def get_regularizers(self, **kwargs):  #FIXME:
        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            L1TimePlanes(kwargs.get('l1_time_planes', 0.0), what='field'),
            L1TimePlanes(kwargs.get('l1_time_planes_proposal_net', 0.0), what='proposal_network'),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0), what='field'),
            TimeSmoothness(kwargs.get('time_smoothness_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
        ]

    @property
    def calc_metrics_every(self):
        return 5


def init_tr_data(data_downsample, data_dir, **kwargs):
    isg = kwargs.get('isg', False)
    ist = kwargs.get('ist', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    log.info(f"Loading Video360Dataset with downsample={data_downsample}")
    tr_dset = Video360Dataset(
        data_dir, split='train', downsample=data_downsample,
        batch_size=batch_size,
        max_cameras=kwargs.get('max_train_cameras', None),
        max_tsteps=kwargs['max_train_tsteps'] if keyframes else None,
        isg=isg, keyframes=keyframes, contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],n_frame_per_cam=kwargs['n_frame_per_cam']
    )
    if ist:
        tr_dset.switch_isg2ist()  # this should only happen in case we're reloading

    g = torch.Generator()
    g.manual_seed(0)
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, num_workers=4,  prefetch_factor=4, pin_memory=True,
        worker_init_fn=init_dloader_random, generator=g)
    return {"tr_loader": tr_loader, "tr_dset": tr_dset}


def init_ts_data(data_dir, split, **kwargs):
    downsample = 2.0 # Both D-NeRF and DyNeRF use downsampling by 2
    ts_dset = Video360Dataset(
        data_dir, split=split, downsample=downsample,
        hold_every=kwargs.get('hold_every', 8),
        max_cameras=kwargs.get('max_test_cameras', None), max_tsteps=kwargs.get('max_test_tsteps', None),
        contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],n_frame_per_cam=kwargs['n_frame_per_cam']
    )
    return {"ts_dset": ts_dset}


def load_data(data_downsample, data_dirs, validate_only, render_only, **kwargs):
    assert len(data_dirs) == 1
    od: Dict[str, Any] = {}
    if not validate_only and not render_only:
        od.update(init_tr_data(data_downsample, data_dirs[0], **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    test_split = 'render' if render_only else 'test'
    od.update(init_ts_data(data_dirs[0], split=test_split, **kwargs))
    return od
