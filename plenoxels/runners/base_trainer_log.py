import abc
import random
import logging as log
import math
import os
from copy import copy
from typing import Iterable, Optional, Union, Dict, Tuple, Sequence, MutableMapping

import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from plenoxels.utils.timer import CudaTimer
from plenoxels.utils.ema import EMA
from plenoxels.models_all.lowrank_model import LowrankModel  #@remind
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_png
from plenoxels.runners.regularization import Regularizer
from plenoxels.ops.lr_scheduling import (
    get_cosine_schedule_with_warmup, get_step_schedule_with_warmup
)
# from models_all.poses import LearnPose
from models_all.porf import PoRF
from models_all.poses_noise import LearnPose
from datasets.ray_utils import get_rays
from models_all.cal_pose import make_c2w, convert3x4_4x4
from utils_poses.align_traj import align_ate_c2b_use_a2b
from utils_poses.comp_ate import compute_ATE, compute_rpe, compute_ate_sim3
from utils_poses.util_vis import *
import plenoxels.utils.camera_barf as camera_barf
class BaseTrainer(abc.ABC):
    def __init__(self,
                 train_data_loader: Iterable,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 device: Union[str, torch.device],
                 num_cams,  # train cam
                 render_every,
                 **kwargs):  # kwargs中包含load进来的data(train and test)
        self.train_data_loader = train_data_loader
        self.num_steps = num_steps
        self.train_fp16 = train_fp16
        self.save_every = save_every
        self.valid_every = valid_every
        self.save_outputs = save_outputs
        self.device = device
        self.eval_batch_size = kwargs.get('eval_batch_size', 8129)
        self.extra_args = kwargs
        self.timer = CudaTimer(enabled=False)

        self.log_dir = os.path.join(logdir, expname)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.logger_py = log.getLogger()
        curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
        self.logfile = os.path.join(self.log_dir, curTime + '.log')
        formatter = log.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fileHandler = log.FileHandler(self.logfile, mode='a')
        fileHandler.setFormatter(formatter)
        self.logger_py.addHandler(fileHandler)



        self.render_every = render_every
        if self.render_every > -1:
            os.makedirs(os.path.join(self.log_dir, f"train_res"), exist_ok=True)

        self.global_step: Optional[int] = None
        self.loss_info: Optional[Dict[str, EMA]] = None

        # breakpoint()
        self.model = self.init_model(**self.extra_args)
        init_poses = convert3x4_4x4(self.train_dataset.per_cam_poses)
        # init_poses[:,0,:] = init_poses[:,0,:] * -1
        self.model_pose = self.init_posenet(num_cams=num_cams, init_c2ws=init_poses, **self.extra_args)
        # self.model_pose = self.init_posenet(num_cams=num_cams, **self.extra_args)

        self.optimizer = self.init_optim(**self.extra_args)
        self.optimizer_pose = self.init_pose_optim(**self.extra_args)
        self.scheduler = self.init_lr_scheduler(**self.extra_args)
        self.scheduler_pose = self.init_pose_lr_scheduler(**self.extra_args)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.regularizers = self.init_regularizers(**self.extra_args)
        self.gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)

        self.num_cams=num_cams  # num_cams for train

        self.model.to(self.device)
        self.model_pose.to(self.device)

    @abc.abstractmethod
    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        self.model_pose.eval()
        return None  # noqa

    def get_pose_metrics(self):
        if "keyframes" in self.extra_args:  # dynamic
            gtposes = self.train_dataset.per_cam_poses
            gtposes = convert3x4_4x4(gtposes)  # (self.num_cams, 4, 4)
        else:  # static
            gtposes = self.train_dataset.poses
            gtposes = convert3x4_4x4(gtposes)  # (self.num_cams, 4, 4)

        self.model_pose.eval()
        learned_poses = torch.stack([self.model_pose(torch.tensor([i])) for i in range(self.num_cams)]).squeeze(1)[:,:3,:].detach()

        # gtposes[:,0,:] = gtposes[:,0,:] * -1  #@remind
        # poses_ = self.model_pose.get_params()['pose']
        # c2w = self.model_pose(data['camera_id'])
        cam_id_tensor = torch.arange(self.num_cams)
        # breakpoint()
        
        # learned_poses = make_c2w(poses_[0][cam_id_tensor], poses_[1][cam_id_tensor])[:,:3,:]  # (self.num_cams, 4, 4)
        #! c2w in 1 d
        # learned_poses = make_c2w(poses_[1][cam_id_tensor], poses_[2][cam_id_tensor])[:,:3,:]  # (self.num_cams, 4, 4)

        # breakpoint()
        pose_aligned, _ = self.prealign_cameras(
            learned_poses.to(device=self.device), gtposes.to(device=self.device)
        )
        error = self.evaluate_camera_alignment(pose_aligned.to(device=self.device), gtposes[:,:3,:].to(device=self.device))
        # print("--------------------------")
        # print("rot:   {:8.3f}".format(np.rad2deg(error.R.mean().cpu())))
        # print("trans: {:10.5f}".format(error.t.mean()))
        # print("--------------------------")
        if self.global_step % 200 == 0 or self.global_step <= 10:
            fig = plt.figure(figsize=(10, 10))
            pose_aligned, pose_ref = pose_aligned.detach().cpu(), gtposes.detach().cpu()
            output_path = self.log_dir
            cam_path = "{}/poses".format(output_path)
            os.makedirs(cam_path, exist_ok=True)
            # plot_save_poses_blender(fig,pose_aligned,pose_ref= pose_ref,path=cam_path,ep=self.global_step)
            plot_save_poses(fig, pose_aligned, pose_ref= pose_ref, path=cam_path, ep=self.global_step)
            plt.close()
        c2ws_est_aligned = align_ate_c2b_use_a2b(learned_poses, gtposes)
        # ate = compute_ATE(gtposes.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        stats_tran, stats_rot, _ = compute_ate_sim3(learned_poses, gtposes)
        rpe_trans, rpe_rot = compute_rpe(gtposes.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        rpe_trans = rpe_trans * 100
        rpe_rot = rpe_rot * 180 / np.pi
    
        return stats_tran, stats_rot


    def train_step(self, data, **kwargs) -> bool:  #@todo 在这部分采光线(rays_o, rays_d, etc.)
        self.model.train()
        self.model_pose.train()
        data = self._move_data_to_device(data) 
        if "timestamps" not in data:
            data["timestamps"] = None
        self.timer.check("move-to-device")

        with torch.cuda.amp.autocast(enabled=self.train_fp16):  # 混合精度训练
            c2w = self.model_pose(data['camera_id'])  # 过model_pose得到的c2w都是乘了-1的，对应kplanes origin中的gt pose, 用于采光线
            # c2w = self.train_dataset.poses[data['camera_id']]
            # except RuntimeError:
            #     breakpoint()
            camera_dirs = data['camera_dirs'].to(self.device)
            rays_o, rays_d = get_rays(
                camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=data['intrinsics'], normalize_rd=True
            )
            fwd_out = self.model(rays_o=rays_o, rays_d=rays_d, near_far=data['near_fars'], bg_color=data['bg_color'], timestamps=data['timestamps'], global_step=self.global_step)
            # fwd_out = self.model(
            #     intrinsics=data['intrinsics'], camera_dirs=data['camera_dirs'], camera_id=data['camera_id'], 
            #     bg_color=data['bg_color'], near_far=data['near_fars'], timestamps=data['timestamps'])
            self.timer.check("model-forward")
            # Reconstruction loss
            recon_loss = self.criterion(fwd_out['rgb'], data['imgs'])
            # Regularization
            loss = recon_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, model_out=fwd_out)
                loss = loss + reg_loss
            self.timer.check("regularizaion-forward")        
        # Update weights
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_pose.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        try:
            self.gscaler.scale(loss).backward()
        except RuntimeError:
            breakpoint()

        self.timer.check("backward")
        self.gscaler.step(self.optimizer)
        self.gscaler.step(self.optimizer_pose)
        scale = self.gscaler.get_scale()
        self.gscaler.update()
        self.timer.check("scaler-step")

        # Report on losses
        # if self.global_step % self.calc_metrics_every == 0:
        #     with torch.no_grad():
        #         recon_loss_val = recon_loss.item()
        #         self.loss_info[f"mse"].update(recon_loss_val)
        #         self.loss_info[f"psnr"].update(-10 * math.log10(recon_loss_val))
        #         for r in self.regularizers:
        #             r.report(self.loss_info)

        # gtposes = data['poses']
        # Report on losses
        if self.global_step % self.calc_metrics_every == 0:
            with torch.no_grad():

                stats_tran, stats_rot = self.get_pose_metrics()

                recon_loss_val = recon_loss.item()
                #! barf eval
                # self.loss_info[f"t"].update(error.t.mean())
                # self.loss_info[f"r"].update(error.R.mean().cpu())
                #! nerfmm eval
                self.loss_info[f"t"].update(stats_tran['mean'])
                self.loss_info[f"r"].update(stats_rot['mean'])
                # self.loss_info([f"RPEt"].update(rpe_trans))
                self.loss_info[f"mse"].update(recon_loss_val)
                self.loss_info[f"psnr"].update(-10 * math.log10(recon_loss_val))
                # for r in self.regularizers:
                #     r.report(self.loss_info)

        return scale <= self.gscaler.get_scale()
    
    @torch.no_grad()
    def evaluate_camera_alignment(self, pose_aligned, pose_GT):
        # measure errors in rotation and translation
        R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)
        R_GT, t_GT = pose_GT.split([3, 1], dim=-1)
        R_error = camera_barf.rotation_distance(R_aligned, R_GT)
        t_error = (t_aligned - t_GT)[..., 0].norm(dim=-1)
        error = edict(R=R_error, t=t_error)
        return error
    
    @torch.no_grad()
    def prealign_cameras(self, pose, pose_GT):
        # compute 3D similarity transform via Procrustes analysis
        center = torch.zeros(1, 1, 3, device=self.device)
        center_pred = camera_barf.cam2world(center, pose)[:, 0]  # [N,3]
        center_GT = camera_barf.cam2world(center, pose_GT)[:, 0]  # [N,3]
        try:
            sim3 = camera_barf.procrustes_analysis(center_GT, center_pred)
        except:
            print("warning: SVD did not converge...")
            sim3 = edict(
                t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=self.device)
            )
        # align the camera poses
        center_aligned = (
            center_pred - sim3.t1
        ) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
        R_aligned = pose[..., :3] @ sim3.R.t()
        t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
        pose_aligned = camera_barf.pose(R=R_aligned, t=t_aligned)
        return pose_aligned, sim3
    
    def post_step(self, progress_bar):
        self.model.step_after_iter(self.global_step)
        if self.global_step % self.calc_metrics_every == 0:
            progress_bar.set_postfix_str(
                losses_to_postfix(self.loss_info, lr=self.lr, lr_pose=self.pose_lr), refresh=False)
            now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
            message = now + " | step"+str(self.global_step)+": | "
            for loss_name, loss_val in self.loss_info.items():
                message += loss_name +": "+ str(loss_val.value)+" | "
                self.writer.add_scalar(f"train/loss/{loss_name}", loss_val.value, self.global_step)
                if self.timer.enabled:
                    tsum = 0.
                    tstr = "Timings: "
                    for tname, tval in self.timer.timings.items():
                        tstr += f"{tname}={tval:.1f}ms  "
                        tsum += tval
                    tstr += f"tot={tsum:.1f}ms"
                    # log.info(tstr)
                    # self.logger_py.info(tstr)
            # self.logger_py.warn(message) 
            with open(self.logfile, "a") as file:
                file.write(message+"\n")   #FIXME:
        progress_bar.update(1)
        if self.render_every > -1 and self.global_step % self.render_every == 0:
            print()
            self.render_train_res()
        if self.save_every > -1 and self.global_step % self.save_every == 0:
            print()
            self.save_model()
            self.save_posenet()
        if self.valid_every > -1 and self.global_step % self.valid_every == 0:
            print()
            self.validate()


    def pre_epoch(self):
        self.loss_info = self.init_epoch_info()

    def train(self):
        """Override this if some very specific training procedure is needed."""
        if self.global_step is None:
            self.global_step = 0
        # log.info(f"Starting training from step {self.global_step + 1}")
        self.logger_py.info(f"Starting training from step {self.global_step + 1}")
        pb = tqdm(initial=self.global_step, total=self.num_steps)
        try:
            self.pre_epoch()
            batch_iter = iter(self.train_data_loader)

            while self.global_step < self.num_steps:
                self.timer.reset()
                self.model.step_before_iter(self.global_step)
                self.global_step += 1
                self.timer.check("step-before-iter")
                try:
                    data = next(batch_iter)
                    self.timer.check("dloader-next")
                except StopIteration:  # switch isgtoist 也会 raise stopiteration
                    self.pre_epoch()  # dataset reset_iter and set loss ema
                    batch_iter = iter(self.train_data_loader)
                    data = next(batch_iter)
                    # log.info("Reset data-iterator")
                    self.logger_py.info("Reset data-iterator")
                try:
                    step_successful = self.train_step(data)
                except StopIteration:
                    self.pre_epoch()
                    batch_iter = iter(self.train_data_loader)
                    # # log.info("Reset data-iterator")
                    self.logger_py.info("Reset data-iterator")
                    step_successful = True

                if step_successful and self.scheduler is not None:
                    self.scheduler.step()
                if step_successful and self.scheduler_pose is not None:
                    self.scheduler_pose.step()
                for r in self.regularizers:
                    r.step(self.global_step)
                self.post_step(progress_bar=pb)
                self.timer.check("after-step")
        finally:
            pb.close()
            self.writer.close()

    def _move_data_to_device(self, data):
        # data["rays_o"] = data["rays_o"].to(self.device)
        # data["rays_d"] = data["rays_d"].to(self.device)
        data["imgs"] = data["imgs"].to(self.device)
        data["near_fars"] = data["near_fars"].to(self.device)
        data["camera_id"] = data["camera_id"].to(self.device)
        data["camera_dirs"] = data["camera_dirs"].to(self.device)
        data["intrinsics"] = data["intrinsics"].to(self.device)
        if "timestamps" in data:
            data["timestamps"] = data["timestamps"].to(self.device)
        bg_color = data["bg_color"]
        if isinstance(bg_color, torch.Tensor):
            bg_color = bg_color.to(self.device)
        data["bg_color"] = bg_color
        return data

    def _normalize_err(self, preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        err = torch.abs(preds - gt)
        err = err.mean(-1, keepdim=True)  # mean over channels
        # normalize between 0, 1 where 1 corresponds to the 90th percentile
        # err = err.clamp_max(torch.quantile(err, 0.9))
        err = self._normalize_01(err)
        return err.repeat(1, 1, 3)

    @staticmethod
    def _normalize_01(t: torch.Tensor) -> torch.Tensor:
        return (t - t.min()) / t.max()

    def _normalize_depth(self, depth: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        return (
            self._normalize_01(depth)
        ).cpu().reshape(img_h, img_w)[..., None]

    def calc_metrics(self, preds: torch.Tensor, gt: torch.Tensor):
        if gt.shape[-1] == 4:
            gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])

        err = (gt - preds) ** 2
        return {
            "mse": torch.mean(err),
            "psnr": metrics.psnr(preds, gt),
            "ssim": metrics.ssim(preds, gt),
            "ms-ssim": metrics.msssim(preds, gt),
            #"alex_lpips": metrics.rgb_lpips(preds, gt, net_name='alex', device=err.device),
            #"vgg_lpips": metrics.rgb_lpips(preds, gt, net_name='vgg', device=err.device)
        }

    def calc_metrics_grad(self, preds: torch.Tensor, gt: torch.Tensor):
        if gt.shape[-1] == 4:
            gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])

        err = (gt - preds) ** 2
        return {
            "mse": torch.mean(err),
            "psnr": metrics.psnr(preds, gt),
            "ssim": metrics.ssim(preds.detach(), gt),
            "ms-ssim": metrics.msssim(preds, gt),
            #"alex_lpips": metrics.rgb_lpips(preds, gt, net_name='alex', device=err.device),
            #"vgg_lpips": metrics.rgb_lpips(preds, gt, net_name='vgg', device=err.device)
        }

    def evaluate_metrics(self,
                         gt: Optional[torch.Tensor],
                         preds: MutableMapping[str, torch.Tensor],
                         dset,
                         img_idx: int,
                         name: Optional[str] = None,
                         save_outputs: bool = True) -> Tuple[dict, np.ndarray, Optional[np.ndarray]]:
        if isinstance(dset.img_h, int):
            img_h, img_w = dset.img_h, dset.img_w
        else:
            img_h, img_w = dset.img_h[img_idx], dset.img_w[img_idx]
        preds_rgb = (
            preds["rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        if not torch.isfinite(preds_rgb).all():
            log.warning(f"Predictions have {torch.isnan(preds_rgb).sum()} NaNs, "
                        f"{torch.isinf(preds_rgb).sum()} infs.")
            preds_rgb = torch.nan_to_num(preds_rgb, nan=0.0)
        out_img = preds_rgb
        summary = dict()

        out_depth = None
        if "depth" in preds:
            out_depth = preds["depth"].cpu().reshape(img_h, img_w)[..., None]
        preds.pop("depth")

        # This is used for proposal-depth keys
        for k in preds.keys():
            if "depth" in k:
                prop_depth = preds[k].cpu().reshape(img_h, img_w)[..., None]
                out_depth = torch.cat((out_depth, prop_depth)) if out_depth is not None else prop_depth

        if gt is not None:
            gt = gt.reshape(img_h, img_w, -1).cpu()
            if gt.shape[-1] == 4:
                gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
            summary.update(self.calc_metrics(preds_rgb, gt))
            out_img = torch.cat((out_img, gt), dim=0)
            out_img = torch.cat((out_img, self._normalize_err(preds_rgb, gt)), dim=0)

        out_img_np: np.ndarray = (out_img * 255.0).byte().numpy()
        out_depth_np: Optional[np.ndarray] = None
        if out_depth is not None:
            out_depth = self._normalize_01(out_depth)
            out_depth_np = (out_depth * 255.0).repeat(1, 1, 3).byte().numpy()

        if save_outputs:
            out_name = f"step{self.global_step}-{img_idx}"
            if name is not None and name != "":
                out_name += "-" + name
            write_png(os.path.join(self.log_dir, out_name + ".png"), out_img_np)
            if out_depth is not None:
                depth_name = out_name + "-depth"
                write_png(os.path.join(self.log_dir, depth_name + ".png"), out_depth_np)

        return summary, out_img_np, out_depth_np
    
    def evaluate_metrics_learnpose(self,
                         gt: Optional[torch.Tensor],
                         preds: MutableMapping[str, torch.Tensor],
                         dset,
                         img_idx: int,) -> Tuple[dict, np.ndarray, Optional[np.ndarray]]:
        if isinstance(dset.img_h, int):
            img_h, img_w = dset.img_h, dset.img_w
        else:
            img_h, img_w = dset.img_h[img_idx], dset.img_w[img_idx]
        preds_rgb = (
            preds["rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        if not torch.isfinite(preds_rgb).all():
            log.warning(f"Predictions have {torch.isnan(preds_rgb).sum()} NaNs, "
                        f"{torch.isinf(preds_rgb).sum()} infs.")
            preds_rgb = torch.nan_to_num(preds_rgb, nan=0.0)

        summary = dict()

        if gt is not None:
            gt = gt.reshape(img_h, img_w, -1).cpu()
            if gt.shape[-1] == 4:
                gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
            summary.update(self.calc_metrics(preds_rgb, gt))

        return summary

    @abc.abstractmethod
    def validate(self):
        pass

    # @abc.abstractmethod
    def render_train_res(self):
        pass

    def report_test_metrics(self, scene_metrics: Dict[str, Sequence[float]], extra_name: Optional[str]):
        log_text = f"step {self.global_step}/{self.num_steps}"
        if extra_name is not None:
            log_text += f" | {extra_name}"
        scene_metrics_agg: Dict[str, float] = {}
        for k in scene_metrics:
            ak = f"{k}_{extra_name}"
            scene_metrics_agg[ak] = np.mean(np.asarray(scene_metrics[k])).item()
            log_text += f" | {k}: {scene_metrics_agg[ak]:.4f}"
            self.writer.add_scalar(f"test/{ak}", scene_metrics_agg[ak], self.global_step)

        # log.info(log_text)
        self.logger_py.info(log_text)
        return scene_metrics_agg

    def get_save_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "global_step": self.global_step
        }
    
    def get_save_dict_pose(self):
        return {
            "model_pose": self.model_pose.state_dict(),
            "optimizer_pose": self.optimizer_pose.state_dict(),
            "sheduler_pose": self.scheduler_pose.state_dict() if self.scheduler_pose is not None else None,
            "global_step": self.global_step
        }

    def save_posenet(self):
        if self.global_step >= self.num_steps - 1:
            model_fname = os.path.join(self.log_dir, f'posenet.pth')
        else:
            model_fname = os.path.join(self.log_dir, f'posenet_step{self.global_step}.pth')
        # log.info(f'Saving model checkpoint to: {model_fname}')
        self.logger_py.info(f'Saving model checkpoint to: {model_fname}')
        torch.save(self.get_save_dict_pose(), model_fname)

    def save_model(self):
        if self.global_step >= self.num_steps - 1:
            model_fname = os.path.join(self.log_dir, f'model.pth')
        else:
            model_fname = os.path.join(self.log_dir, f'model_step{self.global_step}.pth')
        # log.info(f'Saving model checkpoint to: {model_fname}')
        self.logger_py.info(f'Saving model checkpoint to: {model_fname}')
        torch.save(self.get_save_dict(), model_fname)

    def load_model(self, checkpoint_data, training_needed: bool = True):
        self.model.load_state_dict(checkpoint_data["model"], strict=False)
        # log.info("=> Loaded model state from checkpoint")
        self.logger_py.info("=> Loaded model state from checkpoint")
        if training_needed:
            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
            # log.info("=> Loaded optimizer state from checkpoint")
            self.logger_py.info("=> Loaded optimizer state from checkpoint")
        if training_needed and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint_data['lr_scheduler'])
            # log.info("=> Loaded scheduler state from checkpoint")
            self.logger_py.info("=> Loaded scheduler state from checkpoint")
        self.global_step = checkpoint_data["global_step"]
        # log.info(f"=> Loaded step {self.global_step} from checkpoints")
        self.logger_py.info(f"=> Loaded step {self.global_step} from checkpoints")
    def load_posenet(self, checkpoint_data, training_needed: bool = True):
        self.model_pose.load_state_dict(checkpoint_data["model_pose"], strict=False)
        # log.info("=> Loaded model_pose from checkpoint")

        if training_needed:
            self.optimizer_pose.load_state_dict(checkpoint_data["optimizer_pose"])
            # log.info("=> Loaded pose optimizer state from checkpoint")

        if training_needed and self.scheduler is not None:
            self.scheduler_pose.load_state_dict(checkpoint_data['scheduler_pose'])
            # log.info("=> Loaded pose scheduler state from checkpoint")

        self.global_step = checkpoint_data["global_step"]
        # log.info(f"=> Loaded step {self.global_step} from checkpoints")

    @abc.abstractmethod
    def init_epoch_info(self) -> Dict[str, EMA]:
        pass

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_lr_scheduler(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 0
        lr_sched = None
        max_steps = self.num_steps
        scheduler_type = kwargs['scheduler_type']
        # log.info(f"Initializing LR Scheduler of type {scheduler_type} with "
                 # f"{max_steps} maximum steps.")
        self.logger_py.info(f"Initializing LR Scheduler of type {scheduler_type} with "
                 f"{max_steps} maximum steps.")
        if scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=eta_min)
        elif scheduler_type == "warmup_cosine":
            lr_sched = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=512, num_training_steps=max_steps)
        elif scheduler_type == "step":
            lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                # milestones=[
                #     max_steps // 2,
                #     max_steps * 3 // 4,
                #     max_steps * 5 // 6,
                #     max_steps * 9 // 10,
                # ],
                # gamma=0.33)
                milestones=list(range(2000, max_steps, 100)),
                gamma=0.9)
        elif scheduler_type == "warmup_step":
            lr_sched = get_step_schedule_with_warmup(
                self.optimizer, milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
                num_warmup_steps=512)
        return lr_sched
    
    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_pose_lr_scheduler(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 0
        lr_sched = None
        max_steps = self.num_steps
        scheduler_type = kwargs['scheduler_type_pose']
        # log.info(f"Initializing LR Scheduler of type {scheduler_type} with "
                 # f"{max_steps} maximum steps.")
        self.logger_py.info(f"Initializing LR Scheduler of type {scheduler_type} with "
                 f"{max_steps} maximum steps.")
        if scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_pose,
                T_max=max_steps,
                eta_min=eta_min)
        elif scheduler_type == "warmup_cosine":
            lr_sched = get_cosine_schedule_with_warmup(
                self.optimizer_pose, num_warmup_steps=512, num_training_steps=max_steps)
        elif scheduler_type == "step":
            lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_pose,
                # milestones=[
                #     max_steps // 2,
                #     max_steps * 3 // 4,
                #     max_steps * 5 // 6,
                #     max_steps * 9 // 10,
                # ],
                milestones=list(range(500, max_steps, 100)),  #@todo
                gamma=0.99)
        elif scheduler_type == "warmup_step":
            # lr_sched = get_step_schedule_with_warmup(
            #     self.optimizer_pose, milestones=[
            #         max_steps // 2,
            #         max_steps * 3 // 4,
            #         max_steps * 5 // 6,
            #         max_steps * 9 // 10,
            #     ],
            #     gamma=0.33,
            #     num_warmup_steps=512)
            lr_sched = get_step_schedule_with_warmup(
                self.optimizer_pose, milestones=
                    list(range(1000, max_steps, 100))
                ,
                gamma=0.99,
                num_warmup_steps=512)
        return lr_sched 
    
    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_pose_lr_scheduler_eval(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 0
        lr_sched = None
        #max_steps = self.num_steps
        max_steps = kwargs['eval_steps']
        scheduler_type = kwargs['eval_scheduler_type']
        # log.info(f"Initializing LR Scheduler of type {scheduler_type} with "
                #  f"{max_steps} maximum steps.")
        self.logger_py.info(f"Initializing LR Scheduler of type {scheduler_type} with "
                 f"{max_steps} maximum steps.")
        if scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.eval_optimizer,
                T_max=max_steps,
                eta_min=eta_min)
        elif scheduler_type == "warmup_cosine":
            lr_sched = get_cosine_schedule_with_warmup(
                self.eval_optimizer, num_warmup_steps=512, num_training_steps=max_steps)
        elif scheduler_type == "step":
            lr_sched = torch.optim.lr_scheduler.MultiStepLR(self.eval_optimizer,
                                                            milestones=list(range(0, int(max_steps), int(max_steps/5))),
                                                            gamma=0.5)  
            # lr_sched = torch.optim.lr_scheduler.MultiStepLR(
            #     self.eval_optimizer,
            #     milestones=[
            #         max_steps // 2,
            #         max_steps * 3 // 4,
            #         max_steps * 5 // 6,
            #         max_steps * 9 // 10,
            #     ],
            #     gamma=0.33)
        elif scheduler_type == "warmup_step":
            lr_sched = get_step_schedule_with_warmup(
                self.eval_optimizer, milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
                num_warmup_steps=512)
        return lr_sched

    def init_optim(self, **kwargs) -> torch.optim.Optimizer:
        optim_type = kwargs['optim_type']
        if optim_type == 'adam':
            # optim = torch.optim.Adam(params=self.model.get_params(kwargs['lr'], kwargs['pose_lr']), eps=1e-15)
            # optim = torch.optim.Adam(params=self.model.get_params_ms(kwargs['lr'], kwargs['field_lr']), eps=1e-15)
            optim = torch.optim.Adam(params=self.model.get_params(kwargs['lr']), eps=1e-15)
        else:
            raise NotImplementedError()
        return optim
    
    def init_pose_optim(self, **kwargs) -> torch.optim.Optimizer:
        """For Train Posenet"""
        optim_type = kwargs['optim_type']
        if optim_type == 'adam':
            optim = torch.optim.Adam(params=self.model_pose.parameters(), lr=kwargs['pose_lr'])
        else:
            raise NotImplementedError()
        return optim
    
    def init_pose_optim_eval(self, **kwargs) -> torch.optim.Optimizer:
        """Only optimize eval pose params for evaluation."""
        optim_type = kwargs['optim_type']
        if optim_type == 'adam':
            optim = torch.optim.Adam(params=self.eval_posenet.parameters(), lr=kwargs['eval_pose_lr'], eps=1e-15)
        else:
            raise NotImplementedError()
        return optim

    @abc.abstractmethod
    def init_model(self, **kwargs) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def init_posenet(self, **kwargs) -> torch.nn.Module:
        pass

    def get_regularizers(self, **kwargs) -> Sequence[Regularizer]:
        return ()

    def init_regularizers(self, **kwargs):
        # Keep only the regularizers with a positive weight
        regularizers = [r for r in self.get_regularizers(**kwargs) if r.weight > 0]
        return regularizers

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    @property
    def pose_lr(self):
        return self.optimizer_pose.param_groups[0]['lr']
    
    @property
    def calc_metrics_every(self):
        return 1


def losses_to_postfix(loss_dict: Dict[str, EMA], lr: Optional[float], lr_pose: Optional[float]) -> str:
    pfix = [f"{lname}={lval}" for lname, lval in loss_dict.items()]
    if lr is not None:
        pfix.append(f"lr={lr:.2e}")
        pfix.append(f"pose_lr={lr_pose:.2e}")
    return "  ".join(pfix)



def init_dloader_random(_):
    # seed = torch.initial_seed() % 2**32  # worker-specific seed initialized by pytorch
    seed = 42
    np.random.seed(seed)
    random.seed(seed)


def initialize_model(
        runner: Union['StaticTrainer', 'PhototourismTrainer', 'VideoTrainer'],
        **kwargs) -> LowrankModel:
    """Initialize a `LowrankModel` according to the **kwargs parameters.

    Args:
        runner: The runner object which will hold the model.
                Needed here to fetch dataset parameters.
        **kwargs: Extra parameters to pass to the model

    Returns:
        Initialized LowrankModel.
    """
    from .phototourism_trainer import PhototourismTrainer
    extra_args = copy(kwargs)
    extra_args.pop('global_scale', None)
    extra_args.pop('global_translation', None)

    dset = runner.test_dataset
    try:
        global_translation = dset.global_translation
    except AttributeError:
        global_translation = None
    try:
        global_scale = dset.global_scale
    except AttributeError:
        global_scale = None

    num_images = None
    num_cams = None  #@todo
    if runner.train_dataset is not None:
        try:
            num_images = runner.train_dataset.num_images
            num_cams = num_images
        except AttributeError:
            num_images = None
    else:
        try:
            num_images = runner.test_dataset.num_images
            num_cams = runner.test_dataset.num_cams_train  # validate only
        except AttributeError:
            num_images = None
    model = LowrankModel(
        grid_config=extra_args.pop("grid_config"),
        aabb=dset.scene_bbox,
        is_ndc=dset.is_ndc,
        is_contracted=dset.is_contracted,
        global_scale=global_scale,
        global_translation=global_translation,
        use_appearance_embedding=isinstance(runner, PhototourismTrainer),
        num_images=num_images,
        num_cams=num_cams,
        **extra_args)
    log.info(f"Initialized {model.__class__} model with "
             f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters, "
             f"using ndc {model.is_ndc} and contraction {model.is_contracted}. "
             f"Linear decoder: {model.linear_decoder}.")
    return model

def initialize_posenet(
        runner: Union['StaticTrainer', 'PhototourismTrainer', 'VideoTrainer'], 
        # for_train: bool = False, 
        num_cams: Optional[int] = None,
        for_eval: bool = False,
        learn_R : bool = False, learn_t : bool = False,
        init_c2ws : Optional[torch.Tensor] = None,
        **kwargs) -> LearnPose:
    
    pose_params_net = LearnPose(num_cams, learn_R=learn_R, learn_t=learn_t, init_c2w=init_c2ws)
    # pose_params_net = PoRF(num_cams, learn_R=learn_R, learn_t=learn_t, init_c2w=init_c2ws)
    cam_type = "eval" if for_eval else "train"
    print(f"Learn {num_cams} cam poses for {cam_type}!!!")
    return pose_params_net
