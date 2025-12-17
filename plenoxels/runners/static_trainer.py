import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Sequence, Any

import pandas as pd
import torch
import torch.utils.data

from plenoxels.datasets import SyntheticNerfDataset, LLFFDataset
from plenoxels.models_all.lowrank_model import LowrankModel
from plenoxels.utils.ema import EMA
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.utils.parse_args import parse_optint
from .base_trainer import BaseTrainer, init_dloader_random, initialize_model, initialize_posenet, initialize_posenet_synthetic
from .regularization import (
    PlaneTV, HistogramLoss, L1ProposalNetwork, DepthTV, DistortionLoss,
    PlaneTV_kp,
)
from models_all.poses import LearnPose
from models_all.zeroporf import ZeroPoRF
from datasets.ray_utils import get_rays
from models_all.cal_pose import make_c2w
import logging as log

from typing import Iterable, Optional, Union, Dict, Tuple, Sequence, MutableMapping
from plenoxels.ops.image.io import write_png
import numpy as np
import time
from utils_poses.comp_ate import compute_ate
from utils_poses.align_traj import align_scale_c2b_use_a2b
import plenoxels.utils.camera_barf as camera_barf
class StaticTrainer(BaseTrainer):
    def __init__(self,
                ts_loader: torch.utils.data.DataLoader,
                tr_loader: torch.utils.data.DataLoader,
                ts_dset: torch.utils.data.TensorDataset,
                tr_dset: torch.utils.data.TensorDataset,
                num_steps: int,
                logdir: str,
                expname: str,
                train_fp16: bool,
                save_every: int,
                valid_every: int,
                save_outputs: bool,
                device: Union[str, torch.device],
                tr_render_dset: torch.utils.data.TensorDataset = None,
                **kwargs
                ):
        self.test_dataset = ts_dset
        self.train_dataset = tr_dset
        self.train_render_dataset = tr_render_dset
        self.is_ndc = self.test_dataset.is_ndc
        self.is_contracted = self.test_dataset.is_contracted
        self.n_frame_per_cam = 1

        super().__init__(
            train_data_loader=tr_loader,
            test_data_loader=ts_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=save_outputs,
            device=device,
            num_cams=self.test_dataset.num_cams_train,
            dino_feat=self.train_dataset.first_img_feature,
            **kwargs
        )

    def render_train_metrics(self,
                        gt: Optional[torch.Tensor],
                        preds: MutableMapping[str, torch.Tensor],
                        dset,
                        img_idx: int,
                        name: Optional[str] = None,
                        save_outputs: bool = True,
                        save_dir: Optional[str] = None,) -> Tuple[dict, np.ndarray, Optional[np.ndarray]]:
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
        # summary = dict()

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
            # summary.update(self.calc_metrics(preds_rgb, gt))
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
            write_png(os.path.join(save_dir, out_name + ".png"), out_img_np)
            if out_depth is not None:
                depth_name = out_name + "-depth"
                write_png(os.path.join(save_dir, depth_name + ".png"), out_depth_np)

        # return summary
        # return summary, out_img_np, out_depth_np
    
    def render_train_res(self):
        cam_id = 0  # in train list
        # save_path = os.path.join(self.log_dir, f"train_res/step_{self.global_step}-{cam_id}.png")
        save_path = os.path.join(self.log_dir, f"train_res")
        self.model.eval()
        self.model_pose.eval()
        dataset = self.train_render_dataset
        # pb = tqdm(total=len(dataset), desc=f"Test scene {dataset.name}")
        data = dataset[cam_id]
        # for img_idx, data in enumerate(dataset):
        #     ts_render = self.eval_step(data)
        #     out_metrics, _, _ = self.evaluate_metrics(
        #         data["imgs"], ts_render, dset=dataset, img_idx=img_idx,
        #         name=None, save_outputs=self.save_outputs)

        channels = {"rgb", "depth", "proposal_depth"}
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            near_far = data["near_fars"].to(self.device)
            camera_id = data['camera_id'].to(self.device)
            # timestamps = data['timestamps'].to(self.device)
            intrinsics = data['intrinsics'].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)

            # try: 
            # breakpoint()
            c2w = self.model_pose(camera_id)
            # except RuntimeError:
            #     breakpoint()
            camera_dirs = data['camera_dirs'].to(self.device)
            rays_o, rays_d = get_rays(
                camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=intrinsics, normalize_rd=True
            )  # h*w, 3
            preds = defaultdict(list)
            outputs = self.model(rays_o=rays_o, rays_d=rays_d, near_far=near_far,
                                 bg_color=bg_color)
            for k, v in outputs.items():
                if k in channels or "depth" in k:
                    preds[k].append(v.cpu())
        tr_render = {k: torch.cat(v, 0) for k, v in preds.items()}

        # out_metrics = self.render_train_metrics(
        #         data["imgs"], tr_render, dset=dataset, img_idx=0,
        #         name=None, save_outputs=self.save_outputs, save_dir=save_path)
        self.render_train_metrics(
                data["imgs"], tr_render, dset=dataset, img_idx=cam_id,
                name=None, save_outputs=self.save_outputs, save_dir=save_path)


    def eval_step_origin(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        batch_size = self.eval_batch_size
        channels = {"rgb", "depth", "proposal_depth"}
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            # near_far and bg_color are constant over mini-batches
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):  # rays_o.shape[0]:h*w
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                outputs = self.model(rays_o_b, rays_d_b, near_far=near_far,
                                     bg_color=bg_color)
                for k, v in outputs.items():
                    if k in channels or "depth" in k:
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)  # self.model.eval() and self.model_pose.eval()
        batch_size = self.eval_batch_size   
        channels = {"rgb", "depth", "proposal_depth"}
        with torch.cuda.amp.autocast(enabled=self.train_fp16) and torch.no_grad():
            # near_far, camera_id, intrinsics and bg_color are constant over mini-batches
            near_far = data["near_fars"].to(self.device)
            camera_id = data['camera_id'].to(self.device)
            # timestamps = data['timestamps'].to(self.device)
            intrinsics = data['intrinsics'].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)

            # try: 
            c2w = self.eval_posenet(camera_id)
            # breakpoint()
            # c2w = data["c2w"].to(self.device)   #FIXME: using gt poses
            # c2w = c2ws.to(self.device)
            # except RuntimeError:
            #     breakpoint()
            # breakpoint()
            camera_dirs = data['camera_dirs'].to(self.device)
            rays_o, rays_d = get_rays(camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=intrinsics, normalize_rd=True)  # h*w, 3
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):  # rays_o.shape[0]:h*w
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                outputs = self.model(rays_o=rays_o_b, rays_d=rays_d_b, near_far=near_far,
                                    bg_color=bg_color)
                for k, v in outputs.items():
                    if k in channels or "depth" in k:
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def eval_step_trainpose(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:  #@todo
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        # outputs = self.model(data['intrinsics'].to(self.device), data['camera_dirs'].to(self.device), 
        #                          data['image_id'].to(self.device), bg_color=bg_color,
        #                          near_far=data['near_fars'].to(self.device), timestamps=data['timestamps'].to(self.device),
        #                          process_tsdata=True, bs=batch_size)
        super().eval_step(data, **kwargs)  # self.model.eval() and self.model_pose.eval()
        batch_size = self.eval_batch_size
        # channels = {"rgb", "depth", "proposal_depth"}
        # channels = {"rgb"}
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            # near_far, camera_id, intrinsics and bg_color are constant over mini-batches
            # near_far = data["near_fars"].to(self.device)
            # camera_id = data['camera_id'].to(self.device)
            # intrinsics = data['intrinsics'].to(self.device)
            # bg_color = data["bg_color"]
            # if isinstance(bg_color, torch.Tensor):
            #     bg_color = bg_color.to(self.device)
            # breakpoint()
            data = self._move_data_to_device(data)

            # try: 
            c2w = self.eval_posenet(data["camera_id"])
            # except RuntimeError:
            #     breakpoint()
            camera_dirs = data['camera_dirs']#.to(self.device)
            rays_o, rays_d = get_rays(
                camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=data["intrinsics"], normalize_rd=True
            )  # h*w, 3
            
            eval_pixels = torch.randperm(rays_o.shape[0])[:batch_size]
            rays_o = rays_o[eval_pixels]
            rays_d = rays_d[eval_pixels]
            # breakpoint()
            outputs = self.model(rays_o=rays_o, rays_d=rays_d, near_far=data['near_fars'],
                                    bg_color=data['bg_color'])
            # breakpoint()
        
            gt = data['imgs'][eval_pixels]#.to(self.device)
            recon_loss = self.criterion(outputs['rgb'], gt)
        # with torch.no_grad():
        #     recon_loss_val = recon_loss.item()
        #     self.loss_info_eval["mse"].update(recon_loss_val)
        #     self.loss_info_eval["psnr"].update(-10 * math.log10(recon_loss_val))
        return recon_loss
        #     for k, v in outputs.items():
        #         if k in channels:# or "depth" in k:
        #             preds[k].append(v.cpu())
        
        # preds = {k: torch.cat(v, 0) for k, v in preds.items()}
            
        # preds_rgb = (
        #     preds["rgb"]
        #     # .reshape(img_h, img_w, 3)
        #     .cpu()
        #     .clamp(0, 1)
        # )
        # if not torch.isfinite(preds_rgb).all():
        #     log.warning(f"Predictions have {torch.isnan(preds_rgb).sum()} NaNs, "
        #                 f"{torch.isinf(preds_rgb).sum()} infs.")
        #     preds_rgb = torch.nan_to_num(preds_rgb, nan=0.0)

        # gt = data['imgs'][eval_pixels]
        # loss_dict = dict()

        # if gt is not None:
        #     gt = gt.cpu()
        #     if gt.shape[-1] == 4:
        #         gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
        #     loss_dict.update(self.calc_metrics_grad(preds_rgb, gt))

        # return loss_dict
        # return {k: torch.cat(v, 0) for k, v in preds.items()}

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        return super().train_step(data, **kwargs)

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        self.train_dataset.reset_iter()

    @torch.no_grad()
    def validate_origin(self):
        dataset = self.test_dataset
        per_scene_metrics = defaultdict(list)
        pb = tqdm(total=len(dataset), desc=f"Test scene {dataset.name}")
        for img_idx, data in enumerate(dataset):
            ts_render = self.eval_step(data)
            out_metrics, _, _ = self.evaluate_metrics(
                data["imgs"], ts_render, dset=dataset, img_idx=img_idx,
                name=None, save_outputs=self.save_outputs)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
        pb.close()
        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name="")
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def validate(self):
        
        self.model_pose.eval()
        self.model.eval()
        dataset = self.test_dataset
        per_scene_metrics = defaultdict(list)
        
        self.logger_py.info("\n")
        self.logger_py.info(f"Begin validate at step{self.global_step} ---------\n")

        #FIXME:
        if hasattr(self.train_dataset, 'max_frames'):  # synthetic
            # init_c2ws = self.get_pose_align()
            self.eval_posenet = self.init_posenet(
                num_cams=dataset.num_images, 
                for_eval=True,
                # init_c2ws=learn_train_poses,
                init_c2ws=dataset.poses.to(self.device),
                camera_noise=0.0,#0.01,#0.0,#0.05,  #TODO:
                **self.extra_args).to(self.device)
        else:  # llff
            with torch.no_grad():
                # learn_train_poses = torch.stack([self.model_pose([i]) for i in range(self.num_cams)])[[0,7,14]].squeeze(1).detach()
                # learn_train_poses = torch.stack([self.model_pose([i]) for i in range(self.num_cams)])[::(dataset.hold_every-1)].squeeze(1).detach()
            # same as nerfmm
            # learn_train_poses = torch.stack([self.model_pose([i]) for i in range(self.num_cams)]).squeeze(1)
            # init_c2ws = dataset.poses.to(self.device)
            # colmap_c2ws_train = self.train_dataset.poses.to(self.device)
            # init_c2ws, _ = align_scale_c2b_use_a2b(colmap_c2ws_train, learn_train_poses, init_c2ws)
                init_c2ws = self.get_pose_align()

            self.eval_posenet = self.init_posenet(
                num_cams=dataset.num_images,
                for_eval=True,
                init_c2ws=init_c2ws.to(self.device),#learn_train_poses,
                # init_c2ws=dataset.poses.to(self.device),
                camera_noise=0.0,
                **self.extra_args).to(self.device)
            
        self.eval_optimizer = self.init_pose_optim_eval(**self.extra_args)
        self.eval_scheduler = self.init_pose_lr_scheduler_eval(**self.extra_args)
        self.eval_gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)
        
        self.eval_posenet.eval()
        with torch.no_grad():
            stats_tran, stats_rot = self.get_pose_metrics_eval(0)
            self.logger_py.info(f"Step0: t={stats_tran['mean']} | r={stats_rot['mean']}\n")
        eval_steps = self.extra_args["eval_steps"]
        pb_learnpose = [tqdm(total=eval_steps, desc=f"Train eval cam{idx} poses") for idx in range(dataset.num_images)]
        
        self.eval_posenet.train()
        # self.loss_info_eval = Dict[str, EMA]
        for eval_step in range(1, eval_steps):
            batch_iter = iter(self.test_data_loader)
            with torch.cuda.amp.autocast(enabled=self.train_fp16):  # 混合精度训练
                for img_idx in range(dataset.num_images):
                    data = next(batch_iter)
                    # data["bg_color"] = self.train_dataset.bg_color
                    # img_idx = data["idx"]
                    # for img_idx, data in enumerate(dataset):
                    loss = self.eval_step_trainpose(data)
                    with torch.no_grad():
                        psnr = -10 * math.log10(loss.item())
                    #! #########
                    with torch.no_grad():
                        learned_pose = torch.tensor(self.eval_posenet([img_idx]))[:,:3,:].detach()
                        gtpose = dataset.poses[img_idx]
                        # if hasattr(self.train_dataset, "max_frames"):
                        #     P = camera_barf.Pose()
                        #     pose_flip = P(R=torch.diag(torch.tensor([1, -1, -1])))
                        #     gtpose = P.compose([pose_flip, gtpose[:3,:]])
                        #     gtpose = P.invert(gtpose)
                        stats_tran, stats_rot, _ = compute_ate(learned_pose, gtpose.unsqueeze(0))
                    pb_learnpose[img_idx].set_postfix_str(f"t={stats_tran['mean']} r={stats_rot['mean']} mse={loss} PSNR={psnr:.2f}", refresh=False)
                    pb_learnpose[img_idx].update(1) 

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

                if eval_step < 10 or eval_step % 5 == 0:
                    with torch.no_grad():
                        stats_tran, stats_rot = self.get_pose_metrics_eval(eval_step)

            step_successful = scale <= self.eval_gscaler.get_scale()
            #@remind
            if step_successful and self.eval_scheduler is not None:
                self.eval_scheduler.step()
            
        for img_idx in range(dataset.num_images):
            pb_learnpose[img_idx].close()
        self.logger_py.info(f"Train eval cam poses end!\nStart eval........")
        
        # 测试和渲染阶段, 不用保存梯度, 所以丢进去整张图
        # self.eval_posenet.eval()
        pb = tqdm(total=len(dataset), desc=f"Test scene {dataset.name}")
        now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        message = now + " | step" + str(self.global_step)+ ":"
        
        batch_iter = iter(self.test_data_loader)
        # for img_idx, data in enumerate(dataset):
        for img_idx in range(dataset.num_images):
            data = next(batch_iter)
            ts_render = self.eval_step(data)
            out_metrics, _, _ = self.evaluate_metrics(
                data["imgs"], ts_render, dset=dataset, img_idx=img_idx,
                name=None, save_outputs=self.save_outputs)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
            
            message += f" | cam {img_idx}: | mse={out_metrics['mse']:.3f} | PSNR={out_metrics['psnr']:.3f}"
        pb.close()
        with open(self.logfile, "a") as file:
            file.write(message+"\n")
        print()
        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name="")
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    def load_model(self, checkpoint_data, ckpt_path: str = None, training_needed: bool = True):
        super().load_model(checkpoint_data, ckpt_path, training_needed)

    def init_epoch_info(self):
        ema_weight = 0.9  # higher places higher weight to new observations
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankModel:
        return initialize_model(self, **kwargs)
    
    def init_posenet(self, **kwargs): #@remind
        if hasattr(self.train_dataset, 'max_frames'):
            if kwargs.get("for_eval", False):
                return initialize_posenet(self, **kwargs)
            else:
                return initialize_posenet_synthetic(self, **kwargs)
        else:
            return initialize_posenet(self, **kwargs)   

    def get_regularizers(self, **kwargs):
        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            # L1ProposalNetwork(kwargs.get('l1_proposal_net_weight', 0.0)),
            DepthTV(kwargs.get('depth_tv_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
        ]
        
    def get_regularizers_kp(self, **kwargs):
        return [
            PlaneTV_kp(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV_kp(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            # L1ProposalNetwork(kwargs.get('l1_proposal_net_weight', 0.0)),
            DepthTV(kwargs.get('depth_tv_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
        ]

    @property
    def calc_metrics_every(self):
        return 5


def decide_dset_type(dd) -> str:
    if ("chair" in dd or "drums" in dd or "ficus" in dd or "hotdog" in dd
            or "lego" in dd or "materials" in dd or "mic" in dd
            or "ship" in dd):
        return "synthetic"
    elif ("fern" in dd or "flower" in dd or "fortress" in dd
          or "horns" in dd or "leaves" in dd or "orchids" in dd
          or "room" in dd or "trex" in dd):
        return "llff"
    else:
        raise RuntimeError(f"data_dir {dd} not recognized as LLFF or Synthetic dataset.")


def init_tr_data(data_downsample: float, data_dirs: Sequence[str], **kwargs):
    batch_size = int(kwargs['batch_size'])
    assert len(data_dirs) == 1
    data_dir = data_dirs[0]

    dset_type = decide_dset_type(data_dir)  # synthetic or llff
    # dset_type = 'llff'  #@remind
    if dset_type == "synthetic":
        max_tr_frames = parse_optint(kwargs.get('max_tr_frames'))
        dset = SyntheticNerfDataset(
            data_dir, split='train', downsample=data_downsample,
            max_frames=max_tr_frames, batch_size=batch_size)
    elif dset_type == "llff":
        hold_every = parse_optint(kwargs.get('hold_every'))  # 8 in llff_hybrid.py
        dset = LLFFDataset(
            data_dir, split='train', downsample=int(data_downsample), hold_every=hold_every,
            batch_size=batch_size, contraction=kwargs['contract'], ndc=kwargs['ndc'],
            ndc_far=float(kwargs['ndc_far']), near_scaling=float(kwargs['near_scaling']))
    else:
        raise ValueError(f"Dataset type {dset_type} invalid.")
    dset.reset_iter()

    tr_loader = torch.utils.data.DataLoader(
        dset, num_workers=4, prefetch_factor=4, pin_memory=True,
        batch_size=None, worker_init_fn=init_dloader_random)

    return {
        "tr_dset": dset,
        "tr_loader": tr_loader,
    }


def init_ts_data(data_dirs: Sequence[str], split: str, data_downsample=4, **kwargs):
    assert len(data_dirs) == 1
    data_dir = data_dirs[0]
    dset_type = decide_dset_type(data_dir)
    if dset_type == "synthetic":
        max_ts_frames = parse_optint(kwargs.get('max_ts_frames'))
        dset = SyntheticNerfDataset(
            data_dir, split=split, downsample=1, max_frames=max_ts_frames)
    elif dset_type == "llff":
        hold_every = parse_optint(kwargs.get('hold_every'))
        dset = LLFFDataset(
            data_dir, split=split, downsample=int(data_downsample), hold_every=hold_every,
            contraction=kwargs['contract'], ndc=kwargs['ndc'],
            ndc_far=float(kwargs['ndc_far']), near_scaling=float(kwargs['near_scaling']))
    else:
        raise ValueError(f"Dataset type {dset_type} invalid.")
    if split == 'train_render':
        keyname = "tr_render_dset"
        return {
            "tr_render_dset": dset,
        }
    else:
        keyname = "ts_dset"
        ts_loader = torch.utils.data.DataLoader(
            dset, batch_size=None, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4
        )
        return {
            "ts_dset": dset,
            "ts_loader": ts_loader,
        }
    
    # return {"ts_dset": dset}
    # return {keyname: dset}


def load_data(data_downsample, data_dirs, validate_only, render_only, **kwargs):
    od: Dict[str, Any] = {}
    # if not validate_only:
    od.update(init_tr_data(data_downsample, data_dirs, **kwargs))
    # else:
    #     od.update(tr_loader=None, tr_dset=None)
    test_split = 'render' if render_only else 'test'
    od.update(init_ts_data(data_dirs, split=test_split, **kwargs))
    if kwargs.get('render_every', -1) > -1:  # for train render
        if not validate_only:
            od.update(init_ts_data(data_dirs, split='train_render', data_downsample=data_downsample, **kwargs))
    return od
