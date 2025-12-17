from typing import List, Sequence, Optional, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from plenoxels.models_all.density_fields import KPlaneDensityField
from plenoxels.models_all.kplane_field import KPlaneField
# from plenoxels.models_all.zero_kplanes import KPlaneField  #FIXME:
from plenoxels.ops.activations import init_density_activation
from plenoxels.raymarching.ray_samplers import (
    UniformLinDispPiecewiseSampler, UniformSampler,
    ProposalNetworkSampler, RayBundle, RaySamples
)
from plenoxels.raymarching.spatial_distortions import SceneContraction, SpatialDistortion
from plenoxels.utils.timer import CudaTimer

from models_all.poses import LearnPose
from datasets.ray_utils import get_rays

from mmgen.models.builder import MODULES, build_module
from einops import rearrange


class LowrankModel_kp(nn.Module):
    def __init__(self,
                grid_config: Union[str, List[Dict]],
                # boolean flags
                is_ndc: bool,
                is_contracted: bool,
                aabb: torch.Tensor,
                # Model arguments
                multiscale_res: Sequence[int],
                generator: dict(
                    type='TensorialGenerator'),  #@remind 一定要这个！！！！！还有在static_trainer/video_trainer的init model里面注意！！！！
                decoder=dict(
                    type='TriPlaneDecoder'),
                density_activation: Optional[str] = 'trunc_exp',
                concat_features_across_scales: bool = False,
                linear_decoder: bool = True,
                linear_decoder_layers: Optional[int] = 1,
                # Spatial distortion
                global_translation: Optional[torch.Tensor] = None,
                global_scale: Optional[torch.Tensor] = None,
                # proposal-sampling arguments
                num_proposal_iterations: int = 1,
                use_same_proposal_network: bool = False,
                proposal_net_args_list: List[Dict] = None,
                proposal_generator_args_list: List[Dict] = None,
                num_proposal_samples: Optional[Tuple[int]] = None,
                num_samples: Optional[int] = None,
                single_jitter: bool = False,
                proposal_warmup: int = 5000,
                proposal_update_every: int = 5,
                use_proposal_weight_anneal: bool = True,
                proposal_weights_anneal_max_num_iters: int = 1000,
                proposal_weights_anneal_slope: float = 10.0,
                # appearance embedding (phototourism)
                use_appearance_embedding: bool = False,
                appearance_embedding_dim: int = 0,
                num_images: Optional[int] = None,
                pts_spacetime_detach: bool = False,  # 是否将space-time中的xyz进行detach(对于动态场景的nopose联合优化需要它为True)
                scale_step: Optional[int] = None,
                code_mode_ds_s2: Optional[str] = "mul",
                **kwargs,
                ):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        print(f"pts spacetime detach: {pts_spacetime_detach}")
        self.multiscale_res = multiscale_res
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.concat_features_across_scales = concat_features_across_scales
        self.linear_decoder = linear_decoder
        self.linear_decoder_layers = linear_decoder_layers
        self.density_act = init_density_activation(density_activation)
        self.timer = CudaTimer(enabled=False)

        self.spatial_distortion: Optional[SpatialDistortion] = None
        if self.is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float('inf'), global_scale=global_scale,
                global_translation=global_translation)
            
        self.preprocessor = build_module(generator) 
        # 2024/1/12 use original field to do color decoder
        
        decoder["aabb"]=aabb
        decoder["spatial_distortion"]=self.spatial_distortion
        decoder["sub_shapes"] = self.preprocessor.sub_shapes
        del self.preprocessor
        for i in range(len(decoder["sub_shapes"])):
            decoder["sub_shapes"][i][0] *= len(multiscale_res)
        # 2024/1/11
        # decoder["sub_shapes"] = [[64, 320, 320], [64, 320, 320], [64, 320, 320]]    #FIXME: hard coded 
        self.decoder = build_module(decoder)    # keep original decoder
        self.code = None
        self.code_proposal = None

        # # 2024/1/11
        # self.linear_project = nn.Sequential(
        #     nn.Linear(16, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 64),
        #     nn.ReLU(inplace=True)
        # )
        
        self.field = KPlaneField(   # original kplanes field
            aabb,
            grid_config=self.config,
            concat_features_across_scales=self.concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.density_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
            pts_spacetime_detach=pts_spacetime_detach,  #加的
            scale_step=scale_step,
            base_channels=kwargs['model_out_ch_']//len(multiscale_res),
            base_out_channels=kwargs['model_out_ch_'],
        )
        

        # Initialize proposal-sampling nets
        self.density_preproc = []
        self.density_fns = []
        self.precompute_fns = []    # 2024/1/12
        self.num_proposal_iterations = num_proposal_iterations
        self.proposal_net_args_list = proposal_net_args_list
        self.proposal_generator_args_list = proposal_generator_args_list
        self.proposal_warmup = proposal_warmup
        self.proposal_update_every = proposal_update_every
        self.use_proposal_weight_anneal = use_proposal_weight_anneal
        self.proposal_weights_anneal_max_num_iters = proposal_weights_anneal_max_num_iters
        self.proposal_weights_anneal_slope = proposal_weights_anneal_slope
        self.proposal_preproc_nets = torch.nn.ModuleList()
        self.proposal_networks = torch.nn.ModuleList()
        # 2024/1/12
        self.proposal_networks_getcode = torch.nn.ModuleList()
        if use_same_proposal_network:
            assert len(self.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.proposal_net_args_list[0]
            network = KPlaneDensityField(
                aabb, spatial_distortion=self.spatial_distortion,
                density_activation=self.density_act, linear_decoder=self.linear_decoder, 
                pts_spacetime_detach=pts_spacetime_detach, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for _ in range(self.num_proposal_iterations)])
        else:
            for i in range(self.num_proposal_iterations):  # 2
                # 1/12
                prop_net_args = self.proposal_net_args_list[min(i, len(self.proposal_net_args_list) - 1)]
                network = KPlaneDensityField(
                    aabb, spatial_distortion=self.spatial_distortion,
                    density_activation=self.density_act, linear_decoder=self.linear_decoder, 
                    pts_spacetime_detach=pts_spacetime_detach, **prop_net_args,
                )
                self.proposal_networks_getcode.append(network)
                
                prop_gen_args = self.proposal_generator_args_list[min(i, len(self.proposal_generator_args_list) - 1)]
                prop_generator = dict(
                    type='TensorialGenerator',
                    in_ch=prop_gen_args['model_ch'], out_ch=prop_gen_args['model_out_ch'], 
                    noise_res=torch.tensor(prop_gen_args['model_res']),
                    tensor_config=generator['tensor_config'],
                    up_block_num=prop_gen_args['up_block_num'],
                    block_out_channels=prop_gen_args['block_out_channels'],
                )
                encoder = build_module(prop_generator)


                prop_decoder = dict(
                    type='TensorialDecoder',
                    in_ch=prop_gen_args['model_out_ch'],  # 16
                    subreduce=3,    #1 if args.load_image else 2, TODO:
                    reduce='cat',
                    separate_density_and_color=False,
                    sh_coef_only=False,
                    sdf_mode=False,
                    max_steps=1024,# if not args.load_image else 320,
                    # n_images=args.n_views,
                    image_h=decoder['image_h'],#pic_h,
                    image_w=decoder['image_w'],#pic_w,
                    tensor_config=generator['tensor_config'],
                    aabb=aabb,
                    spatial_distortion=self.spatial_distortion,
                    sub_shapes=encoder.sub_shapes,  # TODO:
                    code_mode_ds=code_mode_ds_s2,
                    # has_time_dynamics=False,
                    # visualize_mesh=True
                )
            #     self.proposal_preproc_nets.append(encoder)
                network = build_module(prop_decoder)    # decoder only 
                self.proposal_networks.append(network)
            # self.density_preproc.extend([encoder for encoder in self.proposal_preproc_nets])
            self.precompute_fns.extend([network.precompute_code for network in self.proposal_networks_getcode])
            self.density_fns.extend([network.point_density_decode_precomputed_code for network in self.proposal_networks])  # need precomputed point feature
            # self.density_fns.extend([network.get_density for _ in range(self.num_proposal_iterations)]) # original field density fns
            

        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.proposal_warmup], [0, self.proposal_update_every]),
            1,
            self.proposal_update_every,
        )
        if self.is_contracted or self.is_ndc:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
            # initial_sampler = UniformLinDispPiecewiseSampler(num_samples=256, single_jitter=single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=single_jitter)
        # self.initial_sampler = initial_sampler
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=num_samples,
            num_proposal_samples_per_ray=num_proposal_samples,
            num_proposal_network_iterations=self.num_proposal_iterations,
            single_jitter=single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler
        )

    def step_before_iter(self, step):
        if self.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / N, 0, 1)
            bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, self.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(anneal)

    def step_after_iter(self, step):
        if self.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    @staticmethod
    def render_rgb(rgb: torch.Tensor, weights: torch.Tensor, bg_color: Optional[torch.Tensor]):
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)
        if bg_color is None:
            pass
        else:
            comp_rgb = comp_rgb + (1.0 - accumulated_weight) * bg_color
        return comp_rgb

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples, rays_d: torch.Tensor):
        steps = (ray_samples.starts + ray_samples.ends) / 2
        one_minus_transmittance = torch.sum(weights, dim=-2)
        depth = torch.sum(weights * steps, dim=-2) + one_minus_transmittance * rays_d[..., -1:]
        return depth

    @staticmethod
    def render_accumulation(weights: torch.Tensor):
        accumulation = torch.sum(weights, dim=-2)
        return accumulation

    def preproc(self):
        # Avoid preprocessing the same code more than once
        # This implementation is not very robust though (changes to the model
        # parameters or inplace changes to the code will not update the buffer)
        # lu: encoder
        assert self.preprocessor is not None, "...plz check the preprocessor in zero_kplanes_field..."
        return self.preprocessor()
    
    def init_grids_with_code(self, code: Optional[torch.Tensor], code_proposal: Optional[torch.Tensor]):
        # code: (k, l*ch, *res)
        # self.field.init_with_code_split_channels(code)
        # 2024/01/24 code: l(k(1,ch,*res))
        self.field.init_with_code(code)
        for i, pn in enumerate(self.proposal_networks_getcode):
            pn.init_with_code(code_proposal[i])
        
    def forward(self, bg_color, near_far: torch.Tensor, rays_o=None, rays_d=None, timestamps=None, global_step=None, dino_feat=None):  #@note
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """

        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions are 3D.
            
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples_precompute_codes(   # use original to generate samples
            ray_bundle, timestamps=timestamps, density_fns=self.density_fns, precompute_fns=self.precompute_fns)

        # ray_samples, weights_list, ray_samples_list, code_list = self.proposal_sampler.generate_ray_samples(
        #         ray_bundle, timestamps=timestamps, preproc=self.density_preproc, density_fns=self.density_fns)
        # self.code_proposal = code_list
        # breakpoint()

        # 1/13
        precompute_code = self.field.precompute_code(ray_samples.get_positions(), timestamps)
        # breakpoint()
        field_out = self.decoder.render_with_precompute_code(ray_samples,
                                precompute_code, 
                                ray_bundle.directions, timestamps)

        # field_out = self.field(ray_samples.get_positions(), ray_bundle.directions, timestamps, global_step)
        # field_out = self.field(ray_samples, ray_bundle.directions, timestamps)

        rgb, density = field_out["rgb"], field_out["density"]

        weights = ray_samples.get_weights(density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        # breakpoint()
        rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color)
        depth = self.render_depth(weights=weights, ray_samples=ray_samples, rays_d=ray_bundle.directions)
        accumulation = self.render_accumulation(weights=weights)
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        # for i in range(len(weights_list)):
        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.render_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i], rays_d=ray_bundle.directions)
        return outputs

    def get_params(self, lr: float):
        model_params = self.field.get_params()
        pn_params = [pn.get_params() for pn in self.proposal_networks]  # density_field
        field_params = model_params["field"] + [p for pnp in pn_params for p in pnp["field"]]
        nn_params = model_params["nn"] + [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_params["other"] + [p for pnp in pn_params for p in pnp["other"]]
        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
            
        ]
    def get_params_field_decoder(self, lr: float, field_lr: Optional[float]):
        model_params = self.field.get_params()
        pn_params = [pn.get_params() for pn in self.proposal_networks_getcode]  # density_field
        field_params = model_params["field"] + [p for pnp in pn_params for p in pnp["field"]]
        decoder_params = list([v for k,v in self.decoder.named_parameters(prefix="decoder")])
        pn_decoder_params = [list([v for k,v in pn.named_parameters()]) for pn in self.proposal_networks]  # density_field
        pn_decoder_params = [p for pnp in pn_decoder_params for p in pnp]

        if field_lr is None:
            return [
                {"params": field_params, "lr": lr},
                {"params": decoder_params, "lr": lr},
                {"params": pn_decoder_params, "lr": lr},
            ]
        elif isinstance(field_lr, float):
            return [
                {"params": field_params, "lr": field_lr},
                {"params": decoder_params, "lr": lr},
                {"params": pn_decoder_params, "lr": lr},
            ]
        else:
            params_ms = [{"params": field_params[i*3:(i+1)*3], "lr": field_lr[i]} for i in range(len(field_lr))]
            
            return params_ms + [{"params": field_params[len(field_lr)*3:], "lr": lr},{"params": decoder_params, "lr": lr}, {"params": pn_decoder_params, "lr": lr}]
        
        
        
    def get_params_zero(self, lr: float, density_lr: float = None):
        preproc_params = list([v for k,v in self.preprocessor.named_parameters(prefix="preprocessor")])
        decoder_params = list([v for k,v in self.decoder.named_parameters(prefix="decoder")])
        pn_preproc_params = [list([v for k,v in pn.named_parameters()]) for pn in self.proposal_preproc_nets]  # density_field
        pn_decoder_params = [list([v for k,v in pn.named_parameters()]) for pn in self.proposal_networks]  # density_field
        pn_preproc_params = [p for pnp in pn_preproc_params for p in pnp]
        pn_decoder_params = [p for pnp in pn_decoder_params for p in pnp]
        if density_lr is None:
            density_lr = lr
        return [
            {"params": preproc_params, "lr": lr},
            {"params": decoder_params, "lr": lr},
            {"params": pn_preproc_params, "lr": density_lr},
            {"params": pn_decoder_params, "lr": density_lr},
        ]
        # preproc_params = preproc_params + [p for pnp in pn_preproc_params for p in pnp]
        # decoder_params = decoder_params + [p for pnp in pn_decoder_params for p in pnp]
        # return [
        #     {"params": preproc_params, "lr": lr},
        #     {"params": decoder_params, "lr": lr},
        # ]
        
    def get_params_uniform(self, lr: float):
        model_params = self.field.get_params()
        field_params = model_params["field"]
        nn_params = model_params["nn"]
        other_params = model_params["other"]
        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
        ]
    
    def get_params_ms(self, lr: float, field_lr: list):
        model_params = self.field.get_params()
        pn_params = [pn.get_params() for pn in self.proposal_networks]  # density_field
        field_params = model_params["field"] + [p for pnp in pn_params for p in pnp["field"]]
        nn_params = model_params["nn"] + [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_params["other"] + [p for pnp in pn_params for p in pnp["other"]]
        return [
            {"params": field_params[12:], "lr": lr},            # kp_density_field
            {"params": field_params[0:3], "lr": field_lr[0]},   # kp_field
            {"params": field_params[3:6], "lr": field_lr[1]},
            {"params": field_params[6:9], "lr": field_lr[2]},
            {"params": field_params[9:12], "lr": field_lr[3]},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
        ]
    
    def get_params_nopose_separately(self, lr: float, spaceonly_lr: float, spacetime_lr: float):
        model_params = self.field.get_params_separately()
        pn_params = [pn.get_params_separately() for pn in self.proposal_networks]  # density_field
        field_spaceonly_params = model_params["field_spaceonly"] + [p for pnp in pn_params for p in pnp["field_spaceonly"]]
        field_spacetime_params = model_params["field_spacetime"] + [p for pnp in pn_params for p in pnp["field_spacetime"]] if model_params["field_spacetime"] else None
        nn_params = model_params["nn"] + [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_params["other"] + [p for pnp in pn_params for p in pnp["other"]]

        if field_spacetime_params:
            return [
                {"params": field_spaceonly_params, "lr": spaceonly_lr},
                {"params": field_spacetime_params, "lr": spacetime_lr},
                {"params": nn_params, "lr": lr},
                {"params": other_params, "lr": lr},
            ]
        else:
            return [
                {"params": field_spaceonly_params, "lr": spaceonly_lr},
                {"params": nn_params, "lr": lr},
                {"params": other_params, "lr": lr},
            ]
    
    def get_params_separately(self, lr: float, spaceonly_lr: float, spacetime_lr: float, pose_lr: float):
        model_params = self.field.get_params_separately()
        pose_params = self.pose_params_net.get_params()["pose"]
        pn_params = [pn.get_params_separately() for pn in self.proposal_networks]  # density_field
        field_spaceonly_params = model_params["field_spaceonly"] + [p for pnp in pn_params for p in pnp["field_spaceonly"]]
        field_spacetime_params = model_params["field_spacetime"] + [p for pnp in pn_params for p in pnp["field_spacetime"]] if model_params["field_spacetime"] else None
        nn_params = model_params["nn"] + [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_params["other"] + [p for pnp in pn_params for p in pnp["other"]]

        if field_spacetime_params:
            return [
                {"params": field_spaceonly_params, "lr": spaceonly_lr},
                {"params": field_spacetime_params, "lr": spacetime_lr},
                {"params": pose_params, "lr": pose_lr},
                {"params": nn_params, "lr": lr},
                {"params": other_params, "lr": lr},
            ]
        else:
            return [
                {"params": field_spaceonly_params, "lr": spaceonly_lr},
                {"params": pose_params, "lr": pose_lr},
                {"params": nn_params, "lr": lr},
                {"params": other_params, "lr": lr},
            ]
