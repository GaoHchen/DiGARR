import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy

from plenoxels.ops.interpolation import grid_sample_wrapper
from plenoxels.raymarching.spatial_distortions import SpatialDistortion
import math
import torch.nn.functional as F

from mmgen.models.builder import MODULES, build_module


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):  # pts从aabb缩放至[-1,1]
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            pts_spacetime_detach: bool,
                            cur_level=3) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    has_time_planes = pts.shape[-1] == 4  #
    if num_levels is None:
        num_levels = len(ms_grids)
    cur_level = min(cur_level, num_levels - 1)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            if pts_spacetime_detach and has_time_planes and 3 in coo_comb:  # space_time planes, detach xyz
                interp_out_plane = (
                    grid_sample_wrapper(grid[ci], pts[..., coo_comb].detach())
                    .view(-1, feature_dim)
                )
            else:
                interp_out_plane = (
                    grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                    .view(-1, feature_dim)
                )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if scale_id > cur_level:
            if concat_features:
                multi_scale_interp.append(0. * interp_space.detach())  #@todo .detach() or not
        else:
            if concat_features:
                multi_scale_interp.append(interp_space)
            else:
                multi_scale_interp = multi_scale_interp + interp_space

    # coarse-to-fine training
    # weights = [1.] * (cur_level + 1) + [0.] * (num_levels - cur_level - 1)
    # weights = [1. if i == cur_level else 0. for i in range(num_levels)]  #@remind for debug
    # if global_step is None:
    #     weights = [1.] * num_levels
    # else:
    #     valid_levels = min(num_levels, math.ceil((global_step+1)/9_000))
    #     weights = [1.] * valid_levels + [0.] * (num_levels - valid_levels)

    if concat_features:
        # multi_scale_interp = [multi_scale_interp[i] * weights[i] for i in range(num_levels)]
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        
    return multi_scale_interp


class KPlaneField(nn.Module):
    def __init__(
        self,
        aabb,
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        multiscale_res: Optional[Sequence[int]],
        use_appearance_embedding: bool,
        appearance_embedding_dim: int,
        spatial_distortion: Optional[SpatialDistortion],
        density_activation: Callable,
        linear_decoder: bool,
        linear_decoder_layers: Optional[int],
        num_images: Optional[int],
        pts_spacetime_detach: bool,
        scale_step: Optional[int],
        generator: Optional[dict],
        decoder: Optional[dict],
        occlusion_culling_th = 1e-4,
    ) -> None:
        super().__init__()

        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = grid_config

        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]
        self.concat_features = concat_features_across_scales
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder

        self.pts_spacetime_detach = pts_spacetime_detach
        self.scale_step= scale_step
        
        # self.generator = build_module(generator)
        self.preprocessor = build_module(generator)
        # self.preprocessor_configured = build_module(generator)  #@todo
        self.decoder = build_module(decoder)
        self.occlusion_culling_th = occlusion_culling_th
        
        self.code_proc_buffer = None
        self.code_buffer = None
        
        """
        # 1. Init planes
        self.cur_level = 0
        self.grids = nn.ModuleList()
        self.feature_dim = 0
        self.feat_dims = []
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
                self.feat_dims.append(gp[-1].shape[1])
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)
        log.info(f"Initialized model grids: {self.grids}")  # self.grids.shape: l, k, [1, out_dim, *reso]

        # 2. Init appearance code-related parameters
        self.use_average_appearance_embedding = True  # for test-time
        self.use_appearance_embedding = use_appearance_embedding
        self.num_images = num_images
        self.appearance_embedding = None
        if use_appearance_embedding:
            assert self.num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            # this will initialize as normal_(0.0, 1.0)
            self.appearance_embedding = nn.Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.appearance_embedding_dim = 0

        # 3. Init decoder params

        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        # breakpoint()

        # 3. Init decoder network
        if self.linear_decoder: #
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for
            # combining the color features into RGB
            # This architecture is based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,#self.direction_encoder.n_output_dims,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.geo_feat_dim = 15
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.sigma_nets = []
            for feat_dim in self.feat_dims:
                self.sigma_nets.append(
                    tcnn.Network(
                        n_input_dims=feat_dim,
                        n_output_dims=self.geo_feat_dim + 1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "None",
                            "n_neurons": 64,
                            "n_hidden_layers": 1,
                        },
                    )
                )
            #FIXME: modified 
            self.in_dim_color = (
                    # self.direction_encoder.n_output_dims# + 3    # 
                    3
                    + self.geo_feat_dim # 15
                    + self.appearance_embedding_dim # 0
            )
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
        """
    
    """
    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None, cur_level=3):
        # Computes and returns the densities.
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        # breakpoint()
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None, pts_spacetime_detach=self.pts_spacetime_detach,
            cur_level=cur_level)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features)  # [batch, 1]
        else:
            features = self.sigma_net(features)
            features, density_before_activation = torch.split(
                features, [self.geo_feat_dim, 1], dim=-1)

        density = self.density_activation(
            density_before_activation.to(pts)
        ).view(n_rays, n_samples, 1)
        return density, features

    # @torch.no_grad()
    def upsample_grid(self):
        for ci in range(len(self.grids[self.cur_level])):
            # breakpoint()
            h,w = self.grids[self.cur_level+1][ci].shape[-2:]
            self.grids[self.cur_level+1][ci].data = F.interpolate(self.grids[self.cur_level][ci].data, size=(h,w), mode='bilinear', align_corners=True)
            # rand_noise = torch.zeros((h,w)).normal_(mean=0,std=0.1).to(self.grids[self.cur_level][ci].device)
            # self.grids[self.cur_level+1][ci].data = self.grids[self.cur_level+1][ci].data + rand_noise
    """
            
    def code_proc_pr(self, code):
        code_proc = code
        if self.code_permute is not None:
            code_proc = code_proc.permute([0] + [axis + 1 for axis in self.code_permute])  # add batch dimension
        if self.code_reshape is not None:
            code_proc = code_proc.reshape(code.size(0), *self.code_reshape)  # add batch dimension
        return code_proc

    def code_proc_pr_inv(self, code_proc):
        code = code_proc
        if self.code_reshape is not None:
            code = code.reshape(code.size(0), *self.code_reshape_inv)
        if self.code_permute_inv is not None:
            code = code.permute([0] + [axis + 1 for axis in self.code_permute_inv])
        return code

    def preproc(self):
        # Avoid preprocessing the same code more than once
        # This implementation is not very robust though (changes to the model
        # parameters or inplace changes to the code will not update the buffer)
        # lu: encoder
        assert self.preprocessor is not None, "...plz check the preprocessor in zero_kplanes_field..."
        return self.preprocessor()
        # if self.preprocessor is None:
        #     return code
        # else:
        #     dtype = code.dtype
        #     preproc_dtype = next(self.preprocessor.parameters()).dtype
        #     if self.code_buffer is not code:
        #         if code.requires_grad:
        #             self.code_buffer = code
        #             self.code_proc_buffer = self.code_proc_pr_inv(
        #                 self.preprocessor(self.code_proc_pr(code.to(preproc_dtype)))).to(dtype)  #TODO: preprocessor直接用的noise过的网络,没有用到传进去的code
        #         else:
        #             if isinstance(self.code_buffer, torch.Tensor) and torch.all(self.code_buffer.data == code):
        #                 return self.code_proc_buffer.data.to(dtype)
        #             else:
        #                 self.code_buffer = code
        #                 self.code_proc_buffer = self.code_proc_pr_inv(
        #                     self.preprocessor(self.code_proc_pr(code.to(preproc_dtype)))).to(dtype)
        #     return self.code_proc_buffer
        
    def forward(self,
                ray_samples,
                # pts: torch.Tensor,
                directions: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                # code: Optional[torch.Tensor] = None,
                ): #FIXME:
        pts = ray_samples.get_positions()
        
        # density, features = self.get_density(pts, timestamps, global_step)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)  # (n_rays, n_samples, 3)
        
        """if not self.linear_decoder:
            #TODO: evaluate on this part
            # directions = get_normalized_directions(directions)  # salmon需要, cutbeef不需要
            # encoded_directions = self.direction_encoder(directions)
            
            # directions_kp = get_normalized_directions(directions)
            # encoded_directions = self.direction_encoder(directions_kp)
            # encoded_directions = torch.cat([directions, encoded_directions], -1)
            
            encoded_directions = directions"""
            
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        if timestamps is not None:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])          # [n_rays*n_samples, 3 or 4]
        
        code = self.preproc().to(directions)  # encoder, shape:(1, len(subs)=k * self.out_ch * noise_res**2)
        
        # debug
        if torch.isnan(code).any():
            breakpoint()
            self.get_params()
        # assert not torch.isnan(code).any(), ("Codes become nan!!!!!", self.preprocessor.parameters())
        
        
        # with occl, debuging @todo
        with torch.no_grad():
            sub_shapes = self.preprocessor.sub_shapes
            sigmas, rgbs, num_points = self.decoder.point_decode(pts, None, code, sub_shapes)  #self.point_decode(pts, None, code)
            sigmas = sigmas.to(directions)
        rgbs = sigmas.new_zeros(sigmas.shape[0], 3)
        with torch.no_grad():
            weights = ray_samples.get_weights(sigmas.view(n_rays, n_samples, -1))  # (n_rays, n_samples, 1)
            weights = weights.view(-1)
        occl = weights > self.occlusion_culling_th  # (n_rays*n_samples)
        pts_filt = pts[occl]
        directions_filt = directions[occl]
        sigmas_occl, rgbs_occl, _ = self.decoder.point_decode(pts_filt, directions_filt, code, sub_shapes)  #self.point_decode(xyzs_filt, dirs_filt, code)
        
        sigmas = torch.zeros_like(sigmas)
        # occl = torch.cat(occls_filt)
        sigmas[occl] = sigmas_occl.to(sigmas)
        rgbs[occl] = rgbs_occl.to(rgbs)
        
        
        # sub_shapes = self.preprocessor.sub_shapes
        # sigmas, rgbs, num_points = self.decoder.point_decode(pts, directions, code, sub_shapes)  #self.point_decode(pts, None, code)
        
        sigmas = sigmas.to(directions).view(n_rays, n_samples, 1)
        rgbs = rgbs.to(directions).view(n_rays, n_samples, 3)
        # breakpoint()

        return {"rgb": rgbs, "density": sigmas}

    def get_params(self):
        preproc_params = {k: v for k,v in self.preprocessor.named_parameters(prefix="preprocessor")}
        decoder_params = {k: v for k,v in self.decoder.named_parameters(prefix="decoder")}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in preproc_params.keys() and k not in decoder_params.keys()
        )}
        # breakpoint()
        return {
            "preprocessor": list(preproc_params.values()),
            "decoder": list(decoder_params.values()),
            "other": list(other_params.values()),
        }
    
    def get_params_ms(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = [
            self.sigma_net.named_parameters(prefix="sigma_net"),
            self.direction_encoder.named_parameters(prefix="direction_encoder"),
        ]
        if self.linear_decoder:
            nn_params.append(self.color_basis.named_parameters(prefix="color_basis"))
        else:
            nn_params.append(self.color_net.named_parameters(prefix="color_net"))
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),  # sigma_net, direction_encoder, color_net
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }
    
    def get_params_separately(self):
        """Separate space and space-time fields."""

        # for res in self.multiscale_res_multipliers:
        #     # initialize coordinate grid
        #     config = self.grid_config[0].copy()
        #     # Resolution fix: multi-res only on spatial planes
        #     config["resolution"] = [
        #         r * res for r in config["resolution"][:3]
        #     ] + config["resolution"][3:]
        #     gp = init_grid_param(
        #         grid_nd=config["grid_dimensions"],
        #         in_dim=config["input_coordinate_dim"],
        #         out_dim=config["output_coordinate_dim"],
        #         reso=config["resolution"],
        #     )

        # has_time_planes = in_dim == 4
        # assert grid_nd <= in_dim
        # coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        # grid_coefs = nn.ParameterList()
        # for ci, coo_comb in enumerate(coo_combs):
        #     new_grid_coef = nn.Parameter(torch.empty(
        #         [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        #     ))
        #     if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
        #         nn.init.ones_(new_grid_coef)
        #     else:
        #         nn.init.uniform_(new_grid_coef, a=a, b=b)
        #     grid_coefs.append(new_grid_coef)

        config = self.grid_config[0].copy()
        grid_nd = config["grid_dimensions"]
        in_dim = config["input_coordinate_dim"]
        field_spacetime_params = None
        field_spaceonly_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        if in_dim == 4:
            coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
            field_spacetime_params = {}
            for res in range(len(self.grids)):
                for ci, coo_comb in enumerate(coo_combs):
                    tmp_key = 'grids.'+str(res)+'.'+str(ci)
                    if 3 in coo_comb:
                        field_spacetime_params.update({tmp_key: field_spaceonly_params[tmp_key]})
                        del field_spaceonly_params[tmp_key]

        nn_params = [
            self.sigma_net.named_parameters(prefix="sigma_net"),
            self.direction_encoder.named_parameters(prefix="direction_encoder"),
        ]
        if self.linear_decoder:
            nn_params.append(self.color_basis.named_parameters(prefix="color_basis"))
        else:
            nn_params.append(self.color_net.named_parameters(prefix="color_net"))
        nn_params = {k: v for plist in nn_params for k, v in plist}


        if field_spacetime_params == None:
            other_params = {k: v for k, v in self.named_parameters() if (
                k not in nn_params.keys() and k not in field_spaceonly_params.keys()
            )}
        else:
            other_params = {k: v for k, v in self.named_parameters() if (
                k not in nn_params.keys() and k not in field_spaceonly_params.keys() and k not in field_spacetime_params.keys()
            )}

        return {
            "nn": list(nn_params.values()),
            "field_spaceonly": list(field_spaceonly_params.values()),
            "field_spacetime": list(field_spacetime_params.values()) if field_spacetime_params else None,
            "other": list(other_params.values()),
        }
