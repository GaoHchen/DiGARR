import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import tinycudann as tcnn

from plenoxels.ops.interpolation import grid_sample_wrapper
from plenoxels.raymarching.spatial_distortions import SpatialDistortion
import math
import torch.nn.functional as F
import numpy as np


def positional_encoding(input, L, global_step):  # [B,...,N] 4
    shape = input.shape
    freq = (
        2 ** torch.arange(L, dtype=torch.float32, device=input.device)
        * np.pi
    )  # [L]
    spectrum = input[..., None] * freq  # [B,...,N,L]   pi, 2*pi
    sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
    # input_enc = torch.stack([sin,sin, cos,cos], dim=-2)  # [B,...,N,2,L]
    # breakpoint()
    input_enc = torch.stack([ sin,  cos], dim=-2)  # [B,...,N,2,L]
    input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
    # input_enc = torch.cat([input_enc, input], dim=-1)

    return input_enc

def encoding_masking(L, input_enc, global_step, barf_c2f=[0.1, 0.5]):
    # barf_c2f = [0., 0.4]
    # coarse-to-fine: smoothly mask positional encoding for BARF
    
    if barf_c2f is not None and global_step is not None:
        progress = global_step / 70000.
        # breakpoint()
        # set weights for different frequency bands
        start, end = barf_c2f
        alpha = (progress - start) / (end - start) * L
        k = torch.arange(L, dtype=torch.float32, device=input_enc.device)
        weight = (
            1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()
        ) / 2
        # apply weights
        shape = input_enc.shape
        input_enc = (input_enc.view(-1, L) * weight).view(*shape)
    return input_enc


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
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

        # 1. Init planes
        self.cur_level = 0
        self.grids = nn.ModuleList()
        self.feature_dim = 0
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
        # #FIXME: 
        # self.direction_encoder = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "Frequency",
        #         "n_frequencies": 2,
        #     },
        # )
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
            #FIXME: modified #TODO: DO NOT ENCODE AT ALL
            self.in_dim_color = (
                    self.direction_encoder.n_output_dims    # 
                    # 3
                    # 12
                    + self.geo_feat_dim # 15
                    + self.appearance_embedding_dim # 0
            )
            self.in_dim_color_dir = (
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
            self.color_net_dir = tcnn.Network(
                n_input_dims=self.in_dim_color_dir,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None, cur_level=3):
        """Computes and returns the densities."""
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
    def scale_grid_param(
            self,
            grid_nd: int,
            in_dim: int,
            reso: Sequence[int],):
        assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
        assert grid_nd <= in_dim
        coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        grid_coefs = nn.ParameterList()
        for ci, coo_comb in enumerate(coo_combs):
            # breakpoint()
            # new_grid_coef = nn.Parameter(
            #     F.interpolate(self.grids[self.cur_level-1][ci].data, size=tuple([reso[cc] for cc in coo_comb[::-1]]), mode='bilinear', align_corners=True)
            # )
            new_grid_coef = nn.Parameter(
                F.interpolate(self.grids[self.cur_level-1][ci].data.clone(), scale_factor=2., mode='bilinear', align_corners=True)
            )
            rand_noise = torch.zeros(new_grid_coef.data.shape).normal_(mean=0,std=0.1).to(new_grid_coef.device)
            new_grid_coef.data = new_grid_coef.data + rand_noise
            grid_coefs.append(new_grid_coef)

        return grid_coefs

    # @torch.no_grad()
    def scale_grid(self):
        self.cur_level = self.cur_level + 1
        res = self.multiscale_res_multipliers[self.cur_level]
        # initialize coordinate grid
        config = self.grid_config[0].copy()
        # Resolution fix: multi-res only on spatial planes
        config["resolution"] = [
            r * res for r in config["resolution"][:3]
        ] + config["resolution"][3:]
        self.grids[self.cur_level] = self.scale_grid_param(
            grid_nd=config["grid_dimensions"],
            in_dim=config["input_coordinate_dim"],
            reso=config["resolution"],
        )
        
    def upsample_grid(self):
        for ci in range(len(self.grids[self.cur_level])):
            # breakpoint()
            h,w = self.grids[self.cur_level+1][ci].shape[-2:]
            self.grids[self.cur_level+1][ci].data = F.interpolate(self.grids[self.cur_level][ci].data, size=(h,w), mode='bilinear', align_corners=True)
            # rand_noise = torch.zeros(new_grid_coef.data.shape).normal_(mean=0,std=0.1).to(new_grid_coef.device)
            # new_grid_coef.data = new_grid_coef.data + rand_noise

    def forward(self,
                pts: torch.Tensor,
                directions: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                global_step=None):
        camera_indices = None
        
        if self.use_appearance_embedding:
            if timestamps is None:
                raise AttributeError("timestamps (appearance-ids) are not provided.")
            camera_indices = timestamps
            timestamps = None
            
        if self.scale_step is not None:
            if global_step in self.scale_step:
                log.info(f"Grids scale up to level {self.cur_level+1}")
                # self.scale_grid()
                self.upsample_grid()
                self.cur_level = self.cur_level + 1
            density, features = self.get_density(pts, timestamps, self.cur_level)
        else:
            density, features = self.get_density(pts, timestamps)
        # density, features = self.get_density(pts, timestamps, global_step)
        # density, features = self.get_density(pts, timestamps, self.cur_level)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
        if not self.linear_decoder:

            # directions = get_normalized_directions(directions)  #FIXME: whats wrong with this line
            # # #TODO: evaluate on this part
            # encoded_directions = positional_encoding(directions, 4)
            
            encoded_directions = self.direction_encoder(directions)
            # breakpoint()
            # encoded_directions = encoding_masking(4, encoded_directions, global_step, barf_c2f=[0.1, 0.5])
            # breakpoint()
            # encoded_directions = torch.cat([encoded_directions, directions], dim=-1)
            # breakpoint()
            # encoded_directions = directions
        # else: encoded_directions = self.direction_encoder(directions.detach())
        # breakpoint()

        if self.linear_decoder:
            color_features = [features]
        else:
            color_features = [encoded_directions, features.view(-1, self.geo_feat_dim)] #TODO: debug the direction influence to the pose+nerf
            color_features_dir = [directions, features.view(-1, self.geo_feat_dim)] #TODO: debug the direction influence to the pose+nerf
        if self.use_appearance_embedding:
            if camera_indices.dtype == torch.float32:
                # Interpolate between two embeddings. Currently they are hardcoded below.
                #emb1_idx, emb2_idx = 100, 121  # trevib
                emb1_idx, emb2_idx = 11, 142  # sacre
                emb_fn = self.appearance_embedding
                emb1 = emb_fn(torch.full_like(camera_indices, emb1_idx, dtype=torch.long))
                emb1 = emb1.view(emb1.shape[0], emb1.shape[2])
                emb2 = emb_fn(torch.full_like(camera_indices, emb2_idx, dtype=torch.long))
                emb2 = emb2.view(emb2.shape[0], emb2.shape[2])
                embedded_appearance = torch.lerp(emb1, emb2, camera_indices)
            elif self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                if hasattr(self, "test_appearance_embedding"):
                    embedded_appearance = self.test_appearance_embedding(camera_indices)
                elif self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.appearance_embedding.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

            # expand embedded_appearance from n_rays, dim to n_rays*n_samples, dim
            ea_dim = embedded_appearance.shape[-1]
            embedded_appearance = embedded_appearance.view(-1, 1, ea_dim).expand(n_rays, n_samples, -1).reshape(-1, ea_dim)
            if not self.linear_decoder:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)
        color_features_dir = torch.cat(color_features_dir, dim=-1)
        if self.linear_decoder:
            if self.use_appearance_embedding:
                basis_values = self.color_basis(torch.cat([directions, embedded_appearance], dim=-1))
            else:
                basis_values = self.color_basis(directions)  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = rgb.to(directions)
            rgb = torch.sigmoid(rgb).view(n_rays, n_samples, 3)
        else:
            rgb_encode = self.color_net(color_features).to(directions).view(n_rays, n_samples, 3)
            rgb_dironly = self.color_net_dir(color_features_dir).to(directions).view(n_rays, n_samples, 3)
            
            if timestamps is not None:
                if global_step and global_step <= 1500:
                    # rgb = 0.7 * rgb_dironly + 0.3 * rgb_encode
                    rgb = rgb_dironly
                else: rgb = rgb_encode

            else: 
                if global_step and global_step <= 1000:
                    # rgb =  1*  rgb_dironly + 1 *  rgb_encode
                    rgb =  rgb_dironly # + 0.2 * rgb_encode
                else: rgb = rgb_encode
                # breakpoint()
            # rgb = rgb_dironly
            # rgb = rgb_encode

        return {"rgb": rgb, "density": density}

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = [
            self.sigma_net.named_parameters(prefix="sigma_net"),
            self.direction_encoder.named_parameters(prefix="direction_encoder"),
        ]
        if self.linear_decoder:
            nn_params.append(self.color_basis.named_parameters(prefix="color_basis"))
        else:
            nn_params.append(self.color_net.named_parameters(prefix="color_net"))
            nn_params.append(self.color_net_dir.named_parameters(prefix="color_net_dir"))
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
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
