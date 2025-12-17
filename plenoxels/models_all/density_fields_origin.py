"""
Density proposal field
"""
from typing import Optional, Callable
import logging as log

import torch
import torch.nn as nn
import tinycudann as tcnn
import torch.nn.functional as F

from plenoxels.models_all.kplane_field import interpolate_ms_features, normalize_aabb, init_grid_param  #@remind
from plenoxels.raymarching.spatial_distortions import SpatialDistortion


class KPlaneDensityField(nn.Module):
    def __init__(self,
                 aabb,
                 resolution,
                 num_input_coords,
                 num_output_coords,
                 density_activation: Callable,
                 spatial_distortion: Optional[SpatialDistortion] = None,
                 linear_decoder: bool = True,
                 pts_spacetime_detach: bool = False,):
        super().__init__()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.hexplane = num_input_coords == 4
        self.feature_dim = num_output_coords
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder
        self.pts_spacetime_detach = pts_spacetime_detach
        activation = "ReLU"
        if self.linear_decoder:
            activation = "None"

        self.grids = init_grid_param(
            grid_nd=2, in_dim=num_input_coords, out_dim=num_output_coords, reso=resolution,
            a=0.1, b=0.15)
        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        log.info(f"Initialized KPlaneDensityField. hexplane={self.hexplane} - "
                 f"resolution={resolution}")
        log.info(f"KPlaneDensityField grids: \n{self.grids}")
        
    def init_with_code(self, code: Optional[torch.Tensor] = None):
        """
        Initialize KPlaneDensityField grids with code.
        Args:
            code: from zero_pose
        """
        if code is not None:
            for i, grid in enumerate(self.grids):
                h, w = grid.data.shape[-2:]
                grid.data = F.interpolate(code[i].data, size=(h,w), mode='bilinear', align_corners=True)
            log.info(f"Initialized KPlaneDensityField grids with code.")

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None and self.hexplane:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts.detach(), ms_grids=[self.grids], grid_dimensions=2, concat_features=False, num_levels=None, pts_spacetime_detach=self.pts_spacetime_detach)  #@todo 231124: 给pts加了.detach()
        # features = interpolate_ms_features(
        #     pts, ms_grids=[self.grids], grid_dimensions=2, concat_features=False, num_levels=None)
        density = self.density_activation(
            self.sigma_net(features).to(pts)
            #features.to(pts)
        ).view(n_rays, n_samples, 1)
        return density

    def forward(self, pts: torch.Tensor):
        return self.get_density(pts)

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = {k: v for k, v in self.sigma_net.named_parameters(prefix="sigma_net")}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }
        
    def get_params_separately(self):
        """Separate space and space-time fields."""
        
        grid_nd = 2
        in_dim = 4 if self.hexplane else 3
        field_spacetime_params = None
        field_spaceonly_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        if in_dim == 4:
            import itertools
            coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
            field_spacetime_params = {}
            for ci, coo_comb in enumerate(coo_combs):
                if 3 in coo_comb:
                    tmp_key = 'grids.'+str(ci)
                    try:
                        field_spacetime_params.update({tmp_key: field_spaceonly_params[tmp_key]})
                    except KeyError:
                        breakpoint()
                    del field_spaceonly_params[tmp_key]

        nn_params = {k: v for k, v in self.sigma_net.named_parameters(prefix="sigma_net")}

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

