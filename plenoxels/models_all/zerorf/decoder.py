import os
import numpy
import torch
import torch.nn as nn
import tinycudann as tcnn
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable


# from plenoxels.models_zerorf.ops import SHEncoder, TruncExp
# from plenoxels.models_zerorf.decoders.base_volume_renderer import PointBasedVolumeRenderer
# from plenoxels.models_zerorf.zerorf.generators import TensorialGenerator, CubemapGenerator

# from plenoxels.models_all.zerorf.generators import TensorialGenerator
# from plenoxels.models_all.zerorf.base_volume_renderer import VolumeRenderer
from plenoxels.models_all.ops import TruncExp, SHEncoder
from plenoxels.raymarching.spatial_distortions import SpatialDistortion
# from plenoxels.ops.activations import init_density_activation

from mmgen.models import MODULES, build_module
from mmcv.cnn import xavier_init, constant_init
import mcubes
import trimesh
import matplotlib.pyplot as plotlib
# from mmgen.models.builder import MODULES, build_module


class DepthRegularizer(nn.Module):
    noise: torch.Tensor

    def __init__(self, noise_ch, n_images, target_h, target_w, loss_weight=1.0) -> None:
        super().__init__()
        self.noise_ch, self.target_w, self.target_h = noise_ch, target_w, target_h
        self.n_images = n_images
        self.upx = 24
        self.register_buffer("noise", torch.randn(n_images, noise_ch, target_h // self.upx, target_w // self.upx))
        self.nets = nn.ModuleList([build_module(dict(
            type='VAEDecoder',
            in_channels=noise_ch,
            out_channels=8,
            up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'),
            block_out_channels=(32, 64, 128, 128),
            layers_per_block=0                                                                                                                                                                                                                                                  
        )) for _ in range(n_images)])
        self.net_needs_jit = True
        self.ups = nn.UpsamplingBilinear2d((target_h, target_w))
        self.head = nn.Sequential(
            nn.Linear(8, 32),
            nn.SiLU(True),
            nn.Linear(32, 1)
        )
        self.crit = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, results):
        sample_inds = results['sample_inds']
        if sample_inds is None:
            return results['depth'].new_zeros(1)
        if self.net_needs_jit:
            for i in range(self.n_images):
                self.nets[i] = torch.jit.trace(self.nets[i], self.noise[i: i + 1])
            self.net_needs_jit = False
        with torch.jit.optimized_execution(False):
            depth_features = torch.cat([self.nets[i](self.noise[i: i + 1]) for i in range(self.n_images)])
        ups_features = self.ups(depth_features).reshape(self.n_images, 8, self.target_h * self.target_w).permute(0, 2, 1).flatten(0, 1)
        ups_features = ups_features[sample_inds]
        depths = self.head(ups_features)
        return self.loss_weight * self.crit(results['depth'].reshape_as(depths), depths)


def positional_encoding(positions, freqs):    
    freq_bands = (2 ** torch.arange(freqs, device=positions.device, dtype=torch.float32))  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(*positions.shape[:-1], freqs * positions.shape[-1])  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class CommonDecoder(nn.Module):

    def __init__(self, point_channels, sh_coef_only=False, dir_pe=False, sdf_mode=False):
        super().__init__()
        
        self.sh_coef_only = sh_coef_only
        self.dir_pe = dir_pe
        
        # self.dir_encoder = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "SphericalHarmonics",
        #         "degree": 3,
        #     },
        # )  
        self.dir_encoder = SHEncoder(degree=3) # FIXME:
        self.base_net = nn.Linear(point_channels, 64)
        self.base_activation = nn.SiLU()
        if sdf_mode:
            self.variance = nn.Parameter(torch.tensor(0.3))
            self.variance_act = TruncExp()#init_density_activation('trunc_exp') #TruncExp()
            self.density_net = nn.Linear(64, 1)
        else:
            self.density_net = nn.Sequential(
                nn.Linear(64, 1),
                # init_density_activation('trunc_exp') # TruncExp()
                TruncExp()
            )
        if self.sh_coef_only:
            self.dir_net = None
            self.color_net = nn.Linear(64, 27)
        else:
            if dir_pe:
                self.pe = 5
                self.dir_net = nn.Linear(self.pe * 6, 64)
            else:
                self.dir_net = nn.Linear(9, 64)
            self.color_net = nn.Sequential(
                nn.Linear(64, 3),
                nn.Sigmoid()
            )
            self.color_net_wodir = nn.Sequential(
                nn.Linear(64, 3),
                nn.Sigmoid()
            )
        self.sigmoid_saturation = 0.001
        self.interp_mode = 'bilinear'
        self.sdf_mode = sdf_mode
    
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        if self.dir_net is not None:
            constant_init(self.dir_net, 0)

    def forward(self, point_code, dirs, out_sdf=False):
        """
        Args:
            point_code: torch.Tensor, shape:(n_rays*n_samples, point_channels)
            dirs:       torch.Tensor, shape:(n_rays*n_samples, 3)
        """
        base_x = self.base_net(point_code)
        base_x_act = self.base_activation(base_x)
        sigmas = self.density_net(base_x_act).squeeze(-1)
        if self.sdf_mode:
            if not out_sdf:
                s = self.variance_act(10 * self.variance).clamp(1e-6, 1e6)
                cov = self.variance_act(-s * sigmas).clamp(1e-6, 1e6)
                sigmas = s * cov / (1 + cov) ** 2
        if dirs is None:
            rgbs = None
        else:
            # dirs = torch.cat(dirs, dim=0) if len(dirs) > 1 else dirs[0]  # num_scenes*3?
            if self.sh_coef_only:
                sh_enc = self.dir_encoder(dirs).to(base_x.dtype)
                coefs = self.color_net(base_x_act).reshape(*base_x_act.shape[:-1], 9, 3)
                rgbs = torch.relu(0.5 + (coefs * sh_enc[..., None]).sum(-2))
            else:
                if self.dir_pe:
                    sh_enc = positional_encoding(dirs, self.pe).to(base_x.dtype)
                else:
                    sh_enc = self.dir_encoder(dirs).to(base_x.dtype)  #@note 也用了SH encode dir, shape:(n_rays*n_samples, sh_degree**2=9)
                # breakpoint()
                # print(base_x.shape, sh_enc.shape)
                color_in_nodir = self.base_activation(base_x)
                color_in = self.base_activation(base_x + self.dir_net(sh_enc))
                rgbs_nodir = self.color_net_wodir(color_in_nodir)
                rgbs = 0.8 * self.color_net(color_in) + 0.2 * rgbs_nodir
                if self.sigmoid_saturation > 0:
                    rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        return sigmas, rgbs


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):  # pts从aabb缩放至[-1,1]
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


@MODULES.register_module()
class TensorialDecoder(nn.Module):  #@note
    # preprocessor: TensorialGenerator

    def __init__(self, *args, separate_density_and_color: bool, in_ch, subreduce, reduce, image_h, image_w, tensor_config, 
                aabb,
                spatial_distortion: Optional[SpatialDistortion],
                sub_shapes,    # need to change
                ms_decode=False,
                code_mode="mul",
                sh_coef_only=False, sdf_mode=False, 
                occlusion_culling_th = 1e-4,
                 **kwargs):
        # super().__init__(*args, preprocessor=preprocessor, **kwargs)
        super().__init__()
        # assert isinstance(self.preprocessor, TensorialGenerator)
        if reduce == 'cat':
            in_chs = in_ch * len(tensor_config) // subreduce
        else:
            in_chs = in_ch
        
        # print(reduce,in_chs, len(tensor_config), tensor_config,subreduce)
        self.in_chs = in_chs
        self.separate_density_and_color = separate_density_and_color
        self.pe = 5
        if separate_density_and_color:
            self.density_decoder = CommonDecoder(in_chs // 2, sh_coef_only, sdf_mode=sdf_mode)
            self.color_decoder = CommonDecoder(in_chs // 2, sh_coef_only, sdf_mode=sdf_mode)
        else:
            if reduce == 'cat':
                self.common_decoder = CommonDecoder(in_chs, sh_coef_only, sdf_mode=sdf_mode)
            else:
                self.common_decoder = CommonDecoder(in_chs, sh_coef_only, sdf_mode=sdf_mode)
        self.subreduce = subreduce
        self.reduce = reduce
        self.ms_decode = ms_decode
        self.code_mode = code_mode
        self.sdf_mode = sdf_mode
        # self.visualize_mesh = visualize_mesh
        self.tensor_config = tensor_config
        self.hexplane = False
        for cfg in self.tensor_config:
            if 't' in cfg: 
                self.hexplane = True
                break
        self.occlusion_culling_th = occlusion_culling_th
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.sub_shapes = sub_shapes
        # self.agg_linear = nn.Linear(len(tensor_config), 1)
        # self.preprocessor_needs_jit = True
        # self.preprocessor_configured = self.preprocessor

    # def preproc(self, code):
    #     with torch.jit.optimized_execution(False):
    #         return super().preproc(code)
    
    def get_point_code(self, code, xyzs):
        # preprocessor = self.preprocessor_configured
        # if self.preprocessor_needs_jit:
        #     self.preprocessor_needs_jit = False
        #     self.preprocessor = torch.jit.trace(self.preprocessor, self.code_buffer)  # jit编译器相关部分, 和梯度反传或者参数更新无关
        
        # interpolation
        sub_shapes = self.sub_shapes
        codes = []
        codes_posedetach = []
        for i, cfg in enumerate(self.tensor_config):
            start = sum(map(numpy.prod, sub_shapes[:i]))
            end = sum(map(numpy.prod, sub_shapes[:i + 1]))
            got: torch.Tensor = code[..., start: end].reshape(code.shape[0], *sub_shapes[i])  # lu:(1, out_ch, *noise_res)
            assert len(cfg) + 2 == got.ndim, [len(cfg), got.ndim]
            if got.ndim == 3:
                got = got.unsqueeze(-1)
                cfg += 'x'  # TODO: 'z','x','y' -> 'zx','xx','yx', why?
            coords = xyzs[..., ['xyzt'.index(axis) for axis in cfg]]
            coords = coords.reshape(code.shape[0], 1, xyzs.shape[-2], 2)
            codes.append(
                F.grid_sample(got, coords, mode='bilinear', padding_mode='border', align_corners=True)  #TODO: align_corners=False
                .reshape(code.shape[0], got.shape[1], xyzs.shape[-2]).transpose(1, 2)  # (1, num_points_per_scene, out_ch)
            )
            codes_posedetach.append(
                F.grid_sample(got, coords.detach(), mode='bilinear', padding_mode='border', align_corners=True)  #TODO: align_corners=False
                .reshape(code.shape[0], got.shape[1], xyzs.shape[-2]).transpose(1, 2)  # (1, num_points_per_scene, out_ch)
            )
        
        """
        # get kplanes feats, concat(feat_xy * feat_yz * feat_zx, feat_xt * feat_yt * feat_zt)
        codes_subred = []
        codes_stage = None
        for i, c in enumerate(codes):
            if codes_stage is None:
                codes_stage = c
            else:
                if self.code_mode == "mul":
                    codes_stage = codes_stage * c
                else:
                    assert self.code_mode == "add"
                    codes_stage = codes_stage + c
            if i % self.subreduce == self.subreduce - 1:
                codes_subred.append(codes_stage)
                codes_stage = None
        codes = codes_subred
        """

        #@todo
        assert self.code_mode == "mul"
        if self.code_mode == "mul":
            codes = [(codes_posedetach[0] * codes_posedetach[1] * codes[2] + codes_posedetach[0] * codes_posedetach[2] * codes[1] + codes_posedetach[1] * codes_posedetach[2] * codes[0])/3.0]
            # ?
            # codes = [codes[0] * codes[1] * codes[2]]

            # codes = torch.cat([codes_posedetach[0] * codes_posedetach[1] * codes[2], codes_posedetach[0] * codes_posedetach[2] * codes[1], codes_posedetach[1] * codes_posedetach[2] * codes[0]], dim=0)  #(3, n, out_ch)
            # codes = [self.agg_linear(codes.permute(1,2,0).reshape(-1,3)).reshape(1, xyzs.shape[-2], -1)]  # [(1, n, out_ch)]

            # codes = [codes[0].data*codes[1].data*codes[2] + codes[0].data*codes[1]*codes[2].data + codes[0]*codes[1].data*codes[2].data]
            
            # codes = [0.001*codes[0].data*codes[1].data*codes[2] + 0.001*codes[0].data*codes[1]*codes[2].data + 0.001*codes[0]*codes[1].data*codes[2].data]
            
            # codes = [codes[0]+codes[1]+codes[2]]

        # if not self.separate_density_and_color and self.reduce == 'cat':
        #     codes.append(positional_encoding(xyzs.reshape(code.shape[0], xyzs.shape[-2], 3), self.pe))
        if self.reduce == 'cat':
            try:
                return torch.cat(codes, dim=-1).reshape(code.shape[0] * xyzs.shape[-2], self.in_chs)
            except RuntimeError:
                import ipdb; ipdb.set_trace()
        else:
            assert self.reduce == 'sum'
            return sum(codes).reshape(code.shape[0] * xyzs.shape[-2], self.in_chs)

    def get_ms_point_code(self, code, xyzs):
        """_summary_

        Args:
            code (list): multiscale codes from generator, k (level (1, ch, res, res))
            xyzs torch.tensor: _description_

        Returns:
            _type_: _description_
        """
        # interpolation
        # sub_shapes = self.sub_shapes
        # codes = []
        ms_codes = [] if self.reduce == 'cat' else 0.
        num_levels = len(code[0])
        # field_lr=[1e-2,5e-3,1e-3,5e-4]
        for l in range(num_levels):  # code[l] res low 2 high
            interp_space = 1. if self.code_mode=="mul" else 0.
            for k, cfg in enumerate(self.tensor_config):  # planes
                got: torch.Tensor = code[k][l]  # lu:(1, out_ch, *noise_res)
                coords = xyzs[..., ['xyzt'.index(axis) for axis in cfg]]
                coords = coords.reshape(1, 1, xyzs.shape[-2], 2)
                interp_out_plane = (
                    F.grid_sample(got, coords, mode='bilinear', padding_mode='border', align_corners=True)  #TODO: align_corners=False
                    .reshape(1, got.shape[1], xyzs.shape[-2]).transpose(1, 2)  # (1, num_points_per_scene, out_ch)
                )
                if self.code_mode == "mul":
                    interp_space = interp_space * interp_out_plane
                else:
                    assert self.code_mode == "add"
                    interp_space = interp_space + interp_out_plane
            if self.reduce == 'cat':
                ms_codes.append((0.05**(num_levels-1-l)) * interp_space)  #@todo
            else:
                assert self.reduce == 'sum'
                ms_codes = ms_codes + interp_space
                
        if self.reduce == 'cat':
            # breakpoint()
            ms_codes = torch.cat(ms_codes, dim=-1).reshape(xyzs.shape[-2], self.in_chs)

        return ms_codes

    def point_decode(self, pts, dirs, code, density_only=False):
        """
        Args:
            pts:    torch.Tensor, xyz(t), Shape: (n_rays * n_samples, 4)
            dirs:   torch.Tensor, dir_norm, Shape: (n_rays * n_samples, 3)
            code:   Shape (num_scenes=1, len(subs) * self.out_ch * noise_res**2)
            sub_shapes: of preprocessor
        """
        # num_scenes = code.size(0)  # num_scenes=1
        assert isinstance(pts, torch.Tensor), pts
        
        # assert pts.dim() == 3
        # num_points = pts.shape[0] #xyzs.size(-2)

        if self.ms_decode:  # feats
            point_code = self.get_ms_point_code(code, pts)  # (n_rays * n_samples, self.in_chs = len(subs)//subreduce*out_ch)
        else:  # density fields
            point_code = self.get_point_code(code, pts)  # (n_rays * n_samples, self.in_chs = len(subs)//subreduce*out_ch)

        # num_points = [num_points] * num_scenes
        sigmas, rgbs = self.point_code_render(point_code, dirs)
        return sigmas, rgbs#, num_points
    
    def point_decode_precomputed(self, pts, dirs, precomputed_code, density_only=False):
        """
        Args:
            pts:    torch.Tensor, xyz(t), Shape: (n_rays * n_samples, 4)
            dirs:   torch.Tensor, dir_norm, Shape: (n_rays * n_samples, 3)
            # code:   Shape (num_scenes=1, len(subs) * self.out_ch * noise_res**2)
            precomputed_code:  Shape (num_scenes=1, len(subs) * self.out_ch * noise_res**2)
            sub_shapes: of preprocessor
        """
        # num_scenes = code.size(0)  # num_scenes=1
        assert isinstance(pts, torch.Tensor), pts
        
        # assert pts.dim() == 3
        num_points = pts.shape[1] #xyzs.size(-2)
        # point_code = self.get_point_code(code, pts)  # (n_rays * n_samples, self.in_chs = len(subs)//subreduce*out_ch)
        # num_points = [num_points] * num_scenes
        sigmas, rgbs = self.point_code_render(precomputed_code, dirs)
        return sigmas, rgbs, num_points

    def point_density_decode(self, pts, timestamps, code, **kwargs):
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

        # print(code.shape)
        point_code = self.get_point_code(code, pts)  # (n_rays * n_samples, self.in_chs = len(subs)//subreduce*out_ch)
        sigmas, rgbs = self.point_code_render(point_code, None)
        sigmas = sigmas.to(pts).view(n_rays, n_samples, 1)
        
        return sigmas
    def point_density_decode_precomputed_code(self, pts, timestamps, precomputed_code, **kwargs):
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

        # print(code.shape)
        # point_code = self.get_point_code(code, pts)  # (n_rays * n_samples, self.in_chs = len(subs)//subreduce*out_ch)
        sigmas, rgbs = self.point_code_render(precomputed_code, None)
        sigmas = sigmas.to(pts).view(n_rays, n_samples, 1)
        
        return sigmas

    def point_code_render(self, point_code, dirs):
        if self.separate_density_and_color:
            density_code, color_code = torch.chunk(point_code, 2, -1)  # both (num_scenes*num_points_per_scene, self.in_chs/2)
            sigmas, _ = self.density_decoder(density_code, None)
            if dirs is not None:
                _, rgbs = self.color_decoder(color_code, dirs)
            else:
                rgbs = None
            return sigmas, rgbs
        else:
            return self.common_decoder(point_code, dirs)
        
    def forward(self,
                ray_samples,
                # pts: torch.Tensor,
                code: Optional[torch.Tensor],  #torch.Tensor,
                # sub_shapes,
                directions: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                # code: Optional[torch.Tensor] = None,
                ): #FIXME:
        pts = ray_samples.get_positions()
        
        # density, features = self.get_density(pts, timestamps, global_step)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)  # (n_rays, n_samples, 3)
            
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)    # normalize pts to [-1, 1]
        if timestamps is not None:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])          # [n_rays*n_samples, 3 or 4]
        
        
        # with occl, debuging @todo
        with torch.no_grad():
            sigmas, rgbs = self.point_decode(pts, None, code)  #self.point_decode(pts, None, code)
            
            sigmas = sigmas.to(directions)
        rgbs = sigmas.new_zeros(sigmas.shape[0], 3)
        with torch.no_grad():
            weights = ray_samples.get_weights(sigmas.view(n_rays, n_samples, -1))  # (n_rays, n_samples, 1)
            weights = weights.view(-1)
        occl = weights > self.occlusion_culling_th  # (n_rays*n_samples)
        pts_filt = pts[occl]
        directions_filt = directions[occl]
        sigmas_occl, rgbs_occl = self.point_decode(pts_filt, directions_filt, code)  #self.point_decode(xyzs_filt, dirs_filt, code)
        
        sigmas = torch.zeros_like(sigmas)
        # occl = torch.cat(occls_filt)
        sigmas[occl] = sigmas_occl.to(sigmas)
        rgbs[occl] = rgbs_occl.to(rgbs)
        
        
        sigmas, rgbs = self.point_decode(pts, directions, code)  #self.point_decode(pts, None, code)
        
        sigmas = sigmas.to(directions).view(n_rays, n_samples, 1)
        rgbs = rgbs.to(directions).view(n_rays, n_samples, 3)


        return {"rgb": rgbs, "density": sigmas}
    

    def render_with_precompute_code(self,
                ray_samples,
                # pts: torch.Tensor,
                # code: Optional[torch.Tensor],  #torch.Tensor,
                precomputed_pointcode: Optional[torch.Tensor],  #torch.Tensor,
                # sub_shapes,
                directions: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None,
                # code: Optional[torch.Tensor] = None,
                ): #FIXME:
        pts = ray_samples.get_positions()
        
        # density, features = self.get_density(pts, timestamps, global_step)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)  # (n_rays, n_samples, 3)
            
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)    # normalize pts to [-1, 1]
        if timestamps is not None:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])          # [n_rays*n_samples, 3 or 4]
        
        
        # # with occl, debuging @todo
        # with torch.no_grad():
        #     sigmas, rgbs, num_points = self.point_decode_precomputed(pts, None, precomputed_pointcode)  #self.point_decode(pts, None, code)
            
        #     sigmas = sigmas.to(directions)
        # rgbs = sigmas.new_zeros(sigmas.shape[0], 3)
        # with torch.no_grad():
        #     weights = ray_samples.get_weights(sigmas.view(n_rays, n_samples, -1))  # (n_rays, n_samples, 1)
        #     weights = weights.view(-1)
        # occl = weights > self.occlusion_culling_th  # (n_rays*n_samples)
        # pts_filt = pts[occl]
        # directions_filt = directions[occl]
        # # breakpoint()
        # sigmas_occl, rgbs_occl, _ = self.point_decode_precomputed(pts_filt, directions_filt, precomputed_pointcode[occl])  #self.point_decode(xyzs_filt, dirs_filt, code)
        
        # sigmas = torch.zeros_like(sigmas)
        # # occl = torch.cat(occls_filt)
        # sigmas[occl] = sigmas_occl.to(sigmas)
        # rgbs[occl] = rgbs_occl.to(rgbs)
        
        
        sigmas, rgbs, _ = self.point_decode_precomputed(pts, directions, precomputed_pointcode)  #self.point_decode(pts, None, code)
        
        sigmas = sigmas.to(directions).view(n_rays, n_samples, 1)
        rgbs = rgbs.to(directions).view(n_rays, n_samples, 3)


        return {"rgb": rgbs, "density": sigmas}

    @torch.no_grad()
    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        if 'mean' in str(scene_name):
            return
        buffer = self.code_proc_buffer.to(code)
        preprocessor = self.preprocessor_configured
        for i, cfg in enumerate(preprocessor.tensor_config):
            start = sum(map(numpy.prod, preprocessor.sub_shapes[:i]))
            end = sum(map(numpy.prod, preprocessor.sub_shapes[:i + 1]))
            got: torch.Tensor = buffer[..., start: end].reshape(code.shape[0], *preprocessor.sub_shapes[i])
            assert len(cfg) + 2 == got.ndim, [len(cfg), got.ndim]
            if got.ndim == 3:
                got = got.unsqueeze(-1)
            got = got.permute(0, 2, 3, 1).squeeze(0).transpose(-1, -2).flatten(1)  # HWC -> HW'
            got = got.detach().cpu().numpy()
            plotlib.imsave(os.path.join(viz_dir, "features-%s.png" % cfg), got)
        if not self.visualize_mesh:
            return
        basis = torch.linspace(-1, 1, 500, device=code.device, dtype=code.dtype)
        x, y, z = torch.meshgrid(basis, basis, basis, indexing='ij')
        xyz = torch.stack([x, y, z], dim=-1)
        xyz_f = xyz.flatten(0, -2)
        batches = []
        for batch_split in torch.split(xyz_f, 2 ** 17):
            pc = self.get_point_code(buffer, batch_split)
            if self.separate_density_and_color:
                density_code, color_code = torch.chunk(pc, 2, -1)
                sdf = self.density_decoder(density_code, None, True)[0]
            else:
                sdf = self.common_decoder(pc, None, True)[0]
            if self.sdf_mode:
                batches.append(sdf.cpu())
            else:
                batches.append((sdf > 5).cpu())
        batches = torch.cat(batches, 0).reshape(xyz.shape[:-1])
        if not self.sdf_mode:
            field = mcubes.smooth_constrained(batches.numpy())
            verts, tris = mcubes.marching_cubes(field, 0.5)
        else:
            verts, tris = mcubes.marching_cubes(batches.numpy(), 0.0)
        verts = verts / 250 - 1
        # mcubes.smooth()
        trimesh.Trimesh(verts, tris).export(os.path.join(viz_dir, str(scene_name[0]) + ".glb"))


@MODULES.register_module()
class FreqFactorizedDecoder(TensorialDecoder):

    def __init__(self, *args, freq_bands, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq_bands = freq_bands

    def get_point_code(self, code, xyzs):
        preprocessor = self.preprocessor_configured
        if self.preprocessor_needs_jit:
            self.preprocessor_needs_jit = False
            self.preprocessor = torch.jit.trace(self.preprocessor, self.code_buffer)
        codes = []
        for i, (cfg, band) in enumerate(zip(preprocessor.tensor_config, self.freq_bands)):
            start = sum(map(numpy.prod, preprocessor.sub_shapes[:i]))
            end = sum(map(numpy.prod, preprocessor.sub_shapes[:i + 1]))
            got: torch.Tensor = code[..., start: end].reshape(code.shape[0], *preprocessor.sub_shapes[i])
            assert len(cfg) + 2 == got.ndim == 5, [len(cfg), got.ndim]
            coords = xyzs[..., ['xyzt'.index(axis) for axis in cfg]]
            if band is not None:
                coords = ((coords % band) / (band / 2) - 1)
            coords = coords.reshape(code.shape[0], 1, 1, xyzs.shape[-2], 3)
            codes.append(
                F.grid_sample(got, coords, mode='bilinear', padding_mode='border', align_corners=False)
                .reshape(code.shape[0], got.shape[1], xyzs.shape[-2]).transpose(1, 2)
            )
        codes_subred = []
        codes_stage = None
        for i, c in enumerate(codes):
            if codes_stage is None:
                codes_stage = c
            else:
                codes_stage = codes_stage * c
            if i % self.subreduce == self.subreduce - 1:
                codes_subred.append(codes_stage)
                codes_stage = None
        codes = codes_subred
        # if not self.separate_density_and_color and self.reduce == 'cat':
        #     codes.append(positional_encoding(xyzs.reshape(code.shape[0], xyzs.shape[-2], 3), self.pe))
        if self.reduce == 'cat':
            return torch.cat(codes, dim=-1).reshape(code.shape[0] * xyzs.shape[-2], self.in_chs)
        else:
            assert self.reduce == 'sum'
            return sum(codes).reshape(code.shape[0] * xyzs.shape[-2], self.in_chs)

    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        if 'mean' in str(scene_name):
            return
        if not self.visualize_mesh:
            return
        buffer = self.code_proc_buffer.to(code)
        basis = torch.linspace(-1, 1, 500, device=code.device, dtype=code.dtype)
        x, y, z = torch.meshgrid(basis, basis, basis, indexing='ij')
        xyz = torch.stack([x, y, z], dim=-1)
        xyz_f = xyz.flatten(0, -2)
        batches = []
        for batch_split in torch.split(xyz_f, 2 ** 17):
            pc = self.get_point_code(buffer, batch_split)
            if self.separate_density_and_color:
                density_code, color_code = torch.chunk(pc, 2, -1)
                sdf = self.density_decoder(density_code, None, True)[0]
            else:
                sdf = self.common_decoder(pc, None, True)[0]
            if self.sdf_mode:
                batches.append(sdf.cpu())
            else:
                batches.append((sdf > 5).cpu())
        batches = torch.cat(batches, 0).reshape(xyz.shape[:-1])
        if not self.sdf_mode:
            field = mcubes.smooth_constrained(batches.numpy())
            verts, tris = mcubes.marching_cubes(field, 0.5)
        else:
            verts, tris = mcubes.marching_cubes(batches.numpy(), 0.0)
        verts = verts / 250 - 1
        # mcubes.smooth()
        trimesh.Trimesh(verts, tris).export(os.path.join(viz_dir, str(scene_name[0]) + ".glb"))
