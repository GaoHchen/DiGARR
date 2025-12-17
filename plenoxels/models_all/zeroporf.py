import torch
import torch.nn as nn
from models_all.cal_pose import make_c2w
import logging as log
import numpy as np
from utils.camera_barf import Lie, Pose
from utils_poses.lie_group_helper import convert3x4_4x4
import torch.nn.functional as F

from mmgen.models.builder import MODULES, build_module
from mmcv.cnn import xavier_init, constant_init


class ZeroPoRF(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, generator: dict(type='TensorialGenerator'), tensor_config, in_ch, subreduce=1, reduce='cat', camera_noise=0., init_c2w=None, hidden_unit=128, alpha=0.0001):
        """
        :param num_cams: total n_cams (include train and test splits)
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        # super(ZeroPoRF, self).__init__()
        super().__init__()
        self.Lie = Lie()
        self.Pose = Pose()
        self.num_cams = num_cams
        self.init_c2w = None
        self.camera_noise = camera_noise
        self.pos_enc_levels = 4#10
        self.tensor_config = tensor_config[:3]
        self.reduce = reduce
        self.subreduce = subreduce
        
        if reduce == 'cat':
            in_chs = in_ch * len(self.tensor_config) // subreduce
        else:
            in_chs = in_ch
        self.in_chs = in_chs

        if self.camera_noise != 0.:
            se3_noise = torch.randn(size=(num_cams, 6),device="cuda") * self.camera_noise
            self.pose_noise = self.Lie.se3_to_SE3(se3_noise)  # n,3,4

        if init_c2w is not None:    # if has ext, use it. or give noise
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)#.to(device) n,4,4
            #TODO: add pose noise, cleanup needed
            if self.camera_noise != 0.: self.init_c2w = nn.Parameter(self.Pose.compose([self.pose_noise, init_c2w[:,:3,:].to('cuda')]),requires_grad=False) # add noise
        # else:   # no ext given, use zero initialization
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=False)#.to(device)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=True)#.to(device)  # (N, 3)
            # self.init_c2w = make_c2w(self.r.data, self.t.data).to('cuda').detach()   # initial with llff

        """
        pos_in_dims = (2 * self.pos_enc_levels + 1) * 3  # (2L + 0 or 1) * 3
        # learn this thing
        self.porf = nn.Sequential(
            nn.Linear(pos_in_dims , hidden_unit),
            nn.ELU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.ELU(),
            # nn.Linear(hidden_unit, hidden_unit),
            # nn.ELU(),
            nn.Linear(hidden_unit, 3)
        )
        """
        
        self.preprocessor = build_module(generator)
        self.decoder = nn.Sequential(
            nn.Linear(self.in_chs, hidden_unit),
            nn.SiLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.SiLU(),
            nn.Linear(hidden_unit, 3)
        )
        self.init_weights()

        self.alpha = alpha
        # self.alpha_r = 0.001
        # self.alpha_t = 0.01

    def init_weights(self):
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                # nn.init.xavier_normal_(layer.weight)
                # xavier_init(layer, distribution='uniform')
                nn.init.kaiming_normal_(layer.weight)
                # nn.init.constant_(layer.weight, 0)
                # nn.init.constant_(layer.bias, 0)

    def one_hot_encode_pytorch(self, number, num_classes=11):
        return F.one_hot(torch.tensor([number]), num_classes=num_classes)
    
    def normalize_(self, cam_id, maxid):
        # breakpoint()
        return (torch.tensor(cam_id) / maxid).clamp_(0,1)

    def forward(self, cam_id):  #@todo
        """
        cam_id : torch.Tensor, [n_rays] 
        """
        # r = self.r[cam_id]  # (n_rays, 3)
        t = self.t[cam_id]  # (n_rays, 3)
        # # Rt = make_c2w(r.data, t.data).to('cuda')[:,:3,:] # get pose
        # # se3 = self.Lie.SE3_to_se3(Rt=Rt) # nrays,3,4->nrays,6
        # # id_embedded = self.normalize_(cam_id, self.num_cams)
        # # id_encoded = self.positional_encoding(id_embedded.unsqueeze(-1), L=4)
        # posenc = self.encode_position(input=t.data, levels=self.pos_enc_levels, inc_input=True)
        # # x = torch.cat([t.data, id_encoded.to(t.device)], dim=-1)
        # # concat r,t
        # # breakpoint()
        # # x = torch.cat([r.data, posenc], dim=-1)
        # # x = torch.cat([posenc], dim=-1)
        # # x = torch.cat([se3, id_embedded.unsqueeze(-1).to(se3.device)], dim=-1)
        # rot = self.porf(posenc) * self.alpha
        
        pose_code = self.preprocessor()
        codes = []
        for i,cfg in enumerate(self.tensor_config):
            got: torch.Tensor = pose_code[i]
            coords = t.data[..., ['xyz'.index(axis) for axis in cfg]]  #@remind t.data
            coords = coords.reshape(1, 1, t.shape[-2], 2)
            codes.append(
                F.grid_sample(got, coords, mode='bilinear', padding_mode='border', align_corners=True)  #@todo align_corners=False
                .reshape(1, got.shape[1], t.shape[-2]).transpose(1, 2)  # (1, n_rays, out_ch)
            )
        if self.reduce == 'cat':
            codes = torch.cat(codes, dim=-1).reshape(t.shape[-2], self.in_chs)
        
        rot = self.decoder(codes) * self.alpha

        # c2ws = self.Lie.se3_to_SE3(se3_res)
        c2ws = make_c2w(rot,t)
        if self.init_c2w is not None:
            try:
                c2ws = c2ws @ convert3x4_4x4(self.init_c2w[cam_id])
                # c2ws = c2ws @ self.init_c2w[cam_id]
            except RuntimeError:
                # breakpoint()
                cam_id = torch.tensor(cam_id, device=c2ws.device)
                c2ws = c2ws @ self.init_c2w[cam_id]
        return c2ws

    def get_t(self):
       return self.t
    
    def get_params(self):
        pose_params = {k: v for k, v in self.named_parameters()}
        return {
            "pose": list(pose_params.values()),
        }
    
    def get_eval_params(self, eval_pose_lr):
        # pose_params = {k: v for k, v in self.named_parameters()}
        # return [
        #     {"params": list(pose_params.values()), "lr": eval_pose_lr}
        # ]
        # return [
        #     {"params": list([self.r, self.t]), "lr": eval_pose_lr}
        # ] #FIXME:
    
        return [
            {"params": list([self.decoder]), "lr": eval_pose_lr}
        ]
    def get_params_seperate_lr(self, t_lr, porf_lr):
        porf_params = list(self.preprocessor.parameters()) + list(self.decoder.parameters())
        trans_params = [self.t]
        # return trans_params, porf_params
        return [
                {"params": porf_params, "lr": porf_lr},
                {"params": trans_params, "lr": t_lr},
        ]
        
    def get_params_seperate(self):
        porf_params = list(self.preprocessor.parameters()) + list(self.decoder.parameters())
        trans_params = [self.t]
        return trans_params, porf_params
    
    def positional_encoding(self, input, L):  # [B,...,N] 4
        shape = input.shape
        freq = (
            2 ** torch.arange(L, dtype=torch.float32, device=input.device)
            * np.pi
        )  # [L]
        spectrum = input[..., None] * freq  # [B,...,N,L]   pi, 2*pi
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        # breakpoint()
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]

        return input_enc
        
    def encode_position(self, input, levels, inc_input):
        """
        For each scalar, we encode it using a series of sin() and cos() functions with different frequency.
            - With L pairs of sin/cos function, each scalar is encoded to a vector that has 2L elements. Concatenating with
            itself results in 2L+1 elements.
            - With C channels, we get C(2L+1) channels output.

        :param input:   (..., C)            torch.float32
        :param levels:  scalar L            int
        :return:        (..., C*(2L+1))     torch.float32
        """

        # this is already doing 'log_sampling' in the official code.
        result_list = [input] if inc_input else []
        for i in range(levels):
            temp = 2.0**i * input  # (..., C)
            result_list.append(torch.sin(temp))  # (..., C)
            result_list.append(torch.cos(temp))  # (..., C)

        result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1)) The list has (2L+1) elements, with (..., C) shape each.
        return result_list  # (..., C*(2L+1))