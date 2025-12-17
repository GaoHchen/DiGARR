import torch
import torch.nn as nn
from models_all.cal_pose import make_c2w
from utils_poses.lie_group_helper import convert3x4_4x4
import logging as log
from utils.camera_barf import Lie, Pose


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, cfg=None, init_c2w=None, camera_noise=0.0):
        """
        :param num_cams: total n_cams (include train and test splits)
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.Lie = Lie()
        self.Pose = Pose()
        self.init_c2w = None
        self.camera_noise = camera_noise

        if self.camera_noise != 0.:
            se3_noise = torch.randn(size=(num_cams, 6),device="cuda") * self.camera_noise
            self.pose_noise = self.Lie.se3_to_SE3(se3_noise)  # n,3,4

        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)#.to(device)
            if self.camera_noise != 0.: self.init_c2w = nn.Parameter(self.Pose.compose([self.pose_noise, init_c2w[:,:3,:].to('cuda')]),requires_grad=False)
            
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)#.to(device)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)#.to(device)  # (N, 3)

        log.info(f"Initialized Pose Net for {self.num_cams} cam poses on {self.r.device}")

    def forward(self, cam_id):  #@todo
        """
        cam_id : torch.Tensor, [n_rays] 
        """
        r = self.r[cam_id]  # (n_rays, 3)
        t = self.t[cam_id]  # (n_rays, 3)
        c2ws = make_c2w(r, t)  # (n_rays, 3, 4)
        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            try:
                c2ws = c2ws @ convert3x4_4x4(self.init_c2w[cam_id])
                # c2ws = c2ws @ self.init_c2w[cam_id]
            except RuntimeError:
                c2ws = c2ws @ self.init_c2w[cam_id]
            # breakpoint()
        # c2ws[:,0,:] = c2ws[:,0,:] * -1  #@todo
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
        return [
            {"params": list([self.r, self.t]), "lr": eval_pose_lr}
        ]
    
     
class LearnPose_synthetic(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, cfg=None, init_c2w=None, camera_noise=0.0):
        """
        initrt directly
        :param num_cams: total n_cams (include train and test splits)
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose_synthetic, self).__init__()
        self.num_cams = num_cams
        self.Lie = Lie()
        self.Pose = Pose()
        self.init_c2w = None
        self.camera_noise = camera_noise
        self.pose_noise = None
        self.noise_r = 0.07
        self.noise_t = 0.3

        
        if self.camera_noise != 0.:
            # print(num_cams)
            # se3_noise = torch.randn(size=(num_cams, 6),device="cuda") * self.camera_noise
            # self.pose_noise = self.Lie.se3_to_SE3(se3_noise)  # n,3,4
            self.rot_noise = torch.randn(size=(num_cams, 3),device="cuda") * self.noise_r
            self.t_noise = torch.randn(size=(num_cams, 3),device="cuda") * self.noise_t
            self.pose_noise = torch.cat([self.rot_noise, self.t_noise], dim=-1)
            self.pose_noise = self.Lie.se3_to_SE3(self.pose_noise)  # n,3,4
            # self.pose_noise = make_c2w(se3_noise[..., :3], se3_noise[..., 3:])  # n,4,4
        

        # if init_c2w is not None:
        #     self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)#.to(device)
        #     if self.camera_noise != 0.: self.init_c2w = nn.Parameter(self.Pose.compose([self.pose_noise[:,:3,:], init_c2w[:,:3,:].to('cuda')]),requires_grad=False)
        
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)#.to(device)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)#.to(device)  # (N, 3)
        
        if init_c2w is not None:
            if self.pose_noise is not None:
                init_c2w = self.Pose.compose([self.pose_noise, init_c2w[:,:3,:].to('cuda')])
            else:
                cam_ids = torch.tensor([0,100]) if num_cams > 100 else torch.tensor([0])
                se3_noise = torch.randn(size=(cam_ids.shape[0], 6),device="cuda") * 0.01
                pose_noise = self.Lie.se3_to_SE3(se3_noise)  # n,3,4
                init_c2w[cam_ids,:3,:] = self.Pose.compose([pose_noise, init_c2w[cam_ids,:3,:].to('cuda')])
            # self.r.data = Log(init_c2w[:,:3,:3].to('cuda')).data
            # self.t.data = init_c2w[:,:3,3].to('cuda').data
            log.info(init_c2w[0,:3,:])
            rt = self.Lie.SE3_to_se3(init_c2w[:,:3,:].to('cuda'))
            log.info(rt[0,:])
            self.r.data = rt[..., :3].data
            self.t.data = rt[..., 3:].data
            del rt

        log.info(f"Initialized Pose Net for {self.num_cams} cam poses on {self.r.device}:\n{self.init_c2w}, {self.r}, {self.t}")

    def forward(self, cam_id):  #@todo
        """
        cam_id : torch.Tensor, [n_rays] 
        """
        r = self.r[cam_id]  # (n_rays, 3)
        t = self.t[cam_id]  # (n_rays, 3)
        
        # c2ws = make_c2w(r, t)  # (n_rays, 4, 4)
        
        c2ws = self.Lie.se3_to_SE3(torch.cat([r,t], dim=-1))  # (n_rays, 3, 4)
        c2ws = convert3x4_4x4(c2ws)
        
        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            try:
                c2ws = c2ws @ convert3x4_4x4(self.init_c2w[cam_id])
                # c2ws = c2ws @ self.init_c2w[cam_id]
            except RuntimeError:
                c2ws = c2ws @ self.init_c2w[cam_id]
        
            # breakpoint()
        # c2ws[:,0,:] = c2ws[:,0,:] * -1  #@todo
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
        return [
            {"params": list([self.r, self.t]), "lr": eval_pose_lr}
        ]

    
    

