import torch
import torch.nn as nn
from models_all.cal_pose import make_c2w
import logging as log


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, cfg=None, init_c2w=None):
        """
        :param num_cams: total n_cams (include train and test splits)
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)#.to(device)
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)#.to(device)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)#.to(device)  # (N, 3)

        log.info(f"Initialized Pose Net for {self.num_cams} cam poses on {self.r.device}:\n{self.init_c2w}, {self.r}, {self.t}")

    def forward(self, cam_id):  #@todo
        """
        cam_id : torch.Tensor, [n_rays] 
        """
        r = self.r[cam_id]  # (n_rays, 3)
        t = self.t[cam_id]  # (n_rays, 3)
        c2ws = make_c2w(r, t)  # (n_rays, 3, 4)
        # learn a delta pose between init pose and target pose, if a init pose is provided
        try:
            if self.init_c2w is not None:  
                c2ws = c2ws @ self.init_c2w[cam_id]
        except RuntimeError:
            breakpoint()
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
    

   
    
    

