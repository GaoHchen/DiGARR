import torch
import torch.nn as nn
from models.cal_pose import make_c2w
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
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)
        log.info(f"Initialized Pose Net for {self.num_cams} cam poses:\n{self.init_c2w}\n{self.r}\n{self.t}\n")

    def forward(self, cam_id):  #@todo
        """
        cam_id : torch.Tensor, [n_rays or 1] 
        """
        # cam_id = int(cam_id)
        # c2ws = []
        # for i in cam_id:
        #     r = self.r[i]  # (3, ) axis-angle
        #     t = self.t[i]  # (3, )
        #     c2w = make_c2w(r, t)  # (4, 4)
        # # learn a delta pose between init pose and target pose, if a init pose is provided
        #     if self.init_c2w is not None:
        #         c2w = c2w @ self.init_c2w[i]
        #     c2ws.append(c2w)
        # c2ws = torch.stack(c2ws)  # [n_rays, 4, 4]
        
        r = self.r[cam_id]  # (n_rays, 3)
        t = self.t[cam_id]  # (n_rays, 3)
        c2ws = make_c2w(r, t)  # (n_rays, 3, 4)
        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:  #@todo
            c2ws = c2ws @ self.init_c2w[cam_id]
        return c2ws
    
    def get_t(self):
       return self.t
    
    def get_params(self):
        pose_params = {k: v for k, v in self.named_parameters()}
        return {
            "pose": list(pose_params.values()),
        }

   
    
    

