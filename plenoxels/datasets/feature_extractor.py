from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from einops import rearrange
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms

class FrozenDinoV2ImageEmbedder(nn.Module):

    def __init__(
            self,
            version='dinov2_vitb14',
            dinov2_dirx=None,
            img_size=224,
        ):
        super().__init__()
        assert version in ['dinov2_vitb14', 'dinov2_vits14', 'dinov2_vitl14', 'dinov2_vitg14'] and dinov2_dirx is not None
        #if model == 'vit_base':
        #    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') 
        #elif model == 'vit_small':
        #    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') 
        #elif model == 'vit_large':
        #    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') 
        #elif model == 'vit_giant':
        #    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') 
        #else:
        #    raise NotImplementedError('Invalid dinov2 backbone')
        self.model = torch.hub.load(dinov2_dirx, version, trust_repo=True, source='local', pretrained=False)
        # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.load_state_dict(torch.load(f'{dinov2_dirx}/{version}_pretrain.pth', map_location='cpu'))
        # self.model.patch_embed.proj.stride = int(img_size / 20)
        print('Load dinov2 encoder successfully')
        self.frozen()

    def frozen(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False


    def forward(self, x):
        # return self.model(self.preprocess(x))
        B,C,H,W = x.shape[:4]
        Nv = 1
        ret = self.model.forward_features(x)
        feature = torch.cat([ret['x_norm_clstoken'].unsqueeze(1), ret['x_norm_patchtokens']], dim=1)

        # reshape dino tokens to image-like size
        # feature = rearrange(feature, "B (Nv Nt) C -> (B Nv) Nt C", Nv=Nv)
        # feature = feature[:, 1:].reshape(B * Nv, H // 14, W // 14, -1).permute(0, 3, 1, 2).contiguous()
        # feature = F.interpolate(feature, size=(20, 20), mode='bilinear', align_corners=False)

        return feature
    
class DinoWrapper(nn.Module):
    def __init__(self,path) -> None:
        super().__init__()
        self.transform = torchvision.transforms.ToTensor()
        self.x = Image.open(path).convert('RGB')
        self.x = self.transform(self.x)  # (c,h,w)

        self.extractor = FrozenDinoV2ImageEmbedder(version="dinov2_vits14", 
                                                dinov2_dirx="/media/public/disk5/doublez/NeRF-Pose/DyNeRF/zeropose/plenoxels/datasets/dinov2/pretrained",
                                                img_size=max(self.x.shape[-1],self.x.shape[-2]))
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.eval()

    def forward(self):
        
        self.x = self.x.unsqueeze(0)  # (b,c,h,w)
        self.x = self.x.to(self.device)
        x = self.extractor(self.x)
        print("Dino Feature Extracted") 

        return x


if __name__ == "__main__":
    # transform = torchvision.transforms.ToTensor()
    # img = Image.open("/media/public/disk5/doublez/NeRF-Pose/DyNeRF/zeropose/image000.png").convert('RGB')
    # img = transform(img)
    # # breakpoint()
    # dino = FrozenDinoV2ImageEmbedder(version="dinov2_vits14", dinov2_dirx="/media/public/disk5/doublez/NeRF-Pose/DyNeRF/zeropose/plenoxels/datasets/dinov2/pretrained")
    # img = torch.rand(8,3,504,504)
    Dino = DinoWrapper("/media/public/disk5/doublez/NeRF-Pose/DyNeRF/zeropose/image000.png")
    out = Dino()
    breakpoint()
