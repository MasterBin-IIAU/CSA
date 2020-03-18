# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.utils.model_load import load_pretrain
from common_path import project_path_

'''Capsule SiamRPN++(We can use it as one component in higher-level task)'''
class SiamRPNPP():
    def __init__(self,dataset=''):
        if 'OTB' in dataset:
            cfg_file = os.path.join(project_path_,'pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/config.yaml')
            snapshot = os.path.join(project_path_,'pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/model.pth')
        elif 'LT' in dataset:
            cfg_file = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
            snapshot = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/model.pth')
        else:
            cfg_file = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
            snapshot = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth')
        # load config
        cfg.merge_from_file(cfg_file)
        # create model
        self.model = ModelBuilder()# A Neural Network.(a torch.nn.Module)
        # load model
        self.model = load_pretrain(self.model, snapshot).cuda().eval()

    def get_heat_map(self, X_crop, softmax=False):
        score_map = self.model.track(X_crop)['cls']#(N,2x5,25,25)
        score_map = score_map.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)#(5HWN,2)
        if softmax:
            score_map = F.softmax(score_map, dim=1).data[:, 1]#(5HWN,)
        return score_map
    def get_cls_reg(self, X_crop, softmax=False):
        outputs = self.model.track(X_crop)#(N,2x5,25,25)
        score_map = outputs['cls'].permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)#(5HWN,2)
        reg_res = outputs['loc'].permute(1, 2, 3, 0).contiguous().view(4, -1)
        if softmax:
            score_map = F.softmax(score_map, dim=1).data[:, 1]#(5HWN,)
        return score_map, reg_res

