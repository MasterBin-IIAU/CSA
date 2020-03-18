# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        self.center_size = center_size
        # print('AdjustLayer generated.')

    def forward(self, x):
        x = self.downsample(x) # reduce # of channels
        '''the spatial size of template is 15, so it'll be reduced to 7'''
        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        # print(x.size())
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustAllLayer, self).__init__()
        # self.num = 3
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0],
                                          out_channels[0],
                                          center_size)
        else:
            for i in range(self.num):
                '''add_module method: add a series layers to a nn.Module object.
                This operation is analogous to nn.sequential(),but this can give each sub-module a name!'''
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i],
                                            out_channels[i],
                                            center_size))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                '''getattr: get an object's attribute value.'''
                adj_layer = getattr(self, 'downsample'+str(i+2))
                # Template (1,512,15,15),(1,1024,15,15),(1,2048,15,15) ---> 3x (1,256,7,7)
                out.append(adj_layer(features[i]))
            return out
