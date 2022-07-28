from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .hourglass import *
from .submodule import *


class PSMNet(nn.Module):
    def __init__(self, params):
        super(PSMNet, self).__init__()

        self.maxdisp = int(params['maxdisp'])
        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right, train_opt: bool = None):

        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)

        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).type_as(refimg_fea)

        for i in range(self.maxdisp//4):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3 = self.dres4(out2, pre1, post2)[0]
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if train_opt:
            cost1 = F.interpolate(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
            cost2 = F.interpolate(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.interpolate(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if train_opt:
            return pred1, pred2, pred3
        else:
            return pred3, cost3
