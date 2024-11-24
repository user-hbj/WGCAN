import torch
import torch.nn as nn
from models.CBAM import *
# from CBAM import *
# 定义了通道引导模块，用于引导通道之间的信息传递和特征改进。这个模块包括了一系列卷积和通道注意力操作
class CGB(nn.Module):
    def __init__(self, nfeats = 16):
        super(CGB, self).__init__()

        self.nfeats = nfeats
       
        self.conv_feat_cof =nn.Sequential(nn.Conv2d(self.nfeats, self.nfeats, 3, 1, 1, bias=True), nn.PReLU(), \
            nn.Conv2d(self.nfeats, self.nfeats, 3, 1, 1, bias=True), nn.PReLU(), CBAMBlock(channel=self.nfeats, reduction=16, kernel_size=7),\
            nn.Conv2d(self.nfeats, self.nfeats*2, 3, 1, 1, bias=True), nn.PReLU())
        
        self.conv_filter_1 = nn.Sequential(nn.Conv2d(self.nfeats, self.nfeats, 3, 1, 1, dilation=1), nn.PReLU()) 
        self.conv_filter_2 = nn.Sequential(nn.Conv2d(self.nfeats, self.nfeats, 3, 1, 2, dilation=2), nn.PReLU()) 
        self.conv_filter_3 = nn.Sequential(nn.Conv2d(self.nfeats, self.nfeats, 3, 1, 5, dilation=5), nn.PReLU()) 

        self.conv_filter_post = nn.Sequential(nn.Conv2d(self.nfeats*3, self.nfeats, 1, 1, 0), nn.PReLU())

    def forward(self, x, guide_tensor):
        cof = self.conv_feat_cof(guide_tensor)
        scale = torch.sigmoid(cof[:, :self.nfeats, :, :])
        bias = torch.tanh(cof[:, self.nfeats:, :, :])
        x = x.mul(scale) + bias

        out1 = self.conv_filter_1(x)
        out2 = self.conv_filter_2(x)
        out3 = self.conv_filter_3(x)
        out = self.conv_filter_post(torch.cat([out1, out2, out3], 1))

        return out+x