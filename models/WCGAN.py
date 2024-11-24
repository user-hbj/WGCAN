import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
# from Residual_RECA import *
# from .Residual_RECA import *

from .common import *
# from common import *ss

# 定义了一个离散小波变换（Discrete Wavelet Transform, DWT）和逆离散小波变换（Inverse DWnw）的操作，它们用于对图像进行小波变换和逆变换
# 小波变换是一种在信号和图像处理中常用的技术，用于分析信号和图像的不同频率成分
def dwt_init(x):
    """
    将输入的图像张量 x 分解为四个子图像,分别代表小波变换的低频部分(LL)、水平高频部分(HL)、垂直高频部分(LH)、和对角高频部分(HH)。
    这些子图像经过适当的加权和相加操作后，返回一个包含这四个部分的张量
    """
    # x01 和 x02 是 x 的两个子图像，它们分别包含了 x 中的偶数行和奇数行
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    # x1 包含了 x01 中的偶数列，而 x2 包含了 x02 中的偶数列
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    # x3 包含了 x01 中的奇数列，而 x4 包含了 x02 中的奇数列
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    # 低频部分
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # 函数返回一个张量，其中包含了上述四个部分，它们在深度维度上连接在一起
    """
    如果x_LL、x_HL、x_LH、x_HH都是形状为(batch_size, num_channels, height, width)的张量，
    那么使用torch.cat((x_LL, x_HL, x_LH, x_HH), 1)将它们连接在一起，
    得到一个形状为(batch_size, 4 * num_channels, height, width)的张量。
    """
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    """
    用于执行逆离散小波变换，将四个子图像合并还原成原始图像。
    它接受一个包含四个小波变换部分的输入张量，然后执行逆变换操作，返回还原后的原始图像。
    """
    r = 2
    # 
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
    # 将四个子图像的信息合并还原成原始图像h
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    """
    离散小波变换的 PyTorch 模块，它继承自 nn.Module。在其 forward 方法中，它调用了 dwt_init 函数执行小波变换操作，并返回变换后的图像
    """
    def __init__(self):
        super(DWT, self).__init__()
        # 该模块的参数不会进行梯度计算，因为小波变换操作是固定的
        self.requires_grad = False

    def forward(self, x):
        # 执行离散小波变换操作，并将变换后的图像作为结果返回
        return dwt_init(x)

class IWT(nn.Module):
    """执行逆离散小波变换：执行逆变换操作，并返回还原后的图像"""
    def __init__(self):
        super(IWT, self).__init__()
        # 该模块的参数不会进行梯度计算，因为小波变换操作是固定的
        self.requires_grad = False

    def forward(self, x):
        # 执行逆离散小波变换操作，将还原后的图像作为结果返回
        return iwt_init(x)
    
# 激活函数
class LeakyReLU(nn.Module):
    def __init__(self):
        super(LeakyReLU, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(x)

# 双层卷积 
class UNetConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNetConvBlock, self).__init__()
        self.UNetConvBlock = torch.nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            LeakyReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            LeakyReLU()
        )
    def forward(self, x):
        return self.UNetConvBlock(x)
class CALayer(nn.Module):
    def __init__(self,in_ch,reduction=16):
        super(CALayer,self).__init__()
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch,in_ch // reduction,1),
            nn.Conv2d(in_ch // reduction,in_ch,1),
            nn.Sigmoid(),
        )
        
    def forward(self,x):
        return x * self.a(x)

class RCAB(nn.Module):
    def __init__(self,in_ch,reduction=16):
        super(RCAB, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_ch,in_ch,3,padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch,in_ch,3,padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            CALayer(in_ch,reduction),
        #nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self,x):
        res = self.res(x) + x
        return res
    
class RG(nn.Module):
    def __init__(self, in_ch, reduction=16, num_rcab=2):
        super(RG, self).__init__()
        self.layers = nn.Sequential(
            *[RCAB(in_ch, reduction) for _ in range(num_rcab)]
        )

    def forward(self, x):
        out = self.layers(x)
        return out + x
    
class StackedRG(nn.Module):
    def __init__(self, in_ch, reduction=16, num_rcab=2, num_rg=4):
        super(StackedRG, self).__init__()
        self.stack = nn.Sequential(
            *[RG(in_ch, reduction, num_rcab) for _ in range(num_rg)]
        )

    def forward(self, x):
        out = self.stack(x)
        return out + x
    
def split_feature_map(feature_map, num_groups=4):
    input_channels = feature_map.size(1)
    group_size = input_channels // num_groups
    groups = []
    for i in range(num_groups):
        start_channel = i * group_size
        end_channel = (i + 1) * group_size
        group = feature_map[:, start_channel:end_channel, :, :]
        groups.append(group)
    LL = groups[0]
    LH = groups[1]
    HL = groups[2]
    HH = groups[3]
    return LL,LH,HL,HH

class High_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = UNetConvBlock(12, 64) 
        self.conv2 = UNetConvBlock(64, 128)
        self.conv3 = UNetConvBlock(128, 256)
        # self.conv4 = UNetConvBlock(128, 256)
        
    def forward(self,x):
        H1 = self.conv1(x)  
        # print(f"H1的大小为{H1.shape},而它的类型为{type(H1)}") # conv1的大小为torch.Size([1, 64, 256, 256]),而它的类型为<class 'torch.Tensor'>
        pool1 = F.max_pool2d(H1,kernel_size=2)  
        # print(f"H1的大小为{H1.shape},而它的类型为{type(H1)}") # H1的大小为torch.Size([1, 64, 128, 128]),而它的类型为<class 'torch.Tensor'>
        
        H2 = self.conv2(pool1)  
        # print(f"H2的大小为{H2.shape},而它的类型为{type(H2)}") # conv2的大小为torch.Size([1, 64, 128, 128]),而它的类型为<class 'torch.Tensor'>
        pool2 = F.max_pool2d(H2,kernel_size=2)  
        # print(f"pool2的大小为{pool2.shape},而它的类型为{type(pool2)}") # pool2的大小为torch.Size([1, 64, 64, 64]),而它的类型为<class 'torch.Tensor'>
        
        H3 = self.conv3(pool2)  
        # print(f"H3的大小为{H3.shape},而它的类型为{type(H3)}") # conv3的大小为torch.Size([1, 128, 64, 64]),而它的类型为<class 'torch.Tensor'>
        
        H4 = F.max_pool2d(H3,kernel_size=2)  
        # print(f"H3的大小为{H3.shape},而它的类型为{type(H3)}") # H3的大小为torch.Size([1, 128, 32, 32]),而它的类型为<class 'torch.Tensor'>
        
        return H1,H2,H3,H4
    
class Low_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = UNetConvBlock(4, 64)  # We have 4 Channel (R,G,B,G)- Bayer Pattern Input
        self.conv2 = UNetConvBlock(64, 128)
        self.conv3 = UNetConvBlock(128, 256)
        # self.conv4 = UNetConvBlock(128, 256)
        
    def forward(self,x):
        L1 = self.conv1(x)  
        # print(f"conv1的大小为{conv1.shape},而它的类型为{type(conv1)}") # conv1的大小为torch.Size([1, 32, 256, 256]),而它的类型为<class 'torch.Tensor'>
        pool1 = F.max_pool2d(L1,kernel_size=2)  
        # print(f"H1的大小为{H1.shape},而它的类型为{type(H1)}") # H1的大小为torch.Size([1, 32, 128, 128]),而它的类型为<class 'torch.Tensor'>
        
        L2 = self.conv2(pool1)  
        # print(f"conv2的大小为{conv2.shape},而它的类型为{type(conv2)}") # conv2的大小为torch.Size([1, 64, 128, 128]),而它的类型为<class 'torch.Tensor'>
        pool2 = F.max_pool2d(L2,kernel_size=2)  
        # print(f"pool2的大小为{pool2.shape},而它的类型为{type(pool2)}") # pool2的大小为torch.Size([1, 64, 64, 64]),而它的类型为<class 'torch.Tensor'>
        
        L3 = self.conv3(pool2)  
        # print(f"conv3的大小为{conv3.shape},而它的类型为{type(conv1)}") # conv3的大小为torch.Size([1, 128, 64, 64]),而它的类型为<class 'torch.Tensor'>
        L4 = F.max_pool2d(L3,kernel_size=2)  
        # print(f"H3的大小为{H3.shape},而它的类型为{type(H3)}") # H3的大小为torch.Size([1, 128, 32, 32]),而它的类型为<class 'torch.Tensor'>
        
        # L4 = self.conv4(pool3)  
        # # print(f"conv4的大小为{conv4.shape},而它的类型为{type(conv4)}") # conv4的大小为torch.Size([1, 256, 32, 32]),而它的类型为<class 'torch.Tensor'>
        
        # poolL = F.max_pool2d(L4,kernel_size=2)  
        # # print(f"poolL的大小为{poolL.shape},而它的类型为{type(poolL)}") # poolL的大小为torch.Size([1, 256, 16, 16]),而它的类型为<class 'torch.Tensor'>
        return L1,L2,L3,L4
    
# 定义空间注意模块
# class Spatial_Attention(nn.Module):
#     def __init__(self):
#         super(Spatial_Attention, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return x * self.sigmoid(x)

class Spatial_Attention(nn.Module):
    def __init__(self, in_channels):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        # print(f"out的大小为{out.shape}")
        return out
      
class MGCC(nn.Module):
    def __init__(self):
    # def __init__(self):
        super(MGCC, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        # 频域
        # self.before_unet = Before_Unet(wavelet_type='haar')
        self.high_Encoder = High_Encoder()
        self.low_Encoder = Low_Encoder()
        # self.hlMerge_Deep_Features = HLMerge_Deep_Features(512,512)
        self.hlMerge_Deep_Features = UNetConvBlock(512, 512)
        
        # self.conv = UNetConvBlock(512, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = UNetConvBlock(768, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = UNetConvBlock(384, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.conv_up3 = UNetConvBlock(192, 64)
        
        # self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.convlast1 = UNetConvBlock(192,64)
        self.convlast2 = UNetConvBlock(64,16)
        # self.convFirstlast = UNetConvBlock(4,4)
        
        self.second_stagebefore = UNetConvBlock(4,64)
        self.Convg = UNetConvBlock(32,16)
        self.CGB_r_1 = CGB(16)
        self.CGB_r_2 = CGB(16)
        self.CGB_g_1 = CGB(16)
        self.CGB_g_2 = CGB(16)
        self.CGB_b_1 = CGB(16)
        self.CGB_b_2 = CGB(16)
        self.g_conv_post = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1), LeakyReLU())
        self.spatial_attention_r = Spatial_Attention(16)
        self.spatial_attention_g = Spatial_Attention(16)
        self.spatial_attention_b = Spatial_Attention(16)
        
        self.rcan = StackedRG(64)
        
        # self.convlastsecond1 = nn.Sequential(nn.Conv2d(192, 64, 3, 1, 1), LeakyReLU())
        # self.convlastsecond2 = nn.Sequential(nn.Conv2d(64, 12, 3, 1, 1), LeakyReLU())
        self.convlastsecond1 = nn.Sequential(nn.Conv2d(192, 64, 3, 1, 1),LeakyReLU())
        self.convlastsecond2 = nn.Sequential(nn.Conv2d(64, 12, 3, 1, 1),LeakyReLU())

        self.finish_second = UNetConvBlock(3,3) 
        self.up = nn.PixelShuffle(2)
        
    def forward(self,x):
        # space = self.conv_space(x)
        # print(f"space图像的大小为{space.shape},而它的类型为{type(space)}") # space图像的大小为torch.Size([1, 16, 512, 512]),而它的类型为<class 'torch.Tensor'>
        # L,H = self.before_unet(x)
        # original = x
        L,H = self.dwt(x)
        # print(f"低频图像的大小为{L.shape},而它的类型为{type(L)}") # 低频图像的大小为torch.Size([1, 16, 256, 256]),而它的类型为<class 'torch.Tensor'>
        # print(f"高频图像的大小为{H.shape},而它的类型为{type(H)}") # 高频图像的大小为torch.Size([1, 48, 256, 256]),而它的类型为<class 'torch.Tensor'>
        H1,H2,H3,H4 = self.high_Encoder(H)
        L1,L2,L3,L4 = self.low_Encoder(L)
        x = torch.cat([L4,H4],dim=1) 
        x = self.hlMerge_Deep_Features(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 512, 32, 32]),而x的类型为<class 'torch.Tensor'>
        x = self.up1(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 256, 64, 64]),而x的类型为<class 'torch.Tensor'>
        x = torch.cat([x,L3,H3],dim=1)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 768, 64, 64]),而x的类型为<class 'torch.Tensor'>
        # print(f"L4的大小为{L4.shape},而L4的类型为{type(L4)}") # x的大小为torch.Size([1, 768, 32, 32]),而x的类型为<class 'torch.Tensor'>
        # print(f"H4的大小为{H4.shape},而H4的类型为{type(H4)}") # x的大小为torch.Size([1, 768, 32, 32]),而x的类型为<class 'torch.Tensor'>
        x = self.conv_up1(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 256, 64, 64]),而x的类型为<class 'torch.Tensor'>
        
        x = self.up2(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 128, 128, 128]),而x的类型为<class 'torch.Tensor'>
        x = torch.cat([x,L2,H2],dim=1)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 384, 128, 128]),而x的类型为<class 'torch.Tensor'>
        x = self.conv_up2(x)
        
        
        x = self.up3(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 64, 256, 256]),而x的类型为<class 'torch.Tensor'>
        x = torch.cat([x,L1,H1],dim=1)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 192, 256, 256]),而x的类型为<class 'torch.Tensor'>
    
        x = self.convlast1(x)
        x = self.convlast2(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 16, 256, 256]),而x的类型为<class 'torch.Tensor'>
        x = self.iwt(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}")     # x的大小为torch.Size([1, 4, 512, 512]),而x的类型为<class 'torch.Tensor'>
        # long_raw = self.convFirstlast(x)
        # x = get_detail(x)
        x = self.second_stagebefore(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}")
        r,G1,G2,b = split_feature_map(x)
        # print(f"R的大小为{R.shape},而x的类型为{type(R)}") # R的大小为torch.Size([1, 16, 512, 512]),而R的类型为<class 'torch.Tensor'>
        # print(f"B的大小为{B.shape},而B的类型为{type(B)}") # R的大小为torch.Size([1, 16, 512, 512]),而R的类型为<class 'torch.Tensor'>
        g = torch.cat([G1,G2],dim=1)
        # print(f"g的大小为{g.shape},而g的类型为{type(g)}")  # g的大小为torch.Size([1, 32, 512, 512]),而g的类型为<class 'torch.Tensor'>
        g = self.Convg(g)
        # print(f"g的大小为{g.shape},而g的类型为{type(g)}")  # g的大小为torch.Size([1, 16, 512, 512]),而g的类型为<class 'torch.Tensor'>
        
        r2 = self.CGB_r_1(r, g)
        r2 = self.CGB_r_2(r2, b)

        b2 = self.CGB_b_1(b, g)
        b2 = self.CGB_b_2(b2, r)

        g2_1 = self.CGB_g_1(g, r2)
        g2_2 = self.CGB_g_2(g, b2)
        
        # 通过将 g2_1 和 g2_2 进行拼接后通过卷积层处理得到的
        g2 = self.g_conv_post(torch.cat([g2_1, g2_2], 1))
        
        r = r2 + r
        g = g2 + g
        b = b2 + b
        # print(f"r的大小为{r.shape},而r的类型为{type(r)}") # r的大小为torch.Size([1, 16, 512, 512]),而r的类型为<class 'torch.Tensor'>
        # print(f"g的大小为{g.shape},而g的类型为{type(g)}") # g的大小为torch.Size([1, 16, 512, 512]),而g的类型为<class 'torch.Tensor'>
        # print(f"b的大小为{b.shape},而b的类型为{type(b)}") # b的大小为torch.Size([1, 16, 512, 512]),而b的类型为<class 'torch.Tensor'>
        
        # 空间注意力
        r = self.spatial_attention_r(r)
        b = self.spatial_attention_b(b)
        g = self.spatial_attention_g(g)
        
    
        x = self.rcan(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}")     # x的大小为torch.Size([1, 64, 512, 512]),而x的类型为<class 'torch.Tensor'>
        
        # 扩展形状与x相同
        # 将第一个张量的通道维度扩展为与第二个张量相匹配
        # r_expanded = r.expand(-1, 64, -1, -1)
        # b_expanded = b.expand(-1, 64, -1, -1)
        # g_expanded = g.expand(-1, 64, -1, -1)
        
        # print(f"g_expanded的大小为{g_expanded.shape},而g_expanded的类型为{type(g_expanded)}") # g_expanded的大小为torch.Size([1, 64, 512, 512]),而g_expanded的类型为<class 'torch.Tensor'>
        
        # 执行相乘操作
        
        result_g = torch.mul(x, g)
        result_b = torch.mul(x, b)
        result_r = torch.mul(x, r)
        
        result = torch.cat([result_g,result_b,result_r],dim=1)
        # print(f"result的大小为{result.shape},而result的类型为{type(result)}")     # result的大小为torch.Size([1, 192, 512, 512]),而result的类型为<class 'torch.Tensor'>
        
        
        # Three stage 图像重建阶段
        x = self.convlastsecond1(result)
        x = self.convlastsecond2(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}")     # x的大小为torch.Size([1, 12, 512, 512]),而x的类型为<class 'torch.Tensor'>
        
        gt = self.up(x)   
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}")     # x的大小为torch.Size([1, 3, 1024, 1024]),而x的类型为<class 'torch.Tensor'>
        gt = self.finish_second(gt)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}")     # x的大小为torch.Size([1, 3, 1024, 1024]),而x的类型为<class 'torch.Tensor'>
    
        return gt

if __name__ == '__main__':
    
    
    # 创建了一个大小为 (1, 3, 1024, 1024) 的 NumPy 数组，表示输入数据。这是一个包含 1 个样本、3 个通道、大小为 1024 x 1024 的图像
    input = np.ones((1,4,512,512),dtype=np.float32)
    
    # 这里将 NumPy 数组转换为 PyTorch 张量，并将其移动到 CUDA 设备（GPU）上
    input = torch.tensor(input, dtype=torch.float32, device='cuda')
    
    # print(f"input的大小为{input.shape},而它的类型为{type(input)}") # input的大小为torch.Size([1, 4, 512, 512]),而它的类型为<class 'torch.Tensor'>
    print(f"input的大小为{input.shape},而它的类型为{type(input)}")
   
    net = MGCC()
    # 将模型移动到 CUDA 设备（GPU）上，如果可用的话。这意味着后续的计算将在 GPU 上执行
    net.cuda()
    with torch.no_grad():
        gt = net(input)
    # torch.Size([1, 3, 1024, 1024])
    # print(f"long_raw的大小为{long_raw.shape},而它的类型为{type(long_raw)}")
    print(f"gt的大小为{gt.shape},而它的类型为{type(gt)}")
    