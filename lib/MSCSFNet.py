import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from lib.pvtv2 import pvt_v2_b2


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CSFF(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(CSFF, self).__init__()
        self.cat2 = BasicConv2d(hidden_channels * 2, out_channels, kernel_size=3, padding=1)
        self.param_free_norm = nn.BatchNorm2d(hidden_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)


    def forward(self, x, y, edge):
        xy = self.cat2(torch.cat((x, y), dim=1)) + y + x
        normalized = self.param_free_norm(xy)

        edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(edge)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out


class DenseInteractionDecoder(nn.Module):

    def __init__(self, channel):
        super(DenseInteractionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3, x4):

        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample3(self.upsample(x3_1)) * self.conv_upsample7(self.upsample(x3)) * x4

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        #
        x4_2 = torch.cat((x4_1, self.conv_upsample6(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)
        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x

class CropLayer(nn.Module):
    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

class asyConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(asyConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.initialize()
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs

class ERF(nn.Module):
    def __init__(self, x, y):
        super(ERF, self).__init__()
        self.asyConv = asyConv(in_channels=x, out_channels=y, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False)
        self.oriConv = nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1)
        self.atrConv = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, dilation=3, padding=3, stride=1), nn.BatchNorm2d(y), nn.PReLU()
        )
        self.conv2d = nn.Conv2d(y*2, y, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(y)
        self.res = BasicConv2d(x, y, 1)

    def forward(self, f):
        p2 = self.asyConv(f)
        p3 = self.atrConv(f)
        p  = torch.cat((p2, p3), 1)
        p  = F.relu(self.bn2d(self.conv2d(p)), inplace=True)

        return p

#
class Network(nn.Module):
    def __init__(self, channel=64):
        super(Network, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/media/liuyu/PycharmProjects/lhh/MSCSF-Net/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.ERF2_0 = ERF(64, channel)
        self.ERF2_1 = ERF(128, channel)
        self.ERF3_1 = ERF(320, channel)
        self.ERF4_1 = ERF(512, channel)

        self.linearr1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.CSFF1 = CSFF(64, 64)
        self.CSFF2 = CSFF(64, 64)
        self.CSFF3 = CSFF(64, 64)
        self.CSFF4 = CSFF(64, 64)

        self.DID = DenseInteractionDecoder(channel)

    def forward(self, x):
        image_shape = x.size()[2:]
        pvt = self.backbone(x)

        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x2_t = self.Translayer2_1(x2)#[1, 64, 44, 44]
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)

        wz1 = self.DID(x4_t, x3_t, x2_t, x1)
        E_1 = self.ERF2_0(x1)
        E_2 = self.ERF2_1(x2)  # [1, 64, 44, 44]
        E_3 = self.ERF3_1(x3)
        E_4 = self.ERF4_1(x4)

        clm = F.interpolate(wz1, size=image_shape, mode='bilinear')

        E22 = F.interpolate(E_2, scale_factor=2, mode='bilinear')
        E23 = F.interpolate(E_3, scale_factor=4, mode='bilinear')
        E24 = F.interpolate(E_4, scale_factor=8, mode='bilinear')

        R_4 = self.CSFF4(E24, E24, wz1)
        R_3 = self.CSFF3(E23, R_4, wz1)
        R_2 = self.CSFF2(E22, R_3, wz1)
        R_1 = self.CSFF1(E_1, R_2, wz1)

        map_4 = self.linearr4(R_4)
        map_3 = self.linearr3(R_3) + map_4
        map_2 = self.linearr2(R_2) + map_3
        map_1 = self.linearr1(R_1) + map_2

        out_1 = F.interpolate(map_1, size=image_shape, mode='bilinear')
        out_2 = F.interpolate(map_2, size=image_shape, mode='bilinear')
        out_3 = F.interpolate(map_3, size=image_shape, mode='bilinear')
        out_4 = F.interpolate(map_4, size=image_shape, mode='bilinear')

        return out_1, out_2, out_3, out_4, clm







if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network().cuda()
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352).cuda()
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        torch.cuda.synchronize()
        start = time()
        y = net(dump_x)
        torch.cuda.synchronize()
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        # print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    # print(np.mean(frame_rate))
    # print(y)