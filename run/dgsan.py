import torch
import torch.nn as nn
from common import *
from numpy import random


class SAM(nn.Module):
    def __init__(self, featrues_channels, act_fun='LeakyReLU'):
        super().__init__()
        self.gen_se_weights1 = nn.Sequential(
            conv(featrues_channels, featrues_channels, 1, bias=True, pad='reflection'),
            act(act_fun),
            nn.Sigmoid())
        self.conv_1 = conv(featrues_channels, featrues_channels, 1, bias=True, pad='reflection')
        self.norm_1 = bn(featrues_channels)
        self.conv_2 = nn.Sequential(
            conv(featrues_channels, featrues_channels, 3, bias=True, pad='reflection'),
            bn(featrues_channels),
            act(act_fun))

    def forward(self, guide, x):
        se_weights1 = self.gen_se_weights1(guide)
        dx = self.conv_1(x)
        dx = torch.mul(dx, se_weights1)
        dx = self.norm_1(dx)
        out = self.conv_2(dx)
        return out


class DRM(nn.Module):
    def __init__(self, featrues_channels, act_fun='LeakyReLU', upsample_mode='bilinear', align_corners=True,
                 need_bias=True, pad='reflection'):
        super().__init__()

        self.weight_map = nn.Sequential(
            conv(featrues_channels, featrues_channels, 3, bias=True, pad='reflection'),
            bn(featrues_channels),
            conv(featrues_channels, featrues_channels, 1, bias=need_bias, pad=pad),
            nn.Sigmoid())
        self.upsample_norm = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=align_corners),
            conv(featrues_channels, featrues_channels, 1, bias=need_bias, pad=pad),
            bn(featrues_channels),
            act(act_fun))
        self.norm = bn(featrues_channels)

    def forward(self, guide, x):
        x_upsample = self.upsample_norm(x)
        weight = self.weight_map(guide)
        out = x_upsample + weight
        out = self.norm(out)
        return out


class DGSAN(nn.Module):
    def __init__(self, input_channels=31, output_channels=10, guide_channels=7, featrues_channels=64, need_bias=True,
                 pad='reflection', upsample_mode='bilinear', align_corners=True, downsample_mode='stride', act_fun='LeakyReLU'):
        super().__init__()

        self.SAM = SAM(featrues_channels)
        self.DRM = DRM(featrues_channels)
        self.guide_enc = nn.Sequential(
            conv(guide_channels, featrues_channels, 3, bias=need_bias, pad=pad),
            bn(featrues_channels),
            act(act_fun))
        self.enc = nn.Sequential(
            conv(featrues_channels, featrues_channels, 3, 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode),
            bn(featrues_channels),
            act(act_fun),
            conv(featrues_channels, featrues_channels, 1, bias=need_bias, pad=pad),
            bn(featrues_channels),
            act(act_fun))
        self.dc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=align_corners),
            conv(featrues_channels, featrues_channels, 3, 1, bias=need_bias, pad=pad),
            bn(featrues_channels),
            act(act_fun))
        self.bn = bn(featrues_channels)
        self.enc_ew0 = nn.Sequential(
            conv(input_channels, featrues_channels, 3, 1, bias=need_bias, pad=pad),
            bn(featrues_channels),
            act(act_fun))
        self.dc_conv = nn.Sequential(
            conv(featrues_channels, featrues_channels, 3, 1, bias=need_bias, pad=pad),
            bn(featrues_channels),
            act(act_fun))
        self.lat = nn.Sequential(
            conv(featrues_channels, featrues_channels, 1, bias=need_bias, pad=pad),
            bn(featrues_channels),
            act(act_fun))
        self.output = nn.Sequential(conv(featrues_channels, output_channels, 1, bias=need_bias, pad=pad))

    def forward(self, guide, noise):
        guide_en0 = self.guide_enc(guide)
        guide_en1 = self.enc(guide_en0)
        guide_en2 = self.enc(guide_en1)
        guide_en3 = self.enc(guide_en2)
        guide_en4 = self.enc(guide_en3)
        guide_en5 = self.enc(guide_en4)

        guide_dc1 = self.dc(guide_en5)
        guide_dc2 = self.dc(self.bn(self.lat(guide_en4) + guide_dc1))
        guide_dc3 = self.dc(self.bn(self.lat(guide_en3) + guide_dc2))
        guide_dc4 = self.dc(self.bn(self.lat(guide_en2) + guide_dc3))
        guide_dc5 = self.dc(self.bn(self.lat(guide_en1) + guide_dc4))

        x_en5 = self.enc_ew0(noise)
        x_dc0 = self.SAM(guide_en5, x_en5)
        
        x_dc1 = self.DRM(guide_en4, x_dc0)
        x_dc1 = self.dc_conv(x_dc1)
        x_dc1 = self.SAM(guide_dc1, x_dc1)

        x_dc2 = self.DRM(guide_en3, x_dc1)
        x_dc2 = self.dc_conv(x_dc2)
        x_dc2 = self.SAM(guide_dc2, x_dc2)

        x_dc3 = self.DRM(guide_en2, x_dc2)
        x_dc3 = self.dc_conv(x_dc3)
        x_dc3 = self.SAM(guide_dc3, x_dc3)

        x_dc4 = self.DRM(guide_en1, x_dc3)
        x_dc4 = self.dc_conv(x_dc4)
        x_dc4 = self.SAM(guide_dc4, x_dc4)
        
        x_dc5 = self.DRM(guide_en0, x_dc4)
        x_dc5 = self.dc_conv(x_dc5)
        x_dc5 = self.SAM(guide_dc5, x_dc5)

        out = self.output(x_dc5)

        return out
