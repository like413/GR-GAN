import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pretrainedmodels import inceptionresnetv2
# from torchsummary import summary
import torch.nn.functional as F

# from modules.attention import se_block, cbam_block, eca_block
# attention_blocks = [DAG, se_block, cbam_block, eca_block] #attention_blocks[0, 1, 2]

from modules.DAG import DAG

class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x

class ConvBlock(nn.Module):
    def __init__(self, num_in, num_out, norm_layer):
        super().__init__()

        self.block = nn.Sequential(nn.Conv2d(num_in, num_out, kernel_size=3, padding=1),
                                 norm_layer(num_out),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class FPNInception(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=128, num_filters_fpn=256,
                 is_attention=0, attention_num=0, attention_location=0):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer,
                       is_attention=is_attention, attention_num=attention_num,attention_location=attention_location)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        map0, map1, map2, map3, map4, lateral4, lateral3, lateral2, lateral1, lateral0 = self.fpn(x)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed1 = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed2 = nn.functional.upsample(smoothed1, scale_factor=2, mode="nearest")
        smoothed3 = self.smooth2(smoothed2 + map0)
        smoothed = nn.functional.upsample(smoothed3, scale_factor=2, mode="nearest")

        final = self.final(smoothed)
        # res = torch.tanh(final) + x
        #
        # return torch.clamp(res, min = -1,max = 1)
        return final[:, 0:3, :, :], final[:, 3:6, :, :], lateral4, lateral3, lateral2, lateral1, lateral0, smoothed1,smoothed2,smoothed3,smoothed

class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=256, is_attention=0, attention_num=0, attention_location=0):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
          is_attention: if 1, add attention block, default 0 not add
          attention_num: 0 se_block, 1 cbam_block, 2 eca_block
        """

        super().__init__()
        self.inception = inceptionresnetv2(num_classes=1000, pretrained='imagenet')

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(
            self.inception.conv2d_2a,
            self.inception.conv2d_2b,
            self.inception.maxpool_3a,
        ) # 64
        self.enc2 = nn.Sequential(
            self.inception.conv2d_3b,
            self.inception.conv2d_4a,
            self.inception.maxpool_5a,
        )  # 192
        self.enc3 = nn.Sequential(
            self.inception.mixed_5b,
            self.inception.repeat,
            self.inception.mixed_6a,
        )   # 1088
        self.enc4 = nn.Sequential(
            self.inception.repeat_1,
            self.inception.mixed_7a,
        ) #2080
        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.pad = nn.ReflectionPad2d(1)

        self.is_attention = is_attention
        self.attention_location = attention_location
        # --------------------is_attention[0, 1]----------------------
        # --------------------attention_num[0, 1, 2]------------------
        if is_attention == 1 and attention_location == 0:
            self.enc0_attention = DAG(32,32)
            self.enc1_attention = DAG(64,64)
            self.enc2_attention = DAG(192,192)
            self.enc3_attention = DAG(1088,1088)
            self.enc4_attention = DAG(2080,2080)
        # ------------------------------------------------------------

        self.lateral4 = nn.Conv2d(2080, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(32, num_filters // 2, kernel_size=1, bias=False)

        # --------------------is_attention[0, 1]----------------------
        # --------------------attention_num[0, 1, 2]------------------
        if is_attention == 1 and self.attention_location == 1:
            self.lateral4_attention = DAG(256,256)
            self.lateral3_attention = DAG(256,256)
            self.lateral2_attention = DAG(256,256)
            self.lateral1_attention = DAG(256,256)
            self.lateral0_attention = DAG(256,256)
        # ------------------------------------------------------------

        for param in self.inception.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0) # 256

        enc2 = self.enc2(enc1) # 512

        enc3 = self.enc3(enc2) # 1024

        enc4 = self.enc4(enc3) # 2048

        # #add attention before-------------------
        if self.is_attention == 1 and self.attention_location == 0:
            enc0 = self.enc0_attention(enc0)
            enc1 = self.enc1_attention(enc1)
            enc2 = self.enc2_attention(enc2)
            enc3 = self.enc3_attention(enc3)
            enc4 = self.enc4_attention(enc4)

        # Lateral connections

        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)

        # #add attention later-------------------
        if self.is_attention == 1 and self.attention_location == 1:
            lateral4 = self.lateral4_attention(lateral4)
            lateral3 = self.lateral3_attention(lateral3)
            lateral2 = self.lateral2_attention(lateral2)
            lateral1 = self.lateral1_attention(lateral1)
            lateral0 = self.lateral0_attention(lateral0)

        # Top-down pathway
        pad = (1, 2, 1, 2)  # pad last dim by 1 on each side
        pad1 = (0, 1, 0, 1)
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"))
        map2 = self.td2(F.pad(lateral2, pad, "reflect") + nn.functional.upsample(map3, scale_factor=2, mode="nearest"))
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest"))
        # return F.pad(lateral0, pad1, "reflect"), map1, map2, map3, map4
        return F.pad(lateral0, pad1, "reflect"), map1, map2, map3, map4, lateral4, lateral3, lateral2, lateral1, lateral0
