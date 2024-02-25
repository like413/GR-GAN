import torch
import torch.nn as nn

from modules.mobilenet_v2 import MobileNetV2
from modules.attention import se_block, cbam_block, eca_block

attention_blocks = [se_block, cbam_block, eca_block] #attention_blocks[0, 1, 2]

class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class FPNMobileNet(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=64, num_filters_fpn=128, pretrained=True,
                 is_attention=0, attention_num=0, attention_location=0):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.

        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer = norm_layer, pretrained=pretrained,
                       is_attention=is_attention, attention_num=attention_num, attention_location=attention_location)

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

        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")

        final = self.final(smoothed)
        # res = torch.tanh(final) + x

        # return torch.clamp(res, min=-1, max=1)
        return final



class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=128, pretrained=True, is_attention=0, attention_num=0, attention_location=0):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
          is_attention: if 1, add attention block, default 0 not add
          attention_num: 0 se_block, 1 cbam_block, 2 eca_block
        """

        super().__init__()
        net = MobileNetV2(n_class=1000)

        if pretrained:
            #Load weights into the project directory
            # state_dict = torch.load('mobilenet_v2.pth.tar', map_location='cpu') # add map_location='cpu' if no gpu
            state_dict = torch.load('modules/mobilenet_v2.pth.tar', map_location='cpu') # add map_location='cpu' if no gpu
            net.load_state_dict(state_dict)
        self.features = net.features

        self.enc0 = nn.Sequential(*self.features[0:2])
        self.enc1 = nn.Sequential(*self.features[2:4])
        self.enc2 = nn.Sequential(*self.features[4:7])
        self.enc3 = nn.Sequential(*self.features[7:11])
        self.enc4 = nn.Sequential(*self.features[11:16])

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))

        # --------------------is_attention[0, 1]----------------------
        # --------------------attention_num[0, 1, 2]------------------

        self.is_attention = is_attention
        self.attention_location = attention_location
        if is_attention == 1 and self.attention_location == 0:
            self.enc0_attention = attention_blocks[attention_num](16)
            self.enc1_attention = attention_blocks[attention_num](24)
            self.enc2_attention = attention_blocks[attention_num](32)
            self.enc3_attention = attention_blocks[attention_num](64)
            self.enc4_attention = attention_blocks[attention_num](160)
        # ------------------------------------------------------------

        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(32, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(24, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(16, num_filters // 2, kernel_size=1, bias=False)

        # --------------------is_attention[0, 1]----------------------
        # --------------------attention_num[0, 1, 2]------------------
        if is_attention == 1 and self.attention_location == 1:
            self.lateral4_attention = attention_blocks[attention_num](128)
            self.lateral3_attention = attention_blocks[attention_num](128)
            self.lateral2_attention = attention_blocks[attention_num](128)
            self.lateral1_attention = attention_blocks[attention_num](128)
            self.lateral0_attention = attention_blocks[attention_num](64)
        # ------------------------------------------------------------

        for param in self.features.parameters():
            param.requires_grad = False


    def unfreeze(self):
        for param in self.features.parameters():
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

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        # # add attention later-------------------
        if self.is_attention == 1 and self.attention_location == 1:
            lateral4 = self.lateral4_attention(lateral4)
            lateral3 = self.lateral3_attention(lateral3)
            lateral2 = self.lateral2_attention(lateral2)
            lateral1 = self.lateral1_attention(lateral1)
            lateral0 = self.lateral0_attention(lateral0)


        # Top-down pathway
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"))
        map2 = self.td2(lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest"))
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest"))
        return lateral0, map1, map2, map3, map4


