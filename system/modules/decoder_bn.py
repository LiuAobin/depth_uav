import torch
from torch import nn
from torch.nn import functional as F

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features,
                                            kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(output_features, output_features,
                                            kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(inplace=True),)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x,
                             size=[concat_with.size(2), concat_with.size(3)],
                             mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

class DecoderBN(nn.Module):
    """
    逐步上采样 + 跳跃连接，恢复至输入分辨率
    解码器网络，使用批归一化(BatchNorm)进行特征处理
    该网络接收来自编码器的不同尺度特征，并通过一系列上采样层进行逐步恢复
    最终输出深度特征
    """
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=512):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        # 1x1 卷积，调整通道数
        self.conv2 = nn.Conv2d(bottleneck_features, features,
                               kernel_size=1, stride=1, padding=1)
        # for res50 逐步恢复分辨率
        self.up1 = UpSampleBN(skip_input=features // 1 + 1024, output_features=features // 2) # in:512+1024 out:256
        self.up2 = UpSampleBN(skip_input=features // 2 + 512, output_features=features // 4) # in:256+512 out:128
        self.up3 = UpSampleBN(skip_input=features // 4 + 256, output_features=features // 8) # in:128+256 out:64
        self.up4 = UpSampleBN(skip_input=features // 8 + 64, output_features=features // 16) # in:64+64 out:32

        # self.up5 = UpSampleBN(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes,
                               kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        """
        前向传播过程：
        1. 接收编码器的五层特征图（x_block0 至 x_block4）。C=[64,256,512,1024,2048]
        2. 通过 1x1 卷积调整最深层特征的通道数。
        3. 逐步进行上采样，并在每一步拼接对应尺度的跳跃连接特征。
        4. 最终通过 3x3 卷积生成输出。
        """
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[0], features[1], features[2], features[3], features[4]
        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        # x_d5 = self.up5(x_d4, features[0])
        # out = self.conv3(x_d5)
        out = self.conv3(x_d4)
        return out



class UpSampleBN_One(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpSampleBN_One, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DecoderBN_One(nn.Module):
    """
    逐步上采样 + 跳跃连接，恢复至输入分辨率
    解码器网络，使用批归一化(BatchNorm)进行特征处理
    该网络接收来自编码器的不同尺度特征，并通过一系列上采样层进行逐步恢复
    最终输出深度特征
    """
    def __init__(self, num_features=2048, num_classes=1, encoder_channels=[64, 256, 512, 1024, 2048]):
        super(DecoderBN_One, self).__init__()
        features = int(num_features)
        # 1x1 卷积，调整通道数
        self.reduce_conv = nn.Conv2d(encoder_channels[4], 512, kernel_size=1)
        self.up1 = UpSampleBN_One(in_channels=512, skip_channels=encoder_channels[3], out_channels=256)
        self.up2 = UpSampleBN_One(in_channels=256, skip_channels=encoder_channels[2], out_channels=128)
        self.up3 = UpSampleBN_One(in_channels=128, skip_channels=encoder_channels[1], out_channels=64)
        self.up4 = UpSampleBN_One(in_channels=64, skip_channels=encoder_channels[0], out_channels=32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(bottleneck_features, features,
        #                        kernel_size=1, stride=1, padding=1)
        # for res50 逐步恢复分辨率
        # self.up1 = UpSampleBN(skip_input=features // 1 + 1024, output_features=features // 2) # in:512+1024 out:256
        # self.up2 = UpSampleBN(skip_input=features // 2 + 512, output_features=features // 4) # in:256+512 out:128
        # self.up3 = UpSampleBN(skip_input=features // 4 + 256, output_features=features // 8) # in:128+256 out:64
        # self.up4 = UpSampleBN(skip_input=features // 8 + 64, output_features=features // 16) # in:64+64 out:32

        # self.up5 = UpSampleBN(skip_input=features // 16 + 3, output_features=features//16)
        # self.conv3 = nn.Conv2d(features // 16, num_classes,
        #                        kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        """
        前向传播过程：
        1. 接收编码器的五层特征图（x_block0 至 x_block4）。C=[64,256,512,1024,2048]
        2. 通过 1x1 卷积调整最深层特征的通道数。
        3. 逐步进行上采样，并在每一步拼接对应尺度的跳跃连接特征。
        4. 最终通过 3x3 卷积生成输出。
        """
        # x_block0, x_block1, x_block2, x_block3, x_block4 = features[0], features[1], features[2], features[3], features[4]
        x_block0, x_block1, x_block2, x_block3, x_block4 = features

        x = self.reduce_conv(x_block4)
        x = self.up1(x, x_block3)
        x = self.up2(x, x_block2)
        x = self.up3(x, x_block1)
        x = self.up4(x, x_block0)
        out = self.final_conv(x)
        # x_d0 = self.conv2(x_block4)
        #
        # x_d1 = self.up1(x_d0, x_block3)
        # x_d2 = self.up2(x_d1, x_block2)
        # x_d3 = self.up3(x_d2, x_block1)
        # x_d4 = self.up4(x_d3, x_block0)
        # # x_d5 = self.up5(x_d4, features[0])
        # # out = self.conv3(x_d5)
        # out = self.conv3(x_d4)
        return out