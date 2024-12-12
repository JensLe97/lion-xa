import sparseconvnet as scn
import torch
import torch.nn as nn
import torch.nn.functional as F

from lion_xa.models.salsanext_seg import ResBlock


class Discriminator_2d_UNetResNet(nn.Module):
    # From AUDA
    def __init__(self,
                 input_channels,
                 middle_channel=64
                 ):
        super(Discriminator_2d_UNetResNet, self).__init__()
    
        self.fc_dis = nn.Sequential(
            nn.Conv2d(input_channels, middle_channel, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(middle_channel, middle_channel * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(middle_channel * 2, middle_channel * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(middle_channel * 4, middle_channel * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(middle_channel * 8, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.fc_dis(x)
        return x

class Discriminator_2d_SalsaNext(nn.Module):
    # From LiDARNet
    def __init__(self, 
                 input_channels,
                 middle_channel=64):
        super(Discriminator_2d_SalsaNext, self).__init__()

        self.resBlock1 = ResBlock(32, middle_channel, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(middle_channel, 2 * middle_channel, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * middle_channel, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2 * 4 * 32, 1)

    def forward(self, downCntx):
        input_size = downCntx.shape
        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)
        avg_res = self.avgpool(down5c)
        res = self.fc(avg_res.reshape((input_size[0],-1)))

        return res

class Discriminator_3d(nn.Module):
    def __init__(self,
                 input_channels,
                 num_classes=1,
                 middle_channel=64
                 ):
        super(Discriminator_3d, self).__init__()

        # segmentation head
        self.linear_point = nn.Linear(input_channels, num_classes)

        self.linear_batch_1 = nn.Linear(input_channels, middle_channel)
        self.linear_batch_2 = nn.Linear(middle_channel, middle_channel*2)
        self.linear_batch_3 = nn.Linear(middle_channel*2, middle_channel*4)
        self.linear_batch_3_1 = nn.Linear(middle_channel * 4, middle_channel * 4)
        self.linear_batch_3_2 = nn.Linear(middle_channel * 4, middle_channel * 4)
        self.linear_batch_4 = nn.Linear(middle_channel*4, middle_channel*8)
        self.linear_batch_5 = nn.Linear(middle_channel*8, num_classes)
        self.down = nn.AdaptiveAvgPool1d(8)

        self.bn1 = nn.BatchNorm1d(middle_channel)
        self.bn2 = nn.BatchNorm1d(middle_channel*2)
        self.bn3 = nn.BatchNorm1d(middle_channel*4)
        self.bn4 = nn.BatchNorm1d(middle_channel*8)

    def forward(self, input):
        x = F.relu(self.bn1(self.linear_batch_1(input)))

        x = torch.transpose(x, 0, 1)
        x = x.unsqueeze(0)
        batch_wise_data = self.down(x).squeeze(0)
        batch_wise_data = torch.transpose(batch_wise_data, 0, 1)
        x = F.relu(self.bn2(self.linear_batch_2(batch_wise_data)))
        x = F.relu(self.bn3(self.linear_batch_3(x)))
        x = F.relu(self.bn3(self.linear_batch_3_1(x)))
        x = F.relu(self.bn3(self.linear_batch_3_2(x)))
        x = F.relu(self.bn4(self.linear_batch_4(x)))
        x = self.linear_batch_5(x)

        return x
        

class FCDiscriminator_3d(nn.Module):
    def __init__(self,
                 DIMENSION,
                 in_channels=1,
                 m=8,  # number of unet features (multiplied in each layer)
                 block_reps=1,  # depth
                 residual_blocks=False,  # ResNet style basic blocks
                 full_scale=4096,
                 num_planes=7
                 ):
        super(FCDiscriminator_3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = m
        n_planes = [(n + 1) * m for n in range(num_planes)]

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(DIMENSION, full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(DIMENSION, in_channels, m, 3, False)).add(
            scn.UNet(DIMENSION, block_reps, n_planes, residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(DIMENSION))

    def forward(self, x):
        x = self.sparseModel(x)
        return x

class Discriminator_3d_(nn.Module):
    def __init__(self,
                 input_channels,
                 num_classes=1,
                 ):
        super(Discriminator_3d_, self).__init__()

        # 3D network
        self.net_3d = FCDiscriminator_3d(DIMENSION=input_channels-1)
        # segmentation head
        self.linear_point = nn.Linear(self.net_3d.out_channels, num_classes)
        self.linear_batch = nn.Linear(self.net_3d.out_channels, num_classes)
        self.linear_batch = nn.Linear(self.net_3d.out_channels, num_classes)
        self.down = nn.AdaptiveAvgPool1d(8)

    def forward(self, input):
        out_3D_feature = self.net_3d(input)
        x_point = self.linear_point(out_3D_feature)

        x = torch.transpose(out_3D_feature, 0, 1)
        x = x.unsqueeze(0)
        batch_wise_data = self.down(x).squeeze(0)
        batch_wise_data = torch.transpose(batch_wise_data, 0, 1)

        x_batch = self.linear_batch(batch_wise_data)

        preds = {
            'Dis_out_point': x_point,
            'Dis_out_batch': x_batch
        }

        return preds
