import spconv.pytorch as spconv
import torch
import torch.nn as nn

from lion_xa.models.resnet34_unet import UNetResNet34
from lion_xa.models.salsanext_seg import SalsaNextSeg
from lion_xa.models.scn_unet import UNetSCN
from lion_xa.models.pvd.cylinder_spconv_3d import CylinderAsym
from lion_xa.models.pvd.segmentator_3d_asymm_spconv import Asym3dSpconv
from lion_xa.models.pvd.cylinder_fea_generator import CylinderFeat


class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 backbone_2d_kwargs
                 ):
        super(Net2DSeg, self).__init__()

        # 2D image network
        if backbone_2d == 'UNetResNet34':
            self.net_2d = UNetResNet34(**backbone_2d_kwargs)
            feat_channels = 64
        elif backbone_2d == 'SalsaNextSeg':
            self.net_2d = SalsaNextSeg(num_classes, **backbone_2d_kwargs)
            feat_channels = 32
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        # Segmentation head for 2d features
        self.seg_head_2d = nn.Conv2d(feat_channels, num_classes, kernel_size=(1, 1))

        # segmentation head
        self.linear = nn.Linear(feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(feat_channels, num_classes)

    def forward(self, data_batch):
        # (batch_size, 5, H, W)
        img = data_batch['range_img']
        range_img_indices = data_batch['range_img_indices']

        # 2D network
        x = self.net_2d(img)

        feats_2d = x.clone()

        # 2D-3D feature lifting
        feats_3d = []
        for i in range(x.shape[0]):
            feats_3d.append(x.permute(0, 2, 3, 1)[i][range_img_indices[i][:, 1], range_img_indices[i][:, 0]])
        feats_3d = torch.cat(feats_3d, 0)

        pixel_seg_logit = self.seg_head_2d(x)
        
        # linear
        point_seg_logit = self.linear(feats_3d)

        preds = {
            # 'feats_3d': feats_3d,
            'feats_2d': feats_2d,
            'pixel_seg_logit': pixel_seg_logit,
            'point_seg_logit': point_seg_logit,
        }

        if self.dual_head:
            preds['point_seg_logit2'] = self.linear2(feats_3d)

        return preds


class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 ):
        super(Net3DSeg, self).__init__()
        # 2nd segmentation head
        self.dual_head = dual_head

        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = UNetSCN(**backbone_3d_kwargs)
            self.linear = nn.Linear(self.net_3d.out_channels, num_classes)

            if dual_head:
                self.linear2 = nn.Linear(self.net_3d.out_channels, num_classes)
        elif backbone_3d == 'PVD':
            cylinder_3d_spconv_seg = Asym3dSpconv(**backbone_3d_kwargs, nclasses=num_classes)

            cy_fea_net = CylinderFeat(**backbone_3d_kwargs)

            self.net_3d = CylinderAsym(cylin_model=cy_fea_net, segmentator_spconv=cylinder_3d_spconv_seg)
            
            # segmentation head
            self.logits = spconv.SubMConv3d(4 * self.net_3d.init_size, num_classes, indice_key="logit", kernel_size=3, stride=1, 
                                        padding=1, bias=True)
            if dual_head:
                self.logits2 = spconv.SubMConv3d(4 * self.net_3d.init_size, num_classes, indice_key="logit", kernel_size=3, stride=1, 
                                                 padding=1, bias=True)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

    def forward(self, data_batch):
        preds = {}
        if isinstance(self.net_3d, UNetSCN):
            feats = self.net_3d(data_batch['x'])
            x = self.linear(feats)
            if self.dual_head:
                preds['seg_logit2'] = self.linear2(feats)
        elif isinstance(self.net_3d, CylinderAsym):
            # data_batch['x'][1] = point_feat (N, 9 + 1)
            # data_batch['x'][0] = voxel_grid_ind (N, 3 + 1)
            # batch_size = data_batch['x'][0][-1][0].item() + 1 (last value of batch indices +1)
            feats, indices = self.net_3d(data_batch['x'][1], data_batch['x'][0], data_batch['x'][0][-1][0].item() + 1)
            unq_x = self.logits(feats).features
            # Remap voxel to points
            x = unq_x[indices]
            if self.dual_head:
                unq_x2 = self.logits2(feats).features
                preds['seg_logit2'] = unq_x2[indices]

        preds.update({
            # 'feats': feats,
            'seg_logit': x,
        })

        return preds


def test_Net2DSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      backbone_2d='UNetResNet34',
                      backbone_2d_kwargs={},
                      dual_head=True)

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)


def test_Net3DSeg():
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)

    feats = feats.cuda()

    net_3d = Net3DSeg(num_seg_classes,
                      dual_head=True,
                      backbone_3d='SCN',
                      backbone_3d_kwargs={'in_channels': in_channels})

    net_3d.cuda()
    out_dict = net_3d({
        'x': [coords, feats],
    })
    for k, v in out_dict.items():
        print('Net3DSeg:', k, v.shape)

if __name__ == '__main__':
    test_Net2DSeg()
    test_Net3DSeg()
