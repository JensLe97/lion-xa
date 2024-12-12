from lion_xa.models.lion_xa_arch import Net2DSeg, Net3DSeg
from lion_xa.models.discriminator import Discriminator_2d_UNetResNet, Discriminator_2d_SalsaNext, Discriminator_3d
from lion_xa.models.metric import SegIoU


def build_model_2d(cfg):
    model = Net2DSeg(num_classes=cfg.MODEL_2D.NUM_CLASSES,
                     backbone_2d=cfg.MODEL_2D.TYPE,
                     backbone_2d_kwargs=cfg.MODEL_2D[cfg.MODEL_2D.TYPE],
                     dual_head=cfg.MODEL_2D.DUAL_HEAD
                     )
    train_metric = SegIoU(cfg.MODEL_2D.NUM_CLASSES, name='seg_iou_2d')
    return model, train_metric


def build_model_3d(cfg):
    model = Net3DSeg(num_classes=cfg.MODEL_3D.NUM_CLASSES,
                     backbone_3d=cfg.MODEL_3D.TYPE,
                     backbone_3d_kwargs=cfg.MODEL_3D[cfg.MODEL_3D.TYPE],
                     dual_head=cfg.MODEL_3D.DUAL_HEAD
                     )
    train_metric = SegIoU(cfg.MODEL_3D.NUM_CLASSES, name='seg_iou_3d')
    return model, train_metric

def build_model_dis_feat(cfg):
    if cfg.MODEL_2D.TYPE == 'UNetResNet34':
        feat_channels = 64
        model = Discriminator_2d_UNetResNet(input_channels=feat_channels)
    elif cfg.MODEL_2D.TYPE == 'SalsaNextSeg':
        feat_channels = 32
        model = Discriminator_2d_SalsaNext(input_channels=feat_channels)
    else:
        raise NotImplementedError('2D backbone {} not supported'.format(cfg.MODEL_2D.TYPE))
    return model

def build_model_dis_pred(cfg):
    model = Discriminator_3d(input_channels=cfg.MODEL_2D.NUM_CLASSES)
    return model