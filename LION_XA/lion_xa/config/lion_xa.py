"""LION_XA experiments configuration"""
import os.path as osp

from lion_xa.common.config.base import CN, _C

# public alias
cfg = _C
_C.VAL.METRIC = 'seg_iou'

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN.CLASS_WEIGHTS = []

# ---------------------------------------------------------------------------- #
# LION_XA options
# ---------------------------------------------------------------------------- #
_C.TRAIN.LION_XA = CN()
_C.TRAIN.LION_XA.lambda_xm_src = 0.0
_C.TRAIN.LION_XA.lambda_xm_tgl = 0.0
_C.TRAIN.LION_XA.lambda_xm_trg = 0.0
_C.TRAIN.LION_XA.lambda_minent = 0.0
_C.TRAIN.LION_XA.lambda_logcoral = 0.0
_C.TRAIN.LION_XA.lambda_pixels = 0.0
_C.TRAIN.LION_XA.lambda_G_trg_2d_pred = 0.0
_C.TRAIN.LION_XA.lambda_G_trg_3d_pred = 0.0
_C.TRAIN.LION_XA.lambda_G_trg_2d_feat = 0.0
_C.TRAIN.LION_XA.lambda_D_src_2d_pred = 0.0
_C.TRAIN.LION_XA.lambda_D_src_3d_pred = 0.0
_C.TRAIN.LION_XA.lambda_D_src_2d_feat = 0.0
_C.TRAIN.LION_XA.lambda_D_trg_2d_pred = 0.0
_C.TRAIN.LION_XA.lambda_D_trg_3d_pred = 0.0
_C.TRAIN.LION_XA.lambda_D_trg_2d_feat = 0.0

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASET_SOURCE = CN()
_C.DATASET_SOURCE.TYPE = ''
_C.DATASET_SOURCE.TRAIN = tuple()

_C.DATASET_TARGET = CN()
_C.DATASET_TARGET.TYPE = ''
_C.DATASET_TARGET.TRAIN = tuple()
_C.DATASET_TARGET.VAL = tuple()
_C.DATASET_TARGET.TEST = tuple()

# NuScenesSCN
_C.DATASET_SOURCE.NuScenesSCN = CN()
_C.DATASET_SOURCE.NuScenesSCN.root_dir = ''
_C.DATASET_SOURCE.NuScenesSCN.merge_classes = True
_C.DATASET_SOURCE.NuScenesSCN.lidarseg = False
# 3D
_C.DATASET_SOURCE.NuScenesSCN.scale = 20
_C.DATASET_SOURCE.NuScenesSCN.full_scale = 4096
# 3D augmentation
_C.DATASET_SOURCE.NuScenesSCN.augmentation = CN()
_C.DATASET_SOURCE.NuScenesSCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.NuScenesSCN.augmentation.flip_x = 0.5
_C.DATASET_SOURCE.NuScenesSCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.NuScenesSCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.NuScenesSCN.augmentation.width = 1024
_C.DATASET_SOURCE.NuScenesSCN.augmentation.height = 32
_C.DATASET_SOURCE.NuScenesSCN.augmentation.cut_image = True
_C.DATASET_SOURCE.NuScenesSCN.augmentation.cut_range = [512, 512]
_C.DATASET_SOURCE.NuScenesSCN.augmentation.remove_large = True
_C.DATASET_SOURCE.NuScenesSCN.augmentation.random_flip = True

# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.NuScenesSCN = CN(_C.DATASET_SOURCE.NuScenesSCN)

# SemanticKITTISCN
_C.DATASET_SOURCE.SemanticKITTISCN = CN()
_C.DATASET_SOURCE.SemanticKITTISCN.root_dir = ''
_C.DATASET_SOURCE.SemanticKITTISCN.trgl_dir = ''
_C.DATASET_SOURCE.SemanticKITTISCN.merge_classes = True
_C.DATASET_SOURCE.SemanticKITTISCN.nuscenes = False
# 3D
_C.DATASET_SOURCE.SemanticKITTISCN.scale = 20
_C.DATASET_SOURCE.SemanticKITTISCN.full_scale = 4096
# 3D augmentation
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation = CN()
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.width = 2048
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.height = 64
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.cut_image = True
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.cut_range = [512, 512]
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.remove_large = True
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.random_flip = True

# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.SemanticKITTISCN = CN(_C.DATASET_SOURCE.SemanticKITTISCN)

# SemanticPOSSSCN
_C.DATASET_SOURCE.SemanticPOSSSCN = CN()
_C.DATASET_SOURCE.SemanticPOSSSCN.preprocess_dir = ''
_C.DATASET_SOURCE.SemanticPOSSSCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.SemanticPOSSSCN.scale = 20
_C.DATASET_SOURCE.SemanticPOSSSCN.full_scale = 4096
# 3D augmentation
_C.DATASET_SOURCE.SemanticPOSSSCN.augmentation = CN()
_C.DATASET_SOURCE.SemanticPOSSSCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.SemanticPOSSSCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.SemanticPOSSSCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.SemanticPOSSSCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.SemanticPOSSSCN.augmentation.width = 1800
_C.DATASET_SOURCE.SemanticPOSSSCN.augmentation.height = 40
_C.DATASET_SOURCE.SemanticPOSSSCN.augmentation.cut_image = True
_C.DATASET_SOURCE.SemanticPOSSSCN.augmentation.cut_range = [512, 512]
_C.DATASET_SOURCE.SemanticPOSSSCN.augmentation.remove_large = True
_C.DATASET_SOURCE.SemanticPOSSSCN.augmentation.random_flip = True

# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.SemanticPOSSSCN = CN(_C.DATASET_SOURCE.SemanticPOSSSCN)

# ---------------------------------------------------------------------------- #
# Model 2D
# ---------------------------------------------------------------------------- #
_C.MODEL_2D = CN()
_C.MODEL_2D.TYPE = ''
_C.MODEL_2D.CKPT_PATH = ''
_C.MODEL_2D.NUM_CLASSES = 5
_C.MODEL_2D.DUAL_HEAD = False
# ---------------------------------------------------------------------------- #
# UNetResNet34 options
# ---------------------------------------------------------------------------- #
_C.MODEL_2D.UNetResNet34 = CN()
_C.MODEL_2D.UNetResNet34.pretrained = True
# ---------------------------------------------------------------------------- #
# SalsaNextSeg options
# ---------------------------------------------------------------------------- #
_C.MODEL_2D.SalsaNextSeg = CN()
_C.MODEL_2D.SalsaNextSeg.in_channels = 5

# ---------------------------------------------------------------------------- #
# Model 3D
# ---------------------------------------------------------------------------- #
_C.MODEL_3D = CN()
_C.MODEL_3D.TYPE = ''
_C.MODEL_3D.CKPT_PATH = ''
_C.MODEL_3D.NUM_CLASSES = 5
_C.MODEL_3D.DUAL_HEAD = False
# ----------------------------------------------------------------------------- #
# SCN options
# ----------------------------------------------------------------------------- #
_C.MODEL_3D.SCN = CN()
_C.MODEL_3D.SCN.in_channels = 1
_C.MODEL_3D.SCN.m = 16  # number of unet features (multiplied in each layer)
_C.MODEL_3D.SCN.block_reps = 1  # block repetitions
_C.MODEL_3D.SCN.residual_blocks = False  # ResNet style basic blocks
_C.MODEL_3D.SCN.full_scale = 4096
_C.MODEL_3D.SCN.num_planes = 7
# ----------------------------------------------------------------------------- #
# PVD options
# ----------------------------------------------------------------------------- #
_C.MODEL_3D.PVD = CN()
_C.MODEL_3D.PVD.output_shape = [480, 360, 32]
_C.MODEL_3D.PVD.fea_dim = 8
_C.MODEL_3D.PVD.out_fea_dim = 128
_C.MODEL_3D.PVD.num_input_features = 16
_C.MODEL_3D.PVD.use_norm = True
_C.MODEL_3D.PVD.init_size = 16
_C.MODEL_3D.PVD.num_planes = 7

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# @ will be replaced by config path
_C.OUTPUT_DIR = osp.expanduser('/workspace/lion-xa/LION_XA/output/@')