#!/usr/bin/env python
import os
import os.path as osp
import numpy as np
import argparse
import logging
import time
import socket
import warnings

import torch

import sys
sys.path.append('./LION_XA')
from lion_xa.common.utils.checkpoint import CheckpointerV2
from lion_xa.common.utils.logger import setup_logger
from lion_xa.common.utils.metric_logger import MetricLogger
from lion_xa.common.utils.torch_util import set_random_seed
from lion_xa.models.build import build_model_2d, build_model_3d
from lion_xa.data.build import build_dataloader
from lion_xa.eval.validate import validate


def parse_args():
    parser = argparse.ArgumentParser(description='LION_XA test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('ckpt2d', type=str, help='path to checkpoint file of the 2D model')
    parser.add_argument('ckpt3d', type=str, help='path to checkpoint file of the 3D model')
    args = parser.parse_args()
    return args


def test(cfg, args, output_dir=''):
    logger = logging.getLogger('xmuda.test')

    # build 2d model
    model_2d = build_model_2d(cfg)[0]

    # build 3d model
    model_3d = build_model_3d(cfg)[0]

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build checkpointer
    checkpointer_2d = CheckpointerV2(model_2d, save_dir=output_dir, logger=logger)
    if args.ckpt2d:
        # load weight if specified
        weight_path = args.ckpt2d.replace('@', output_dir)
        checkpointer_2d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_2d.load(None, resume=True)
    checkpointer_3d = CheckpointerV2(model_3d, save_dir=output_dir, logger=logger)
    if args.ckpt3d:
        # load weight if specified
        weight_path = args.ckpt3d.replace('@', output_dir)
        checkpointer_3d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_3d.load(None, resume=True)

    # build dataset
    nusc = None
    if cfg.DATASET_TARGET.TYPE == 'NuScenesSCN' and cfg.DATASET_SOURCE.TYPE == 'SemanticKITTISCN':
        dataroot = cfg.DATASET_TARGET.NuScenesSCN.root_dir
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
        test_dataloader = build_dataloader(cfg, mode='val', domain='target', nusc=nusc)
    else:
        test_dataloader = build_dataloader(cfg, mode='test', domain='target', nusc=nusc)

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #

    set_random_seed(cfg.RNG_SEED)
    test_metric_logger = MetricLogger(delimiter='  ')
    model_2d.eval()
    model_3d.eval()

    beta = 2
    weights_class = np.array(cfg.TRAIN.CLASS_WEIGHTS, float)
    m = np.max(weights_class)
    for i in range(np.size(weights_class)):
        weights_class[i] = ((beta - 1) * weights_class[i] + m - beta) / (m - 1)

    class_weights = torch.tensor(weights_class.tolist()).cuda() if cfg.DATASET_TARGET.TYPE == 'SemanticPOSSSCN' else None
    validate(cfg, model_2d, model_3d, test_dataloader, test_metric_logger, knn=False, class_weights=class_weights, visualize=False)


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from lion_xa.common.config import purge_cfg
    from lion_xa.config.lion_xa import cfg
    cfg.merge_from_file(args.config_file)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = f'{timestamp}.{hostname}'

    logger = setup_logger('lion_xa', output_dir, comment='test.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    test(cfg, args, output_dir)


if __name__ == '__main__':
    main()
