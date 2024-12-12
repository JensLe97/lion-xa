import sys
sys.path.append('./LION_XA')
import argparse
import os
import os.path as osp
import logging
import time
import socket
import json

import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from lion_xa.common.solver.build import build_optimizer, build_scheduler
from lion_xa.common.utils.checkpoint import CheckpointerV2
from lion_xa.models.build import build_model_2d, build_model_3d
from lion_xa.data.build import build_dataloader
from lion_xa.common.utils.logger import setup_logger
from lion_xa.common.utils.metric_logger import MetricLogger
from lion_xa.common.utils.torch_util import set_random_seed
from lion_xa.eval.validate import validate
from lion_xa.models.losses import entropy_loss

"""LiDAR-Only Cross-Modal Adversarial Training (LION_XA)

This class can be used as a trainer for LiDAR point cloud Domain Adaptation
Inputs are 3D LiDAR point clouds from vehicle LiDAR sensors.
One Source domain with labeled point (semantic class for each point).
One Target domain with no labeled data.
Domains (cross-sensor and cross-city/country)
    1. SemanticPOSS <-> SemanticKITTI
    2. AD2D <-> SemanticKITTI
    3. nuScenes <-> SemanticKITTI
    4. nuScenes Boston (USA) <-> nuScenes Singapore

This approach is based on adversarial training using generators and discriminators
It can be used for semantic segmentation and could be applied to instance segmentation
when adding an instance head to the segmentor network.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='LION_XA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='Path to config file',
        type=str,
    )
    return parser.parse_args()

def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger

def train(cfg, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, tensorboard logger
    # ---------------------------------------------------------------------------- #
    logger = logging.getLogger('lion_xa.init')
    logger.info('Initialize LION_XA Trainer')

    set_random_seed(cfg.RNG_SEED)

    # Build 2D model
    model_2d, train_metric_2d = build_model_2d(cfg)

    # Build 3D model
    model_3d, train_metric_3d = build_model_3d(cfg)
    
    # model_2d = DataParallel(model_2d)
    # model_3d = DataParallel(model_3d)

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # Build optimizer
    optimizer_2d = build_optimizer(cfg.OPTIMIZER_2D, model_2d)
    optimizer_3d = build_optimizer(cfg.OPTIMIZER_3D, model_3d)

    # Build lr scheduler
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)

    # Build checkpointer
    checkpointer_2d = CheckpointerV2(model_2d,
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d = checkpointer_2d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_3d = CheckpointerV2(model_3d,
                                     optimizer=optimizer_3d,
                                     scheduler=scheduler_3d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d = checkpointer_3d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # Build tensorboard logger
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Training Loop
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_2d.get('iteration', 0)
    # Build Dataloader
    # Reset the random seed again in case the initialization of models changed the random state.
    set_random_seed(cfg.RNG_SEED)

    nusc = None
    if cfg.DATASET_SOURCE.TYPE == 'NuScenesSCN' and cfg.DATASET_TARGET.TYPE == 'SemanticKITTISCN':
        dataroot = cfg.DATASET_SOURCE.NuScenesSCN.root_dir
        nusc = True
    elif cfg.DATASET_TARGET.TYPE == 'NuScenesSCN' and cfg.DATASET_SOURCE.TYPE == 'SemanticKITTISCN':
        dataroot = cfg.DATASET_TARGET.NuScenesSCN.root_dir
        nusc = True
    if nusc:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration, nusc=nusc)
    train_dataloader_tgl = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration, nusc=nusc, trg_like=True)
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration, nusc=nusc)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target', nusc=nusc)
    
    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d_point': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None)
    }

    best_metric_iter = {'2d_point': -1, '3d': -1}

    logger = logging.getLogger('lion_xa.train')
    logger.info(f'Start training from iteration {start_iteration}')

    # add metrics
    train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d])
    val_metric_logger = MetricLogger(delimiter='  ')

    def setup_train():
        # set training mode
        model_2d.train()
        model_3d.train()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_2d.eval()
        model_3d.eval()
        # reset metric
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    setup_train()
    end = time.time()

    train_iter_src = enumerate(train_dataloader_src)
    train_iter_tgl = enumerate(train_dataloader_tgl)
    train_iter_trg = enumerate(train_dataloader_trg)

    for iteration in range(start_iteration, max_iteration):
        _, data_batch_src = next(train_iter_src)
        _, data_batch_tgl = next(train_iter_tgl)
        _, data_batch_trg = next(train_iter_trg)
        data_time = time.time() - end

        # source
        if cfg.MODEL_3D.TYPE == 'PVD':
            data_batch_src['x'][0] = data_batch_src['x'][0].cuda()
        data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
        data_batch_src['seg_label_3d'] = data_batch_src['seg_label_3d'].cuda()
        data_batch_src['range_img'] = data_batch_src['range_img'].cuda()
        data_batch_src['seg_label_2d'] = data_batch_src['seg_label_2d'].cuda()
        # target-like
        if cfg.MODEL_3D.TYPE == 'PVD':
            data_batch_tgl['x'][0] = data_batch_tgl['x'][0].cuda()
        data_batch_tgl['x'][1] = data_batch_tgl['x'][1].cuda()
        data_batch_tgl['seg_label_3d'] = data_batch_tgl['seg_label_3d'].cuda()
        data_batch_tgl['range_img'] = data_batch_tgl['range_img'].cuda()
        data_batch_tgl['seg_label_2d'] = data_batch_tgl['seg_label_2d'].cuda()
        # target
        if cfg.MODEL_3D.TYPE == 'PVD':
            data_batch_trg['x'][0] = data_batch_trg['x'][0].cuda()
        data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
        data_batch_trg['seg_label_3d'] = data_batch_trg['seg_label_3d'].cuda()
        data_batch_trg['range_img'] = data_batch_trg['range_img'].cuda()
        data_batch_trg['seg_label_2d'] = data_batch_trg['seg_label_2d'].cuda()

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()

        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch_src)
        preds_3d = model_3d(data_batch_src)

        # Segmentation loss: cross entropy
        seg_loss_src_2d_points = F.cross_entropy(preds_2d['point_seg_logit'], data_batch_src['seg_label_3d'], weight=class_weights)
        seg_loss_src_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_src['seg_label_3d'], weight=class_weights)
        train_metric_logger.update(seg_loss_src_2d_points=seg_loss_src_2d_points, seg_loss_src_3d=seg_loss_src_3d)
        loss_2d = seg_loss_src_2d_points
        loss_3d = seg_loss_src_3d

        if cfg.TRAIN.LION_XA.lambda_pixels > 0:
            seg_loss_src_2d_pixels = F.cross_entropy(preds_2d['pixel_seg_logit'], data_batch_src['seg_label_2d'], weight=class_weights)
            train_metric_logger.update(seg_loss_src_2d_pixels=seg_loss_src_2d_pixels)
            loss_2d += cfg.TRAIN.LION_XA.lambda_pixels * seg_loss_src_2d_pixels

        if cfg.TRAIN.LION_XA.lambda_xm_src > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['point_seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['point_seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_src_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['point_seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_src_2d=xm_loss_src_2d,
                                       xm_loss_src_3d=xm_loss_src_3d)
            loss_2d += cfg.TRAIN.LION_XA.lambda_xm_src * xm_loss_src_2d
            loss_3d += cfg.TRAIN.LION_XA.lambda_xm_src * xm_loss_src_3d

        # update metric (e.g. IoU)
        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d, data_batch_src)
            train_metric_3d.update_dict(preds_3d, data_batch_src)

        # backward
        loss_2d.backward()
        loss_3d.backward()

        # ---------------------------------------------------------------------------- #
        # Train on target-like
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch_tgl)
        preds_3d = model_3d(data_batch_tgl)

        # Segmentation loss: cross entropy
        seg_loss_tgl_2d_points = F.cross_entropy(preds_2d['point_seg_logit'], data_batch_tgl['seg_label_3d'], weight=class_weights)
        seg_loss_tgl_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_tgl['seg_label_3d'], weight=class_weights)
        train_metric_logger.update(seg_loss_tgl_2d_points=seg_loss_tgl_2d_points, seg_loss_tgl_3d=seg_loss_tgl_3d)
        loss_2d = seg_loss_tgl_2d_points
        loss_3d = seg_loss_tgl_3d

        if cfg.TRAIN.LION_XA.lambda_pixels > 0:
            seg_loss_tgl_2d_pixels = F.cross_entropy(preds_2d['pixel_seg_logit'], data_batch_tgl['seg_label_2d'], weight=class_weights)
            train_metric_logger.update(seg_loss_tgl_2d_pixels=seg_loss_tgl_2d_pixels)
            loss_2d += cfg.TRAIN.LION_XA.lambda_pixels * seg_loss_tgl_2d_pixels

        if cfg.TRAIN.LION_XA.lambda_xm_tgl > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['point_seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['point_seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_tgl_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_tgl_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['point_seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_tgl_2d=xm_loss_tgl_2d,
                                       xm_loss_tgl_3d=xm_loss_tgl_3d)
            loss_2d += cfg.TRAIN.LION_XA.lambda_xm_tgl * xm_loss_tgl_2d
            loss_3d += cfg.TRAIN.LION_XA.lambda_xm_tgl * xm_loss_tgl_3d

        # update metric (e.g. IoU)
        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d, data_batch_tgl)
            train_metric_3d.update_dict(preds_3d, data_batch_tgl)

        # backward
        loss_2d.backward()
        loss_3d.backward()

        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch_trg)
        preds_3d = model_3d(data_batch_trg)
        
        loss_2d = []
        loss_3d = []

        if cfg.TRAIN.LION_XA.lambda_xm_trg > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['point_seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['point_seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_trg_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['point_seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                       xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.LION_XA.lambda_xm_trg * xm_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.LION_XA.lambda_xm_trg * xm_loss_trg_3d)
        if cfg.TRAIN.LION_XA.lambda_minent > 0:
            # MinEnt
            minent_loss_trg_2d_points = entropy_loss(F.softmax(preds_2d['point_seg_logit'], dim=1))
            minent_loss_trg_3d = entropy_loss(F.softmax(preds_3d['seg_logit'], dim=1))
            train_metric_logger.update(minent_loss_trg_2d_points=minent_loss_trg_2d_points,
                                       minent_loss_trg_3d=minent_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.LION_XA.lambda_minent * minent_loss_trg_2d_points)
            loss_3d.append(cfg.TRAIN.LION_XA.lambda_minent * minent_loss_trg_3d)
            
            if cfg.TRAIN.LION_XA.lambda_pixels > 0:
                minent_loss_trg_2d_pixels = entropy_loss(F.softmax(preds_2d['pixel_seg_logit'], dim=1))
                loss_2d.append(cfg.TRAIN.LION_XA.lambda_pixels * minent_loss_trg_2d_pixels)
                train_metric_logger.update(minent_loss_trg_2d_pixels=minent_loss_trg_2d_pixels)
        
        sum(loss_2d).backward()
        sum(loss_3d).backward()

        optimizer_2d.step()
        optimizer_3d.step()

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)

        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr=optimizer_2d.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric['2d_point']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)

        # ---------------------------------------------------------------------------- #
        # Validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()

            validate(cfg,
                     model_2d,
                     model_3d,
                     val_dataloader,
                     val_metric_logger)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in ['2d_point', '3d']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter

            # restore training
            setup_train()

        scheduler_2d.step()
        scheduler_3d.step()
        end = time.time()
    for modality in ['2d_point', '3d']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(modality.upper(),
                                                                     cfg.VAL.METRIC,
                                                                     best_metric[modality] * 100,
                                                                     best_metric_iter[modality]))

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
        os.makedirs(output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = f'{timestamp}.{hostname}'

    logger = setup_logger('lion_xa', output_dir, comment=f'train.{run_name}')
    logger.info(f'{torch.cuda.device_count()} GPUs available')
    logger.info(f'Loaded configuration file {args.config_file}')
    logger.info(f'Running with config:\n{cfg}')
    
    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    assert cfg.TRAIN.LION_XA.lambda_xm_src > 0 or cfg.TRAIN.LION_XA.lambda_xm_trg > 0 or \
           cfg.TRAIN.LION_XA.lambda_minent > 0 or cfg.TRAIN.LION_XA.lambda_pixels > 0
    train(cfg, output_dir, run_name)
    train(cfg, output_dir, run_name)

if __name__ == '__main__':
    main()
    