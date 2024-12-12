import logging
import time
import os

import torch
import torch.nn.functional as F

from lion_xa.eval.evaluate import Evaluator

def validate(cfg,
             model_2d,
             model_3d,
             dataloader,
             val_metric_logger,
             knn=False,
             class_weights=None,
             visualize=False):
    logger = logging.getLogger('lion_xa.validate')
    logger.info('Validation')

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d_pixel = Evaluator(class_names)
    evaluator_2d_point = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if model_3d else None

    if knn:
        from lion_xa.models.KNN import KNN
        post_cfg = cfg.get('VAL').get('KNN')
        post = KNN(cfg.MODEL_2D.NUM_CLASSES, **post_cfg)

    frames = [96]
    pred_folder = 'predictions' # 'predictions', 'baseline'
    batch_size = cfg.VAL.BATCH_SIZE
    iters = [(frame // batch_size) for frame in frames]
    finished = False

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            
            if visualize:
                if finished:
                    break
                if iteration not in iters:
                    continue
                frame = frames[iters.index(iteration)]
                if frame == frames[-1]:
                    finished = True

            data_time = time.time() - end

            seg_label_2d = data_batch['seg_label_2d']
            seg_label_2d[data_batch['inpainted_mask'] == 0] = -100

            # copy data from cpu to gpu
            if cfg.MODEL_3D.TYPE == 'PVD':
                data_batch['x'][0] = data_batch['x'][0].cuda()
            data_batch['x'][1] = data_batch['x'][1].cuda()
            data_batch['seg_label_3d'] = data_batch['seg_label_3d'].cuda()
            data_batch['range_img'] = data_batch['range_img'].cuda()
            data_batch['seg_label_2d'] = data_batch['seg_label_2d'].cuda()

            # predict
            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch) if model_3d else None

            probs_2d = F.softmax(preds_2d['point_seg_logit'], dim=1)
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            
            if class_weights is not None:
                probs_2d = probs_2d * class_weights
                probs_3d = probs_3d * class_weights
                
            pred_label_pixel_grid_2d = preds_2d['pixel_seg_logit'].argmax(1)
            if knn:
                pred_label_pixel_2d = torch.zeros_like(data_batch['seg_label_3d'])
                # softmax average (ensembling)
                pred_label_voxel_ensemble = torch.mul(probs_2d, 0.5) + probs_3d if model_3d else None
            else:
                pred_label_pixel_2d = pred_label_pixel_grid_2d.cpu().numpy()
                # softmax average (ensembling)
                pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if model_3d else None
            pred_label_voxel_2d = preds_2d['point_seg_logit'].argmax(1).cpu().numpy()
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # get original point cloud from before voxelization
            seg_label_3d = data_batch['orig_seg_label_3d']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(seg_label_3d)):

                if visualize and batch_ind != frame % batch_size:
                    continue
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                # assert np.all(curr_points_idx)

                curr_seg_label_3d = seg_label_3d[batch_ind][curr_points_idx]
                right_idx = left_idx + curr_points_idx.sum()
                if knn:
                    # get depth (range) of all points
                    depth = torch.linalg.norm(data_batch['points'][batch_ind], 2, dim=1)
                    depth[depth==0] = depth[depth==0] + 1e-10
                    pred_label_pixel_2d, probs_2d_knn = post(data_batch['org_proj_range'][batch_ind].cuda(),
                                               depth.cuda(),
                                               pred_label_pixel_grid_2d[batch_ind],
                                               torch.from_numpy(data_batch['range_img_indices'][batch_ind][:, 0]).cuda(),
                                               torch.from_numpy(data_batch['range_img_indices'][batch_ind][:, 1]).cuda())
                    pred_label_pixel_2d = pred_label_pixel_2d.cpu().numpy()
                    pred_label_ensemble = (pred_label_voxel_ensemble[left_idx:right_idx] + torch.mul(probs_2d_knn, 0.5)).argmax(1).cpu().numpy() if model_3d else None
                else:
                    curr_seg_label_2d = seg_label_2d[batch_ind]
                    pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if model_3d else None

                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None

                # evaluate
                if knn:
                    evaluator_2d_pixel.update(pred_label_pixel_2d, curr_seg_label_3d)
                else:
                    evaluator_2d_pixel.update(pred_label_pixel_2d[batch_ind], curr_seg_label_2d)
                evaluator_2d_point.update(pred_label_2d, curr_seg_label_3d)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label_3d)
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label_3d)

                if visualize:
                    save_path = cfg.DATASET_TARGET.NuScenesSCN.root_dir + f"/{pred_folder}/point_cloud_" + str(iteration*batch_size+batch_ind) 
                    if pred_folder == 'baseline':
                        pred_label_ensemble.tofile(save_path)
                    else:
                        predictions = [pred_label_2d, pred_label_3d, pred_label_ensemble]
                        for i, pred in enumerate(predictions):
                            pred.tofile(save_path + "_" + str(i) + ".label")

                left_idx = right_idx

            seg_loss_2d_pixel = F.cross_entropy(preds_2d['pixel_seg_logit'], data_batch['seg_label_2d'])
            seg_loss_2d_point = F.cross_entropy(preds_2d['point_seg_logit'], data_batch['seg_label_3d'])
            seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch['seg_label_3d']) if model_3d else None
            val_metric_logger.update(seg_loss_2d_point=seg_loss_2d_point, seg_loss_2d_pixel=seg_loss_2d_pixel)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        val_metric_logger.update(seg_iou_2d_point=evaluator_2d_point.overall_iou, seg_iou_2d_pixel=evaluator_2d_pixel.overall_iou)
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
        eval_list = [('2D Point', evaluator_2d_point), ('2D Pixel', evaluator_2d_pixel)]
        if model_3d:
            eval_list.extend([('3D', evaluator_3d), ('2D+3D', evaluator_ensemble)])
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))