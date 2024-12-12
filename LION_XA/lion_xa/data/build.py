from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader, default_collate
from yacs.config import CfgNode as CN

from lion_xa.common.utils.torch_util import worker_init_fn
from lion_xa.data.collate import get_collate_scn
from lion_xa.common.utils.sampler import IterationBasedBatchSampler

def build_dataloader(cfg, mode='train', domain='source', start_iteration=0, halve_batch_size=False, nusc=None, trg_like=False):
    assert mode in ['train', 'val', 'test']
    dataset_cfg = cfg.get('DATASET_' + domain.upper())
    split = dataset_cfg[mode.upper()]
    is_train = 'train' in mode
    batch_size = cfg['TRAIN'].BATCH_SIZE if is_train else cfg['VAL'].BATCH_SIZE
    if halve_batch_size:
        batch_size = batch_size // 2

    # build dataset
    # Make a copy of dataset_kwargs so that we can pop augmentation afterwards without destroying the cfg.
    # Note that the build_dataloader fn is called twice for train and val.
    dataset_kwargs = CN(dataset_cfg.get(dataset_cfg.TYPE, dict()))
    if 'SCN' in cfg.MODEL_3D.keys():
        assert dataset_kwargs.full_scale == cfg.MODEL_3D.SCN.full_scale
    augmentation = dataset_kwargs.pop('augmentation')
    augmentation = augmentation if is_train else dict()
    
    if dataset_cfg.TYPE == 'NuScenesSCN':
        if nusc and domain == 'source':
            from lion_xa.data.nuscenes.nuscenes_dataloader_otf import TGLNuScenesSCN as NuScenesSCN
        elif nusc and domain == 'target': 
            from lion_xa.data.nuscenes.nuscenes_dataloader_otf import NuScenesSCN
        else:
            from lion_xa.data.nuscenes.nuscenes_dataloader import NuScenesSCN
        dataset = NuScenesSCN(split=split,
                              output_orig=not is_train,
                              model_type=cfg.MODEL_3D.TYPE,
                              nusc=nusc,
                              **dataset_kwargs,
                              **augmentation)
    elif dataset_cfg.TYPE == 'SemanticKITTISCN':
        if trg_like:
            from lion_xa.data.semantic_kitti.semantic_kitti_dataloader_otf import TGLSemanticKITTISCN as SemanticKITTISCN
        else:
            from lion_xa.data.semantic_kitti.semantic_kitti_dataloader_otf import SemanticKITTISCN
        dataset = SemanticKITTISCN(split=split,
                                   output_orig=not is_train,
                                   model_type=cfg.MODEL_3D.TYPE,
                                   **dataset_kwargs,
                                   **augmentation)
    elif dataset_cfg.TYPE == 'SemanticPOSSSCN':
        if domain == 'source':
            from lion_xa.data.semantic_poss.semantic_poss_dataloader import TGLSemanticPOSSSCN as SemanticPOSSSCN
        elif domain == 'target':
            from lion_xa.data.semantic_poss.semantic_poss_dataloader import SemanticPOSSSCN
        else:
            raise NotImplementedError('No other domain than source and target available for {}'.format(dataset_cfg.TYPE))
        dataset = SemanticPOSSSCN(split=split,
                                   output_orig=not is_train,
                                   model_type=cfg.MODEL_3D.TYPE,
                                   **dataset_kwargs,
                                   **augmentation)
    else:
        raise ValueError('Unsupported type of dataset: {}.'.format(dataset_cfg.TYPE))

    if 'SCN' in dataset_cfg.TYPE:
        collate_fn = get_collate_scn(is_train)
    else:
        collate_fn = default_collate

    if is_train:
        sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=cfg.DATALOADER.DROP_LAST)
        batch_sampler = IterationBasedBatchSampler(batch_sampler, cfg.SCHEDULER.MAX_ITERATION, start_iteration)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )

    return dataloader
