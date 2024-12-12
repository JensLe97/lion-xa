import os
import os.path as osp
from pathlib import Path
import numpy as np
import pickle
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import sys
sys.path.append('./LION_XA')
import yaml

from lion_xa.data.semantic_poss import splits
from lion_xa.data.range_img.laserscan import SemLaserScan
from lion_xa.data.range_img.range_img import get_range_image
# prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')

SENSOR_FILE_NAME = "Pandora.yml"
DATASET_FILE_NAME = "semantic_poss.yaml"

class DummyDataset(Dataset):
    """Use torch dataloader for multiprocessing"""
    def __init__(self, root_dir, split_name, with_inpaint=True):
        self.root_dir = root_dir
        self.data = []
        self.glob_frames(getattr(splits, split_name))
        self.with_inpaint = with_inpaint

        with open(Path(__file__).parent / SENSOR_FILE_NAME) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            sensor_fov_up = data["v_start_angle"]
            sensor_fov_down = data["v_end_angle"]
            sensor_v_angles = data['v_angles']
            sensor_img_H = data["num_beams"]
            sensor_img_W = data["num_hsize"]
            self.norm_remission = data['norm_remission']

        DA = False
        flip_sign = False
        rot = False
        drop_points = False
        if split_name == 'train' and with_inpaint:
            rot = True

        with open(Path(__file__).parent / DATASET_FILE_NAME) as f:
            self.label_config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.mapper = SemLaserScan(self.label_config["color_map"],
                                    project=True,
                                    H=sensor_img_H,
                                    W=sensor_img_W,
                                    fov_up=sensor_fov_up,
                                    fov_down=sensor_fov_down,
                                    DA=DA,
                                    flip_sign=flip_sign,
                                    drop_points=drop_points,
                                    rot=rot,
                                    v_angles=sensor_v_angles)

    def glob_frames(self, scenes):
        for scene in scenes:
            glob_path = osp.join(self.root_dir, 'dataset', 'sequences', scene, 'velodyne', '*.bin')
            lidar_paths = sorted(glob.glob(glob_path))

            for lidar_path in lidar_paths:
                basename = osp.basename(lidar_path)
                frame_id = osp.splitext(basename)[0]
                assert frame_id.isdigit()
                data = {
                    'lidar_path': lidar_path,
                    'label_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'labels',
                                           frame_id + '.label'),
                }
                for k, v in data.items():
                    if isinstance(v, str):
                        if not osp.exists(v):
                            raise IOError('File not found {}'.format(v))
                self.data.append(data)

    def __getitem__(self, index):
        data_dict = self.data[index].copy()

        scan = np.fromfile(data_dict['lidar_path'], dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, :3]

        label_3d = np.fromfile(data_dict['label_path'], dtype=np.uint32)
        label_3d = label_3d.reshape((-1))
        label_3d = label_3d & 0xFFFF  # get lower half for semantics

        _, _, range_img, label_2d, inpainted_mask, range_img_indices = get_range_image(
            self.mapper, self.norm_remission, self.label_config, data_dict, with_inpaint=self.with_inpaint)

        data_dict['points'] = points
        data_dict['seg_label_3d'] = label_3d.astype(np.int16)
        data_dict['range_img'] = range_img
        data_dict['inpainted_mask'] = inpainted_mask
        data_dict['seg_label_2d'] = label_2d.astype(np.int16)
        data_dict['range_img_indices'] = range_img_indices

        return data_dict

    def __len__(self):
        return len(self.data)

def preprocess(split_name, root_dir, out_dir):
    pkl_data = []

    dataloader = DataLoader(DummyDataset(root_dir, split_name), num_workers=8)

    num_skips = 0
    for i, data_dict in enumerate(dataloader):
        # data error leads to returning empty dict
        if not data_dict:
            print('empty dict, continue')
            num_skips += 1
            continue
        for k, v in data_dict.items():
            data_dict[k] = v[0]
        print('{}/{} {}'.format(i, len(dataloader), data_dict['lidar_path']))

        # convert to relative path
        lidar_path = data_dict['lidar_path'].replace(root_dir + '/', '')

        # append data
        out_dict = {
            'points': data_dict['points'].numpy(),
            'seg_label_3d': data_dict['seg_label_3d'].numpy(),
            'range_img': data_dict['range_img'].numpy(),
            'inpainted_mask': data_dict['inpainted_mask'].numpy(),
            'seg_label_2d': data_dict['seg_label_2d'].numpy(),
            'range_img_indices': data_dict['range_img_indices'].numpy(),
            'lidar_path': lidar_path,
        }
        pkl_data.append(out_dict)

    print('Skipped {} files'.format(num_skips))

    # save to pickle file
    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, '{}.pkl'.format(split_name))
    with open(save_path, 'wb') as f:
        pickle.dump(pkl_data, f)
        print('Wrote preprocessed data to ' + save_path)

def calc_statistics(split_name, root_dir):
    dataloader = DataLoader(DummyDataset(root_dir, split_name, False), num_workers=8)

    psum    = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    count = 0

    num_skips = 0
    for i, data_dict in enumerate(dataloader):
        # data error leads to returning empty dict
        if not data_dict:
            print('empty dict, continue')
            num_skips += 1
            continue
        for k, v in data_dict.items():
            data_dict[k] = v[0]
        print('{}/{} {}'.format(i, len(dataloader), data_dict['lidar_path']))
        
        inputs = data_dict['range_img'].clone()
        count += (inputs[0]!=-1).sum()
        remove = (inputs[0]==-1).repeat(5, 1, 1)
        inputs[remove] = 0

        psum    += inputs.sum(axis        = [1, 2])
        psum_sq += (inputs ** 2).sum(axis = [1, 2])

    # mean and std
    total_mean = psum / count.item()
    total_var  = (psum_sq / count.item()) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    torch.set_printoptions(precision=7)
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))
    print('\t\trange, remission, norm_x,   norm_y,   norm_z')

if __name__ == '__main__':
    root_dir = '/teamspace/studios/this_studio/data/datasets/SemanticPOSS'
    out_dir = '/teamspace/studios/this_studio/data/datasets/SemanticPOSS/semantic_poss_preprocess_lidar'
    create_pkl = True # False = only calculation
    if create_pkl:
        preprocess('val', root_dir, out_dir)
        preprocess('train', root_dir, out_dir)
        preprocess('test', root_dir, out_dir)
    else:
        calc_statistics('train', root_dir)