import os.path as osp
from pathlib import Path
import pickle
import numpy as np
from torch.utils.data import Dataset
import yaml

import sys
sys.path.append('./LION_XA')
from lion_xa.data.utils.augmentation_3d import augment_and_scale_3d
from lion_xa.data.utils.augmentation_2d import augment_and_scale_2d
from lion_xa.data.range_img.range_img import depth_to_normal

DATASET_FILE_NAME = "semantic_poss.yaml"

class SemanticPOSSBase(Dataset):
    """SemanticPOSS dataset"""

    # https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    id_to_class_name = {
        0: "unlabeled",
        4: "1 person",
        5: "2+ person",
        6: "rider",
        7: "car",
        8: "trunk",
        9: "plants",
        10: "traffic sign 1", # standing sign
        11: "traffic sign 2", # hanging sign
        12: "traffic sign 3", # high/big hanging sign
        13: "pole",
        14: "trashcan",
        15: "building",
        16: "cone/stone",
        17: "fence",
        21: "bike",
        22: "ground", # class definition
    }

    class_name_to_id = {v: k for k, v in id_to_class_name.items()}

    # use those categories if merge_classes == True (common with KITTI)
    categories = {
        'person': ['1 person', '2+ person'],
        'bicyclist': ['rider'],
        'car': ['car'],
        'trunk': ['trunk'],
        'vegetation': ['plants'],
        'traffic-sign': ['traffic sign 1', 'traffic sign 2', 'traffic sign 3'],
        'pole': ['pole'],
        'object': ['trashcan', 'cone/stone'],
        'building': ['building'],
        'fence': ['fence'],
        'bike': ['bike'],
        'ground': ['ground'],
    }

    def __init__(self,
                 split,
                 preprocess_dir,
                 merge_classes=False
                 ):

        self.split = split
        self.preprocess_dir = preprocess_dir

        print("Initialize SemanticPOSS dataloader")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))

        if merge_classes:
            highest_id = list(self.id_to_class_name.keys())[-1]
            self.label_mapping = -100 * np.ones(highest_id + 2, dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_name_to_id[class_name]] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class SemanticPOSSSCN(SemanticPOSSBase):
    def __init__(self,
                 split,
                 preprocess_dir,
                 merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_y=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 # 2D augmentation:
                 width=1800,
                 height=40,
                 cut_image=False,
                 cut_range=[512, 512],
                 remove_large=False,
                 random_flip=False,
                 output_orig=False,
                 model_type='SCN'
                 ):
        super().__init__(split,
                         preprocess_dir,
                         merge_classes=merge_classes)

        self.output_orig = output_orig
        self.model_type = model_type

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_y = flip_y
        self.rot_z = rot_z
        self.transl = transl

        # 2D augmentation
        self.opt = {'width': width,
                    'height': height,
                    'cut_image': cut_image,
                    'cut_range': cut_range,
                    'remove_large': remove_large,
                    'random_flip': random_flip,
                    }

        with open(Path(__file__).parent / DATASET_FILE_NAME) as f:
            self.label_config = yaml.load(f, Loader=yaml.FullLoader)

    def __getitem__(self, index):
        data_dict = self.data[index]

        points = data_dict['points'].copy()
        seg_label_3d = data_dict['seg_label_3d']

        out_dict = {}

        range_img = data_dict['range_img'].copy()
        inpainted_mask = data_dict['inpainted_mask'].copy()
        seg_label_2d = data_dict['seg_label_2d'].copy()
        range_img_indices = data_dict['range_img_indices'].copy()

        if self.label_mapping is not None:
            seg_label_3d = self.label_mapping[seg_label_3d]
            seg_label_2d = self.label_mapping[seg_label_2d]
            
        proj_range = range_img[0, :, :]
        proj_remission = range_img[1, :, :]

        proj_range, proj_remission, inpainted_mask, seg_label_2d, range_img_indices, keep_idx = augment_and_scale_2d(
            proj_range, proj_remission, inpainted_mask, seg_label_2d, range_img_indices, self.opt)

        points = points[keep_idx]
        seg_label_3d = seg_label_3d[keep_idx]
        
        data_statics = self.label_config['data_statics']

        org_proj_range = proj_range.copy()

        # Normalize the normal vectors
        proj_normal_mean = data_statics['proj_norm_mean']
        proj_normal_std = data_statics['proj_norm_std']
        proj_normal = depth_to_normal(proj_range)
        proj_normal = (proj_normal - proj_normal_mean) / proj_normal_std
        
        # Normalize range
        proj_range_mean = data_statics['proj_range_mean']
        proj_range_std = data_statics['proj_range_std']
        proj_range = (proj_range-proj_range_mean)/proj_range_std

        # Normalize remission
        proj_remission_mean = data_statics['proj_remission_mean']
        proj_remission_std = data_statics['proj_remission_std']
        proj_remission = (proj_remission-proj_remission_mean) / proj_remission_std
        
        proj_range = np.expand_dims(proj_range, axis=2)
        proj_remission = np.expand_dims(proj_remission, axis=2)

        # Finished Range image
        range_image = np.concatenate((proj_range, proj_remission, proj_normal), axis=2)
        range_image = np.transpose(range_image, axes=[2, 0, 1])

        # Mask the range image
        # range_img *= inpainted_mask.repeat(5, 1, 1)

        out_dict['range_img'] = range_image.astype(np.float32)
        out_dict['inpainted_mask'] = inpainted_mask
        out_dict['seg_label_2d'] = seg_label_2d.astype(np.int64)

        # 3D data augmentation and scaling from points to voxel indices
        # POSS lidar coordinates: x (front), y (left), z (up)
        coords, point_feat = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                        flip_y=self.flip_y, rot_z=self.rot_z, transl=self.transl, 
                                        model_type=self.model_type)

        # only use voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)

        out_dict['coords'] = coords[idxs]
        out_dict['feats'] = point_feat[idxs] if point_feat is not None else \
                            np.ones([len(idxs), 1], np.float32) # simply use 1 as feature
        out_dict['seg_label_3d'] = seg_label_3d[idxs].astype(np.int64)
        out_dict['range_img_indices'] = range_img_indices[idxs]

        if self.output_orig:
            out_dict.update({
                'orig_seg_label_3d': seg_label_3d.astype(np.int64),
                'orig_points_idx': idxs,
                'points': points[idxs],
                'org_proj_range': org_proj_range
            })

        return out_dict

class TGLSemanticPOSSSCN(SemanticPOSSSCN):
    def __init__(self,
                 split,
                 preprocess_dir,
                 **kwargs
                 ):
        width  = 2048
        height = 64
        kwargs.pop('width')
        kwargs.pop('height')
        
        super().__init__(split,
                         preprocess_dir,
                         **kwargs,
                         width=width,
                         height=height,
                         )

def test_SemanticPOSSSCN():
    from xmuda.data.utils.visualize import draw_bird_eye_view
    preprocess_dir = '/_data/datasets/SemanticPOSS/semantic_poss_preprocess_lidar/preprocess'
    # pselab_paths = ("/home/docker_user/workspace/outputs/xmuda/a2d2_semantic_kitti/xmuda_crop_resize/pselab_data/train.npy",)
    # split = ('train',)
    split = ('val',)
    dataset = SemanticPOSSSCN(split=split,
                               preprocess_dir=preprocess_dir,
                               merge_classes=True,
                               noisy_rot=0.1,
                               flip_y=0.5,
                               rot_z=2*np.pi,
                               transl=True,
                               width=1800,
                               height=40,
                               cut_image=True,
                               cut_range=[512, 512],
                               remove_large=True,
                               random_flip=True,
                               )
    for i in [10, 20, 30, 40, 50, 60]:
        data = dataset[i]
        coords = data['coords']
        draw_bird_eye_view(coords)

def compute_class_weights():
    preprocess_dir = '/_data/datasets/SemanticPOSS/semantic_poss_preprocess_lidar/preprocess'
    split = ('train',)
    dataset = SemanticPOSSBase(split,
                                preprocess_dir,
                                merge_classes=True
                                )
    # compute points per class over whole dataset
    num_classes = len(dataset.class_names)
    points_per_class = np.zeros(num_classes, int)
    for i, data in enumerate(dataset.data):
        print('{}/{}'.format(i, len(dataset)))
        labels = dataset.label_mapping[data['seg_label_3d']]
        points_per_class += np.bincount(labels[labels != -100], minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('class weights: ', class_weights)
    log_smoothed_class_weights = class_weights / class_weights.min()
    print('log smoothed class weights: ', log_smoothed_class_weights)

    beta = 1.5
    m = log_smoothed_class_weights.max()

    scaled_class_weights = ((beta - 1) * log_smoothed_class_weights + m - beta) / (m - 1)
    print('scaled class weights: ', scaled_class_weights)

if __name__ == '__main__':
    # test_SemanticPOSSSCN()
    compute_class_weights()
