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

DATASET_FILE_NAME = "nuscenes.yaml"

class NuScenesBase(Dataset):
    """NuScenes dataset"""

    class_names = [
        "car",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "traffic_cone",
        "barrier",
        "background",
    ]
    class_names_lidarseg = [
        "ignore",
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction_vehicle", 
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "driveable_surface",
        "other_flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
    ]

    # use those categories if merge_classes == True
    categories = {
        "vehicle": ["car", "truck", "bus", "trailer", "construction_vehicle"],
        "pedestrian": ["pedestrian"],
        "bike": ["motorcycle", "bicycle"],
        "traffic_boundary": ["traffic_cone", "barrier"],
        "background": ["background"]
    }
    categories_lidarseg = {
        'vehicle': ['car', 'truck', 'bus', 'trailer', 
                    'construction_vehicle', 'bicycle', 'motorcycle'],
        'driveable_surface': ['driveable_surface'],
        'sidewalk': ['sidewalk'],
        'manmade': ['manmade'],
        'terrain': ['terrain'],
        'vegetation': ['vegetation'],
    }

    def __init__(self,
                 split,
                 root_dir,
                 merge_classes=False,
                 lidarseg=False
                 ):

        self.split = split
        self.root_dir = root_dir

        print("Initialize NuScenes dataloader")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.root_dir, curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))

        if lidarseg:
            self.class_names = self.class_names_lidarseg
            self.categories  = self.categories_lidarseg

        if merge_classes:
            self.label_mapping = -100 * np.ones(len(self.class_names), dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_names.index(class_name)] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class NuScenesSCN(NuScenesBase):
    def __init__(self,
                 split,
                 root_dir,
                 merge_classes=False,
                 nusc=None,
                 lidarseg=False,
                 scale=20,
                 full_scale=4096,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_x=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 # 2D augmentation:
                 width=1024,
                 height=32,
                 cut_image=False,
                 cut_range=[512, 512],
                 remove_large=False,
                 random_flip=False,
                 output_orig=False,
                 model_type='SCN'
                 ):
        super().__init__(split,
                         root_dir,
                         merge_classes=merge_classes,
                         lidarseg=lidarseg)

        self.output_orig = output_orig
        self.model_type = model_type
        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_x = flip_x
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
        seg_label_3d = data_dict['seg_label_3d'].astype(np.int64)

        range_img = data_dict['range_img'].copy()
        inpainted_mask = data_dict['inpainted_mask'].copy()
        seg_label_2d = data_dict['seg_label_2d'].copy()
        range_img_indices = data_dict['range_img_indices'].copy()

        if self.label_mapping is not None:
            seg_label_3d = self.label_mapping[seg_label_3d]
            seg_label_2d = self.label_mapping[seg_label_2d]
            
        proj_range     = range_img[0, :, :]
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

        out_dict = {}
        out_dict['range_img'] = range_image.astype(np.float32)
        out_dict['inpainted_mask'] = inpainted_mask
        out_dict['seg_label_2d'] = seg_label_2d.astype(np.int64)

        # 3D data augmentation and scaling from points to voxel indices
        # nuscenes lidar coordinates: x (right), y (front), z (up)
        coords, point_feat = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_x=self.flip_x, rot_z=self.rot_z, transl=self.transl,
                                      model_type=self.model_type)

        # only use voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)

        out_dict['coords'] = coords[idxs]
        out_dict['feats'] = point_feat[idxs] if point_feat is not None else \
                            np.ones([len(idxs), 1], np.float32) # simply use 1 as feature
        out_dict['seg_label_3d'] = seg_label_3d[idxs]
        out_dict['range_img_indices'] = range_img_indices[idxs]

        if self.output_orig:
            out_dict.update({
                'orig_seg_label_3d': seg_label_3d,
                'orig_points_idx': idxs,
                'points': points[idxs],
                'org_proj_range': org_proj_range
            })

        return out_dict


def test_NuScenesSCN():
    from xmuda.data.utils.visualize import draw_points_image_labels, draw_points_image_depth, draw_bird_eye_view
    preprocess_dir = '/_data/datasets/nuScenes/nuscenes_preprocess_lidar/preprocess'
    # split = ('train_singapore',)
    # pselab_paths = ('/home/docker_user/workspace/outputs/xmuda/nuscenes/usa_singapore/xmuda/pselab_data/train_singapore.npy',)
    split = ('train_usa',)
    dataset = NuScenesSCN(split=split,
                          root_dir=preprocess_dir,
                          merge_classes=True,
                          noisy_rot=0.1,
                          flip_x=0.5,
                          rot_z=2*np.pi,
                          transl=True,
                          width=1024,
                          height=32,
                          cut_image=True,
                          cut_range=[512, 512],
                          remove_large=True,
                          random_flip=True)
    for i in [10, 20, 30, 40, 50, 60]:
        data = dataset[i]
        coords = data['coords']
        draw_bird_eye_view(coords)

def compute_class_weights():
    preprocess_dir = '/data/datasets/nuScenes/nuscenes_preprocess_lidar/preprocess'
    split = ('train_usa',)
    dataset = NuScenesBase(split,
                           preprocess_dir,
                           merge_classes=True,
                           lidarseg=False,
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
    print('log smoothed class weights: ', class_weights / class_weights.min())


if __name__ == '__main__':
    # test_NuScenesSCN()
    compute_class_weights()
