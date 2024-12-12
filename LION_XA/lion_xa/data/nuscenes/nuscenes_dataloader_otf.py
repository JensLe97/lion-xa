import os.path as osp
from pathlib import Path
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yaml

import sys
sys.path.append('./LION_XA')

from nuscenes.utils.data_io import load_bin_file

from lion_xa.data.utils.augmentation_3d import augment_and_scale_3d
from lion_xa.data.utils.augmentation_2d import augment_and_scale_2d
from lion_xa.data.range_img.range_img import get_range_image, depth_to_normal
from lion_xa.data.range_img.laserscan import SemLaserScan

SENSOR_FILE_NAME = "HDL-32E.yml"
DATASET_FILE_NAME = "nuscenes_all.yaml"

class MapperWrapper:
    def __init__(self, split_name, with_inpaint=True):
        self.data = []

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

class NuScenesBase(Dataset):
    """NuScenes dataset"""

    class_names = [
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
        'vehicle': ['car', 'truck', 'bus', 'trailer', 
                    'construction_vehicle', 'bicycle', 'motorcycle'],
        'driveable_surface': ['driveable_surface'],
        'sidewalk': ['sidewalk'],
        'manmade': ['manmade'],
        'terrain': ['terrain'],
        'vegetation': ['vegetation'],
    }

    #nusc.lidarseg_idx2name_mapping:
    lidarseg_name_mapping = {
        0: 'noise',
        1: 'animal',
        2: 'human.pedestrian.adult',
        3: 'human.pedestrian.child',
        4: 'human.pedestrian.construction_worker',
        5: 'human.pedestrian.personal_mobility',
        6: 'human.pedestrian.police_officer',
        7: 'human.pedestrian.stroller',
        8: 'human.pedestrian.wheelchair',
        9: 'movable_object.barrier',
        10: 'movable_object.debris',
        11: 'movable_object.pushable_pullable',
        12: 'movable_object.trafficcone',
        13: 'static_object.bicycle_rack',
        14: 'vehicle.bicycle',
        15: 'vehicle.bus.bendy',
        16: 'vehicle.bus.rigid',
        17: 'vehicle.car',
        18: 'vehicle.construction',
        19: 'vehicle.emergency.ambulance',
        20: 'vehicle.emergency.police',
        21: 'vehicle.motorcycle',
        22: 'vehicle.trailer',
        23: 'vehicle.truck',
        24: 'flat.driveable_surface',
        25: 'flat.other',
        26: 'flat.sidewalk',
        27: 'flat.terrain',
        28: 'static.manmade',
        29: 'static.other',
        30: 'static.vegetation',
        31: 'vehicle.ego'}

    lidarseg_cat_mapping = {
        'animal': 0,
        'flat.driveable_surface': 11,
        'flat.other': 12,
        'flat.sidewalk': 13,
        'flat.terrain': 14,
        'human.pedestrian.adult': 7,
        'human.pedestrian.child': 7,
        'human.pedestrian.construction_worker': 7,
        'human.pedestrian.personal_mobility': 0,
        'human.pedestrian.police_officer': 7,
        'human.pedestrian.stroller': 0,
        'human.pedestrian.wheelchair': 0,
        'movable_object.barrier': 1,
        'movable_object.debris': 0,
        'movable_object.pushable_pullable': 0,
        'movable_object.trafficcone': 8,
        'noise': 0,
        'static.manmade': 15,
        'static.other': 0,
        'static.vegetation': 16,
        'static_object.bicycle_rack': 0,
        'vehicle.bicycle': 2,
        'vehicle.bus.bendy': 3,
        'vehicle.bus.rigid': 3,
        'vehicle.car': 4,
        'vehicle.construction': 5,
        'vehicle.ego': 0,
        'vehicle.emergency.ambulance': 0,
        'vehicle.emergency.police': 0,
        'vehicle.motorcycle': 6,
        'vehicle.trailer': 9,
        'vehicle.truck': 10}

    def __init__(self,
                 split,
                 root_dir,
                 merge_classes=False,
                 nusc=None,
                 ):
        self.root_dir = root_dir
        print("Initialize NuScenes dataloader")

        assert isinstance(split, tuple) and len(split) == 1
        self.mapper_wrapper = MapperWrapper(split[0])

        print('Load', split)
        with open(osp.join(root_dir, split[0] + '.pkl'), 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        self.nusc = nusc

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
        return len(self.nusc_infos)


class NuScenesSCN(NuScenesBase):
    def __init__(self,
                 split,
                 root_dir,
                 merge_classes=False,
                 nusc=None,
                 lidarseg=False,
                 scale=20,
                 full_scale=4096,
                 # 3D augmentation
                 noisy_rot=0.0,
                 flip_x=0.0,
                 rot_z=0.0,
                 transl=False,
                 # 2D augmentation
                 width=1024,
                 height=32,
                 cut_image=False,
                 cut_range=[512, 512],
                 remove_large=False,
                 random_flip=False,
                 output_orig=False,
                 model_type='SCN',
                 visualize=False,
                 ):
        super().__init__(split,
                         root_dir,
                         merge_classes=merge_classes,
                         nusc=nusc)

        self.output_orig = output_orig
        self.model_type = model_type
        self.visualize = visualize

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

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]

        # load lidar points
        raw_lidar = np.fromfile(osp.join(self.root_dir, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points = raw_lidar[:, :3]
        remissions = raw_lidar[:, 3]

        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_label_filename = osp.join(self.root_dir, self.nusc.get('lidarseg', lidar_sd_token)['filename'])
        
        seg_label_3d = load_bin_file(lidarseg_label_filename, type='lidarseg')

        # Map the lidarseg labels to the 16 classes
        with np.nditer(seg_label_3d, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = self.lidarseg_cat_mapping[self.lidarseg_name_mapping[x.item()]]
         
        # convert to relative path
        lidar_path = lidar_path.replace(self.root_dir + '/', '')

        self.mapper_wrapper.mapper.set_points(points, remissions)
        self.mapper_wrapper.mapper.set_label(seg_label_3d)

        range_img, seg_label_2d, inpainted_mask, range_img_indices = get_range_image(
            self.mapper_wrapper.mapper, self.mapper_wrapper.norm_remission, self.mapper_wrapper.label_config, open_file=False)

        if self.label_mapping is not None:
            seg_label_3d = self.label_mapping[seg_label_3d]
            seg_label_2d = self.label_mapping[seg_label_2d]
            
        proj_range     = range_img[0, :, :]
        proj_remission = range_img[1, :, :]

        proj_range, proj_remission, inpainted_mask, seg_label_2d, range_img_indices, keep_idx = augment_and_scale_2d(
            proj_range, proj_remission, inpainted_mask, seg_label_2d, range_img_indices, self.opt)

        if not self.visualize:
            points = points[keep_idx]
            seg_label_3d = seg_label_3d[keep_idx]

        data_statics = self.mapper_wrapper.label_config['data_statics']

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
        out_dict['seg_label_3d'] = seg_label_3d[idxs].astype(np.int64)
        
        if not self.visualize:
            out_dict['range_img_indices'] = range_img_indices[idxs]

        if self.output_orig:
            out_dict.update({
                'orig_seg_label_3d': seg_label_3d.astype(np.int64),
                'orig_points_idx': idxs,
                'points': points if self.visualize else points[idxs],
                'org_proj_range': org_proj_range
            })

        return out_dict

class NuScenesStatistic(NuScenesBase):
    def __init__(self,
                 split,
                 root_dir,
                 merge_classes=False,
                 nusc=None,
                 ):
        super().__init__(split,
                         root_dir,
                         merge_classes=merge_classes,
                         nusc=nusc)

    def __getitem__(self, index):
        info = self.nusc_infos[index]

        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_label_filename = osp.join(self.root_dir, self.nusc.get('lidarseg', lidar_sd_token)['filename'])
        
        seg_label_3d = load_bin_file(lidarseg_label_filename, type='lidarseg')

        # Map the lidarseg labels to the 16 classes
        with np.nditer(seg_label_3d, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = self.lidarseg_cat_mapping[self.lidarseg_name_mapping[x.item()]]

        if self.label_mapping is not None:
            seg_label_3d = self.label_mapping[seg_label_3d]
            
        out_dict = {}
        out_dict['seg_label_3d'] = seg_label_3d.astype(np.int64)
    
        return out_dict


class TGLNuScenesSCN(NuScenesSCN):
    def __init__(self,
                 split,
                 root_dir,
                 **kwargs
                 ):
        width  = 2048
        height = 64
        kwargs.pop('width')
        kwargs.pop('height')
        
        super().__init__(split,
                         root_dir,
                         **kwargs,
                         width=width,
                         height=height,
                         )

def compute_class_weights():
    dataroot = '/data/datasets/nuScenes/data/nuscenes'
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    split = ('train_all',)
    dataset = NuScenesStatistic(split,
                                dataroot,
                                merge_classes=True,
                                nusc=nusc)
    # compute points per class over whole dataset
    num_classes = len(dataset.class_names)
    points_per_class = np.zeros(num_classes, int)
    dataloader = DataLoader(dataset)
    for i, data in enumerate(dataloader):
        print('{}/{}'.format(i, len(dataloader)))
        labels = data['seg_label_3d']
        points_per_class += np.bincount(labels[labels != -100], minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ', class_weights / class_weights.min())

def visualize_NuScenesSCN():
    from lion_xa.data.utils.visualize import draw_point_cloud
    dataroot = '/_data/datasets/nuScenes/data/nuscenes'
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    # split = ('train_singapore',)
    # pselab_paths = ('/home/docker_user/workspace/outputs/xmuda/nuscenes/usa_singapore/xmuda/pselab_data/train_singapore.npy',)
    split = ('val_all',)
    dataset = NuScenesSCN(split=split,
                          root_dir=dataroot,
                          nusc=nusc,
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
                          random_flip=True,
                          lidarseg=True,
                          output_orig=True,
                          visualize=True)
    for i in [96]:
        data = dataset[i]
        points = data['points']
        ground_truth_labels = data['orig_seg_label_3d']
        prediction_baseline = np.fromfile(f'{dataroot}/baseline/point_cloud_{i}.label', dtype=np.int64)
        draw_point_cloud(points, ground_truth_labels)
        draw_point_cloud(points, prediction_baseline)
        for j in range(3):
            prediction = np.fromfile(f'{dataroot}/predictions/point_cloud_{i}_{j}.label', dtype=np.int64)
            draw_point_cloud(points, prediction)

if __name__ == '__main__':
    visualize_NuScenesSCN()
    #compute_class_weights()
