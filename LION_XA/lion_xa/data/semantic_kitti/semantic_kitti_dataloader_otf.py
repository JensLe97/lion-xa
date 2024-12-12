import glob
import os.path as osp
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yaml

import sys
sys.path.append('./LION_XA')
from lion_xa.data.semantic_kitti import splits
from lion_xa.data.utils.augmentation_3d import augment_and_scale_3d
from lion_xa.data.utils.augmentation_2d import augment_and_scale_2d
from lion_xa.data.range_img.range_img import get_range_image, depth_to_normal
from lion_xa.data.range_img.laserscan import SemLaserScan

SENSOR_FILE_PATH = Path(__file__).parent / "HDL-64E.yml"
DATASET_FILE_PATH = Path(__file__).parent / "semantic_kitti.yaml"

class MapperWrapper:
    def __init__(self, split_name, sensor_file_path, with_inpaint=True):
        self.data = []

        with open(sensor_file_path) as f:
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

        with open(DATASET_FILE_PATH) as f:
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

class SemanticKITTIBase(Dataset):
    """SemanticKITTI dataset"""

    # https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    id_to_class_name = {
        0: "unlabeled",
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }

    class_name_to_id = {v: k for k, v in id_to_class_name.items()}

    # use those categories if merge_classes == True (common with A2D2)
    categories = {
        'person': ['person', 'moving-person'],
        'bicyclist': ['bicyclist', 'motorcyclist',
                      'moving-bicyclist', 'moving-motorcyclist'],  # riders are labeled as bikes in Audi dataset
        'car': ['car', 'moving-car', 'motorcycle', 'bus', 'on-rails', 'truck', 
                'other-vehicle', 'moving-on-rails', 'moving-bus', 
                'moving-truck', 'moving-other-vehicle'],
        'trunk': ['trunk'],
        'vegetation': ['vegetation'],
        'traffic-sign': ['traffic-sign'],
        'pole': ['pole'],
        'object': ['other-object'],
        'building': ['building'],
        'fence': ['fence'],
        'bike': ['bicycle'],
        'ground': ['road', 'sidewalk', 'parking', 'terrain', 'other-ground'],
    }

    categories_nuscenes = {
        'vehicle': ['car', 'bicycle', 'truck', 'motorcycle',
                    'bicyclist', 'motorcyclist', 'moving-car', 'moving-bicyclist',
                    'moving-motorcyclist', 'moving-truck'],
        'driveable_surface': ['road', 'parking', 'lane-marking'],
        'sidewalk': ['sidewalk'],
        'manmade': ['building', 'fence', 'pole', 'traffic-sign', 'other-object'],
        'terrain': ['terrain'],
        'vegetation': ['vegetation', 'trunk'],
    }

    def __init__(self,
                 split,
                 root_dir,
                 merge_classes=False,
                 nuscenes=False):
        self.root_dir = root_dir
        print("Initialize SemanticKITTI dataloader")

        assert isinstance(split, tuple) and len(split) == 1        
        print('Load', split)
        self.data = []
        for curr_split in split:
            self.glob_frames(getattr(splits, curr_split))
        
        if nuscenes:
            self.categories = self.categories_nuscenes

        if merge_classes:
            highest_id = list(self.id_to_class_name.keys())[-1]
            self.label_mapping = -100 * np.ones(highest_id + 2, dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_name_to_id[class_name]] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None

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
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class SemanticKITTISCN(SemanticKITTIBase):
    def __init__(self,
                 split,
                 root_dir,
                 trgl_dir='',
                 merge_classes=False,
                 nuscenes=False,
                 scale=20,
                 full_scale=4096,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_y=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 # 2D augmentation:
                 width=2048,
                 height=64,
                 cut_image=False,
                 cut_range=[512, 512],
                 remove_large=False,
                 random_flip=False,
                 output_orig=False,
                 model_type='SCN',
                 sensor_file_path=SENSOR_FILE_PATH,
                 ):
        super().__init__(split,
                         root_dir,
                         merge_classes=merge_classes,
                         nuscenes=nuscenes)
        self.mapper_wrapper = MapperWrapper(split[0], sensor_file_path)
        
        self.output_orig = output_orig
        self.model_type = model_type
        self.nuscenes = nuscenes

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

    def __getitem__(self, index):
        data_dict = self.data[index].copy()

        points, seg_label_3d, range_img, seg_label_2d, inpainted_mask, range_img_indices = get_range_image(
                    self.mapper_wrapper.mapper, self.mapper_wrapper.norm_remission, self.mapper_wrapper.label_config, data_dict)

        if self.label_mapping is not None:
            seg_label_3d = self.label_mapping[seg_label_3d]
            seg_label_2d = self.label_mapping[seg_label_2d]

        proj_range     = range_img[0, :, :]
        proj_remission = range_img[1, :, :]

        proj_range, proj_remission, inpainted_mask, seg_label_2d, range_img_indices, keep_idx = augment_and_scale_2d(
            proj_range, proj_remission, inpainted_mask, seg_label_2d, range_img_indices, self.opt)

        points = points[keep_idx]
        seg_label_3d = seg_label_3d[keep_idx]

        data_statics = self.mapper_wrapper.label_config['data_statics']

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
        # Kitti lidar coordinates: x (front), y (left), z (up)
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
            })

        return out_dict

class SemanticKITTIStatistic(SemanticKITTIBase):
    def __init__(self,
                 split,
                 root_dir,
                 merge_classes=False,
                 nuscenes=False):
        super().__init__(split,
                         root_dir,
                         merge_classes=merge_classes,
                         nuscenes=nuscenes)

    def __getitem__(self, index):
        data_dict = self.data[index]
        
        seg_label_3d = np.fromfile(data_dict['label_path'], dtype=np.uint32)
        seg_label_3d = seg_label_3d.reshape((-1))
        seg_label_3d = seg_label_3d & 0xFFFF  # get lower half for semantics

        if self.label_mapping is not None:
            seg_label_3d = self.label_mapping[seg_label_3d]

        out_dict = {}
        out_dict['seg_label_3d'] = seg_label_3d.astype(np.int64)
    
        return out_dict

class TGLSemanticKITTISCN(SemanticKITTISCN):
    def __init__(self,
                 split,
                 root_dir,
                 trgl_dir='',
                 nuscenes=False,
                 **kwargs
                 ):
        if nuscenes:
            width = 1024
            height = 32
            sensor_file_path  = Path(__file__).parent.parent / "nuscenes" / "HDL-32E.yml"
        else:
            width = 1800
            height = 40
            sensor_file_path  = Path(__file__).parent.parent / "semantic_poss" / "Pandora.yml"
        
        kwargs.pop('width')
        kwargs.pop('height')
        
        super().__init__(split,
                         trgl_dir,
                         **kwargs,
                         nuscenes=nuscenes,
                         width=width,
                         height=height,
                         sensor_file_path=sensor_file_path,
                         )

def test_SemanticKITTISCN():
    from xmuda.data.utils.visualize import draw_bird_eye_view
    root_dir = '/_data/datasets/SemanticKITTI/'
    # pselab_paths = ("/home/docker_user/workspace/outputs/xmuda/a2d2_semantic_kitti/xmuda_crop_resize/pselab_data/train.npy",)
    split = ('train',)
    # split = ('val',)
    dataset = SemanticKITTISCN(split=split,
                               root_dir=root_dir,
                               merge_classes=True,
                               noisy_rot=0.1,
                               flip_y=0.5,
                               rot_z=2*np.pi,
                               transl=True,
                               width=2048,
                               height=64,
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
    dataroot = '/data/datasets/SemanticKITTI'
    split = ('train',)
    dataset = SemanticKITTIStatistic(split,
                                     dataroot,
                                     merge_classes=True,
                                     nuscenes=True)
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

if __name__ == '__main__':
    # test_SemanticKITTISCN()
    compute_class_weights()
