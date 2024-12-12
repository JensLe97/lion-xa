import os
import os.path as osp
from pathlib import Path
import numpy as np
import pickle
from PIL import Image
import sys
sys.path.append('./LION_XA')
import torch
import yaml
import faulthandler

faulthandler.enable()

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.eval.detection.utils import category_to_detection_name

from lion_xa.data.nuscenes import splits
from lion_xa.data.nuscenes.nuscenes_dataloader import NuScenesBase
from lion_xa.data.range_img.laserscan import SemLaserScan
from lion_xa.data.range_img.range_img import get_range_image

# prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')

SENSOR_FILE_NAME = "HDL-32E.yml"
DATASET_FILE_NAME = "nuscenes.yaml"

class_names_to_id = dict(zip(NuScenesBase.class_names, range(len(NuScenesBase.class_names))))
if 'background' in class_names_to_id:
    del class_names_to_id['background']

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

def preprocess(nusc, split_names, root_dir, out_dir,
               keyword=None, keyword_action=None, subset_name=None,
               location=None, lidarseg=True):
    # cannot process day/night and location at the same time
    assert not (bool(keyword) and bool(location))
    if keyword:
        assert keyword_action in ['filter', 'exclude']

    # init dict to save
    pkl_dict = {}
    mapper_wrappers = {}
    for split_name in split_names:
         pkl_dict[split_name] = []
         mapper_wrappers[split_name] = MapperWrapper(split_name)

    #nusc.lidarseg_idx2name_mapping:
    lidarseg_name_mapping = {0: 'noise',
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

    lidarseg_cat_mapping = {'animal': 0,
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

    for i, sample in enumerate(nusc.sample):
        curr_scene_name = nusc.get('scene', sample['scene_token'])['name']

        # get if the current scene is in train, val or test
        curr_split = None
        for split_name in split_names:
            if curr_scene_name in getattr(splits, split_name):
                curr_split = split_name
                break
        if curr_split is None:
            continue

        if subset_name == 'night':
            if curr_split == 'train':
                if curr_scene_name in splits.val_night:
                    curr_split = 'val'
        if subset_name == 'singapore':
            if curr_split == 'train':
                if curr_scene_name in splits.val_singapore:
                    curr_split = 'val'

        if location:
            scene = nusc.get("scene", sample["scene_token"])
            if location not in nusc.get("log", scene['log_token'])['location']:
                continue

        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_path, boxes_lidar, _ = nusc.get_sample_data(lidar_token)

        print('{}/{} {} {}'.format(i + 1, len(nusc.sample), curr_scene_name, lidar_path))

        # load lidar points
        raw_lidar = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4].T
        pts = raw_lidar[:3, :]
        remissions = raw_lidar[3, :]

        lidar_sd_token = nusc.get('sample', sample['token'])['data']['LIDAR_TOP']
        lidarseg_label_filename = osp.join(root_dir, nusc.get('lidarseg', lidar_sd_token)['filename'])
        
        if lidarseg:
            seg_label_3d = load_bin_file(lidarseg_label_filename, type='lidarseg')

            # Map the lidarseg labels to the 10 classes
            with np.nditer(seg_label_3d, op_flags=['readwrite']) as it:
                for x in it:
                    x[...] = lidarseg_cat_mapping[lidarseg_name_mapping[x.item()]]
        else:
            seg_label_3d = np.full(pts.shape[1], fill_value=len(class_names_to_id), dtype=np.uint8)
            boxes = [box for box in boxes_lidar]
            for box in boxes:
                # get points that lie inside of the box
                fg_mask = points_in_box(box, pts)
                det_class = category_to_detection_name(box.name)
                if det_class is not None:
                    seg_label_3d[fg_mask] = class_names_to_id[det_class]
         
        # convert to relative path
        lidar_path = lidar_path.replace(root_dir + '/', '')

        # transpose to yield shape (num_points, 3)
        pts = pts.T
        remissions = remissions.T

        mapper_wrapper = mapper_wrappers[curr_split]
        mapper = mapper_wrapper.mapper
        mapper.set_points(pts, remissions)
        mapper.set_label(seg_label_3d)

        range_img, seg_label_2d, inpainted_mask, range_img_indices = get_range_image(
            mapper, mapper_wrapper.norm_remission, mapper_wrapper.label_config, open_file=False)

        # append data to train, val or test list in pkl_dict
        data_dict = {
            'points': pts,
            'seg_label_3d': seg_label_3d,
            'range_img': range_img.astype(np.float32),
            'inpainted_mask': inpainted_mask,
            'seg_label_2d': seg_label_2d.astype(np.uint8),
            'range_img_indices': range_img_indices,
            'lidar_path': lidar_path,
        }
        pkl_dict[curr_split].append(data_dict)

    # save to pickle file
    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    for split_name in split_names:
        save_path = osp.join(save_dir, '{}{}.pkl'.format(split_name, '_' + subset_name if subset_name else ''))
        with open(save_path, 'wb') as f:
            pickle.dump(pkl_dict[split_name], f)
            print('Wrote preprocessed data to ' + save_path)

def calc_statistics(nusc, split_names, keyword=None, keyword_action=None, subset_name=None, location=None):
    # cannot process day/night and location at the same time
    assert not (bool(keyword) and bool(location))
    if keyword:
        assert keyword_action in ['filter', 'exclude']

    mapper_wrappers = {}
    for split_name in split_names:
         mapper_wrappers[split_name] = MapperWrapper(split_name, False)

    psum    = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    count = 0

    for i, sample in enumerate(nusc.sample):
        curr_scene_name = nusc.get('scene', sample['scene_token'])['name']

        # get if the current scene is in train, val or test
        curr_split = None
        for split_name in split_names:
            if curr_scene_name in getattr(splits, split_name):
                curr_split = split_name
                break
        if curr_split is None:
            continue

        if subset_name == 'night':
            if curr_split == 'train':
                if curr_scene_name in splits.val_night:
                    curr_split = 'val'
        if subset_name == 'singapore':
            if curr_split == 'train':
                if curr_scene_name in splits.val_singapore:
                    curr_split = 'val'

        if location:
            scene = nusc.get("scene", sample["scene_token"])
            if location not in nusc.get("log", scene['log_token'])['location']:
                continue

        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_path, _, _ = nusc.get_sample_data(lidar_token)

        print('{}/{} {} {}'.format(i + 1, len(nusc.sample), curr_scene_name, lidar_path))

        # load lidar points
        raw_lidar = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        pts = raw_lidar[:, :3]
        remissions = raw_lidar[:, 3]

        mapper_wrapper = mapper_wrappers[curr_split]
        mapper = mapper_wrapper.mapper
        mapper.set_points(pts, remissions)

        range_img, _, _, _ = get_range_image(mapper, mapper_wrapper.norm_remission, mapper_wrapper.label_config, open_file=False, with_inpaint=False)

        inputs = torch.from_numpy(range_img)
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

def calc_statistics_all(nusc, split_names):
    mapper_wrappers = {}
    for split_name in split_names:
         mapper_wrappers[split_name] = MapperWrapper(split_name, False)

    psum    = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    count = 0

    for i, sample in enumerate(nusc.sample):
        curr_scene_name = nusc.get('scene', sample['scene_token'])['name']

        # get if the current scene is in train_all
        curr_split = None
        for split_name in split_names:
            if curr_scene_name in getattr(splits, split_name):
                curr_split = split_name
                break
        if curr_split is None:
            continue

        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_path, _, _ = nusc.get_sample_data(lidar_token)

        print('{}/{} {} {}'.format(i + 1, len(nusc.sample), curr_scene_name, lidar_path))

        # load lidar points
        raw_lidar = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        pts = raw_lidar[:, :3]
        remissions = raw_lidar[:, 3]

        mapper_wrapper = mapper_wrappers[curr_split]
        mapper = mapper_wrapper.mapper
        mapper.set_points(pts, remissions)

        range_img, _, _, _ = get_range_image(mapper, mapper_wrapper.norm_remission, mapper_wrapper.label_config, open_file=False, with_inpaint=False)

        inputs = torch.from_numpy(range_img)
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

def create_mini_dataset(split_name, preprocess_dir, total_div_number):
    data = []
    with open(osp.join(preprocess_dir, 'preprocess', split_name + '.pkl'), 'rb') as f:
        data.extend(pickle.load(f))
    
    div_size = len(data) // total_div_number
    for div_number in range(total_div_number+1):
        save_path = osp.join(preprocess_dir, '{}{}.pkl'.format(split_name, '_' + str(div_number)))
        if div_number == total_div_number and len(data) % total_div_number != 0:
            with open(save_path, 'wb') as f:
                pickle.dump(data[div_number*div_size:-1], f)
                print('Wrote preprocessed data to ' + save_path)
                return
        with open(save_path, 'wb') as f:
            pickle.dump(data[div_number*div_size:div_number*div_size+div_size], f)
            print('Wrote preprocessed data to ' + save_path)

if __name__ == '__main__':
    root_dir = '/teamspace/studios/this_studio/data/datasets/nuScenes/data/nuscenes'
    out_dir = '/teamspace/studios/this_studio/data/datasets/nuScenes/nuscenes_preprocess'
    nusc = NuScenes(version='v1.0-trainval', dataroot=root_dir, verbose=True)
    # for faster debugging, the script can be run using the mini dataset
    # nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # We construct the splits by using the meta data of NuScenes:
    # USA/Singapore: We check if the location is Boston or Singapore.
    create_mini = False
    create_pkl = True # False = only calculation
    if create_mini:
        create_mini_dataset('train_usa', out_dir, 20)
        create_mini_dataset('train_singapore', out_dir, 10)
        create_mini_dataset('val_singapore', out_dir, 4)
        create_mini_dataset('test_singapore', out_dir, 4)
    if create_pkl:
        preprocess(nusc, ['train'], root_dir, out_dir, location='boston', subset_name='usa', lidarseg=False)
        preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, location='singapore', subset_name='singapore', lidarseg=False)
    else:
        # calc_statistics(nusc, ['train'], location='boston', subset_name='usa')
        calc_statistics_all(nusc, ['train_all'])
