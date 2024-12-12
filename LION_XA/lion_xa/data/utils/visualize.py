import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import open3d

# all classes
NUSCENES_COLOR_PALETTE = [
    (255, 158, 0),  # car
    (255, 158, 0),  # truck
    (255, 158, 0),  # bus
    (255, 158, 0),  # trailer
    (255, 158, 0),  # construction_vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # motorcycle
    (255, 61, 99),  # bicycle
    (0, 0, 0),  # traffic_cone
    (0, 0, 0),  # barrier
    (200, 200, 200),  # background
]

# classes after merging (as used in xMUDA)
NUSCENES_COLOR_PALETTE_SHORT = [
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # bike
    (0, 0, 0),  # traffic boundary
    (200, 200, 200),  # background
]

NUSCENES_LIDARSEG_COLOR_PALETTE_DICT = OrderedDict([
    ('ignore', (0, 0, 0)),  # Black
    ('barrier', (112, 128, 144)),  # Slategrey
    ('bicycle', (220, 20, 60)),  # Crimson
    ('bus', (255, 127, 80)),  # Coral
    ('car', (255, 158, 0)),  # Orange
    ('construction_vehicle', (233, 150, 70)),  # Darksalmon
    ('motorcycle', (255, 61, 99)),  # Red
    ('pedestrian', (0, 0, 230)),  # Blue
    ('traffic_cone', (47, 79, 79)),  # Darkslategrey
    ('trailer', (255, 140, 0)),  # Darkorange
    ('truck', (255, 99, 71)),  # Tomato
    ('driveable_surface', (0, 207, 191)),  # nuTonomy green
    ('other_flat', (175, 0, 75)),
    ('sidewalk', (75, 0, 75)),
    ('terrain', (112, 180, 60)),
    ('manmade', (222, 184, 135)),  # Burlywood
    ('vegetation', (0, 175, 0))  # Green
])

NUSCENES_LIDARSEG_COLOR_PALETTE = list(NUSCENES_LIDARSEG_COLOR_PALETTE_DICT.values())

NUSCENES_LIDARSEG_COLOR_PALETTE_SHORT = [
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['car'],  # vehicle
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['driveable_surface'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['sidewalk'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['terrain'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['manmade'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['vegetation'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['ignore']
]

# all classes
A2D2_COLOR_PALETTE_SHORT = [
    (255, 0, 0),  # car
    (255, 128, 0),  # truck
    (182, 89, 6),  # bike
    (204, 153, 255),  # person
    (255, 0, 255),  # road
    (150, 150, 200),  # parking
    (180, 150, 200),  # sidewalk
    (241, 230, 255),  # building
    (147, 253, 194),  # nature
    (255, 246, 143),  # other-objects
    (0, 0, 0)  # ignore
]

# colors as defined in https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
SEMANTIC_KITTI_ID_TO_BGR = {  # bgr
  0: [0, 0, 0],
  1: [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0],
}
SEMANTIC_KITTI_COLOR_PALETTE = [SEMANTIC_KITTI_ID_TO_BGR[id] if id in SEMANTIC_KITTI_ID_TO_BGR.keys() else [0, 0, 0]
                                for id in range(list(SEMANTIC_KITTI_ID_TO_BGR.keys())[-1] + 1)]


# classes after merging (as used in xMUDA)
SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [180, 30, 80],  # truck
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [255, 150, 255],  # parking
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
SEMANTIC_KITTI_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR]

def draw_point_cloud(points, seg_labels, color_palette_type='NuScenes'):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE_SHORT
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')

    color_palette = np.array(color_palette) / 255.
    seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels]

    pcd = open3d.geometry.PointCloud()
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = 0.01

    to_reset_view_point = True
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors.astype(np.float64))

    # zoom=0.080000000000000002,
    # front=[ -0.23416626894017176, 0.92057372291598827, 0.31258627475824408 ],
    # lookat=[ -0.0027277739978425614, -1.9340680020235039, -1.2356797864242999 ],
    # up=[ 0.028134927219509137, -0.31497429992851672, 0.94868309579906285 ])
    open3d.visualization.draw_geometries([pcd], 
                                         zoom=0.040000000000000001,
                                         front=[ 0.1428093616290074, -0.90146787167988796, 0.40860881361033879 ],
                                         lookat=[ 4.6201810055391714, -3.3178356616212992, 1.5289759305779336 ],
                                         up=[ -0.045168068497389996, 0.40647430899881726, 0.91254505735999103 ])

    vis.destroy_window()
