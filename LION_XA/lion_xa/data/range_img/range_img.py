import numpy as np
import numpy.linalg as LA
import cv2 as cv
# import torch
# from torchvision.utils import save_image, make_grid


from lion_xa.data.range_img.laserscan import SemLaserScan

def get_range_image(mapper: SemLaserScan, norm_remission: int, label_config, scan_path: dict = {}, open_file=True, with_inpaint=True):
    """The range image consists of:
    1. Range map (proj_range)
    2. Remission map (proj_remission)
    3. Normal map (proj_normal)
    """
    assert (scan_path and open_file) or (not scan_path and not open_file)
    
    if open_file:
        points = mapper.open_scan(scan_path['lidar_path'])
        label_3d = mapper.open_label(scan_path['label_path'])
        label_3d = label_3d & 0xFFFF  # get lower half for semantics

    proj_mask = mapper.proj_mask

    # Save mask to img:
    # inp_mask = proj_mask.copy()
    # inp_mask = torch.from_numpy(inp_mask)
    # inp_mask = inp_mask.float()
    # inp_mask = inp_mask.unsqueeze(dim=0)
    # inp_mask = inp_mask.unsqueeze(dim=0)
    # inp_mask = inp_mask.repeat(1, 3, 1, 1)
    # grid = make_grid(inp_mask, nrow=1)
    # save_image(grid.data,
    #             f"./mask.png", nrow=1)
    
    label = mapper.proj_sem_label
    label = map_label(label, label_config)
    label = label * proj_mask

    # Range image: [Range, remission, normal(x, y, z)]
    proj_range = mapper.proj_range

    proj_remission = mapper.proj_remission
    if norm_remission > 0:
        proj_remission = proj_remission / norm_remission

    if with_inpaint:
        inpainted_mask, proj_range, proj_remission, _, label = inpaint(
            proj_mask, proj_range, proj_remission, label)
    else:
        inpainted_mask = np.zeros_like(proj_range)
    
    proj_normal = depth_to_normal(proj_range)
    
    proj_range = np.expand_dims(proj_range, axis=2)
    proj_remission = np.expand_dims(proj_remission, axis=2)

    # Finished Range image
    range_image = np.concatenate((proj_range, proj_remission, proj_normal), axis=2)
    range_image = np.transpose(range_image, axes=[2, 0, 1])

    range_img_indices = np.transpose(np.vstack((mapper.proj_x, mapper.proj_y)))

    # org_range = mapper.unproj_range
    
    if open_file:
        return points, label_3d, range_image, label, inpainted_mask, range_img_indices
    else:
        return range_image, label, inpainted_mask, range_img_indices


def get_small_hole(mask):
    morph_elem = cv.MORPH_RECT
    morph_size = 5
    element = cv.getStructuringElement(morph_elem, (morph_size, 3))
    inpainted_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, element)
    mask_hole = np.abs(inpainted_mask-mask)
    return mask_hole, inpainted_mask

def inpaint_label(label, mask):
    filter = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1],
                       [0, 1], [1, -1], [1, 0], [1, 1]])
    # print(mask.max())
    mask_copy = mask.copy()
    padded_label = np.pad(label, pad_width=1, mode='reflect')
    padded_mask = np.pad(mask_copy, pad_width=1,
                         mode='constant', constant_values=0)
    padded_mask_copy = padded_mask.copy()
    padded_label_copy = padded_label.copy()
    # print(mask.shape,padded_mask.shape)
    mask_sum = (mask_copy == 255).sum()
    while mask_sum:
        x_id, y_id = np.where(mask_copy == 255)
        for x, y in zip(x_id, y_id):
            nb_id = filter.copy()
            nb_id[:, 0] = nb_id[:, 0]+x+1
            nb_id[:, 1] = nb_id[:, 1]+y+1
            # print(nb_id.shape)
            nb_mask = padded_mask[nb_id[:, 0], nb_id[:, 1]]
            # print(nb_id,mask_copy[x,y],(nb_mask==255).sum())
            if (nb_mask == 255).sum() == 8:
                continue
            else:
                mask_copy[x, y] = 0
                padded_mask_copy[x+1, y+1] = 0
                nb_labels = padded_label[nb_id[:, 0], nb_id[:, 1]]
                padded_label_copy[x+1, y+1] = nb_labels.max()
        padded_mask = padded_mask_copy.copy()
        padded_label = padded_label_copy.copy()
        mask_sum = (mask_copy == 255).sum()
    return padded_label[1:-1, 1:-1]

def inpaint(proj_mask, proj_range, proj_remission, label, proj_xyz=None):
    mask_hole, inpainted_mask = get_small_hole(proj_mask.astype(np.uint8))
    mask_hole = (mask_hole*255).astype(np.uint8)
    inpainted_range = cv.inpaint(proj_range.astype(
        np.float32), mask_hole, 3, cv.INPAINT_NS)
    inpainted_remission = cv.inpaint(proj_remission.astype(
        np.float32), mask_hole, 3, cv.INPAINT_NS)
    inpainted_remission = inpainted_remission
    inpainted_xyz = proj_xyz
    if proj_xyz is not None:

        inpainted_xyz[:, :, 0] = cv.inpaint(
            proj_xyz[:, :, 0].astype(np.float32), mask_hole, 3, cv.INPAINT_NS)
        inpainted_xyz[:, :, 1] = cv.inpaint(
            proj_xyz[:, :, 1].astype(np.float32), mask_hole, 3, cv.INPAINT_NS)
        inpainted_xyz[:, :, 2] = cv.inpaint(
            proj_xyz[:, :, 2].astype(np.float32), mask_hole, 3, cv.INPAINT_NS)
    inpainted_label = inpaint_label(label, mask_hole)

    return inpainted_mask, inpainted_range, inpainted_remission, inpainted_xyz, inpainted_label


def depth_to_normal(depth):
    res = np.zeros(depth.shape + (3,), dtype=np.float32)
    x = np.arange(1, depth.shape[1] - 1)
    y = np.arange(1, depth.shape[0] - 1)
    xv, yv = np.meshgrid(x, y)
    t_depth = depth[yv - 1, xv]
    l_depth = depth[yv, xv - 1]
    c_depth = depth[yv, xv]
    t = np.dstack((xv, yv - 1, t_depth))
    l = np.dstack((xv - 1, yv, l_depth))
    c = np.dstack((xv, yv, c_depth))
    d = np.cross(l - c, t - c, axis=2)
    length = LA.norm(d, axis=2)
    length = length.reshape(length.shape + (1,))
    normal = np.zeros_like(d)

    np.divide(d, length, out=normal, dtype=np.float32)
    res[1:-1, 1:-1] = normal
    return res

def map_label(label, label_config):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    lut_dict = label_config['color_inv_map']
    lut = []
    for key in lut_dict:
        lut.append(lut_dict[key])
    maxkey = 0
    for key, data in label_config["learning_map"].items():
        if isinstance(data, list):
            nel = len(data)
        else:
            nel = 1
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
        lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
        lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in label_config["learning_map"].items():
        try:
            lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # do the mapping
    return lut[label]