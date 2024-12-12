import numpy as np
import random
from scipy import ndimage
from PIL import Image


def augment_and_scale_2d(proj_range, proj_remission, inpainted_mask, label, indices, 
    opt={'width': 2048,
         'height': 64,
         'cut_image': False,
         'cut_range': [512, 512],
         'remove_large': False,
         'random_flip': False,
         }):

    keep_idx = np.ones(len(indices), dtype=np.bool)
    # Resize and Crop to width
    H, W = inpainted_mask.shape
    if (opt['width'] is not None):
        if not (W == opt['width']):
            new_size = (H, opt['width'])
            indices = resize_and_crop_indices(indices, new_size, H, W)
            inpainted_mask = resize_and_crop(inpainted_mask, new_size)
            proj_range = resize_and_crop(proj_range, new_size)
            proj_remission = resize_and_crop(proj_remission, new_size)
            if len(label) != 0:
                label = resize_and_crop_img(label, new_size)
    
    # Cut image to specified size
    H, W = inpainted_mask.shape
    if opt['cut_image']:
        if opt['cut_range'][0] == opt['cut_range'][1]:
            n = opt['cut_range'][1]
            start_ind = np.random.randint(0, W)
            end_ind = start_ind+n
            if end_ind > (W-1):
                h_inds = list(range(start_ind, W)) + \
                    list(range(0, (end_ind-W)))
                end_ind = (end_ind-W)
            else:
                h_inds = list(range(start_ind, end_ind))
        else:
            start_ind = opt['cut_range'][0]
            end_ind = opt['cut_range'][1]
            h_inds = list(range(start_ind, end_ind))
        # update image indices
        if start_ind < end_ind:
            keep_idx = indices[:, 0] >= start_ind
            keep_idx = np.logical_and(keep_idx, indices[:, 0] < end_ind)
            indices = indices[keep_idx]
            indices[:, 0] -= start_ind
        else:
            keep_idx_1 = indices[:, 0] >= start_ind
            keep_idx_2 = indices[:, 0] < end_ind
            keep_idx = np.logical_or(keep_idx_1, keep_idx_2)
            indices = indices[keep_idx]
            a = indices[:, 0]
            indices[:, 0] = np.where(a >= start_ind, a - start_ind, a + W - start_ind)
        inpainted_mask = inpainted_mask[:, h_inds]
        proj_range = proj_range[:, h_inds]
        proj_remission = proj_remission[:, h_inds]
        if len(label) != 0:
            label = label[:, h_inds]
    
    # Resize and Crop to height
    H, W = inpainted_mask.shape
    if (opt["height"] is not None):
        if not (H == opt["height"]):
            new_size = (opt["height"], W)
            indices = resize_and_crop_indices(indices, new_size, H, W)
            inpainted_mask = resize_and_crop(inpainted_mask, new_size)
            proj_range = resize_and_crop(proj_range, new_size)
            proj_remission = resize_and_crop(proj_remission, new_size)
            if len(label) != 0:
                label = resize_and_crop_img(label, new_size)
    
    # Randomly remove large parts
    H, W = inpainted_mask.shape
    remove_large_flag = random.choice([True, False])
    if opt["remove_large"] and remove_large_flag:
        a = np.ones((H, W))
        h_range = [i for i in range(0, H)]
        w_range = [i for i in range(0, W)]
        remove_h = random.choice(range(0, int(H)))
        remove_w = random.choice(range(0, int(W)))
        left_top = (random.choice(h_range), random.choice(w_range))
        a = inpainted_mask
        a[left_top[0]:left_top[0]+remove_h,
            left_top[1]:left_top[1]+remove_w] = 0
        inpainted_mask = np.multiply(inpainted_mask, a)
        proj_range = np.multiply(proj_range, a)
        proj_remission = np.multiply(proj_remission, a)
        label = (label*a).astype(label.dtype)

    # Randomly flip over all dimensions
    flip_flag = random.choice([True, False])
    if opt['random_flip'] and flip_flag:
        flip_flag = 1
        indices[:, 0] = W - 1 - indices[:, 0]
        indices[:, 1] = H - 1 - indices[:, 1]
        inpainted_mask = np.flip(inpainted_mask, flip_flag).copy()
        proj_range = np.flip(proj_range, flip_flag).copy()
        proj_remission = np.flip(proj_remission, flip_flag).copy()
        if len(label) != 0:
            label = np.flip(label, flip_flag).copy()

    return proj_range, proj_remission, inpainted_mask, label, indices, keep_idx

def resize_and_crop(img, new_size):
    h, w = img.shape
    zoom_r = [new_size[0]/h, new_size[1]/w]
    zoomed_img = ndimage.zoom(img, zoom_r, mode='reflect', order=1)
    return zoomed_img

def resize_and_crop_img(img, new_size):
    size = (new_size[1], new_size[0])
    img[img == -100] = 100
    new_img = Image.fromarray(img.astype(np.uint8))
    new_img = new_img.resize(size, Image.NEAREST)
    new_img = np.array(new_img).astype(np.int64)
    new_img[new_img == 100] = -100
    return new_img

def resize_and_crop_indices(indices, new_size, h, w):
    zoom_r = [new_size[1]/w, new_size[0]/h]
    new_indices = indices * zoom_r
    return new_indices.astype(np.int32)
