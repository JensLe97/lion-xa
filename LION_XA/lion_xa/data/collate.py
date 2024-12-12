import torch
from functools import partial


def collate_scn_base(input_dict_list, output_orig, output_image=True):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_image: whether to output range images
    :return: Collated data batch as dict
    """
    locs=[]
    feats=[]
    labels_3d=[]

    if output_image:
        range_imgs = []
        inpainted_masks = []
        labels_2d=[]
        range_img_idxs = []

    if output_orig:
        orig_seg_label_3d = []
        orig_points_idx = []
        points = []
        org_proj_range = []

    for idx, input_dict in enumerate(input_dict_list):
        coords = torch.from_numpy(input_dict['coords'])
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
        locs.append(torch.cat([coords, batch_idxs], 1))
        feats.append(torch.from_numpy(input_dict['feats']))
        if 'seg_label_3d' in input_dict.keys():
            labels_3d.append(torch.from_numpy(input_dict['seg_label_3d']))
        if output_image:
            range_imgs.append(torch.from_numpy(input_dict['range_img']))
            inpainted_masks.append(torch.from_numpy(input_dict['inpainted_mask']))
            labels_2d.append(torch.from_numpy(input_dict['seg_label_2d']))
            range_img_idxs.append(input_dict['range_img_indices'])
        if output_orig:
            orig_seg_label_3d.append(input_dict['orig_seg_label_3d'])
            orig_points_idx.append(input_dict['orig_points_idx'])
            points.append(torch.from_numpy(input_dict['points']))
            org_proj_range.append(torch.from_numpy(input_dict['org_proj_range']))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    out_dict = {'x': [locs, feats]}
    if labels_3d:
        labels_3d = torch.cat(labels_3d, 0)
        out_dict['seg_label_3d'] = labels_3d
    if output_image:
        out_dict['range_img'] = torch.stack(range_imgs)
        out_dict['inpainted_mask'] = torch.stack(inpainted_masks)
        out_dict['seg_label_2d'] = torch.stack(labels_2d)
        out_dict['range_img_indices'] = range_img_idxs
    if output_orig:
        out_dict['orig_seg_label_3d'] = orig_seg_label_3d
        out_dict['orig_points_idx'] = orig_points_idx
        out_dict['points'] = points
        out_dict['org_proj_range'] = org_proj_range
    return out_dict


def get_collate_scn(is_train):
    return partial(collate_scn_base,
                   output_orig=not is_train,
                   )
