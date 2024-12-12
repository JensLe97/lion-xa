# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch_scatter

class CylinderFeat(nn.Module):

    def __init__(self, output_shape, fea_dim=3,
                 out_fea_dim=64, max_pt_per_encode=64, num_input_features=None, **kwargs):
        super(CylinderFeat, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64),#64
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.num_input_features = num_input_features
        self.grid_size = output_shape
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_fea_dim

        # point feature compression
        if self.num_input_features is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.num_input_features),
                nn.ReLU())
            self.pt_fea_dim = self.num_input_features
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, cat_pt_fea, cat_pt_ind):
        # cur_dev = cat_pt_fea[0].get_device()

        # Batch index from index 3 to position 0
        cat_pt_ind = torch.index_select(cat_pt_ind, 1, torch.LongTensor([3,0,1,2]).cuda())

        # pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        # shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        # cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        # cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, _ = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

        if self.num_input_features:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        return unq, unq_inv, processed_pooled_data
