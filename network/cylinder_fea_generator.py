# -*- coding:utf-8 -*-
# author: Xinge
# Modified in 2026 to add optional learnable cylindrical coordinate offsets.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import numba as nb
except ImportError:
    nb = None
import multiprocessing
import torch_scatter


class cylinder_fea(nn.Module):

    def __init__(self, grid_size, fea_dim=3,
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None,
                 learnable_offset=False, offset_scale=None, offset_hidden_dim=32):
        super(cylinder_fea, self).__init__()

        self.learnable_offset = learnable_offset
        pp_input_dim = fea_dim
        if self.learnable_offset:
            if fea_dim < 8:
                raise ValueError("Learnable cylindrical offset requires at least 8 point features")
            if offset_scale is None or len(offset_scale) != 3:
                raise ValueError("offset_scale must contain rho/theta/z limits")

            self.register_buffer("offset_scale", torch.tensor(offset_scale, dtype=torch.float32))
            self.offset_mlp = nn.Sequential(
                nn.Linear(fea_dim, offset_hidden_dim),
                nn.ReLU(),
                nn.Linear(offset_hidden_dim, offset_hidden_dim),
                nn.ReLU(),
                nn.Linear(offset_hidden_dim, 3)
            )
            # Start from the exact fixed-coordinate baseline and learn deformation gradually.
            nn.init.zeros_(self.offset_mlp[-1].weight)
            nn.init.zeros_(self.offset_mlp[-1].bias)
            pp_input_dim += 3

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(pp_input_dim),

            nn.Linear(pp_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def apply_learnable_offset(self, point_features):
        normalized_offset = torch.tanh(self.offset_mlp(point_features))
        metric_offset = normalized_offset * self.offset_scale

        local_cylindrical = point_features[:, 0:3] + metric_offset
        cylindrical = point_features[:, 3:6] + metric_offset
        corrected_xy = torch.stack((
            cylindrical[:, 0] * torch.cos(cylindrical[:, 1]),
            cylindrical[:, 0] * torch.sin(cylindrical[:, 1])
        ), dim=1)

        corrected_features = torch.cat((
            local_cylindrical,
            cylindrical,
            corrected_xy,
            point_features[:, 8:]
        ), dim=1)
        return torch.cat((corrected_features, normalized_offset), dim=1)

    def forward(self, pt_fea, xy_ind):
        cur_dev = pt_fea[0].get_device()

        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # process feature
        if self.learnable_offset:
            cat_pt_fea = self.apply_learnable_offset(cat_pt_fea)
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        return unq, processed_pooled_data
