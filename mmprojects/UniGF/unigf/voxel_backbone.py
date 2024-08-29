import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np

from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmcv.ops import points_in_boxes_part

from .spconv_utils import spconv, replace_feature
from .voxel_backbone_utils import post_act_block_sparse_3d, post_act_block_sparse_2d, SparseBasicBlock3D, SparseBasicBlock2D

norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
norm_fn_2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

def focal_loss_sparse(pred, target):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    loss = 0
    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class SEDLayer(spconv.SparseModule):

    def __init__(self, dim: int, down_kernel_size: list, down_stride: list, num_SBB: list, indice_key, xy_only=False):
        super().__init__()

        block = SparseBasicBlock2D if xy_only else SparseBasicBlock3D
        post_act_block = post_act_block_sparse_2d if xy_only else post_act_block_sparse_3d

        self.encoder = nn.ModuleList(
            [spconv.SparseSequential(
                *[block(dim, indice_key=f"{indice_key}_0") for _ in range(num_SBB[0])])]
        )

        num_levels = len(down_stride)
        for idx in range(1, num_levels):
            cur_layers = [
                post_act_block(
                    dim, dim, down_kernel_size[idx], down_stride[idx], down_kernel_size[idx] // 2,
                    conv_type='spconv', indice_key=f'spconv_{indice_key}_{idx}'),

                *[block(dim, indice_key=f"{indice_key}_{idx}") for _ in range(num_SBB[idx])]
            ]
            self.encoder.append(spconv.SparseSequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                post_act_block(
                    dim, dim, down_kernel_size[idx],
                    conv_type='inverseconv', indice_key=f'spconv_{indice_key}_{idx}'))
            self.decoder_norm.append(norm_fn_1d(dim))

    def forward(self, x):
        feats = []
        for conv in self.encoder:
            x = conv(x)
            feats.append(x)

        x = feats[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, norm(x.features + up_x.features))
        return x

class SparseHEDNet(nn.Module):

    def __init__(self, 
                 input_channels, 
                 grid_size, 
                 class_names, 
                 voxel_size, 
                 point_cloud_range, 
                 model_cfg, 
                 afd,
                 afd_cfg,
                 **kwargs):
        super().__init__()

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        sed_dim = model_cfg.sed_feature_dim
        sed_num_layers = model_cfg.sed_num_layers
        sed_num_SBB = model_cfg.sed_num_sbb
        sed_down_kernel_size = model_cfg.sed_down_kernel_size
        sed_down_stride = model_cfg.sed_down_stride
        assert sed_down_stride[0] == 1
        assert len(sed_num_SBB) == len(sed_down_kernel_size) == len(sed_down_stride)

        afd_dim = afd_cfg.afd_feature_dim
        afd_num_layers = afd_cfg.afd_num_layers
        afd_num_SBB = afd_cfg.afd_num_sbb
        afd_down_kernel_size = afd_cfg.afd_down_kernel_size
        afd_down_stride = afd_cfg.afd_down_stride
        assert afd_down_stride[0] == 1
        assert len(afd_num_SBB) == len(afd_down_stride)

        post_act_block = post_act_block_sparse_3d
        self.stem = spconv.SparseSequential(
            post_act_block(input_channels, 16, 3, 1, 1, indice_key='subm1', conv_type='subm'),

            SparseBasicBlock3D(16, indice_key='conv1'),
            SparseBasicBlock3D(16, indice_key='conv1'),
            post_act_block(16, 32, 3, 2, 1, indice_key='spconv1', conv_type='spconv'),

            SparseBasicBlock3D(32, indice_key='conv2'),
            SparseBasicBlock3D(32, indice_key='conv2'),
            post_act_block(32, 64, 3, 2, 1, indice_key='spconv2', conv_type='spconv'),

            SparseBasicBlock3D(64, indice_key='conv3'),
            SparseBasicBlock3D(64, indice_key='conv3'),
            SparseBasicBlock3D(64, indice_key='conv3'),
            SparseBasicBlock3D(64, indice_key='conv3'),
            post_act_block(64, sed_dim, 3, (1, 2, 2), 1, indice_key='spconv3', conv_type='spconv'),
        )

        self.sed_layers = nn.ModuleList()
        for idx in range(sed_num_layers):
            layer = SEDLayer(
                sed_dim, sed_down_kernel_size, sed_down_stride, sed_num_SBB,
                indice_key=f'sedlayer{idx}', xy_only=kwargs.get('xy_only', False))
            self.sed_layers.append(layer)

        self.transition = spconv.SparseSequential(
            post_act_block(sed_dim, afd_dim, (3, 1, 1), (2, 1, 1), 0, indice_key='spconv4', conv_type='spconv'),
            post_act_block(afd_dim, afd_dim, (3, 1, 1), (2, 1, 1), 0, indice_key='spconv5', conv_type='spconv'),
        )

        self.adaptive_feature_diffusion = afd
        if self.adaptive_feature_diffusion:
            self.class_names = class_names
            self.voxel_size = voxel_size
            self.point_cloud_range = point_cloud_range
            self.fg_thr = afd_cfg.fg_thewshold
            self.featmap_stride = afd_cfg.featmap_stride
            self.group_pooling_kernel_size = afd_cfg.group_pooling_kernel_size
            self.detach_feature = afd_cfg.detach_feature

            self.class_names = class_names
            self.group_class_names = []
            for names in afd_cfg.group_class_names:
                self.group_class_names.append([x for x in names if x in class_names])

            self.cls_conv = spconv.SparseSequential(
                spconv.SubMConv2d(afd_dim, afd_dim, 3, stride=1, padding=1, bias=False, indice_key='conv_cls'),
                norm_fn_1d(afd_dim),
                nn.ReLU(),
                spconv.SubMConv2d(afd_dim, len(self.group_class_names), 1, bias=True, indice_key='cls_out')
            )
            self.forward_ret_dict = {}

        self.afd_layers = nn.ModuleList()
        for idx in range(afd_num_layers):
            layer = SEDLayer(
                afd_dim, afd_down_kernel_size, afd_down_stride, afd_num_SBB,
                indice_key=f'afdlayer{idx}', xy_only=True)
            self.afd_layers.append(layer)

        self.num_point_features = afd_dim

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (spconv.SubMConv2d, spconv.SubMConv3d)):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.adaptive_feature_diffusion:
            self.cls_conv[-1].bias.data.fill_(-2.19)

    def assign_target(self, batch_spatial_indices, batch_gt_instance_3d):
        batch_size = len(batch_gt_instance_3d)
        all_names = np.array([*self.class_names])
        inside_box_target = batch_spatial_indices.new_zeros((len(self.group_class_names), batch_spatial_indices.shape[0]))

        batch_gt_boxes = []
        batch_gt_labels = []
        for idx in range(batch_size):
            batch_gt_boxes.append(batch_gt_instance_3d[idx].bboxes_3d)
            batch_gt_labels.append(batch_gt_instance_3d[idx].labels_3d)

        for gidx, names in enumerate(self.group_class_names):
            batch_inside_box_mask = []

            for bidx in range(len(batch_gt_boxes)):
                spatial_indices = batch_spatial_indices[batch_spatial_indices[:, 0] == bidx][:, [2, 1]]
                points = spatial_indices.clone() + 0.5
                points[:, 0] = points[:, 0] * self.featmap_stride * self.voxel_size[0] + self.point_cloud_range[0]
                points[:, 1] = points[:, 1] * self.featmap_stride * self.voxel_size[1] + self.point_cloud_range[1]
                points = torch.cat([points, points.new_zeros((points.shape[0], 1))], dim=-1)

                gt_boxes = batch_gt_boxes[bidx].clone()
                gt_labels = batch_gt_labels[bidx].clone()
                boxes_mask = (gt_boxes.tensor[:, 3] > 0) & (gt_boxes.tensor[:, 4] > 0)
                gt_boxes = gt_boxes[boxes_mask]
                gt_class_names = all_names[gt_labels[boxes_mask].cpu().long().numpy()]

                gt_boxes_single_head = []
                for _, name in enumerate(gt_class_names):
                    if name in names:
                        gt_boxes_single_head.append(gt_boxes[_].tensor.squeeze())

                inside_box_mask = points.new_zeros((points.shape[0]))
                if len(gt_boxes_single_head) > 0:
                    boxes = torch.stack(gt_boxes_single_head)[:, :7]
                    boxes[:, 2] = 0
                    boxes = LiDARInstance3DBoxes(boxes)
                    inside_box_mask[boxes.points_in_boxes_part(points[None]).long() > -1] = 1
                batch_inside_box_mask.append(inside_box_mask)
            inside_box_target[gidx] = torch.cat(batch_inside_box_mask)
        return inside_box_target

    def feature_diffusion(self, x, batch_gt_instance_3d):
        if not self.adaptive_feature_diffusion:
            return x

        detached_x = x
        if self.detach_feature:
            detached_x = spconv.SparseConvTensor(
                features=x.features.detach(),
                indices=x.indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            )
        inside_box_pred = self.cls_conv(detached_x).features.permute(1, 0)
        if self.training:
            inside_box_target = self.assign_target(x.indices, batch_gt_instance_3d)
            self.forward_ret_dict['batch_size'] = x.batch_size
            self.forward_ret_dict['spatial_indices'] = x.indices
            self.forward_ret_dict['inside_box_pred'] = inside_box_pred
            self.forward_ret_dict['inside_box_target'] = inside_box_target

        group_inside_mask = inside_box_pred.sigmoid() > self.fg_thr
        bg_mask = ~group_inside_mask.max(dim=0, keepdim=True)[0]
        group_inside_mask = torch.cat([group_inside_mask, bg_mask], dim=0)

        one_mask = x.features.new_zeros((x.batch_size, 1, x.spatial_shape[0], x.spatial_shape[1]))
        for gidx, inside_mask in enumerate(group_inside_mask):
            selected_indices = x.indices[inside_mask]
            single_one_mask = spconv.SparseConvTensor(
                features=x.features.new_ones(selected_indices.shape[0], 1),
                indices=selected_indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            ).dense()
            pooling_size = self.group_pooling_kernel_size[gidx]
            single_one_mask = F.max_pool2d(single_one_mask, kernel_size=pooling_size, stride=1, padding=pooling_size // 2)
            one_mask = torch.maximum(one_mask, single_one_mask)

        zero_indices = (one_mask[:, 0] > 0).nonzero().int()
        zero_features = x.features.new_zeros((len(zero_indices), x.features.shape[1]))

        cat_indices = torch.cat([x.indices, zero_indices], dim=0)
        cat_features = torch.cat([x.features, zero_features], dim=0)
        indices_unique, _inv = torch.unique(cat_indices, dim=0, return_inverse=True)
        features_unique = x.features.new_zeros((indices_unique.shape[0], x.features.shape[1]))
        features_unique.index_add_(0, _inv, cat_features)

        x = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )
        return x

    def loss(self):
        spatial_indices = self.forward_ret_dict['spatial_indices']
        batch_size = self.forward_ret_dict['batch_size']
        batch_index = spatial_indices[:, 0]

        inside_box_pred = self.forward_ret_dict['inside_box_pred']
        inside_box_target = self.forward_ret_dict['inside_box_target']
        inside_box_pred = torch.cat([inside_box_pred[:, batch_index == bidx] for bidx in range(batch_size)], dim=1)
        inside_box_pred = torch.clamp(inside_box_pred.sigmoid(), min=1e-4, max=1 - 1e-4)

        cls_loss = 0.0
        loss_dict = {}
        for gidx in range(len(self.group_class_names)):
            group_cls_loss = focal_loss_sparse(inside_box_pred[gidx], inside_box_target[gidx].float())
            cls_loss += group_cls_loss

            # fg_mask = inside_box_target[gidx] > 0
            # pred_mask = inside_box_pred[gidx][fg_mask] > self.fg_thr
            # recall_dict[f'afd_recall_{gidx}'] = (pred_mask.sum() / fg_mask.sum().clamp(min=1.0)).item()
            # recall_dict[f'afd_cls_loss_{gidx}'] = group_cls_loss.item()
        loss_dict['afd_cls_loss'] = cls_loss
        return loss_dict


@MODELS.register_module()
class SparseHEDNet2D(SparseHEDNet):
    
    def __init__(self, 
                 input_channels, 
                 grid_size, 
                 class_names, 
                 voxel_size, 
                 point_cloud_range, 
                 model_cfg, 
                 afd, 
                 afd_cfg, 
                 **kwargs):
        kwargs['xy_only'] = True

        super().__init__(input_channels, 
                         grid_size, 
                         class_names, 
                         voxel_size, 
                         point_cloud_range, 
                         model_cfg, 
                         afd, 
                         afd_cfg, 
                         **kwargs)

        self.sparse_shape = grid_size[:2][::-1]
        sed_dim = model_cfg.sed_feature_dim
        afd_dim = afd_cfg.afd_feature_dim

        del self.stem
        self.transition = post_act_block_sparse_2d(
            sed_dim, afd_dim, 3, 2, 1, conv_type='spconv', indice_key='transition')

    def forward(self, batch_inputs_dict, batch_data_samples):

        batch_gt_instance_3d = []
        for data_sample in batch_data_samples:
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)

        voxel_features = batch_inputs_dict['voxel_features']
        voxel_coords = batch_inputs_dict['voxel_coords'].long()
        batch_size = len(batch_data_samples)

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords[:, [0, 2, 3]].int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        for layer in self.sed_layers:
            x = layer(x)

        x = self.transition(x)
        x = self.feature_diffusion(x, batch_gt_instance_3d)
        for layer in self.afd_layers:
            x = layer(x)

        # batch_inputs_dict.update({'spatial_features_2d': x})
        return x
