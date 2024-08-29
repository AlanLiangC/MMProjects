import copy
from typing import Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmdet3d.models.layers import make_sparse_convmodule
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from mmdet3d.models import circle_nms, gaussian_radius
from mmengine.structures import InstanceData
from mmengine.model import BaseModule
from mmdet.models.utils import multi_apply
from mmdet.models.task_modules import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.models.layers import nms_bev


from .spconv_utils import spconv
from .centernet_utils import draw_gaussian_to_normalized_heatmap
from .utils import clip_sigmoid


def to_dense(self, channels_first: bool = True):

    def scatter_nd(indices, updates, shape):
        ret = - torch.ones(*shape, dtype=updates.dtype, device=updates.device) * 1e6
        ndim = indices.shape[-1]
        output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
        flatted_indices = indices.view(-1, ndim)
        slices = [flatted_indices[:, i] for i in range(ndim)]
        slices += [Ellipsis]
        ret[slices] = updates.view(*output_shape)
        return ret

    output_shape = [self.batch_size] + list(
        self.spatial_shape) + [self.features.shape[1]]
    res = scatter_nd(
        self.indices.to(self.features.device).long(), self.features,
        output_shape)
    if not channels_first:
        return res
    ndim = len(self.spatial_shape)
    trans_params = list(range(0, ndim + 1))
    trans_params.insert(1, ndim + 1)
    return res.permute(*trans_params).contiguous()


@MODELS.register_module()
class SparseTransFusionHead(nn.Module):
    def __init__(
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        use_cross_attn_mask=False,
        num_decoder_layers=3,
        decoder_layer=dict(),
        num_heads=8,
        nms_kernel_size=1,
        bn_momentum=0.1,
        # config for FFN
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        bias='auto',
        # loss
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean'),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        super(SparseTransFusionHead, self).__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.use_cross_attn_mask = use_cross_attn_mask
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        self.shared_conv = make_sparse_convmodule(                
            in_channels,
            hidden_channel,
            3,
            norm_cfg=norm_cfg,
            padding=1,
            indice_key='share_conv',
            conv_type='SubMConv2d',
            order=('conv', ))

        heatmap_head = []
        heatmap_head.append(
            make_sparse_convmodule(
                hidden_channel,
                hidden_channel,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='heatmap_conv',
                conv_type='SubMConv2d',
                order=('conv', 'norm', 'act'))
        )
        # watch here
        heatmap_head.append(spconv.SubMConv2d(hidden_channel, num_classes, 1, bias=True, indice_key='heatmap_out'))
        self.heatmap_head = spconv.SparseSequential(*heatmap_head)
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(MODELS.build(decoder_layer))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                SeparateHead(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                ))
            
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training # noqa: E501
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg[
            'out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg[
            'out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1) # [1, H*W, 2]
        return nn.Parameter(coord_base, requires_grad=False)

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        # if self.train_cfg is None:
        #     return

        # if self.sampling:
        #     self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        # else:
        #     self.bbox_sampler = PseudoSampler()
        # if isinstance(self.train_cfg.assigner, dict):
        #     self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        # elif isinstance(self.train_cfg.assigner, list):
        #     self.bbox_assigner = [
        #         build_assigner(res) for res in self.train_cfg.assigner
        #     ]
        from .pcdet_utils import HungarianAssigner3D
        self.bbox_assigner = HungarianAssigner3D(cls_cost={'gamma': 2.0, 'alpha': 0.25, 'weight': 0.15},
                                                 reg_cost={'weight': 0.25},
                                                 iou_cost={'weight': 0.25})

    def forward_single(self, x, metas):

        batch_size = x.batch_size
        x = self.shared_conv(x)
        x_flatten = x.dense().view(batch_size, x.features.shape[1], -1)  # [B, C, H*W]
        with torch.autocast('cuda', enabled=False):
            dense_heatmap = to_dense(self.heatmap_head(x))                  # [B, num_classes, H, W]
        heatmap = dense_heatmap.detach().sigmoid()

        # perform max pooling on heatmap
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg[
                'dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]

        # query
        self.query_labels = top_proposals_class
        query_feat = x_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, x_flatten.shape[1], -1),dim=-1,)
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1)   # [B, H*W, 2]
        query_pos = bev_pos.gather(index=top_proposals_index[:, :, None].expand(-1, -1, bev_pos.shape[-1]), dim=1) # [B, K, 2]

        ret_dicts = []
        if self.use_cross_attn_mask:
            tensor_mask = spconv.SparseConvTensor(
                features=x.features.new_ones(x.indices.shape[0], 1),
                indices=x.indices, spatial_shape=x.spatial_shape, batch_size=x.batch_size).dense()  # [B, 1, H, W]

            # attn_mask: [B * num_head, K, H*W]
            cross_attn_mask = tensor_mask[:, None].expand(-1, self.num_heads, query_feat.shape[2], -1, -1)
            cross_attn_mask = cross_attn_mask.reshape(batch_size * self.num_heads, query_feat.shape[2], -1).bool()

        for i in range(self.num_decoder_layers):

            query_feat = self.decoder[i](
                query_feat,
                key=x_flatten,
                query_pos=query_pos,
                key_pos=bev_pos,
                cross_attn_mask=cross_attn_mask if self.use_cross_attn_mask else None)

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer['center'] = res_layer['center'] + query_pos.permute(
                0, 2, 1)
            ret_dicts.append(res_layer)
            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        ret_dicts[0]['query_heatmap_score'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1,)  # [bs, num_classes, num_proposals]
        ret_dicts[0]['dense_heatmap'] = dense_heatmap
        if self.use_cross_attn_mask:
            ret_dicts[0]['tensor_mask'] = tensor_mask

        if self.auxiliary is False:
            return [ret_dicts[-1]]
        
        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in [
                    'dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score', 'tensor_mask'
            ]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]
    
    def forward(self, feats, metas):

        if isinstance(feats, spconv.SparseConvTensor) or isinstance(feats, torch.Tensor):
            feats = [feats]
        res = multi_apply(self.forward_single, feats, [metas])
        assert len(res) == 1, 'only support one level features.'
        return res
    
    def loss(self, batch_feats, batch_data_samples):
        """Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        batch_input_metas, batch_gt_instances_3d = [], []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
        preds_dicts = self(batch_feats, batch_input_metas)
        loss = self.loss_by_feat(preds_dicts, batch_gt_instances_3d)

        return loss
    
    def get_targets(self, batch_gt_instances_3d: List[InstanceData],
                    preds_dict: List[dict]):

        list_of_pred_dict = []
        for batch_idx in range(len(batch_gt_instances_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                preds = []
                for i in range(self.num_decoder_layers):
                    pred_one_layer = preds_dict[i][key][batch_idx:batch_idx +
                                                        1]
                    preds.append(pred_one_layer)
                pred_dict[key] = torch.cat(preds)
            list_of_pred_dict.append(pred_dict)

        assert len(batch_gt_instances_3d) == len(list_of_pred_dict)
        res_tuple = multi_apply(
            self.get_targets_single,
            batch_gt_instances_3d,
            list_of_pred_dict,
            np.arange(len(batch_gt_instances_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        )

    def get_targets_single(self, gt_instances_3d, preds_dict, batch_idx):

        # 1. Assignment
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)

        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals *
                                                idx_layer:self.num_proposals *
                                                (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals *
                                idx_layer:self.num_proposals *
                                (idx_layer + 1), ]

            assigned_gt_inds, ious = self.bbox_assigner.assign(
                bboxes_tensor_layer, 
                gt_bboxes_tensor, 
                gt_labels_3d, 
                score_layer, 
                self.train_cfg['point_cloud_range'])

        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        assert len(pos_inds) + len(neg_inds) == num_proposals
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if gt_bboxes_3d.tensor.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_tensor).view(-1, 9)
        else:
            pos_gt_bboxes = gt_bboxes_tensor[pos_assigned_gt_inds.long()]

        # 3. Create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long) + self.num_classes
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(pos_gt_bboxes)
            bbox_targets[pos_inds] = pos_bbox_targets
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        tensor_mask = None
        if self.use_cross_attn_mask:
            tensor_mask = preds_dict['tensor_mask'].squeeze()

        # # compute sparse heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = (grid_size[:2] // self.train_cfg['out_size_factor']
                            )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1],
                                         feature_map_size[0])

        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = ((x - pc_range[0]) / voxel_size[0] /
                          self.train_cfg['out_size_factor'])
                coor_y = ((y - pc_range[1]) / voxel_size[1] /
                          self.train_cfg['out_size_factor'])

                center = torch.tensor([coor_x, coor_y],
                                      dtype=torch.float32,
                                      device=device)
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) # noqa: E501
                # NOTE: fix

                draw_gaussian_to_normalized_heatmap(heatmap[gt_labels_3d[idx]],
                                      center_int, radius, valid_mask=tensor_mask, normalize=True)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)

        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
        )

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        ) = self.get_targets(batch_gt_instances_3d, preds_dicts[0])

        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        loss_dict = dict()

        preds_dicts = preds_dicts[0][0]
        dense_heatmap = clip_sigmoid(preds_dicts['dense_heatmap'])
        if self.use_cross_attn_mask:
            tensor_mask = preds_dicts['tensor_mask']
            tensor_mask = tensor_mask.expand(-1, dense_heatmap.shape[1], -1, -1).bool()
            dense_heatmap = dense_heatmap[tensor_mask]
            heatmap = heatmap[tensor_mask]

        # heatmap loss
        loss_heatmap = self.loss_heatmap(
                                dense_heatmap,
                                heatmap.float(),
                                avg_factor=max(heatmap.eq(1).float().sum().item(), 1),)
        loss_dict['loss_heatmap'] = loss_heatmap

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = preds_dicts['heatmap'].permute(0, 2, 1).reshape(-1, self.num_classes)

        label_targets = torch.zeros(*list(labels.shape), self.num_classes + 1, dtype=cls_score.dtype, device=labels.device)
        label_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        label_targets = label_targets[..., :-1]

        loss_cls = self.loss_cls(cls_score, label_targets, label_weights,
                                 avg_factor=max(num_pos, 1))
        loss_dict['loss_cls'] = loss_cls

        # regression loss
        preds = torch.cat([preds_dicts[head_name] for head_name in ['center', 'height', 'dim', 'rot', 'vel']], dim=1).permute(0, 2, 1)
        code_weights = self.train_cfg.get('code_weights', None)
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)
        loss_bbox = self.loss_bbox(preds, bbox_targets, reg_weights,avg_factor=max(num_pos, 1))
        loss_dict['loss_bbox'] = loss_bbox * 0.25

        loss_dict['matched_ious'] = loss_cls.new_tensor(matched_ious)
        return loss_dict


    def fake_loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        ) = self.get_targets(batch_gt_instances_3d, preds_dicts[0])

        loss_dict = dict()

        preds_dicts = preds_dicts[0][0]
        dense_heatmap = clip_sigmoid(preds_dicts['dense_heatmap'])
        if self.use_cross_attn_mask:
            tensor_mask = preds_dicts['tensor_mask']
            tensor_mask = tensor_mask.expand(-1, dense_heatmap.shape[1], -1, -1).bool()
            dense_heatmap = dense_heatmap[tensor_mask]
            heatmap = heatmap[tensor_mask]

        # heatmap loss
        loss_heatmap = self.loss_heatmap(
                                dense_heatmap,
                                heatmap.float(),
                                avg_factor=max(heatmap.eq(1).float().sum().item(), 1),)

        loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            # classification loss
            layer_labels = labels[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, ].reshape(-1)
            layer_label_weights = label_weights[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, ].reshape(-1)
            layer_score = preds_dicts['heatmap'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)

            layer_label_targets = torch.zeros(*list(layer_labels.shape), self.num_classes + 1, dtype=layer_cls_score.dtype, device=labels.device)
            layer_label_targets.scatter_(-1, layer_labels.unsqueeze(dim=-1).long(), 1.0)
            layer_label_targets = layer_label_targets[..., :-1]
            layer_loss_cls = self.loss_cls(
                                layer_cls_score.float(),
                                layer_label_targets,
                                layer_label_weights,
                                avg_factor=max(num_pos, 1),
                            )

            # regression loss
            layer_center = preds_dicts['center'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, ]
            layer_height = preds_dicts['height'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, ]
            layer_rot = preds_dicts['rot'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, ]
            layer_dim = preds_dicts['dim'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, ]
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1) 

            if 'vel' in preds_dicts.keys():
                layer_vel = preds_dicts['vel'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, ]
                preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :, ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :, ]
            layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox

        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict