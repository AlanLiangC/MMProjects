import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from mmdet3d.structures import Det3DDataSample

from typing import Dict, List, Optional, Tuple

@MODELS.register_module()
class UniGF(Base3DDetector):
    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        voxel_encoder: Optional[dict] = None,
        voxel_backbone: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super(UniGF, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg, **kwargs)

        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.voxel_backbone = MODELS.build(voxel_backbone)
        if bbox_head is not None:

            self.bbox_head = MODELS.build(bbox_head)

        if seg_head is not None:
            self.seg_head = MODELS.build(seg_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _forward(self):
        pass

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: List[Det3DDataSample]) -> tuple:
        """Extract features from images and points.
        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains
                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        batch_inputs_dict = self.voxel_encoder(batch_inputs_dict)
        batch_inputs_dict = self.voxel_backbone(batch_inputs_dict, batch_data_samples)

        return batch_inputs_dict

    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.
                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (B, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pts_feats = self.extract_feat(batch_inputs_dict, batch_data_samples)
        losses = dict()
        afd_loss = self.voxel_backbone.loss()
        losses.update(afd_loss)
        head_loss = self.bbox_head.loss(pts_feats, batch_data_samples)
        losses.update(head_loss)
        return losses
    
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.
                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.
            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        pts_feats = self.extract_feat(batch_inputs_dict, batch_data_samples)
        results_list_3d = self.bbox_head.predict(pts_feats, batch_data_samples)

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d)
        return detsamples