import torch
import torch.nn as nn
from trackit.criteria import CriterionOutput
from trackit.criteria.modules.iou_loss import bbox_overlaps
from trackit.miscellanies.torch.distributed.reduce_mean import reduce_mean_


class SimpleCriteria(nn.Module):
    def __init__(self, cls_loss: nn.Module, bbox_reg_loss: nn.Module,
                 iou_aware_classification_score: bool,
                 cls_loss_weight: float, bbox_reg_loss_weight: float,
                 cls_loss_display_name: str, bbox_reg_loss_display_name: str, warmup_epochs: int,
                 evidence_config: dict = None):
        super().__init__()
        self.cls_loss = cls_loss
        self.bbox_reg_loss = bbox_reg_loss
        self.iou_aware_classification_score = iou_aware_classification_score
        self.cls_loss_weight = cls_loss_weight
        self.bbox_reg_loss_weight = bbox_reg_loss_weight
        self.cls_loss_display_name = cls_loss_display_name
        self.bbox_reg_loss_display_name = bbox_reg_loss_display_name
        self._across_all_nodes_normalization = True
        self._warmup_epochs = warmup_epochs
        self.evidence_enabled = evidence_config is not None and evidence_config.get('enabled', False)
        if self.evidence_enabled:
            self.evidence_loss = nn.BCEWithLogitsLoss()
            self.evidence_iou_threshold = evidence_config.get('iou_threshold', 0.5)
            self.evidence_loss_weight = evidence_config.get('weight', 1.0)
            self.evidence_score_map_with_sigmoid = evidence_config.get('score_map_with_sigmoid', True)
            self.evidence_label_source = evidence_config.get('label_source', 'iou')

    def forward(self, outputs: dict, targets: dict):
        metrics = {}
        extra_metrics = {}
        total_loss = 0
        epoch = targets['epoch']

        for i, output in enumerate(outputs):
            num_positive_samples = targets[f'num_positive_samples_{i}']
            assert isinstance(num_positive_samples, torch.Tensor)

            reduce_mean_(num_positive_samples)  # caution: inplace update
            num_positive_samples.clamp_(min=1.)

            predicted_score_map = output['score_map'].to(torch.float)
            predicted_bboxes = output['boxes'].to(torch.float)
            groundtruth_bboxes = targets[f'boxes_{i}']
            evidence_groundtruth_bboxes = groundtruth_bboxes

            N, H, W = predicted_score_map.shape

            # shape: (num_positive_samples, )
            positive_sample_batch_dim_index = targets[f'positive_sample_batch_dim_indices_{i}']
            # shape: (num_positive_samples, )
            positive_sample_feature_map_dim_index = targets[f'positive_sample_map_dim_indices_{i}']

            has_positive_samples = positive_sample_batch_dim_index is not None

            if has_positive_samples:
                predicted_bboxes = predicted_bboxes.view(N, H * W, 4)
                predicted_bboxes = predicted_bboxes[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index]
                groundtruth_bboxes = groundtruth_bboxes[positive_sample_batch_dim_index]

            with torch.no_grad():
                groundtruth_response_map = torch.zeros((N, H * W),  dtype=torch.float32, device=predicted_score_map.device)
                if has_positive_samples:
                    if self.iou_aware_classification_score:
                        groundtruth_response_map.index_put_(
                            (positive_sample_batch_dim_index, positive_sample_feature_map_dim_index),
                            bbox_overlaps(groundtruth_bboxes, predicted_bboxes, is_aligned=True))
                    else:
                        groundtruth_response_map[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index] = 1.

            cls_loss = self.cls_loss(predicted_score_map.view(N, -1), groundtruth_response_map).sum() / num_positive_samples

            if has_positive_samples:
                reg_loss = self.bbox_reg_loss(predicted_bboxes, groundtruth_bboxes).sum() / num_positive_samples
            else:
                reg_loss = predicted_bboxes.mean() * 0

            if self.cls_loss_weight != 1.:
                cls_loss = cls_loss * self.cls_loss_weight

            bbox_reg_loss_weight = self.bbox_reg_loss_weight
            if bbox_reg_loss_weight != 1. or (self._warmup_epochs > 0 and 0 <= epoch < self._warmup_epochs):
                if self._warmup_epochs > 0 and 0 <= epoch < self._warmup_epochs:
                    bbox_reg_loss_weight = bbox_reg_loss_weight * 10
                reg_loss = reg_loss * bbox_reg_loss_weight

            cls_loss_cpu = cls_loss.detach().cpu().item()
            reg_loss_cpu = reg_loss.detach().cpu().item()

            metrics.update({f'Loss/{self.cls_loss_display_name}_{i}': cls_loss_cpu, f'Loss/{self.bbox_reg_loss_display_name}_{i}': reg_loss_cpu})
            extra_metrics.update({f'Loss/{self.cls_loss_display_name}_unscale_{i}': cls_loss_cpu / self.cls_loss_weight, f'Loss/{self.bbox_reg_loss_display_name}_unscale_{i}': reg_loss_cpu / bbox_reg_loss_weight})

            total_loss += cls_loss
            total_loss += reg_loss
            if self.evidence_enabled:
                target_presence = targets.get(f'target_presence_{i}')
                evidence_loss, evidence_metrics = self._compute_evidence_loss(output, evidence_groundtruth_bboxes,
                                                                              target_presence, N, H, W)
                total_loss += evidence_loss
                metrics.update({f'Loss/evidence_{i}': evidence_metrics['loss']})
                extra_metrics.update({
                    f'Evidence/label_positive_ratio_{i}': evidence_metrics['label_positive_ratio'],
                    f'Evidence/prediction_accuracy_{i}': evidence_metrics['accuracy'],
                    f'Evidence/mean_iou_{i}': evidence_metrics['mean_iou'],
                    f'Evidence/mean_score_{i}': evidence_metrics['mean_score'],
                })

        return CriterionOutput(total_loss, metrics, extra_metrics)

    def _compute_evidence_loss(self, output: dict, groundtruth_bboxes: torch.Tensor,
                               target_presence: torch.Tensor, N: int, H: int, W: int):
        if 'evidence' not in output:
            raise RuntimeError('evidence loss is enabled, but model output does not contain evidence logits')

        evidence_logits = output['evidence'].to(torch.float).view(N)
        predicted_score_map = output['score_map'].detach().to(torch.float)
        predicted_bboxes = output['boxes'].detach().to(torch.float)
        groundtruth_bboxes = groundtruth_bboxes.to(torch.float)

        with torch.no_grad():
            if self.evidence_score_map_with_sigmoid:
                ranking_score = predicted_score_map.sigmoid()
            else:
                ranking_score = predicted_score_map
            best_indices = ranking_score.view(N, H * W).argmax(dim=1)
            selected_bboxes = predicted_bboxes.view(N, H * W, 4)[
                torch.arange(N, device=predicted_bboxes.device), best_indices]
            selected_iou = bbox_overlaps(selected_bboxes, groundtruth_bboxes, is_aligned=True)
            if self.evidence_label_source == 'iou':
                evidence_labels = (selected_iou >= self.evidence_iou_threshold).to(torch.float)
            elif self.evidence_label_source == 'target_presence':
                if target_presence is None:
                    raise RuntimeError('target_presence evidence labels are enabled, but target_presence is missing')
                evidence_labels = target_presence.to(evidence_logits.device, dtype=torch.float).view(N)
            elif self.evidence_label_source == 'presence_and_iou':
                if target_presence is None:
                    raise RuntimeError('presence_and_iou evidence labels are enabled, but target_presence is missing')
                target_presence = target_presence.to(evidence_logits.device, dtype=torch.bool).view(N)
                evidence_labels = (target_presence & (selected_iou >= self.evidence_iou_threshold)).to(torch.float)
            else:
                raise ValueError(f'Unknown evidence label source: {self.evidence_label_source}')

        evidence_loss = self.evidence_loss(evidence_logits, evidence_labels)
        if self.evidence_loss_weight != 1.:
            evidence_loss = evidence_loss * self.evidence_loss_weight

        with torch.no_grad():
            evidence_scores = evidence_logits.sigmoid()
            evidence_predictions = evidence_scores >= 0.5
            metrics = {
                'loss': evidence_loss.detach().cpu().item(),
                'label_positive_ratio': evidence_labels.mean().detach().cpu().item(),
                'accuracy': (evidence_predictions == evidence_labels.to(torch.bool)).to(torch.float).mean().detach().cpu().item(),
                'mean_iou': selected_iou.mean().detach().cpu().item(),
                'mean_score': evidence_scores.mean().detach().cpu().item(),
            }
        return evidence_loss, metrics
