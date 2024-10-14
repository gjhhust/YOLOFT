# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


# Losses
class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        super().__init__()

    def forward(self, pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

from torch import Tensor
def gaussian_focal_loss(pred: Tensor,
                        gaussian_target: Tensor,
                        alpha: float = 2.0,
                        gamma: float = 4.0,
                        pos_weight: float = 1.0,
                        neg_weight: float = 1.0) -> Tensor:
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_weight * pos_loss + neg_weight * neg_loss


# def gaussian_focal_loss_with_pos_inds(
#         pred: Tensor,
#         gaussian_target: Tensor,
#         pos_inds: Tensor,
#         pos_labels: Tensor,
#         alpha: float = 2.0,
#         gamma: float = 4.0,
#         pos_weight: float = 1.0,
#         neg_weight: float = 1.0,
#         reduction: str = 'mean',
#         avg_factor: Optional[Union[int, float]] = None) -> Tensor:
#     """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
#     distribution.

#     Note: The index with a value of 1 in ``gaussian_target`` in the
#     ``gaussian_focal_loss`` function is a positive sample, but in
#     ``gaussian_focal_loss_with_pos_inds`` the positive sample is passed
#     in through the ``pos_inds`` parameter.

#     Args:
#         pred (torch.Tensor): The prediction. The shape is (N, num_classes).
#         gaussian_target (torch.Tensor): The learning target of the prediction
#             in gaussian distribution. The shape is (N, num_classes).
#         pos_inds (torch.Tensor): The positive sample index.
#             The shape is (M, ).
#         pos_labels (torch.Tensor): The label corresponding to the positive
#             sample index. The shape is (M, ).
#         alpha (float, optional): A balanced form for Focal Loss.
#             Defaults to 2.0.
#         gamma (float, optional): The gamma for calculating the modulating
#             factor. Defaults to 4.0.
#         pos_weight(float): Positive sample loss weight. Defaults to 1.0.
#         neg_weight(float): Negative sample loss weight. Defaults to 1.0.
#         reduction (str): Options are "none", "mean" and "sum".
#             Defaults to 'mean`.
#         avg_factor (int, float, optional): Average factor that is used to
#             average the loss. Defaults to None.
#     """
#     eps = 1e-12
#     neg_weights = (1 - gaussian_target).pow(gamma)

#     pos_pred_pix = pred[pos_inds]
#     pos_pred = pos_pred_pix.gather(1, pos_labels.unsqueeze(1))
#     pos_loss = -(pos_pred + eps).log() * (1 - pos_pred).pow(alpha)
#     pos_loss = weight_reduce_loss(pos_loss, None, reduction, avg_factor)

#     neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
#     neg_loss = weight_reduce_loss(neg_loss, None, reduction, avg_factor)

#     return pos_weight * pos_loss + neg_weight * neg_loss


class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """

    def __init__(self,
                 alpha: float = 2.0,
                 gamma: float = 4.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 pos_weight: float = 1.0,
                 neg_weight: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                pos_inds: Tensor = None,
                pos_labels:Tensor = None,
                weight: Tensor = None,
                avg_factor:Tensor = None,
                reduction_override: str = None) -> Tensor:
        """Forward function.

        If you want to manually determine which positions are
        positive samples, you can set the pos_index and pos_label
        parameter. Currently, only the CenterNet update version uses
        the parameter.

        Args:
            pred (torch.Tensor): The prediction. The shape is (N, num_classes).
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution. The shape is (N, num_classes).
            pos_inds (torch.Tensor): The positive sample index.
                Defaults to None.
            pos_labels (torch.Tensor): The label corresponding to the positive
                sample index. Defaults to None.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if pos_inds is not None:
            assert pos_labels is not None
            # Only used by centernet update version
            loss_reg = self.loss_weight * gaussian_focal_loss_with_pos_inds(
                pred,
                target,
                pos_inds,
                pos_labels,
                alpha=self.alpha,
                gamma=self.gamma,
                pos_weight=self.pos_weight,
                neg_weight=self.neg_weight,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            loss_reg = self.loss_weight * gaussian_focal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                gamma=self.gamma,
                pos_weight=self.pos_weight,
                neg_weight=self.neg_weight,
                reduction=reduction,
                avg_factor=avg_factor)
        return loss_reg

def siou_loss(pred, target, eps=1e-7, neg_gamma=False):
    r"""`Implementation of paper `SIoU Loss: More Powerful Learning
    for Bounding Box Regression <https://arxiv.org/abs/2205.12740>`_.

    Code is modified from https://github.com/meituan/YOLOv6.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
        neg_gamma (bool): `True` follows original implementation in paper.

    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    # modified clamp threshold zero to eps to avoid NaN
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=eps)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # angle cost
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps

    sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)

    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(torch.asin(sin_alpha) * 2 - math.pi / 2)

    # distance cost
    rho_x = (s_cw / cw)**2
    rho_y = (s_ch / ch)**2

    # `neg_gamma=True` follows original implementation in paper
    # but setting `neg_gamma=False` makes training more stable.
    gamma = angle_cost - 2 if neg_gamma else 2 - angle_cost
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    # shape cost
    omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
        1 - torch.exp(-1 * omiga_h), 4)

    # SIoU
    sious = ious - 0.5 * (distance_cost + shape_cost)
    loss = 1 - sious.clamp(min=-1.0, max=1.0)
    return loss

class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, bbox_siou=False):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if bbox_siou:
            pass
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
                    # + (F.l1_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                    #            reduction='none') * weight).sum() / target_scores_sum / 100

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()

class CenterpointLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_kpts, gt_kpts , area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        sigmas = torch.ones(pred_kpts.shape[0], device=area.device)/pred_kpts.shape[0]
        e = d / (2 * sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        loss = ((1 - torch.exp(-e))).mean()
        return loss

def wasserstein_loss(pred, target, eps=1e-7, mode='exp', gamma=1, constant=12.8):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    center1 = (pred[:, :2] + pred[:, 2:]) / 2
    center2 = (target[:, :2] + target[:, 2:]) / 2

    whs = center1[:, :2] - center2[:, :2]

    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps #

    w1 = pred[:, 2] - pred[:, 0]  + eps
    h1 = pred[:, 3] - pred[:, 1]  + eps
    w2 = target[:, 2] - target[:, 0]  + eps
    h2 = target[:, 3] - target[:, 1]  + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance

    if mode == 'exp':
        normalized_wasserstein = torch.exp(-torch.sqrt(wasserstein_2)/constant)
        wloss = 1 - normalized_wasserstein
    
    if mode == 'sqrt':
        wloss = torch.sqrt(wasserstein_2)
    
    if mode == 'log':
        wloss = torch.log(wasserstein_2 + 1)

    if mode == 'norm_sqrt':
        wloss = 1 - 1 / (gamma + torch.sqrt(wasserstein_2))

    if mode == 'w2':
        wloss = wasserstein_2

    return wloss.sum()


class WassersteinLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=10.0, mode='exp', gamma=2, constant=5.0):
        super(WassersteinLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mode = mode
        self.gamma = gamma
        self.constant = constant    # constant = 12.8 for AI-TOD

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * wasserstein_loss(
            pred,
            target,
            eps=self.eps,
            mode=self.mode,
            gamma=self.gamma,
            constant=self.constant)
        return loss

class MOVEDetectionLoss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        wasserstein = True if self.hyp.loss_mode == "wasserstein"  else False

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0, wasserstein = wasserstein)
            
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        
        self.centerPoint_loss = CenterpointLoss()
        self.Wasserstein_loss = WassersteinLoss(loss_weight=1.0)
        
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl,move
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores, pred_ismove = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc, 1), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_ismove = pred_ismove.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes'], batch['is_moving'].view(-1, 1)), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes, gt_ismove = targets.split((1, 4, 1), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        # import pdb;pdb.set_trace()
        target_ismove = gt_ismove.long().flatten()[target_gt_idx].unsqueeze(-1)  # (b, h*w)
        target_ismove.clamp_(0)
        loss[-1] = self.bce(pred_ismove, target_ismove.to(dtype)).sum() / max(target_ismove.sum(), 1)  # BCE
        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor

            if self.hyp.loss_mode == "wasserstein":
                loss[0] = self.Wasserstein_loss(pred_bboxes[fg_mask],target_bboxes[fg_mask])
            else:
                loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                                target_scores_sum, fg_mask)               
            
            pred_kpts = (pred_bboxes[...,:2]+(pred_bboxes[...,2:]-pred_bboxes[...,:2])/2).float().clone()
            keypoints = (target_bboxes[...,:2]+(target_bboxes[...,2:]-target_bboxes[...,:2])/2).float().clone()
            for i in range(batch_size):   
                idx = target_gt_idx[i][fg_mask[i]]
                gt_kpt = keypoints[i][idx]  # (n, 2)
                n = gt_kpt.size(0)
                if n < 1:
                    continue
                ones_column = torch.ones((n, 1), dtype=gt_kpt.dtype, device=gt_kpt.device)
                gt_kpt = torch.cat((gt_kpt, ones_column), dim=1)
                area = xyxy2xywh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                pred_kpt = pred_kpts[i][fg_mask[i]]
                loss[3] += self.centerPoint_loss(pred_kpt, gt_kpt, area)  # cpt loss
                # if torch.isnan(loss[3]).any(): 
                    # import pdb;pdb.set_trace()

        if self.hyp.loss_mode == "wasserstein":
            loss[0] *= self.hyp.wasserstein_loss / batch_size 
            loss[1] *= self.hyp.cls
            loss[2] = 0
            loss[3] = self.hyp.mov
        if self.hyp.loss_mode == "cpt":
            loss[0] *= self.hyp.box  # box gain
            loss[1] *= self.hyp.cls  # cls gain
            loss[2] *= self.hyp.dfl  # dfl gain
            loss[3] *= self.hyp.mov
        else:
            loss[0] *= self.hyp.box  # box gain
            loss[1] *= self.hyp.cls  # cls gain
            loss[2] *= self.hyp.dfl  # dfl gain
            loss[3] *= self.hyp.mov
        
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

import cv2
import numpy as np
def draw_rectangle(image, pt1, pt2, color, thickness):
    return cv2.polylines(image, [np.array([pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])], dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

def generate_gaussian_ellipse_masks(targets, imgsz, show=False):
    """
    生成每个目标的外接椭圆的高斯掩码图

    Args:
    - targets (torch.Tensor): 目标信息矩阵，形状为 [bs, n, 5]，包括 cls, bbox(x1, y1, x2, y2)
    - imgsz (tuple): 原图的长宽，如 (height, width)

    Returns:
    - masks (torch.Tensor): 椭圆的高斯掩码图，形状为 [bs, 1, height, width]
    """
    batch_size, max_gt_count, _ = targets.shape
    masks = []

    for i in range(batch_size):
        gt_count = int(torch.sum(targets[i, :, 0] != 0))  # 统计每个 batch 中有效的 GT bbox 数量

        # 获取当前目标的信息
        cls = targets[i, :gt_count, 0]
        bboxes = targets[i, :gt_count, 1:]

        # 计算外接矩形
        centers = (bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2
        major_axes = (bboxes[:, 2] - bboxes[:, 0])
        minor_axes = (bboxes[:, 3] - bboxes[:, 1])
        angles = torch.zeros_like(major_axes)  # 如果椭圆是外接矩形，角度为 0

        # 生成高斯椭圆掩码图
        mask = draw_gaussian_ellipses(imgsz, centers, major_axes, minor_axes, angles, intensity=255)

        # 将 GT bbox 绘制在图上
        if show:
            for j in range(gt_count):
                bbox = bboxes[j].cpu().numpy().astype(int)
                mask = draw_rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=255, thickness=1)

        masks.append(torch.from_numpy(mask))

    masks = torch.stack(masks, dim=0).unsqueeze(1)  # 堆叠成批量的张量

    return masks


def draw_gaussian_ellipses(image_shape, centers, major_axes, minor_axes, angles, intensity=255, sigma=1.0):
    """
    生成多个高斯椭圆灰度图

    Args:
    - image_shape (tuple): 输出图像的形状，如 (height, width)
    - centers (tuple): 椭圆的中心坐标，分别是 x 坐标和 y 坐标
    - major_axes (torch.Tensor): 椭圆的长轴长度
    - minor_axes (torch.Tensor): 椭圆的短轴长度
    - angles (torch.Tensor): 椭圆的旋转角度（以度为单位）
    - intensity (float): 椭圆的最大强度
    - sigma (float): 高斯分布的标准差

    Returns:
    - image (torch.Tensor): 高斯椭圆灰度图，形状为 (height, width)
    """
    image = np.zeros(image_shape, dtype=np.float32)

    for center_x, center_y, major_axis, minor_axis, angle in zip(centers[0], centers[1], major_axes, minor_axes, angles):
        center = (int(center_x.item()), int(center_y.item()))
        axes = (int(major_axis.item() / 2), int(minor_axis.item() / 2))

        # 生成椭圆的权重矩阵
        ellipse = np.zeros(image_shape, dtype=np.float32)
        cv2.ellipse(ellipse, center, axes, 0, 0, 360, intensity, -1)

        # 生成高斯分布，简化计算
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        dist = np.sqrt((x - center[0]) ** 2 / (axes[0] ** 2) + (y - center[1]) ** 2 / (axes[1] ** 2))
        gaussian = np.exp(-(dist ** 2) / (2 * (sigma ** 2)))

        # 将椭圆的权重矩阵与高斯分布相乘
        ellipse = ellipse * gaussian

        # 将图像中的椭圆添加到原始图像中
        image += ellipse

    return image


import torch.nn.functional as F

def get_top_k_indices(scores, tensor, k):
  """
  根据每个 batch 的 scores 排序在前 k 的 index 截取 tensor

  Args:
    scores: scores 的维度为 [batchsize, n]
    tensor2: tensor2 的维度为 [batchsize, n, 4]
    k: 要截取的元素个数

  Returns:
    截取后的 tensor2，维度为 [batchsize, k, 4]
  """

  # 沿着第二维度排序
  value, indices = torch.sort(scores, dim=1, descending=True)

  # 取出每个 batch 排序在前 k 的 index
  top_k_indices = indices[:, :k]

  # 根据 index 截取 tensor2
  top_k_tensor = torch.gather(tensor, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, 4))

  return top_k_tensor, value[:, :k]

def resize_tensor(input_tensor, new_height, new_width):
    """
    将输入张量resize到指定的新高度和宽度。

    Args:
    - input_tensor (torch.Tensor): 输入张量，形状为 [batch_size, 1, height, width]
    - new_height (int): 目标高度
    - new_width (int): 目标宽度

    Returns:
    - resized_tensor (torch.Tensor): 调整大小后的张量，形状为 [batch_size, 1, new_height, new_width]
    """
    batch_size, _, height, width = input_tensor.size()
    resized_tensor = F.interpolate(input_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
    return resized_tensor

# Criterion class for computing Detection training losses
class v8DetectionLoss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module

        self.pred_index = m.nl 
        # save topk bbox
        if hasattr(m, 'buffer'):
            # import pdb;pdb.set_trace()
            self.buffer = m.buffer
            self.topkbbox = m.topkbbox
        
        self.mode = getattr(m, "mode", "normal")
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0, wasserstein = False)
            
            
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        
        self.centerPoint_loss = CenterpointLoss()
        self.Wasserstein_loss = WassersteinLoss(loss_weight=1.0)
        
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def total_mask_loss(self, pred_masks, gt_masks, gt_bbox, original_size):
        """
        计算总的掩膜损失。

        Args:
        pred_masks (torch.Tensor): 预测掩膜，形状为 [bs, 1, h, w]
        gt_masks (torch.Tensor): 真实掩膜，形状为 [bs, 1, h, w]
        gt_bbox (torch.Tensor): ground truth bounding box，形状为 [bs, n, 4]
        original_size (tuple): 原始图像的大小，形状为 (original_height, original_width)

        Returns:
        (torch.Tensor): 总的掩膜损失
        """
        bs, n, _ = gt_bbox.shape

        # 调整bounding box大小和归一化bounding box到掩膜图的尺寸
        adjusted_normalized_gt_box = torch.stack([
            gt_bbox[:, :, 0] / original_size[1],  # x1
            gt_bbox[:, :, 1] / original_size[0],  # y1
            gt_bbox[:, :, 2] / original_size[1],  # x2
            gt_bbox[:, :, 3] / original_size[0],  # y2
        ], dim=2)

        total_loss = 0.0

        for i in range(bs):
            gt_mask = gt_masks[i:i+1]
            pred_mask = pred_masks[i:i+1]
            
            # 计算掩膜损失
            loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='mean')

            # 计算归一化后的面积
            area = (adjusted_normalized_gt_box[i, :, :, 2] - adjusted_normalized_gt_box[i, :, :, 0]) * \
                (adjusted_normalized_gt_box[i, :, :, 3] - adjusted_normalized_gt_box[i, :, :, 1])

            # 根据面积进行归一化
            normalized_loss = crop_mask(loss, adjusted_normalized_gt_box[i]).mean(dim=(1, 2)) / area

            # 将归一化的损失累加到总的损失中
            total_loss += normalized_loss.mean()

        # 对所有batch取平均
        total_loss /= bs

        return total_loss

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        feats = preds[1] if isinstance(preds, tuple) else preds

        aux_losses = feats[self.pred_index:]
        feats = feats[:self.pred_index]

        loss = torch.zeros(3+len(aux_losses), device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)


        ancillary_losses_result = {}
        # if ancillary_losses:
        #     pass
            # for key, value in ancillary_losses.items():
            #     bs,_,h,w = value.shape
            #     masks = generate_gaussian_ellipse_masks(targets, imgsz.cpu().numpy().astype(np.int16), False).to(self.device).float()
            #     if tuple(masks.shape[-2:]) != (h, w):  # downsample
            #         masks = F.interpolate(masks, (h, w), mode='nearest')
            #     ancillary_losses_result[key] = self.total_mask_loss(value, masks, gt_bboxes, imgsz.cpu().numpy())
            # SHOW
            # for i in range(masks.shape[0]):
            #     cv2.imwrite(f"Gaussian_{i}.jpg", masks[i, 0].numpy())


        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # TODO: save topk bbox， uesd by MemoryAtten moudle
        if hasattr(self, 'buffer'):
            max_value, max_index = torch.max(pred_scores,dim=2)
            bbox_topk, score_topk = get_top_k_indices(max_value, pred_bboxes, self.topkbbox)
            # 归一化0-1
            # max_score = score_topk[:, 0].unsqueeze(-1)
            # min_score = score_topk[:, -1].unsqueeze(-1)
            # normalized_score_topk = (score_topk - min_score) / (max_score - min_score)
            # 方差归一化
            std_score_topk = torch.std(score_topk, dim=1, keepdim=True)
            mean_score_topk = torch.mean(score_topk, dim=1, keepdim=True)
            normalized_score_topk = (score_topk - mean_score_topk) / std_score_topk
            self.buffer.update_bbox(bbox_topk, normalized_score_topk)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor

            if self.hyp.loss_mode == "wasserstein":
                loss[0] = self.Wasserstein_loss(pred_bboxes[fg_mask],target_bboxes[fg_mask])
            else:
                loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                                target_scores_sum, fg_mask)               
                
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


# Criterion class for computing training losses
class v8SegmentationLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea)  # seg

                # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
                else:
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box / batch_size  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """Mask loss for one image."""
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


# Criterion class for computing training losses
class v8PoseLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                    idx = target_gt_idx[i][fg_mask[i]]
                    gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    area = xyxy2xywh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    pred_kpt = pred_kpts[i][fg_mask[i]]
                    kpt_mask = gt_kpt[..., 2] != 0
                    loss[1] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
                    # kpt_score loss
                    if pred_kpt.shape[-1] == 3:
                        loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose / batch_size  # pose gain
        loss[2] *= self.hyp.kobj / batch_size  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def kpts_decode(self, anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y


class v8ClassificationLoss:

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / 64
        loss_items = loss.detach()
        return loss, loss_items
