import torch

def compute_focal_loss(pred, gt, alpha=0.25, gamma=2.0, eps=1e-8):
    """
    Compute the focal loss for a batch of predicted and ground truth heatmaps.

    Arguments:
    - pred: Tensor of predicted probabilities, shape [B, C, N]
    - gt: Tensor of ground truth values, shape [B, C, N]
    - alpha: Balancing factor, default 0.25
    - gamma: Focusing parameter, default 2.0
    - eps: Small constant to prevent division by zero

    Returns:
    - Focal loss value as a scalar.
    """
    pred = pred.clamp(min=eps, max=1-eps)

    p_t = gt * pred + (1 - gt) * (1 - pred)
    focal_loss = -alpha * (1 - p_t) ** gamma * torch.log(p_t)

    loss = focal_loss.mean(dim=[0, 1]).sum()

    return loss

def compute_cornernet_focal_loss(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def compute_diou_loss_1d(
    pred_segments: torch.Tensor,
    gt_segments: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    Reference code from
    https://pytorch.org/vision/main/_modules/torchvision/ops/diou_loss.html#distance_box_iou_loss

    Implemented for 1D segments.
    Args:
        pred/gt_segments (Tensor): 1D segments of size (B, N, 2)
        mask (Tensor): mask of size (B, N) replace all padding values with 0
        eps (float): small number to prevent division by zero
    """

    pred_lbound = pred_segments[..., 0]
    pred_rbound = pred_segments[..., 1]
    gt_lbound = gt_segments[..., 0]
    gt_rbound = gt_segments[..., 1]

    intsct = torch.min(pred_rbound, gt_rbound) - torch.max(pred_lbound, gt_lbound)
    union = torch.max(pred_rbound, gt_rbound) - torch.min(pred_lbound, gt_lbound)

    iou = intsct / (union + eps)

    # smallest enclosing box
    enc_lbound = torch.min(pred_lbound, gt_lbound)
    enc_rbound = torch.max(pred_rbound, gt_rbound)

    diagonal_distance_squared = (enc_rbound - enc_lbound) ** 2 + eps

    center_distance_squared = (pred_rbound + pred_lbound - gt_rbound - gt_lbound) ** 2 / 4

    loss = 1 - iou + center_distance_squared / diagonal_distance_squared
    loss[mask] = 0

    loss = loss.sum(dim=1).mean()

    return loss