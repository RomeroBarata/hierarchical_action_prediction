import torch
import torch.nn.functional as F


def mse_loss(input, target, ignore_value=-1, reduction='mean'):
    """MSE loss for PyTorch tensors with optional value to be ignored."""
    mask = (target != ignore_value).float()
    num_nonmissing_elements = mask.sum().item()
    if num_nonmissing_elements == 0:
        return torch.tensor(0.0, dtype=input.dtype, device=input.device)
    criterion = F.mse_loss(input * mask, target * mask, reduction=reduction)
    if reduction == 'mean':
        criterion *= input.numel() / num_nonmissing_elements
    return criterion


def smooth_mae_loss(input, target, ignore_value=-1, reduction='mean'):
    """Smooth MAE loss for PyTorch tensors with optional value to be ignored."""
    mask = (target != ignore_value).float()
    num_nonmissing_elements = mask.sum().item()
    if num_nonmissing_elements == 0:
        return torch.tensor(0.0, dtype=input.dtype, device=input.device)
    criterion = F.smooth_l1_loss(input * mask, target * mask, reduction=reduction)
    if reduction == 'mean':
        criterion *= input.numel() / num_nonmissing_elements
    return criterion


def mae_loss(input, target, ignore_value=-1, reduction='mean'):
    """MAE loss for PyTorch tensors with optional value to be ignored."""
    mask = (target != ignore_value).float()
    num_nonmissing_elements = mask.sum().item()
    if num_nonmissing_elements == 0:
        return torch.tensor(0.0, dtype=input.dtype, device=input.device)
    criterion = F.l1_loss(input * mask, target * mask, reduction=reduction)
    if reduction == 'mean':
        criterion *= input.numel() / num_nonmissing_elements
    return criterion


def binary_cross_entropy_loss_with_logits(input, target, ignore_value=-1, reduction='mean'):
    """BCE with logits loss for PyTorch tensors with optional value to be ignored.

    For the targets to be ignored (target == ignore_value), we set these targets to zero and make the input at these
    positions a large negative value.
    """
    mask = (target != ignore_value).float()
    num_nonmissing_elements = mask.sum().item()
    if num_nonmissing_elements == 0:
        return torch.tensor(0.0, dtype=input.dtype, device=input.device)
    large_negative_tensor = (target == ignore_value).float() * -10000
    criterion = F.binary_cross_entropy_with_logits(input + large_negative_tensor, target * mask, reduction=reduction)
    criterion *= input.numel() / num_nonmissing_elements
    return criterion


def binary_cross_entropy_loss(input, target, positive_class_weight=1, ignore_value=-1, reduction='mean'):
    """BCE loss for PyTorch tensors with optional value to be ignored.

    For the targets to be ignored (target == ignore_value), we multiply these targets and the corresponding predictions
    by zero so that we have zero loss on these terms.
    """
    mask = (target != ignore_value).float()
    num_nonmissing_elements = mask.sum().item()
    if num_nonmissing_elements == 0:
        return torch.tensor(0.0, dtype=input.dtype, device=input.device)
    if positive_class_weight > 1:
        input = torch.where(target == 1.0, input ** positive_class_weight, input)
    criterion = F.binary_cross_entropy(input * mask, target * mask, reduction=reduction)
    criterion *= input.numel() / num_nonmissing_elements
    return criterion


def hera_loss(output, target, reduction='mean', loss_weights=None,
              disable_encoder_loss=False, coarse_class_weights=None,
              fine_class_weights=None, model_v2=False):
    (y_enc_fine_logits, y_enc_fine_lens, y_enc_coarse_logits, y_enc_coarse_lens,
     y_tra_fine_remlen, y_tra_coarse_remlen,
     y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens, dx_fine_steps) = output
    if loss_weights is None:
        loss_weights = [1.0] * (len(target) - 1)
    criteria = []
    if not disable_encoder_loss:
        criterion_a = F.nll_loss(y_enc_fine_logits.transpose(1, 2), target[0], weight=fine_class_weights,
                                 ignore_index=-1, reduction=reduction)
        criterion_b = mse_loss(y_enc_fine_lens, target[1], ignore_value=-1, reduction=reduction)
        criterion_c = F.nll_loss(y_enc_coarse_logits.transpose(1, 2), target[2], weight=coarse_class_weights,
                                 ignore_index=-1, reduction=reduction)
        criterion_d = mse_loss(y_enc_coarse_lens, target[3], ignore_value=-1, reduction=reduction)
        if len(loss_weights) == 2:
            criteria += [criterion_a * loss_weights[1], criterion_b * loss_weights[0],
                         criterion_c * loss_weights[1], criterion_d * loss_weights[0]]
        else:
            criteria += [criterion_a, criterion_b, criterion_c, criterion_d]
    criterion_e = mse_loss(y_tra_fine_remlen, target[4], ignore_value=-1, reduction=reduction)
    criterion_f = mse_loss(y_tra_coarse_remlen, target[5], ignore_value=-1, reduction=reduction)
    if model_v2:
        num_coarse_classes = y_enc_coarse_logits.size(-1)
        dtype, device = y_dec_fine_logits[0].dtype, y_dec_fine_logits[0].device
        criterion_g = torch.tensor(0.0, dtype=dtype, device=device)
        criterion_h = torch.tensor(0.0, dtype=dtype, device=device)
        criterion_i = torch.tensor(0.0, dtype=dtype, device=device)
        criterion_j = torch.tensor(0.0, dtype=dtype, device=device)
        batch_size = len(y_dec_fine_logits)
        effective_batch_size = 0
        for i in range(batch_size):
            # Fine
            criterion_g += F.nll_loss(y_dec_fine_logits[i].transpose(1, 2), target[6][i:i + 1],
                                      weight=fine_class_weights, ignore_index=-1, reduction=reduction)
            criterion_h += mse_loss(y_dec_fine_lens[i], target[7][i:i + 1], ignore_value=-1, reduction=reduction)
            # Coarse
            target_i = target[8][i:i + 1, target[-1][i, 0] == 1.0]
            if target_i.nelement():
                pred_i = match_target_shape(y_dec_coarse_logits[i], target_i, num_features=num_coarse_classes,
                                            dtype=dtype, device=device)
                criterion_i += F.nll_loss(pred_i.transpose(1, 2), target_i, weight=coarse_class_weights,
                                          ignore_index=-1, reduction=reduction)
                effective_batch_size += 1
            target_j = target[9][i:i + 1, target[-1][i, 0] == 1.0]
            if target_j.nelement():
                pred_j = match_target_shape(y_dec_coarse_lens[i], target_j, num_features=1, dtype=dtype, device=device)
                criterion_j += mse_loss(pred_j, target_j, ignore_value=-1, reduction=reduction)
        if not effective_batch_size:
            effective_batch_size = 1
        criterion_g /= batch_size
        criterion_h /= batch_size
        criterion_i /= effective_batch_size
        criterion_j /= effective_batch_size
    else:
        criterion_g = F.nll_loss(y_dec_fine_logits.transpose(1, 2), target[6], weight=fine_class_weights,
                                 ignore_index=-1, reduction=reduction)
        criterion_h = mse_loss(y_dec_fine_lens, target[7], ignore_value=-1, reduction=reduction)
        if torch.all(torch.eq(dx_fine_steps, target[-1][:, :1])).item():  # Training mode
            criterion_i = F.nll_loss(y_dec_coarse_logits.transpose(1, 2), target[8], weight=coarse_class_weights,
                                     ignore_index=-1, reduction=reduction)
            criterion_j = mse_loss(y_dec_coarse_lens, target[9], ignore_value=-1, reduction=reduction)
        else:  # Test mode
            num_coarse_classes = y_enc_coarse_logits.size(-1)
            dtype, device = y_dec_fine_logits[0].dtype, y_dec_fine_logits[0].device
            criterion_i = torch.tensor(0.0, dtype=dtype, device=device)
            criterion_j = torch.tensor(0.0, dtype=dtype, device=device)
            batch_size = len(y_dec_fine_logits)
            effective_batch_size = 0
            for i in range(batch_size):
                # Coarse
                target_i = target[8][i:i + 1, target[-1][i, 0] == 1.0]
                if target_i.nelement():
                    pred_i = y_dec_coarse_logits[i:i + 1, dx_fine_steps[i, 0] == 1.0]
                    pred_i = match_target_shape(pred_i, target_i, num_features=num_coarse_classes,
                                                dtype=dtype, device=device)
                    criterion_i += F.nll_loss(pred_i.transpose(1, 2), target_i, weight=coarse_class_weights,
                                              ignore_index=-1, reduction=reduction)
                    effective_batch_size += 1
                target_j = target[9][i:i + 1, target[-1][i, 0] == 1.0]
                if target_j.nelement():
                    pred_j = y_dec_coarse_lens[i:i + 1, dx_fine_steps[i, 0] == 1.0]
                    pred_j = match_target_shape(pred_j, target_j, num_features=1, dtype=dtype,
                                                device=device)
                    criterion_j += mse_loss(pred_j, target_j, ignore_value=-1, reduction=reduction)
            criterion_i /= effective_batch_size
            criterion_j /= effective_batch_size
    if len(loss_weights) == 2:
        criteria += [criterion_e * loss_weights[0], criterion_f * loss_weights[0],
                     criterion_g * loss_weights[1], criterion_h * loss_weights[0],
                     criterion_i * loss_weights[1], criterion_j * loss_weights[0]]
    else:
        criteria += [criterion_e, criterion_f,
                     criterion_g, criterion_h, criterion_i, criterion_j]
        num_criteria = len(criteria)
        criteria = [criterion * weight for criterion, weight in zip(criteria, loss_weights[-num_criteria:])]
    return criteria


def match_target_shape(pred, target, num_features, dtype, device):
    """Make pred compatible with target.

    pred: either a tensor of shape (1, k*, num_features) or an empty list
    target: a tensor of shape (1, k)
    num_features: in case pred is [], we need to know the number of features

    We need to make k* match k in a sensible way.
    """
    k = target.size(1)
    if not isinstance(pred, list):
        k_star = pred.size(1)
        if k_star < k:
            padding = torch.zeros([1, k - k_star, num_features], dtype=dtype, device=device)
            pred = torch.cat([pred, padding], dim=1)
        elif k_star > k:
            pred = pred[:, :k]
    else:
        pred = torch.zeros([1, k, num_features], dtype=dtype, device=device)
    return pred


def baseline_loss(output, target, baseline_type, action_level, reduction='mean'):
    y_logits_fine, y_lens_fine, y_logits_coarse = output
    criteria = []
    if baseline_type > 0:
        criterion_a = F.nll_loss(y_logits_fine.transpose(1, 2), target[0], ignore_index=-1, reduction=reduction)
        criterion_b = mse_loss(y_lens_fine, target[1], ignore_value=-1, reduction=reduction)
        criteria += [criterion_a, criterion_b]
        if y_logits_coarse is not None:
            criterion_c = F.nll_loss(y_logits_coarse.transpose(1, 2), target[2], ignore_index=-1, reduction=reduction)
            criteria.append(criterion_c)
    else:
        if action_level == 'coarse':
            target_nll, target_mse = target[2:]
        else:
            target_nll, target_mse = target[:2]
        criterion_a = F.nll_loss(y_logits_fine.transpose(1, 2), target_nll, ignore_index=-1, reduction=reduction)
        criterion_b = mse_loss(y_lens_fine, target_mse, ignore_value=-1, reduction=reduction)
        criteria += [criterion_a, criterion_b]
    return criteria
