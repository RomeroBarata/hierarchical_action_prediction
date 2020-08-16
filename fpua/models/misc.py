import math
from bisect import bisect_left
from itertools import accumulate, dropwhile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fpua.data.general import aggregate_actions_and_lengths
from fpua.utils import normalise, one_hot_to_index, nan_to_value


class ActionLayer(nn.Module):
    def __init__(self, in_features, out_features, length_activation='linear', bias=True, ignore_action=False):
        super(ActionLayer, self).__init__()
        self.ignore_action = ignore_action
        self.length_activation = length_activation
        if self.ignore_action:
            self.register_parameter('fc_action', None)
        else:
            self.fc_action = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.fc_length = nn.Linear(in_features=in_features, out_features=1, bias=bias)

    def forward(self, x, mask=None):
        action_length = apply_len_activation(self.fc_length(x), activation=self.length_activation)
        if self.ignore_action:
            return None, action_length
        action_logits = self.fc_action(x)
        if mask is not None:
            # TODO: float('-inf') is the correct way, but doesn't work in practice along with teacher forcing.
            action_logits[mask] = float('-inf')
            # action_logits[mask] = -1e2
        action_logits = F.log_softmax(action_logits, dim=-1)
        return action_logits, action_length


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = (x > 0.5).float()
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = output_gradient * 1
        return grad_x


straight_through_estimator = StraightThroughEstimator.apply


def hard_sigmoid(x, a):
    return F.hardtanh((a * x + 1) / 2, min_val=0.0, max_val=1.0)


class MultiTaskLossLearner(nn.Module):
    def __init__(self, loss_types):
        super(MultiTaskLossLearner, self).__init__()
        self.loss_types = list(loss_types)

        self.log_sds = nn.Parameter(torch.zeros(len(self.loss_types), dtype=torch.float32))

    def forward(self, losses):
        assert len(self.loss_types) == len(losses), 'Specified loss types must match the number of input losses.'
        weighted_losses = []
        for loss_type, log_sd, loss in zip(self.loss_types, self.log_sds, losses):
            weighted_loss = self.compute_weighted_loss(loss, loss_type, log_sd)
            weighted_losses.append(weighted_loss)
        return weighted_losses

    def compute_weighted_loss(self, loss, loss_type, log_sd):
        if loss_type not in {'mae', 'mse', 'softmax'}:
            raise ValueError('loss_type must be one of \'softmax\', \'mae\' or \'mse\'.')
        loss_weight = self.compute_loss_weight(loss_type, log_sd)
        weighted_loss = loss_weight * loss + log_sd
        return weighted_loss

    @staticmethod
    def compute_loss_weight(loss_type, log_sd):
        if loss_type == 'mae':
            loss_weight = math.sqrt(2.0) * torch.exp(-log_sd)
        elif loss_type == 'mse':
            loss_weight = 0.5 * torch.exp(-2 * log_sd)
        else:
            loss_weight = torch.exp(-2 * log_sd)
        return loss_weight

    def get_weights(self):
        weights = [self.compute_loss_weight(loss_type, log_sd).item()
                   for loss_type, log_sd in zip(self.loss_types, self.log_sds)]
        return weights


def apply_len_activation(x, activation):
    if activation == 'relu':
        x = torch.relu(x)
    elif activation == 'elu':
        x = torch.nn.functional.elu(x) + 1
    elif activation == 'sigmoid':
        x = torch.sigmoid(x)
    return x


def compute_positional_encoding(x, out_dim, encoding='both'):
    assert (out_dim % 2) == 0, 'out_dim must be a multiple of two.'
    y = []
    for i in range(out_dim // 2):
        w = 1 / (1e4 ** (2 * i / out_dim))
        y.append(torch.sin(w * x))
        y.append(torch.cos(w * x))
    y = torch.cat(y, dim=-1)
    return y


def next_action_info(y_dec_logits, y_dec_len, id_to_action, num_frames, parent_la_prop=1.0):
    na_id = torch.argmax(y_dec_logits, dim=-1).item()
    na_label = id_to_action.get(na_id)
    na_len = round(y_dec_len.item() * parent_la_prop * num_frames)
    return na_label, na_len


def compute_steps_to_grab(predicted_fine_steps, num_frames_to_grab):
    lengths = [predicted_len if predicted_len is not None else 0
               for predicted_action, predicted_len in predicted_fine_steps]
    acc_lengths = list(accumulate(lengths))
    steps_to_grab = bisect_left(acc_lengths, num_frames_to_grab) + 1
    return steps_to_grab


def scale_num_part(x, scalers, scaler_name):
    scaler = scalers.get(scaler_name)
    if scaler is not None:
        x_cat, x_num = x[..., :-1], x[..., -1:]
        x_num, _ = normalise(x_num, scaler=scaler)
        x = np.concatenate([x_cat, x_num], axis=-1)
    return x


def compute_ground_truth_flushes(transition_coarse_action, transition_fine_action, unobserved_coarse_actions,
                                 unobserved_fine_actions):
    unobserved_fine_actions = list(dropwhile(lambda action: action == transition_fine_action, unobserved_fine_actions))
    unobserved_coarse_actions = unobserved_coarse_actions[-len(unobserved_fine_actions):]
    flushes = []
    if transition_coarse_action != unobserved_coarse_actions[0]:
        flushes.append(1.0)
    actions, _ = aggregate_actions_and_lengths(list(zip(unobserved_coarse_actions, unobserved_fine_actions)))
    for t, (coarse_action, _) in enumerate(actions[:-1]):
        flush = 1.0 if coarse_action != actions[t + 1][0] else 0.0
        flushes.append(flush)
    flushes.append(1.0)
    return flushes


def compute_scaler(x_cat, x_num, quantile_range, to_index=True):
    """x_cat is (batch_size, *, num_actions) and x_num is (batch_size, *, 1)."""
    if to_index:
        x_cat = one_hot_to_index(x_cat)
    x = np.concatenate([x_cat, x_num], axis=-1)
    x = nan_to_value(x, value=-1.0, inplace=False)
    x_unique = np.unique(x.reshape(-1, x.shape[-1]), axis=0)
    x_unique_num = x_unique[..., -1:]
    x_unique_num[x_unique_num == -1.0] = np.nan
    _, scaler = normalise(x_unique_num, strategy='robust', with_centering=False, quantile_range=quantile_range)
    return scaler


def scale_input(x, scalers, quantile_range, scaler_name, to_index=True):
    x_scaler = scalers.get(scaler_name)
    x_cat, x_num = x[..., :-1], x[..., -1:]
    if x_scaler is None:
        scalers[scaler_name] = compute_scaler(x_cat, x_num, quantile_range=quantile_range, to_index=to_index)
    x_num, _ = normalise(x_num, strategy='robust', with_centering=False,
                         quantile_range=quantile_range, scaler=scalers[scaler_name])
    x = np.concatenate([x_cat, x_num], axis=-1)
    return x, scalers


def scale_transition(x_tra, scalers, quantile_range, scaler_name):
    x_tra_cat, x_tra_num = x_tra[..., :-1], x_tra[..., -1:]
    x_tra_num, scalers[scaler_name] = normalise(x_tra_num, strategy='robust',
                                                with_centering=False, quantile_range=quantile_range,
                                                scaler=scalers.get(scaler_name))
    x_tra = np.concatenate([x_tra_cat, x_tra_num], axis=-1)
    return x_tra, scalers
