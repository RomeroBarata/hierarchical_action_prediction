import collections.abc
from functools import partial
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from fpua.models.fetchers import multiple_input_multiple_output
from fpua.models.losses import hera_loss
from fpua.models.rnn import HMGRU, HMGRUV2, HMLSTM
from fpua.models.misc import MultiTaskLossLearner, ActionLayer, apply_len_activation, compute_positional_encoding
from fpua.models.misc import scale_input, scale_transition
from fpua.utils import numpy_to_torch, train, save_checkpoint, create_alias, nan_to_value, logit2one_hot
from fpua.utils import extract_info_from_str, num_workers_from_batch_size, normalise, set_initial_teacher_prob
from fpua.utils import grab_subset


class SingleLevelInputEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, embedding_nonlinearity='relu', partition=None,
                 positional_embedding=False, bias=True, cat_embeddings_to_share=None):
        super(SingleLevelInputEmbedding, self).__init__()

        self.embedding_nonlinearity = embedding_nonlinearity
        if partition is None:
            partition = 0, input_size - 1
        self.partition = partition
        self.num_features = input_size - partition[-1]
        if not isinstance(embedding_size, collections.abc.Sequence):
            embedding_size = [embedding_size] * len(partition)
        EmbeddingLayers = [nn.Linear if embedding_size_ else nn.Identity for embedding_size_ in embedding_size]

        if cat_embeddings_to_share is None:
            self.cat_embeddings = nn.ModuleList()
            for i, (p_a, p_b, emb_size) in enumerate(zip(partition[:-1], partition[1:], embedding_size)):
                self.cat_embeddings.append(EmbeddingLayers[i](p_b - p_a, emb_size, bias=bias))
        else:
            offset = len(cat_embeddings_to_share)
            cat_embeddings = nn.ModuleList()
            for i, (p_a, p_b, emb_size) in enumerate(zip(partition[offset:-1], partition[offset + 1:],
                                                         embedding_size[offset:]), start=offset):
                cat_embeddings.append(EmbeddingLayers[i](p_b - p_a, emb_size, bias=bias))
            self.cat_embeddings = nn.ModuleList(list(cat_embeddings_to_share) + list(cat_embeddings))
        if positional_embedding and embedding_size[-1]:
            self.num_embedding = partial(compute_positional_encoding, out_dim=embedding_size[-1], encoding='both')
        else:
            self.num_embedding = EmbeddingLayers[-1](self.num_features, embedding_size[-1], bias=bias)
        JointEmbeddingLayer, joint_embedding_input_size = self._get_joint_embedding_info(embedding_size)
        self.joint_embedding = JointEmbeddingLayer(joint_embedding_input_size, max(embedding_size), bias=bias)

    def forward(self, x):
        emb_cats = []
        for i, (p_a, p_b) in enumerate(zip(self.partition[:-1], self.partition[1:])):
            emb_cat = self.cat_embeddings[i](x[..., p_a:p_b])
            if isinstance(self.cat_embeddings[i], nn.Linear):
                if self.embedding_nonlinearity == 'relu':
                    emb_cat = F.relu(emb_cat)
                else:
                    emb_cat = torch.tanh(emb_cat)
            emb_cats.append(emb_cat)
        emb_num = self.num_embedding(x[..., -self.num_features:])
        if isinstance(self.num_embedding, nn.Linear):
            if self.embedding_nonlinearity == 'relu':
                emb_num = F.relu(emb_num)
            else:
                emb_num = torch.tanh(emb_num)
        emb_out = self.joint_embedding(torch.cat(emb_cats + [emb_num], dim=-1))
        if isinstance(self.joint_embedding, nn.Linear):
            if self.embedding_nonlinearity == 'relu':
                emb_out = F.relu(emb_out)
            else:
                emb_out = torch.tanh(emb_out)
        return emb_out

    def _get_joint_embedding_info(self, embedding_size: list):
        input_sizes = [p_b - p_a for p_a, p_b in zip(self.partition[:-1], self.partition[1:])] + [self.num_features]
        total_embedding_size = sum(embedding_size)
        if total_embedding_size:
            joint_input_sizes = [emb_size or input_size for input_size, emb_size in zip(input_sizes, embedding_size)]
            return nn.Linear, sum(joint_input_sizes)
        return nn.Identity, 0


class InputEmbedding(nn.Module):
    def __init__(self, input_sizes, embedding_sizes, embedding_nonlinearity='relu', partitions=None,
                 positional_embedding=False, bias=True, share_parent_embedding=False):
        super(InputEmbedding, self).__init__()
        if partitions is None:
            partitions = [None for _ in input_sizes]

        if share_parent_embedding:
            cannot_share_parent_embedding = False
            for parent_partition, child_partition in zip(partitions[-1:0:-1], partitions[-2::-1]):
                if len(parent_partition) == len(child_partition):
                    cannot_share_parent_embedding = True
                    break
            if cannot_share_parent_embedding:
                self.embeddings = nn.ModuleList(
                    [SingleLevelInputEmbedding(input_size, embedding_size, embedding_nonlinearity, partition,
                                               positional_embedding=positional_embedding, bias=bias)
                     for input_size, embedding_size, partition in zip(input_sizes, embedding_sizes, partitions)])
            else:
                self.embeddings, cat_embeddings_to_share = nn.ModuleList(), None
                for input_size, embedding_size, partition in zip(input_sizes[-1::-1],
                                                                 embedding_sizes[-1::-1],
                                                                 partitions[-1::-1]):
                    embedding = SingleLevelInputEmbedding(input_size, embedding_size, embedding_nonlinearity,
                                                          partition, positional_embedding=positional_embedding,
                                                          bias=bias, cat_embeddings_to_share=cat_embeddings_to_share)
                    cat_embeddings_to_share = embedding.cat_embeddings
                    self.embeddings.append(embedding)
                self.embeddings = self.embeddings[-1::-1]
        else:
            self.embeddings = nn.ModuleList([SingleLevelInputEmbedding(input_size, embedding_size,
                                                                       embedding_nonlinearity, partition,
                                                                       positional_embedding=positional_embedding,
                                                                       bias=bias)
                                             for input_size, embedding_size, partition in
                                             zip(input_sizes, embedding_sizes, partitions)])

    def forward(self, *xs):
        embeddings = [self.embeddings[i](x) for i, x in enumerate(xs)]
        return embeddings


class EncoderNet(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, num_actions, embedding_sizes, known_boundaries, partitions=None,
                 len_activation=('linear', 'linear'), embedding_nonlinearity='relu', positional_embedding=False,
                 add_skip_connection=False, weight_initialisation='pytorch', use_plain_gru_cell=False,
                 use_hmgruv2_cell=False, use_lstm_cell=False, do_not_reset_after_flush=False,
                 always_include_parent_state=False, with_final_action=False, bias=True, share_parent_embedding=False):
        super(EncoderNet, self).__init__()
        self.add_skip_connection = add_skip_connection
        self.use_plain_gru_cell = use_plain_gru_cell

        self.embedding_layer = InputEmbedding(input_sizes, embedding_sizes,
                                              embedding_nonlinearity=embedding_nonlinearity,
                                              partitions=partitions, positional_embedding=positional_embedding,
                                              bias=bias, share_parent_embedding=share_parent_embedding)
        input_sizes = [max(embedding_size) or input_size if embedding_size else input_size
                       for input_size, embedding_size in zip(input_sizes, embedding_sizes)]
        if use_plain_gru_cell:
            self.encoder_hmgru = nn.ModuleList()
            self.encoder_hmgru.append(nn.GRU(input_sizes[0], hidden_sizes[0], num_layers=1,
                                             bias=bias, batch_first=True))
            self.encoder_hmgru.append(nn.GRU(input_sizes[1] + hidden_sizes[0], hidden_sizes[1], num_layers=1,
                                             bias=bias, batch_first=True))
        else:
            if use_hmgruv2_cell:
                self.encoder_hmgru = HMGRUV2(input_sizes=input_sizes, hidden_sizes=hidden_sizes,
                                             reset_after_flush=not do_not_reset_after_flush,
                                             always_include_parent_state=always_include_parent_state, bias=bias)
            elif use_lstm_cell:
                self.encoder_hmgru = HMLSTM(input_sizes=input_sizes, hidden_sizes=hidden_sizes, bias=bias)
            else:
                self.encoder_hmgru = HMGRU(input_sizes=input_sizes, hidden_size=hidden_sizes,
                                           known_boundaries=known_boundaries,
                                           weight_initialisation=weight_initialisation, bias=bias)
        num_fine_actions, num_coarse_actions = num_actions
        skip_sizes = input_sizes if add_skip_connection else [0, 0]
        extra_action = 1 if with_final_action else 0
        self.fine_action_layer = ActionLayer(hidden_sizes[0] + skip_sizes[0], num_fine_actions + extra_action,
                                             length_activation=len_activation[0], bias=bias)
        self.coarse_action_layer = ActionLayer(hidden_sizes[1] + skip_sizes[1], num_coarse_actions + extra_action,
                                               length_activation=len_activation[1], bias=bias)

    def forward(self, x_fine, x_coarse, x_fine_mask=None, x_coarse_mask=None, dx=None, hx=None,
                return_all_hidden_states=False, disable_gradient_from_child=False):
        x_fine, x_coarse = self.embedding_layer(x_fine, x_coarse)
        if self.use_plain_gru_cell:
            out_fine, hx_fine = self.encoder_hmgru[0](x_fine, hx=hx)
            out_coarse, hx_coarse = self.encoder_hmgru[1](torch.cat([x_coarse, out_fine], dim=-1), hx=hx)
            out, hx = [out_fine, out_coarse], [hx_fine[0], hx_coarse[0]]
        else:
            out, hx, _, _ = self.encoder_hmgru([x_fine, x_coarse], hx=hx, dx=dx[0], dx_layer_zero=dx[1],
                                               disable_gradient_from_child=disable_gradient_from_child)
        if self.add_skip_connection:
            y_enc_fine_logits, y_enc_fine_lens = self.fine_action_layer(torch.cat([out[0], x_fine], dim=-1),
                                                                        mask=x_fine_mask)
            y_enc_coarse_logits, y_enc_coarse_lens = self.coarse_action_layer(torch.cat([out[1], x_coarse], dim=-1),
                                                                              mask=x_coarse_mask)
        else:
            y_enc_fine_logits, y_enc_fine_lens = self.fine_action_layer(out[0], mask=x_fine_mask)
            y_enc_coarse_logits, y_enc_coarse_lens = self.coarse_action_layer(out[1], mask=x_coarse_mask)
        y_tensors = [y_enc_fine_logits, y_enc_fine_lens, y_enc_coarse_logits, y_enc_coarse_lens]
        if return_all_hidden_states:
            return y_tensors, hx, out
        else:
            return y_tensors, hx


class RefresherNet(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, embedding_sizes, partitions=None, len_activation=('linear', 'linear'),
                 embedding_nonlinearity='relu', add_skip_connection=False, bias=True, embeddings_to_share=None):
        super(RefresherNet, self).__init__()
        self.len_activation = len_activation
        self.add_skip_connection = add_skip_connection

        self.embedding_layer = InputEmbedding(input_sizes, embedding_sizes,
                                              embedding_nonlinearity=embedding_nonlinearity,
                                              partitions=partitions, bias=bias)
        if embeddings_to_share is not None:
            self.update_embedding_layer(embeddings_to_share)
        input_sizes = [max(embedding_size) or input_size if embedding_size else input_size
                       for input_size, embedding_size in zip(input_sizes, embedding_sizes)]
        fine_input_size, coarse_input_size = input_sizes
        hidden_size_fine, hidden_size_coarse = hidden_sizes
        skip_sizes = input_sizes if add_skip_connection else [0, 0]
        self.fine_transition_layer = nn.Linear(fine_input_size + hidden_size_fine + hidden_size_coarse,
                                               hidden_size_fine, bias=bias)
        self.fine_action_len_layer = nn.Linear(hidden_size_fine + skip_sizes[0], 1, bias=bias)
        self.coarse_transition_layer = nn.Linear(coarse_input_size + hidden_size_coarse,
                                                 hidden_size_coarse, bias=bias)
        self.coarse_action_len_layer = nn.Linear(hidden_size_coarse + skip_sizes[1], 1, bias=bias)

    def forward(self, x_fine, x_coarse, hx, disable_gradient_from_child=False):
        x_fine, x_coarse = self.embedding_layer(x_fine, x_coarse)

        hx_coarse = torch.tanh(self.coarse_transition_layer(torch.cat([x_coarse, hx[1]], dim=-1)))
        if self.add_skip_connection:
            y_tra_coarse_rem_len = self.coarse_action_len_layer(torch.cat([hx_coarse, x_coarse], dim=-1))
        else:
            y_tra_coarse_rem_len = self.coarse_action_len_layer(hx_coarse)
        y_tra_coarse_rem_len = apply_len_activation(y_tra_coarse_rem_len, activation=self.len_activation[1])

        if disable_gradient_from_child:
            hx_fine = torch.tanh(self.fine_transition_layer(torch.cat([x_fine, hx[0], hx_coarse.detach()], dim=-1)))
        else:
            hx_fine = torch.tanh(self.fine_transition_layer(torch.cat([x_fine, hx[0], hx_coarse], dim=-1)))
        if self.add_skip_connection:
            y_tra_fine_rem_len = self.fine_action_len_layer(torch.cat([hx_fine, x_fine], dim=-1))
        else:
            y_tra_fine_rem_len = self.fine_action_len_layer(hx_fine)
        y_tra_fine_rem_len = apply_len_activation(y_tra_fine_rem_len, activation=self.len_activation[0])

        hx = [hx_fine, hx_coarse]
        y_tensors = [y_tra_fine_rem_len, y_tra_coarse_rem_len]
        return y_tensors, hx

    def update_embedding_layer(self, embeddings_to_share):
        for i, embedding in enumerate(embeddings_to_share.embeddings):
            self.embedding_layer.embeddings[i].cat_embeddings = embedding.cat_embeddings


class DecoderNet(nn.Module):
    def __init__(self, input_sizes, output_seq_len, num_actions, hidden_sizes, embedding_sizes,
                 known_boundaries, partitions=None, len_activation=('linear', 'linear'), embedding_nonlinearity='relu',
                 positional_embedding=False, add_skip_connection=False, weight_initialisation='pytorch',
                 use_plain_gru_cell=False, use_hmgruv2_cell=False, use_lstm_cell=False, do_not_reset_after_flush=False,
                 always_include_parent_state=False, with_final_action=False, bias=True, parameters_to_share=None):
        super(DecoderNet, self).__init__()
        self.output_seq_len = output_seq_len
        self.len_activation = len_activation
        self.add_skip_connection = add_skip_connection
        self.use_plain_gru_cell = use_plain_gru_cell
        self.with_final_action = with_final_action
        hidden_size_fine, hidden_size_coarse = hidden_sizes
        num_fine_actions, num_coarse_actions = num_actions

        if parameters_to_share is not None and 'embeddings' in parameters_to_share:
            self.embedding_layer = parameters_to_share['embeddings']
        else:
            self.embedding_layer = InputEmbedding(input_sizes, embedding_sizes,
                                                  embedding_nonlinearity=embedding_nonlinearity,
                                                  partitions=partitions, positional_embedding=positional_embedding,
                                                  bias=bias)
        input_sizes = [max(embedding_size) or input_size if embedding_size else input_size
                       for input_size, embedding_size in zip(input_sizes, embedding_sizes)]
        if parameters_to_share is not None and 'encoder_decoder' in parameters_to_share:
            self.decoder_hmgru = parameters_to_share['encoder_decoder']
        else:
            if use_plain_gru_cell:
                self.decoder_hmgru = nn.ModuleList()
                self.decoder_hmgru.append(nn.GRU(input_sizes[0], hidden_sizes[0], num_layers=1, bias=bias,
                                                 batch_first=True))
                self.decoder_hmgru.append(nn.GRU(input_sizes[1] + hidden_sizes[0], hidden_sizes[1], num_layers=1,
                                                 bias=bias, batch_first=True))
            else:
                if use_hmgruv2_cell:
                    self.decoder_hmgru = HMGRUV2(input_sizes=input_sizes, hidden_sizes=hidden_sizes,
                                                 reset_after_flush=not do_not_reset_after_flush,
                                                 always_include_parent_state=always_include_parent_state, bias=bias)
                elif use_lstm_cell:
                    self.decoder_hmgru = HMLSTM(input_sizes=input_sizes, hidden_sizes=hidden_sizes, bias=bias)
                else:
                    self.decoder_hmgru = HMGRU(input_sizes=input_sizes, hidden_size=hidden_sizes,
                                               known_boundaries=known_boundaries,
                                               weight_initialisation=weight_initialisation, bias=bias)
        skip_sizes = input_sizes if add_skip_connection else [0, 0]
        extra_action = 1 if with_final_action else 0
        if parameters_to_share is not None and 'predictions' in parameters_to_share:
            fine_action_layer, coarse_action_layer = parameters_to_share['predictions']
            self.fine_action_layer, self.coarse_action_layer = fine_action_layer, coarse_action_layer
        else:
            self.fine_action_layer = ActionLayer(hidden_size_fine + skip_sizes[0],
                                                 num_fine_actions + extra_action,
                                                 length_activation=len_activation[0], bias=bias)
            self.coarse_action_layer = ActionLayer(hidden_size_coarse + skip_sizes[1],
                                                   num_coarse_actions + extra_action,
                                                   length_activation=len_activation[1], bias=bias)

    def forward(self, x_fine, x_coarse, x_fine_mask=None, x_coarse_mask=None, dx=None, hx=None, teacher_prob=None,
                scaling_values=None, x_enc_coarse=None, x_tra_fine=None, x_tra_coarse=None,
                y_tra_fine_rem_prop=None, y_tra_coarse_rem_prop=None, disable_gradient_from_child=False):
        if teacher_prob is None:
            x_fine, x_coarse = self.embedding_layer(x_fine, x_coarse)
            if self.use_plain_gru_cell:
                out_fine, _ = self.decoder_hmgru[0](x_fine, hx[0].unsqueeze(0))
                out_coarse, _ = self.decoder_hmgru[1](torch.cat([x_coarse, out_fine], dim=-1), hx[1].unsqueeze(0))
                dec_out = [out_fine, out_coarse]
            else:
                dec_out, _, _, _ = self.decoder_hmgru([x_fine, x_coarse], hx=hx, dx=dx,
                                                      disable_gradient_from_child=disable_gradient_from_child)
            if self.add_skip_connection:
                y_dec_fine_logits, y_dec_fine_lens = self.fine_action_layer(torch.cat([dec_out[0], x_fine], dim=-1),
                                                                            mask=x_fine_mask)
                y_dec_coarse_logits, y_dec_coarse_lens = self.coarse_action_layer(torch.cat([dec_out[1], x_coarse],
                                                                                            dim=-1),
                                                                                  mask=x_coarse_mask)
            else:
                y_dec_fine_logits, y_dec_fine_lens = self.fine_action_layer(dec_out[0], mask=x_fine_mask)
                y_dec_coarse_logits, y_dec_coarse_lens = self.coarse_action_layer(dec_out[1], mask=x_coarse_mask)
            output = [y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens, dx[..., :1]]
        else:
            scaling_values = scaling_values if scaling_values is not None else {}
            sc_y_tra_fine = scaling_values.get('y_tra_fine_scaler', 1.0)
            sc_y_tra_coarse = scaling_values.get('y_tra_coarse_scaler', 1.0)
            sc_y_dec_fine = scaling_values.get('y_dec_fine_scaler', 1.0)
            sc_y_dec_coarse = scaling_values.get('y_dec_coarse_scaler', 1.0)
            y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens = [], [], [], []

            fine_la_rem_prop = y_tra_fine_rem_prop.detach() * sc_y_tra_fine
            y_fine_acc_len = x_tra_fine[..., -2:-1].detach() + fine_la_rem_prop  # clamp?
            coarse_la_rem_prop = y_tra_coarse_rem_prop.detach() * sc_y_tra_coarse
            y_coarse_acc_len = x_tra_coarse[..., -2:-1].detach() + coarse_la_rem_prop
            dx_fine_step = 0.0
            dx_fine_steps = []
            children_input_contain_parent = len(self.embedding_layer.embeddings[0].cat_embeddings) > 1
            mask_softmax = x_fine_mask is not None
            for t in range(self.output_seq_len):
                use_ground_truth = random.random() < teacher_prob
                if use_ground_truth:
                    # Fine
                    x_gt_fine_step = x_fine[:, t]
                    dx_gt_fine_step = dx[..., 0, t - 1:t] if t else 0.0
                    x_gt_fine_mask_step = x_fine_mask[:, t] if mask_softmax else None
                    y_fine_step_logits, y_fine_step_lens, hx[0] = \
                        self.single_step_fine(x_gt_fine_step, dx_gt_fine_step,
                                              hx=hx,
                                              x_fine_mask=x_gt_fine_mask_step,
                                              disable_gradient_from_child=disable_gradient_from_child)
                    y_dec_fine_logits.append(y_fine_step_logits)
                    y_dec_fine_lens.append(y_fine_step_lens)
                    # Coarse
                    x_gt_coarse_step = x_coarse[:, t]
                    dx_gt_fine_step = dx[..., 0, t:t + 1]
                    x_gt_coarse_mask_step = x_coarse_mask[:, t] if mask_softmax else None
                    y_coarse_step_logits, y_coarse_step_lens, hx[1] = \
                        self.single_step_coarse(x_gt_coarse_step,
                                                dx_gt_fine_step,
                                                hx=hx,
                                                x_coarse_mask=x_gt_coarse_mask_step)
                    y_dec_coarse_logits.append(y_coarse_step_logits)
                    y_dec_coarse_lens.append(y_coarse_step_lens)
                    # Update rolling predictions for correct teacher forcing behaviour
                    fine_na_rel_prop = y_dec_fine_lens[-1].detach() * sc_y_dec_fine
                    dx_fine_step = (y_fine_acc_len >= 1.0).type_as(x_gt_fine_step)
                    y_fine_acc_len = (y_fine_acc_len + fine_na_rel_prop) * (1 - dx_fine_step)
                    coarse_na_prop = y_dec_coarse_lens[-1].detach() * sc_y_dec_coarse
                    y_coarse_acc_len = y_coarse_acc_len + coarse_na_prop * dx_fine_step
                else:
                    if t == 0:
                        x_fine_cat_step = x_tra_fine[..., :-2].detach()
                        x_coarse_cat_step = x_tra_coarse[..., :-2].detach()
                    else:
                        # Fine
                        x_fine_cat_step = logit2one_hot(y_dec_fine_logits[-1].detach()) * (1 - dx_fine_step)
                        if self.with_final_action:
                            x_fine_cat_step = x_fine_cat_step[..., :-1]
                        if children_input_contain_parent:
                            x_fine_cat_parent_step = logit2one_hot(y_dec_coarse_logits[-1].detach()) * dx_fine_step
                            if self.with_final_action:
                                x_fine_cat_parent_step = x_fine_cat_parent_step[..., :-1]
                            x_fine_cat_parent_step = x_fine_cat_parent_step + x_coarse_cat_step * (1 - dx_fine_step)
                            x_fine_cat_step = torch.cat([x_fine_cat_parent_step, x_fine_cat_step], dim=-1)
                        # Coarse
                        x_coarse_cat_step = logit2one_hot(y_dec_coarse_logits[-1].detach()) * dx_fine_step
                        if self.with_final_action:
                            x_coarse_cat_step = x_coarse_cat_step[..., :-1]
                        x_coarse_cat_step = x_coarse_cat_step + x_coarse_cat_step * (1 - dx_fine_step)
                    x_fine_step = torch.cat([x_fine_cat_step, y_fine_acc_len], dim=-1)
                    y_fine_step_logits, y_fine_step_lens, hx[0] = \
                        self.single_step_fine(x_fine_step, dx_fine_step,
                                              hx=hx,
                                              x_fine_mask=None,
                                              disable_gradient_from_child=disable_gradient_from_child)
                    y_dec_fine_logits.append(y_fine_step_logits)
                    y_dec_fine_lens.append(y_fine_step_lens)
                    fine_na_rel_prop = y_dec_fine_lens[-1].detach() * sc_y_dec_fine
                    dx_fine_step = (y_fine_acc_len >= 1.0).type_as(x_fine_step)
                    y_fine_acc_len = (y_fine_acc_len + fine_na_rel_prop) * (1 - dx_fine_step)
                    x_coarse_step = torch.cat([x_coarse_cat_step, y_coarse_acc_len], dim=-1)
                    y_coarse_step_logits, y_coarse_step_lens, hx[1] = \
                        self.single_step_coarse(x_coarse_step,
                                                dx_fine_step,
                                                hx=hx,
                                                x_coarse_mask=None)
                    y_dec_coarse_logits.append(y_coarse_step_logits)
                    y_dec_coarse_lens.append(y_coarse_step_lens)
                    coarse_na_prop = y_dec_coarse_lens[-1].detach() * sc_y_dec_coarse
                    y_coarse_acc_len = y_coarse_acc_len + coarse_na_prop * dx_fine_step
                if teacher_prob == 1.0:
                    dx_fine_steps.append(dx[:, :1, t])
                else:
                    dx_fine_steps.append(dx_fine_step)
            output = [y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens]
            output = [torch.stack(out, dim=1) for out in output]
            dx_fine_steps = torch.stack(dx_fine_steps, dim=2)
            output.append(dx_fine_steps)
        return output

    def single_step_fine(self, x_fine, dx, hx, x_fine_mask=None, disable_gradient_from_child=False):
        x_fine = self.embedding_layer.embeddings[0](x_fine)
        if self.use_plain_gru_cell:
            _, hx_fine = self.decoder_hmgru[0](x_fine.unsqueeze(1), hx=hx[0].unsqueeze(0))
            hx_fine = hx_fine.squeeze()
        else:
            h_top = hx[1][0] if isinstance(self.decoder_hmgru, HMLSTM) else hx[1]
            if disable_gradient_from_child:
                h_top = h_top.detach()
            hx_fine, _, _ = self.decoder_hmgru.cells[0](x_fine, h_bottom=None, h=hx[0], h_top=h_top, d_bottom=1, d=dx)
        if self.add_skip_connection:
            y_dec_fine_logits, y_dec_fine_lens = self.fine_action_layer(torch.cat([hx_fine, x_fine], dim=-1),
                                                                        mask=x_fine_mask)
        else:
            hx_fine_action = hx_fine[0] if isinstance(self.decoder_hmgru, HMLSTM) else hx_fine
            y_dec_fine_logits, y_dec_fine_lens = self.fine_action_layer(hx_fine_action, mask=x_fine_mask)
        return y_dec_fine_logits, y_dec_fine_lens, hx_fine

    def single_step_coarse(self, x_coarse, dx_fine, hx, x_coarse_mask=None):
        x_coarse = self.embedding_layer.embeddings[1](x_coarse)
        if self.use_plain_gru_cell:
            _, hx_coarse = self.decoder_hmgru[1](torch.cat([x_coarse, hx[0]], dim=-1).unsqueeze(1),
                                                 hx=hx[1].unsqueeze(0))
            hx_coarse = hx_coarse.squeeze()
        else:
            h_bottom = hx[0][0] if isinstance(self.decoder_hmgru, HMLSTM) else hx[0]
            hx_coarse, _, _ = self.decoder_hmgru.cells[1](x_coarse, h_bottom=h_bottom, h=hx[1], h_top=None,
                                                          d_bottom=dx_fine, d=0)
        if self.add_skip_connection:
            y_dec_coarse_logits, y_dec_coarse_lens = self.coarse_action_layer(torch.cat([hx_coarse, x_coarse], dim=-1),
                                                                              mask=x_coarse_mask)
        else:
            hx_coarse_action = hx_coarse[0] if isinstance(self.decoder_hmgru, HMLSTM) else hx_coarse
            y_dec_coarse_logits, y_dec_coarse_lens = self.coarse_action_layer(hx_coarse_action, mask=x_coarse_mask)
        return y_dec_coarse_logits, y_dec_coarse_lens, hx_coarse


class DecoderNetV2(nn.Module):
    def __init__(self, input_sizes, output_seq_len, num_actions, hidden_sizes, embedding_sizes, partitions=None,
                 len_activation=('linear', 'linear'), embedding_nonlinearity='relu', do_not_reset_after_flush=False,
                 always_include_parent_state=False, with_final_action=False, bias=True):
        super(DecoderNetV2, self).__init__()
        self.reset_after_flush = not do_not_reset_after_flush
        self.always_include_parent_state = always_include_parent_state
        self.with_final_action = with_final_action
        self.output_seq_len = output_seq_len
        input_sizes = list(input_sizes)

        self.embedding_layer = InputEmbedding(input_sizes, embedding_sizes,
                                              embedding_nonlinearity=embedding_nonlinearity,
                                              partitions=partitions, positional_embedding=False,
                                              bias=bias)

        input_sizes = [max(embedding_size) or input_size if embedding_size else input_size
                       for input_size, embedding_size in zip(input_sizes, embedding_sizes)]
        self.decoder_hmgru = nn.ModuleList()
        self.decoder_hmgru.append(nn.GRU(input_sizes[0] + hidden_sizes[1], hidden_sizes[0], num_layers=1, bias=bias,
                                         batch_first=True))
        self.decoder_hmgru.append(nn.GRU(input_sizes[1] + hidden_sizes[0], hidden_sizes[1], num_layers=1,
                                         bias=bias, batch_first=True))

        extra_action = 1 if with_final_action else 0
        self.fine_action_layer = ActionLayer(hidden_sizes[0], num_actions[0] + extra_action,
                                             length_activation=len_activation[0], bias=bias)
        self.coarse_action_layer = ActionLayer(hidden_sizes[1], num_actions[1] + extra_action,
                                               length_activation=len_activation[1], bias=bias)

    def forward(self, x_fine, x_coarse, dx, hx, teacher_prob=None, x_fine_initial_step=None,
                x_coarse_initial_step=None):
        if teacher_prob is None:
            pass
        else:
            y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens = [], [], [], []
            batch_size, batch_output = x_fine.size(0), []
            for batch_example in range(batch_size):
                batch_example_hx = [hx[0][batch_example], hx[1][batch_example]]
                batch_example_output = \
                    self.process_batch_example(x_fine[batch_example], x_coarse[batch_example],
                                               dx[batch_example], batch_example_hx,
                                               teacher_prob=teacher_prob,
                                               x_fine_initial_step=x_fine_initial_step[batch_example],
                                               x_coarse_initial_step=x_coarse_initial_step[batch_example])
                y_fine_logits, y_fine_lens, y_coarse_logits, y_coarse_lens = batch_example_output
                y_dec_fine_logits.append(y_fine_logits)
                y_dec_fine_lens.append(y_fine_lens)
                y_dec_coarse_logits.append(y_coarse_logits)
                y_dec_coarse_lens.append(y_coarse_lens)
        output = [y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens]
        return output

    def process_batch_example(self, x_fine, x_coarse, dx, hx, teacher_prob, x_fine_initial_step, x_coarse_initial_step):
        """Process a single batch example.

        x_fine: tensor of shape (output_seq_len, fine_input_size)
        x_coarse: tensor of shape (output_seq_len, coarse_input_size)
        dx: tensor of shape (2, output_seq_len)
        hx: 2-element list with tensors of shape (fine_hidden_size,) and (coarse_hidden_size,)
        teacher_prob: [0.0, 1.0] float
        x_fine_initial_step: tensor of shape (fine_input_size,)
        x_coarse_initial_step: tensor of shape (coarse_input_size,)

        output:
        y_dec_fine_logits: list of length output_seq_len, where each element is a tensor of shape (num_fine_actions,)
        y_dec_fine_lens: list of length output_seq_len, where each element is a tensor of shape (1,)
        y_dec_coarse_logits: list of variable length, where each element is a tensor of shape (num_coarse_actions,)
        y_dec_coarse_lens: list of variable length, where each element is a tensor of shape (1,)
        """
        children_input_contain_parent = len(self.embedding_layer.embeddings[0].cat_embeddings) > 1
        x_fine_step, x_coarse_step = x_fine_initial_step, x_coarse_initial_step
        dx_fine_step = 0.0
        y_fine_acc_len = x_fine_step[-1:].detach().clone()
        y_coarse_acc_len = x_coarse_step[-1:].detach().clone()
        y_dec_fine_logits, y_dec_fine_lens = [], []
        y_dec_coarse_logits, y_dec_coarse_lens = [], []
        for t in range(self.output_seq_len):
            use_ground_truth = random.random() < teacher_prob
            if use_ground_truth:
                # Fine
                x_gt_fine_step = x_fine[t]
                dx_gt_fine_step = dx[0, t - 1] if t else 0.0
                y_fine_step_logits, y_fine_step_lens, hx[0] = \
                    self.single_step_fine(x_gt_fine_step, dx_gt_fine_step, hx=hx)
                y_dec_fine_logits.append(y_fine_step_logits)
                y_dec_fine_lens.append(y_fine_step_lens)
                # Update rolling predictions for correct teacher forcing behaviour
                fine_na_rel_prop = y_dec_fine_lens[-1].detach()[0]
                dx_fine_step = (y_fine_acc_len >= 1.0).type_as(x_gt_fine_step)
                y_fine_acc_len = (y_fine_acc_len + fine_na_rel_prop) * (1 - dx_fine_step)
                # Coarse
                dx_gt_fine_step = dx[0, t]
                if dx_gt_fine_step == 0:
                    continue
                x_gt_coarse_step = x_coarse[t]
                y_coarse_step_logits, y_coarse_step_lens, hx[1] = \
                    self.single_step_coarse(x_gt_coarse_step, hx=hx)
                y_dec_coarse_logits.append(y_coarse_step_logits)
                y_dec_coarse_lens.append(y_coarse_step_lens)
                # Update rolling predictions for correct teacher forcing behaviour
                coarse_na_prop = y_dec_coarse_lens[-1].detach()[0]
                y_coarse_acc_len = y_coarse_acc_len + coarse_na_prop
            else:
                # Fine
                y_fine_step_logits, y_fine_step_lens, hx[0] = \
                    self.single_step_fine(x_fine_step, dx_fine_step, hx=hx)
                y_dec_fine_logits.append(y_fine_step_logits)
                y_dec_fine_lens.append(y_fine_step_lens)
                fine_na_rel_prop = y_dec_fine_lens[-1].detach()[0]
                dx_fine_step = (y_fine_acc_len >= 1.0).type_as(x_fine_step)
                y_fine_acc_len = (y_fine_acc_len + fine_na_rel_prop) * (1 - dx_fine_step)
                x_fine_cat_step = (logit2one_hot(y_dec_fine_logits[-1].detach()) * (1 - dx_fine_step))[0]
                if self.with_final_action:
                    x_fine_cat_step = x_fine_cat_step[:-1]
                x_fine_step = torch.cat([x_fine_cat_step, y_fine_acc_len], dim=-1)
                if children_input_contain_parent:
                    if y_dec_coarse_logits:
                        x_coarse_cat_step = logit2one_hot(y_dec_coarse_logits[-1].detach())[0]
                        if self.with_final_action:
                            x_coarse_cat_step = x_coarse_cat_step[:-1]
                    else:
                        x_coarse_cat_step = x_coarse_step[:-1].detach().clone()
                    x_fine_step = torch.cat([x_coarse_cat_step, x_fine_step], dim=-1)
                # Coarse
                if dx_fine_step == 0:
                    continue
                y_coarse_step_logits, y_coarse_step_lens, hx[1] = self.single_step_coarse(x_coarse_step, hx=hx)
                y_dec_coarse_logits.append(y_coarse_step_logits)
                y_dec_coarse_lens.append(y_coarse_step_lens)
                coarse_na_prop = y_dec_coarse_lens[-1].detach()[0]
                y_coarse_acc_len = y_coarse_acc_len + coarse_na_prop
                x_coarse_cat_step = logit2one_hot(y_dec_coarse_logits[-1].detach())[0]
                if self.with_final_action:
                    x_coarse_cat_step = x_coarse_cat_step[:-1]
                x_coarse_step = torch.cat([x_coarse_cat_step, y_coarse_acc_len], dim=-1)
        output = [y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens]
        output = [torch.stack(out, dim=1) if out else out for out in output]
        return output

    def single_step_fine(self, x_fine_step, dx_fine_step, hx):
        """Single step processing for fine layer.

        x_fine_step: tensor of shape (fine_input_size,)
        dx_fine_step: 0.0 or 1.0 float
        hx: 2-element list with tensors of shape (fine_hidden_size,) and (coarse_hidden_size,)
        """
        x_fine = self.embedding_layer.embeddings[0](x_fine_step.unsqueeze(0))
        if self.always_include_parent_state or dx_fine_step:
            x_fine = torch.cat([x_fine, hx[1].unsqueeze(0)], dim=-1)
        else:
            x_fine = torch.cat([x_fine, torch.zeros_like(hx[1].unsqueeze(0))], dim=-1)
        if self.reset_after_flush and dx_fine_step:
            hx_fine = torch.zeros_like(hx[0][None, None])
        else:
            hx_fine = hx[0][None, None]
        _, hx_fine = self.decoder_hmgru[0](x_fine.unsqueeze(1), hx=hx_fine)
        hx_fine = hx_fine.squeeze()
        y_dec_fine_logits, y_dec_fine_lens = self.fine_action_layer(hx_fine.unsqueeze(0))
        return y_dec_fine_logits, y_dec_fine_lens, hx_fine

    def single_step_coarse(self, x_coarse_step, hx):
        """Single step processing for coarse layer.

        x_coarse_step: tensor of shape (coarse_input_size,)
        hx: 2-element list with tensors of shape (fine_hidden_size,) and (coarse_hidden_size,)
        """
        x_coarse = self.embedding_layer.embeddings[1](x_coarse_step.unsqueeze(0))
        x_coarse = torch.cat([x_coarse, hx[0].unsqueeze(0)], dim=-1)
        _, hx_coarse = self.decoder_hmgru[1](x_coarse.unsqueeze(1), hx=hx[1][None, None])
        hx_coarse = hx_coarse.squeeze()
        y_dec_coarse_logits, y_dec_coarse_lens = self.coarse_action_layer(hx_coarse.unsqueeze(0))
        return y_dec_coarse_logits, y_dec_coarse_lens, hx_coarse


class AnticipatorNet(nn.Module):
    def __init__(self, input_sizes, output_seq_len, num_actions, hidden_sizes, embedding_sizes,
                 partitions=None, len_activation=('linear', 'linear'), embedding_nonlinearity='relu',
                 positional_embedding=False, do_not_reset_after_flush=False, always_include_parent_state=False,
                 with_final_action=False, input_soft_parent=False, bias=True, parameters_to_share=None):
        super(AnticipatorNet, self).__init__()
        self.output_seq_len = output_seq_len
        self.len_activation = len_activation
        self.with_final_action = with_final_action
        self.input_soft_parent = input_soft_parent
        hidden_size_fine, hidden_size_coarse = hidden_sizes
        num_fine_actions, num_coarse_actions = num_actions

        if parameters_to_share is not None and 'embeddings' in parameters_to_share:
            self.embedding_layer = parameters_to_share['embeddings']
        else:
            self.embedding_layer = InputEmbedding(input_sizes, embedding_sizes,
                                                  embedding_nonlinearity=embedding_nonlinearity,
                                                  partitions=partitions, positional_embedding=positional_embedding,
                                                  bias=bias)
        input_sizes = [max(embedding_size) or input_size if embedding_size else input_size
                       for input_size, embedding_size in zip(input_sizes, embedding_sizes)]
        if parameters_to_share is not None and 'encoder_decoder' in parameters_to_share:
            self.decoder_hmgru = parameters_to_share['encoder_decoder']
        else:
            self.decoder_hmgru = HMGRUV2(input_sizes=input_sizes, hidden_sizes=hidden_sizes,
                                         reset_after_flush=not do_not_reset_after_flush,
                                         always_include_parent_state=always_include_parent_state, bias=bias)
        extra_action = 1 if with_final_action else 0
        if parameters_to_share is not None and 'predictions' in parameters_to_share:
            fine_action_layer, coarse_action_layer = parameters_to_share['predictions']
            self.fine_action_layer, self.coarse_action_layer = fine_action_layer, coarse_action_layer
        else:
            self.fine_action_layer = ActionLayer(hidden_size_fine,
                                                 num_fine_actions + extra_action,
                                                 length_activation=len_activation[0], bias=bias)
            self.coarse_action_layer = ActionLayer(hidden_size_coarse,
                                                   num_coarse_actions + extra_action,
                                                   length_activation=len_activation[1], bias=bias)

    def forward(self, x_fine, x_coarse, dx=None, hx=None, teacher_prob=None, x_fine_step=None, x_coarse_step=None,
                disable_gradient_from_child=False):
        if teacher_prob is None:
            x_fine, x_coarse = self.embedding_layer(x_fine, x_coarse)
            dec_out, _, _, _ = self.decoder_hmgru([x_fine, x_coarse], hx=hx, dx=dx,
                                                  disable_gradient_from_child=disable_gradient_from_child)
            y_dec_fine_logits, y_dec_fine_lens = self.fine_action_layer(dec_out[0])
            y_dec_coarse_logits, y_dec_coarse_lens = self.coarse_action_layer(dec_out[1])
            output = [y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens, dx[..., :1]]
        else:
            y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens = [], [], [], []
            dx_fine_steps = []
            dx_fine_step = 0.0
            y_fine_acc_len = x_fine_step[..., -1:].detach().clone()
            y_coarse_acc_len = x_coarse_step[..., -1:].detach().clone()
            children_input_contain_parent = len(self.embedding_layer.embeddings[0].cat_embeddings) > 1
            for t in range(self.output_seq_len):
                use_ground_truth = random.random() < teacher_prob
                if use_ground_truth:
                    # Fine
                    x_gt_fine_step = x_fine[:, t]
                    dx_gt_fine_step = dx[..., 0, t - 1:t] if t > 0 else 0.0
                    y_fine_step_logits, y_fine_step_lens, hx[0] = \
                        self.single_step_fine(x_gt_fine_step, dx_gt_fine_step, hx=hx,
                                              disable_gradient_from_child=disable_gradient_from_child)
                    y_dec_fine_logits.append(y_fine_step_logits)
                    y_dec_fine_lens.append(y_fine_step_lens)
                    # Coarse
                    x_gt_coarse_step = x_coarse[:, t]
                    dx_gt_fine_step = dx[..., 0, t:t + 1]
                    y_coarse_step_logits, y_coarse_step_lens, hx[1] = \
                        self.single_step_coarse(x_gt_coarse_step, dx_gt_fine_step, hx=hx)
                    y_dec_coarse_logits.append(y_coarse_step_logits)
                    y_dec_coarse_lens.append(y_coarse_step_lens)
                    # Update rolling predictions for correct teacher forcing behaviour
                    dx_fine_step = y_fine_acc_len >= 1.0
                    if self.with_final_action:
                        dx_fine_step_ = logit2one_hot(y_dec_fine_logits[-1].detach().clone())[..., -1:] == 1.0
                        dx_fine_step = dx_fine_step | dx_fine_step_
                    dx_fine_step = dx_fine_step.type_as(x_fine_step)
                    fine_na_rel_prop = y_dec_fine_lens[-1].detach().clone()
                    y_fine_acc_len = (y_fine_acc_len + fine_na_rel_prop) * (1 - dx_fine_step)
                    coarse_na_prop = y_dec_coarse_lens[-1].detach().clone()
                    y_coarse_acc_len = y_coarse_acc_len + coarse_na_prop * dx_fine_step
                else:
                    # Fine
                    y_fine_step_logits, y_fine_step_lens, hx[0] = \
                        self.single_step_fine(x_fine_step, dx_fine_step, hx=hx,
                                              disable_gradient_from_child=disable_gradient_from_child)
                    y_dec_fine_logits.append(y_fine_step_logits)
                    y_dec_fine_lens.append(y_fine_step_lens)
                    dx_fine_step = y_fine_acc_len >= 1.0
                    if self.with_final_action:
                        dx_fine_step_ = logit2one_hot(y_dec_fine_logits[-1].detach().clone())[..., -1:] == 1.0
                        dx_fine_step = dx_fine_step | dx_fine_step_
                    dx_fine_step = dx_fine_step.type_as(x_fine_step)
                    # Coarse
                    y_coarse_step_logits, y_coarse_step_lens, hx[1] = \
                        self.single_step_coarse(x_coarse_step, dx_fine_step, hx=hx)
                    y_dec_coarse_logits.append(y_coarse_step_logits)
                    y_dec_coarse_lens.append(y_coarse_step_lens)
                    # Update coarse next step
                    coarse_na_prop = y_dec_coarse_lens[-1].detach().clone()
                    y_coarse_acc_len = y_coarse_acc_len + coarse_na_prop * dx_fine_step
                    x_coarse_cat_step = logit2one_hot(y_dec_coarse_logits[-1].detach().clone()) * dx_fine_step
                    if self.with_final_action:
                        x_coarse_cat_step = x_coarse_cat_step[..., :-1]
                    x_coarse_cat_step = x_coarse_cat_step + (x_coarse_step[..., :-1].detach()) * (1 - dx_fine_step)
                    x_coarse_step = torch.cat([x_coarse_cat_step, y_coarse_acc_len], dim=-1)
                    # Update fine next step
                    fine_na_rel_prop = y_dec_fine_lens[-1].detach().clone()
                    y_fine_acc_len = (y_fine_acc_len + fine_na_rel_prop) * (1 - dx_fine_step)
                    x_fine_cat_step = logit2one_hot(y_dec_fine_logits[-1].detach().clone()) * (1 - dx_fine_step)
                    if self.with_final_action:
                        x_fine_cat_step = x_fine_cat_step[..., :-1]
                    x_fine_step = torch.cat([x_fine_cat_step, y_fine_acc_len], dim=-1)
                    if children_input_contain_parent:
                        if self.input_soft_parent:
                            if self.with_final_action:
                                x_coarse_cat_step = torch.softmax(y_dec_coarse_logits[-1][..., :-1].detach().clone(),
                                                                  dim=-1)
                            else:
                                x_coarse_cat_step = torch.softmax(y_dec_coarse_logits[-1].detach().clone(), dim=-1)
                            x_coarse_cat_step = (x_coarse_cat_step * dx_fine_step +
                                                 (x_coarse_step[..., :-1].detach()) * (1 - dx_fine_step))
                        x_fine_step = torch.cat([x_coarse_cat_step.clone(), x_fine_step], dim=-1)
                if teacher_prob == 1.0:
                    dx_fine_steps.append(dx[:, :1, t])
                else:
                    dx_fine_steps.append(dx_fine_step)
            output = [y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens]
            output = [torch.stack(out, dim=1) for out in output]
            dx_fine_steps = torch.stack(dx_fine_steps, dim=2)
            output.append(dx_fine_steps)
        return output

    def single_step_fine(self, x_fine, dx, hx, x_fine_mask=None, disable_gradient_from_child=False):
        x_fine = self.embedding_layer.embeddings[0](x_fine)
        h_top = hx[1]
        if disable_gradient_from_child:
            h_top = h_top.detach()
        hx_fine, _, _ = self.decoder_hmgru.cells[0](x_fine, h_bottom=None, h=hx[0], h_top=h_top, d_bottom=1, d=dx)
        y_dec_fine_logits, y_dec_fine_lens = self.fine_action_layer(hx_fine)
        return y_dec_fine_logits, y_dec_fine_lens, hx_fine

    def single_step_coarse(self, x_coarse, dx_fine, hx, x_coarse_mask=None):
        x_coarse = self.embedding_layer.embeddings[1](x_coarse)
        h_bottom = hx[0]
        hx_coarse, _, _ = self.decoder_hmgru.cells[1](x_coarse, h_bottom=h_bottom, h=hx[1], h_top=None,
                                                      d_bottom=dx_fine, d=0)
        y_dec_coarse_logits, y_dec_coarse_lens = self.coarse_action_layer(hx_coarse)
        return y_dec_coarse_logits, y_dec_coarse_lens, hx_coarse


class HERA(nn.Module):
    def __init__(self, input_sizes, transition_input_sizes, output_seq_len, hidden_sizes, num_actions,
                 embedding_sizes=(0, 0), len_activation=('linear', 'linear'), embedding_nonlinearity='relu',
                 partitions=None, share_embeddings=False, share_encoder_decoder=False, share_predictions=False,
                 mask_softmax=False, positional_embedding=False, add_skip_connection=False,
                 weight_initialisation='pytorch', use_plain_gru_cell=False, use_hmgruv2_cell=True,
                 disable_transition_layer=False, disable_gradient_from_child=False, use_lstm_cell=False,
                 model_v2=False, model_v3=True, do_not_reset_after_flush=False, always_include_parent_state=True,
                 with_final_action=False, input_soft_parent=False, bias=True):
        super(HERA, self).__init__()
        self.mask_softmax = mask_softmax
        self.disable_transition_layer = disable_transition_layer
        self.disable_gradient_from_child = disable_gradient_from_child
        self.model_v2 = model_v2
        self.model_v3 = model_v3
        self.with_final_action = with_final_action

        self.encoder_net = EncoderNet(input_sizes=input_sizes, hidden_sizes=hidden_sizes, num_actions=num_actions,
                                      embedding_sizes=embedding_sizes, known_boundaries=[True, True],
                                      partitions=partitions, len_activation=len_activation,
                                      embedding_nonlinearity=embedding_nonlinearity,
                                      positional_embedding=positional_embedding,
                                      add_skip_connection=add_skip_connection,
                                      weight_initialisation=weight_initialisation,
                                      use_plain_gru_cell=use_plain_gru_cell, use_hmgruv2_cell=use_hmgruv2_cell,
                                      use_lstm_cell=use_lstm_cell, do_not_reset_after_flush=do_not_reset_after_flush,
                                      always_include_parent_state=always_include_parent_state,
                                      with_final_action=with_final_action, bias=bias, share_parent_embedding=False)
        embeddings_to_share = self.encoder_net.embedding_layer if share_embeddings else None
        self.transition_net = RefresherNet(input_sizes=transition_input_sizes, hidden_sizes=hidden_sizes,
                                           embedding_sizes=embedding_sizes, partitions=partitions,
                                           len_activation=len_activation,
                                           embedding_nonlinearity=embedding_nonlinearity,
                                           add_skip_connection=add_skip_connection, bias=bias,
                                           embeddings_to_share=embeddings_to_share)
        parameters_to_share = {}
        if share_embeddings:
            parameters_to_share['embeddings'] = self.encoder_net.embedding_layer
        if share_encoder_decoder:
            parameters_to_share['encoder_decoder'] = self.encoder_net.encoder_hmgru
        if share_predictions:
            predictions_to_share = self.encoder_net.fine_action_layer, self.encoder_net.coarse_action_layer
            parameters_to_share['predictions'] = predictions_to_share
        if not parameters_to_share:
            parameters_to_share = None
        if model_v2:
            self.decoder_net = DecoderNetV2(input_sizes=input_sizes, output_seq_len=output_seq_len,
                                            num_actions=num_actions, hidden_sizes=hidden_sizes,
                                            embedding_sizes=embedding_sizes, partitions=partitions,
                                            len_activation=len_activation,
                                            embedding_nonlinearity=embedding_nonlinearity,
                                            do_not_reset_after_flush=do_not_reset_after_flush,
                                            always_include_parent_state=always_include_parent_state,
                                            with_final_action=with_final_action, bias=bias)
        elif model_v3:
            self.decoder_net = AnticipatorNet(input_sizes=input_sizes, output_seq_len=output_seq_len,
                                              num_actions=num_actions, hidden_sizes=hidden_sizes,
                                              embedding_sizes=embedding_sizes, partitions=partitions,
                                              len_activation=len_activation,
                                              embedding_nonlinearity=embedding_nonlinearity,
                                              positional_embedding=positional_embedding,
                                              do_not_reset_after_flush=do_not_reset_after_flush,
                                              always_include_parent_state=always_include_parent_state,
                                              with_final_action=with_final_action, input_soft_parent=input_soft_parent,
                                              bias=bias, parameters_to_share=parameters_to_share)
        else:
            self.decoder_net = DecoderNet(input_sizes=input_sizes,
                                          output_seq_len=output_seq_len,
                                          num_actions=num_actions,
                                          hidden_sizes=hidden_sizes, embedding_sizes=embedding_sizes,
                                          known_boundaries=[True, True], partitions=partitions,
                                          len_activation=len_activation, embedding_nonlinearity=embedding_nonlinearity,
                                          positional_embedding=positional_embedding,
                                          add_skip_connection=add_skip_connection,
                                          weight_initialisation=weight_initialisation,
                                          use_plain_gru_cell=use_plain_gru_cell, use_hmgruv2_cell=use_hmgruv2_cell,
                                          use_lstm_cell=use_lstm_cell,
                                          do_not_reset_after_flush=do_not_reset_after_flush,
                                          always_include_parent_state=always_include_parent_state,
                                          with_final_action=with_final_action, bias=bias,
                                          parameters_to_share=parameters_to_share)

    def forward(self, x_enc, x_tra, x_dec, dx_enc, dx_dec, hx=None, teacher_prob=None, scaling_values=None):
        x_enc_fine, x_enc_fine_mask, x_enc_coarse, x_enc_coarse_mask = x_enc
        if not self.mask_softmax:
            x_enc_fine_mask = x_enc_coarse_mask = None
        y_tensors, hx = self.encoder_net(x_enc_fine, x_enc_coarse, x_fine_mask=x_enc_fine_mask,
                                         x_coarse_mask=x_enc_coarse_mask, dx=dx_enc, hx=hx,
                                         disable_gradient_from_child=self.disable_gradient_from_child)
        y_enc_fine_logits, y_enc_fine_lens, y_enc_coarse_logits, y_enc_coarse_lens = y_tensors

        x_tra_fine, x_tra_coarse = x_tra
        hx_tra = [hl[0] for hl in hx] if isinstance(self.encoder_net.encoder_hmgru, HMLSTM) else hx
        y_tensors, hx_tra = self.transition_net(x_tra_fine, x_tra_coarse, hx=hx_tra,
                                                disable_gradient_from_child=self.disable_gradient_from_child)
        if not self.disable_transition_layer:
            if isinstance(self.encoder_net.encoder_hmgru, HMLSTM):
                for i, hl in enumerate(hx_tra):
                    hx[i][0] = hl
            else:
                hx = hx_tra
        y_tra_fine_rem_len, y_tra_coarse_rem_len = y_tensors

        x_dec_fine, x_dec_fine_mask, x_dec_coarse, x_dec_coarse_mask = x_dec
        if not self.mask_softmax:
            x_dec_fine_mask = x_dec_coarse_mask = None
        x_fine_initial_step = x_tra_fine[..., :-1].detach().clone()
        x_fine_initial_step[..., -1:] = x_fine_initial_step[..., -1:] + y_tra_fine_rem_len.detach().clone()
        x_coarse_initial_step = x_tra_coarse[..., :-1].detach().clone()
        x_coarse_initial_step[..., -1:] = x_coarse_initial_step[..., -1:] + y_tra_coarse_rem_len.detach().clone()
        if self.model_v2:
            y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens, *dx_fine_steps = \
                self.decoder_net(x_dec_fine, x_dec_coarse, dx=dx_dec, hx=hx, teacher_prob=teacher_prob,
                                 x_fine_initial_step=x_fine_initial_step, x_coarse_initial_step=x_coarse_initial_step)
        elif self.model_v3:
            y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens, dx_fine_steps = \
                self.decoder_net(x_dec_fine, x_dec_coarse, dx=dx_dec, hx=hx, teacher_prob=teacher_prob,
                                 x_fine_step=x_fine_initial_step, x_coarse_step=x_coarse_initial_step,
                                 disable_gradient_from_child=self.disable_gradient_from_child)
        else:
            y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens, dx_fine_steps = \
                self.decoder_net(x_dec_fine, x_dec_coarse, x_fine_mask=x_dec_fine_mask, x_coarse_mask=x_dec_coarse_mask,
                                 dx=dx_dec, hx=hx, teacher_prob=teacher_prob,
                                 scaling_values=scaling_values, x_enc_coarse=x_enc_coarse,
                                 x_tra_fine=x_tra_fine, x_tra_coarse=x_tra_coarse,
                                 y_tra_fine_rem_prop=y_tra_fine_rem_len, y_tra_coarse_rem_prop=y_tra_coarse_rem_len,
                                 disable_gradient_from_child=self.disable_gradient_from_child)

        output = [y_enc_fine_logits, y_enc_fine_lens, y_enc_coarse_logits, y_enc_coarse_lens,
                  y_tra_fine_rem_len, y_tra_coarse_rem_len,
                  y_dec_fine_logits, y_dec_fine_lens, y_dec_coarse_logits, y_dec_coarse_lens, dx_fine_steps]
        return output


def train_hera(args):
    training_data = args.training_data
    validation_data = args.validation_data
    input_seq_len = args.input_seq_len
    output_seq_len = args.output_seq_len
    hidden_size_fine = args.hidden_size_fine
    hidden_size_coarse = args.hidden_size_coarse
    hidden_sizes = hidden_size_fine, hidden_size_coarse
    fine_embedding_size = args.fine_embedding_size
    coarse_embedding_size = args.coarse_embedding_size
    embedding_sizes = fine_embedding_size, coarse_embedding_size
    embedding_nonlinearity = args.embedding_nonlinearity
    epochs = args.epochs
    learning_rate = args.learning_rate
    transition_learning_rate = args.transition_learning_rate
    batch_size = args.batch_size
    length_activation = args.length_activation
    multi_task_loss_learner = args.multi_task_loss_learner
    loss_weights = args.loss_weights if not multi_task_loss_learner else None
    print_raw_losses = args.print_raw_losses
    normalise_input = args.normalise_input
    normalise_output = args.normalise_output
    quantile_range = 0.0, args.quantile_upper_bound
    teacher_schedule = args.teacher_schedule if args.teacher_schedule != 'None' else None
    teacher_prob = set_initial_teacher_prob(teacher_schedule) if teacher_schedule is not None else None
    disable_parent_input = args.disable_parent_input
    input_soft_parent = args.input_soft_parent
    share_embeddings = args.share_embeddings
    share_encoder_decoder = args.share_encoder_decoder
    share_predictions = args.share_predictions
    disable_encoder_loss = args.disable_encoder_loss
    positional_embedding = args.positional_embedding
    mask_softmax = args.mask_softmax
    add_skip_connection = args.add_skip_connection
    weight_decay = args.weight_decay
    weight_decay_decoder_only = args.weight_decay_decoder_only
    disable_transition_layer = args.disable_transition_layer
    weight_initialisation = args.weight_initialisation
    clip_gradient_at = args.clip_gradient_at
    disable_gradient_from_child = args.disable_gradient_from_child
    pretrain_coarse = args.pretrain_coarse
    do_not_reset_after_flush = args.do_not_reset_after_flush
    always_include_parent_state = args.always_include_parent_state
    with_final_action = '_withfinalaction' in training_data
    train_on_subset_percentage = args.train_on_subset_percentage
    log_dir = args.log_dir
    # Some legacy options
    use_plain_gru_cell = use_lstm_cell = model_v2 = False
    use_hmgruv2_cell = model_v3 = True
    # Data
    num_workers = num_workers_from_batch_size(batch_size)
    with np.load(training_data) as data:
        tensors, scalers, _ = assemble_tensors(data, input_seq_len, output_seq_len,
                                               normalise_input=normalise_input,
                                               normalise_output=normalise_output, quantile_range=quantile_range,
                                               disable_parent_input=disable_parent_input,
                                               with_final_action=with_final_action)
    if train_on_subset_percentage is not None:
        n = round(tensors[0].shape[0] * train_on_subset_percentage / 100)
        tensors = grab_subset(*tensors, n=n)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensors = numpy_to_torch(*tensors)
    dataset = TensorDataset(*tensors)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=False, drop_last=False)
    val_loader = None
    if validation_data is not None:
        with np.load(validation_data) as data:
            val_tensors, _, _ = assemble_tensors(data, input_seq_len, output_seq_len, normalise_input=normalise_input,
                                                 normalise_output=normalise_output, quantile_range=quantile_range,
                                                 scalers=scalers, disable_parent_input=disable_parent_input,
                                                 with_final_action=with_final_action)
        val_tensors = numpy_to_torch(*val_tensors)
        val_dataset = TensorDataset(*val_tensors)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=False, drop_last=False)
    # Model
    input_sizes = tensors[0].size(-1), tensors[2].size(-1)
    transition_input_sizes = tensors[6].size(-1), tensors[7].size(-1)
    num_coarse_actions = input_sizes[1] - 1
    if disable_parent_input:
        num_fine_actions = input_sizes[0] - 1
        partitions = [(0, num_fine_actions),
                      (0, num_coarse_actions)]
    else:
        num_fine_actions = input_sizes[0] - 1 - num_coarse_actions
        partitions = [(0, num_coarse_actions, num_coarse_actions + num_fine_actions),
                      (0, num_coarse_actions)]
    enforce_embedding_size_consistency(fine_embedding_size, partitions[0], layer_name='fine')
    enforce_embedding_size_consistency(coarse_embedding_size, partitions[1], layer_name='coarse')
    num_actions = num_fine_actions, num_coarse_actions
    model_creation_args = {'input_sizes': input_sizes, 'transition_input_sizes': transition_input_sizes,
                           'output_seq_len': output_seq_len, 'hidden_sizes': hidden_sizes,
                           'num_actions': num_actions, 'embedding_sizes': embedding_sizes,
                           'len_activation': length_activation, 'embedding_nonlinearity': embedding_nonlinearity,
                           'partitions': partitions, 'share_embeddings': share_embeddings,
                           'share_encoder_decoder': share_encoder_decoder,
                           'share_predictions': share_predictions, 'mask_softmax': mask_softmax,
                           'positional_embedding': positional_embedding, 'add_skip_connection': add_skip_connection,
                           'weight_initialisation': weight_initialisation, 'use_plain_gru_cell': use_plain_gru_cell,
                           'use_hmgruv2_cell': use_hmgruv2_cell, 'disable_transition_layer': disable_transition_layer,
                           'disable_gradient_from_child': disable_gradient_from_child, 'use_lstm_cell': use_lstm_cell,
                           'model_v2': model_v2, 'model_v3': model_v3,
                           'do_not_reset_after_flush': do_not_reset_after_flush,
                           'always_include_parent_state': always_include_parent_state,
                           'with_final_action': with_final_action, 'input_soft_parent': input_soft_parent,
                           'bias': True}
    model = HERA(**model_creation_args).to(device)

    if pretrain_coarse:
        params = add_coarse_only_params(model)
    else:
        params = [{'params': model.encoder_net.parameters()}]
        add_remaining_params(params, model, transition_learning_rate, share_embeddings, share_encoder_decoder,
                             share_predictions, weight_decay, weight_decay_decoder_only)
    if weight_decay_decoder_only:
        optimizer = torch.optim.Adam(params, lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    if multi_task_loss_learner and not pretrain_coarse:
        loss_types = ['softmax', 'mse', 'softmax', 'mse', 'mse', 'mse', 'softmax', 'mse', 'softmax', 'mse']
        if disable_encoder_loss:
            loss_types = loss_types[4:]
        mtll_model = MultiTaskLossLearner(loss_types=loss_types).to(device)
        optimizer.add_param_group({'params': mtll_model.parameters()})
    else:
        mtll_model = None
    criterion = partial(hera_loss, loss_weights=loss_weights,
                        disable_encoder_loss=disable_encoder_loss, model_v2=model_v2)
    loss_names = ['NLL E F', 'MSE E F', 'NLL E C', 'MSE E C',
                  'MSE T F', 'MSE T C',
                  'NLL D F', 'MSE D F', 'NLL D C', 'MSE D C'
                  ]
    if disable_encoder_loss:
        loss_names = loss_names[4:]
    fetch_model_data = partial(multiple_input_multiple_output, n=13)

    _, nc, _, obs_at_least_k_percent = extract_info_from_str(training_data)
    input_normalisation = 'robust' if normalise_input else None
    output_normalisation = 'robust' if normalise_output else None
    checkpoint_name = create_alias(hidden_size=hidden_sizes, epochs=epochs, batch_size=batch_size,
                                   input_seq_len=input_seq_len, output_seq_len=output_seq_len,
                                   length_activation=length_activation, learning_rate=learning_rate,
                                   transition_learning_rate=transition_learning_rate, nc=nc,
                                   embedding_size=embedding_sizes, teacher_schedule=teacher_schedule,
                                   validation_data=validation_data, multi_task_loss_learner=multi_task_loss_learner,
                                   num_layers=len(input_sizes), normalisation=output_normalisation,
                                   quantile_range=quantile_range, input_normalisation=input_normalisation,
                                   obs_at_least_k_percent=obs_at_least_k_percent,
                                   share_encoder_decoder=share_encoder_decoder, share_embeddings=share_embeddings,
                                   share_predictions=share_predictions, disable_parent_input=disable_parent_input,
                                   disable_encoder_loss=disable_encoder_loss,
                                   embedding_nonlinearity=embedding_nonlinearity,
                                   mask_softmax=mask_softmax, positional_embedding=positional_embedding,
                                   add_skip_connection=add_skip_connection, l2_reg=weight_decay,
                                   weight_initialisation=weight_initialisation, clip_gradient_at=clip_gradient_at,
                                   use_plain_gru_cell=use_plain_gru_cell, use_hmgruv2_cell=use_hmgruv2_cell,
                                   loss_weights=loss_weights, disable_transition_layer=disable_transition_layer,
                                   disable_gradient_from_child=disable_gradient_from_child,
                                   use_lstm_cell=use_lstm_cell, weight_decay_decoder_only=weight_decay_decoder_only,
                                   pretrain_coarse=pretrain_coarse, model_v2=model_v2, model_v3=model_v3,
                                   do_not_reset_after_flush=do_not_reset_after_flush,
                                   always_include_parent_state=always_include_parent_state,
                                   with_final_action=with_final_action, input_soft_parent=input_soft_parent)
    scaling_values = {scaler_name: scaler.scale_.item() for scaler_name, scaler in scalers.items()}
    tensorboard_log_dir = os.path.join(log_dir, 'runs', checkpoint_name) if log_dir is not None else None
    checkpoint = train(model, train_loader, optimizer, criterion, pretrain_coarse or epochs, device,
                       clip_gradient_at=clip_gradient_at,
                       fetch_model_data=fetch_model_data, feed_model_data=feed_model_data, loss_names=loss_names,
                       val_loader=val_loader, mtll_model=mtll_model, print_raw_losses=print_raw_losses,
                       num_main_losses=6, tensorboard_log_dir=tensorboard_log_dir, teacher_schedule=teacher_schedule,
                       teacher_prob=teacher_prob, scaling_values=scaling_values)
    if pretrain_coarse:
        params = [{'params': model.encoder_net.parameters()}]
        add_remaining_params(params, model, transition_learning_rate, share_embeddings, share_encoder_decoder,
                             share_predictions, weight_decay, weight_decay_decoder_only)
        if weight_decay_decoder_only:
            optimizer = torch.optim.Adam(params, lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        if multi_task_loss_learner:
            loss_types = ['softmax', 'mse', 'softmax', 'mse', 'mse', 'mse', 'softmax', 'mse', 'softmax', 'mse']
            if disable_encoder_loss:
                loss_types = loss_types[4:]
            mtll_model = MultiTaskLossLearner(loss_types=loss_types).to(device)
            optimizer.add_param_group({'params': mtll_model.parameters()})
        initial_epoch = 1 + pretrain_coarse
        checkpoint = train(model, train_loader, optimizer, criterion, epochs, device, clip_gradient_at=clip_gradient_at,
                           fetch_model_data=fetch_model_data, feed_model_data=feed_model_data, loss_names=loss_names,
                           val_loader=val_loader, initial_epoch=initial_epoch, mtll_model=mtll_model,
                           print_raw_losses=print_raw_losses, num_main_losses=6,
                           tensorboard_log_dir=tensorboard_log_dir, teacher_schedule=teacher_schedule,
                           teacher_prob=teacher_prob, scaling_values=scaling_values)

    print('HERA model successfully trained.')
    if log_dir is not None:
        checkpoint['input_seq_len'] = input_seq_len
        checkpoint['scalers'] = scalers
        checkpoint['disable_parent_input'] = disable_parent_input
        checkpoint['disable_encoder_loss'] = disable_encoder_loss
        checkpoint['model_creation_args'] = model_creation_args
        save_checkpoint(log_dir, checkpoint, checkpoint_name=checkpoint_name, include_timestamp=False)


def enforce_embedding_size_consistency(embedding_size, partition, layer_name):
    if isinstance(embedding_size, collections.abc.Sequence) and len(embedding_size) != len(partition):
        required_embeddings = len(partition)
        if len(embedding_size) < required_embeddings:
            print(f'Only {len(embedding_size)} embedding sizes specified for the {layer_name} layer, '
                  f'but {required_embeddings} required. Unspecified layers now have the '
                  f'same size as the last specified embedding.')
            embedding_size += [embedding_size[-1]] * (required_embeddings - len(embedding_size))
        else:
            print(f'There were {len(embedding_size)} embedding sizes specified for the {layer_name} layer, but '
                  f'only {required_embeddings} required. Ignoring the extra sizes specified.')
            embedding_size[:] = embedding_size[:required_embeddings]


def assemble_tensors(data, input_seq_len, output_seq_len, normalise_input=True,
                     normalise_output=True, quantile_range=(0.0, 95.0), scalers=None, disable_parent_input=False,
                     with_final_action=False):
    scalers = scalers if scalers is not None else {}
    # ENCODER
    x_enc_coarse = data['x_enc_coarse'][:, -input_seq_len:]  # batch_size, input_seq_len, num_coarse_actions + 1
    x_enc_coarse = nan_to_value(x_enc_coarse, value=0.0)
    x_enc_coarse_mask = nan_to_value(data['x_enc_coarse'][:, -input_seq_len:, :-1], value=0.0).astype(np.bool)

    num_coarse_actions = x_enc_coarse.shape[-1] - 1
    # batch_size, input_seq_len, num_coarse_actions + num_fine_actions + 1
    x_enc_fine = data['x_enc_fine'][:, -input_seq_len:]
    if disable_parent_input:
        x_enc_fine = x_enc_fine[..., num_coarse_actions:]
    x_enc_fine = nan_to_value(x_enc_fine, value=0.0)
    x_enc_fine_mask = nan_to_value(data['x_enc_fine'][:, -input_seq_len:, num_coarse_actions:-1], value=0.0)
    x_enc_fine_mask = x_enc_fine_mask.astype(np.bool)

    y_enc_coarse = data['y_enc_coarse'][:, -input_seq_len:]  # batch_size, input_seq_len, 1 + 1
    if normalise_output:
        y_enc_coarse, scalers = scale_input(y_enc_coarse, scalers, quantile_range,
                                            scaler_name='y_enc_coarse_scaler', to_index=False)
    y_enc_coarse_cat, y_enc_coarse_num = pull_parts(y_enc_coarse, convert_nans=True)

    y_enc_fine = data['y_enc_fine'][:, -input_seq_len:]  # batch_size, input_seq_len, 1 + 1
    if normalise_output:
        y_enc_fine, scalers = scale_input(y_enc_fine, scalers, quantile_range,
                                          scaler_name='y_enc_fine_scaler', to_index=False)
    y_enc_fine_cat, y_enc_fine_num = pull_parts(y_enc_fine, convert_nans=True)

    enc_bottom_layer_boundary = data['enc_boundary'][:, -input_seq_len:]  # batch_size, input_seq_len
    enc_bottom_layer_boundary = nan_to_value(enc_bottom_layer_boundary, value=0.0)
    enc_upper_layer_boundary = np.zeros_like(enc_bottom_layer_boundary)
    enc_boundary = np.stack([enc_bottom_layer_boundary, enc_upper_layer_boundary], axis=1)
    enc_zero_layer_boundary = compute_zero_layer_boundary(data['x_enc_fine'][:, -input_seq_len:])

    # TRANSITION
    x_tra_coarse = data['x_tra_coarse']  # batch_size, num_coarse_actions + 2
    if normalise_input:
        x_tra_coarse, scalers = scale_transition(x_tra_coarse, scalers, quantile_range,
                                                 scaler_name='x_tra_coarse_scaler')
    x_tra_coarse = nan_to_value(x_tra_coarse, value=0.0)

    y_tra_coarse = data['y_tra_coarse']  # batch_size, 1
    if normalise_output:
        y_tra_coarse, scalers['y_tra_coarse_scaler'] = normalise(y_tra_coarse, strategy='robust', with_centering=False,
                                                                 quantile_range=quantile_range,
                                                                 scaler=scalers.get('y_tra_coarse_scaler'))

    x_tra_fine = data['x_tra_fine']  # batch_size, num_coarse_actions + num_fine_actions + 2
    if disable_parent_input:
        x_tra_fine = x_tra_fine[..., num_coarse_actions:]
    if normalise_input:
        x_tra_fine, scalers = scale_transition(x_tra_fine, scalers, quantile_range, scaler_name='x_tra_fine_scaler')
    x_tra_fine = nan_to_value(x_tra_fine, value=0.0)

    y_tra_fine = data['y_tra_fine']  # batch_size, 1
    if normalise_output:
        y_tra_fine, scalers['y_tra_fine_scaler'] = normalise(y_tra_fine, strategy='robust', with_centering=False,
                                                             quantile_range=quantile_range,
                                                             scaler=scalers.get('y_tra_fine_scaler'))

    # DECODER
    x_dec_coarse = data['x_dec_coarse'][:, :output_seq_len]  # batch_size, output_seq_len, num_coarse_actions + 1
    x_dec_coarse = nan_to_value(x_dec_coarse, value=0.0)
    x_dec_coarse_mask = nan_to_value(data['x_dec_coarse'][:, :output_seq_len, :-1], value=0.0).astype(np.bool)
    # batch_size, output_seq_len, num_coarse_actions + num_fine_actions + 1
    x_dec_fine = data['x_dec_fine'][:, :output_seq_len]
    if disable_parent_input:
        x_dec_fine = x_dec_fine[..., num_coarse_actions:]
    x_dec_fine = nan_to_value(x_dec_fine, value=0.0)
    x_dec_fine_mask = nan_to_value(data['x_dec_fine'][:, :output_seq_len, num_coarse_actions:-1], value=0.0)
    x_dec_fine_mask = x_dec_fine_mask.astype(np.bool)

    y_dec_coarse = data['y_dec_coarse'][:, :output_seq_len]  # batch_size, output_seq_len, 1 + 1
    if normalise_output:
        y_dec_coarse, scalers = scale_input(y_dec_coarse, scalers, quantile_range,
                                            scaler_name='y_dec_coarse_scaler', to_index=False)
    y_dec_coarse_cat, y_dec_coarse_num = pull_parts(y_dec_coarse, convert_nans=True)

    y_dec_fine = data['y_dec_fine'][:, :output_seq_len]  # batch_size, output_seq_len, 1 + 1
    if normalise_output:
        y_dec_fine, scalers = scale_input(y_dec_fine, scalers, quantile_range,
                                          scaler_name='y_dec_fine_scaler', to_index=False)
    y_dec_fine_cat, y_dec_fine_num = pull_parts(y_dec_fine, convert_nans=True)

    dec_bottom_layer_boundary = data['dec_boundary'][:, :output_seq_len]  # batch_size, output_seq_len
    dec_bottom_layer_boundary = nan_to_value(dec_bottom_layer_boundary, value=0.0)
    dec_upper_layer_boundary = np.zeros_like(dec_bottom_layer_boundary)
    dec_boundary = np.stack([dec_bottom_layer_boundary, dec_upper_layer_boundary], axis=1)

    num_fine_actions = x_enc_fine.shape[-1] - 1
    if not disable_parent_input:
        num_fine_actions -= num_coarse_actions
    extra_action = 1 if with_final_action else 0
    fine_class_weight = compute_weight_tensor(y_dec_fine_cat, num_classes=num_fine_actions + extra_action)
    coarse_class_weight = compute_weight_tensor(y_dec_coarse_cat, num_classes=num_coarse_actions + extra_action)
    weights = [fine_class_weight, coarse_class_weight]

    tensors = [x_enc_fine, x_enc_fine_mask, x_enc_coarse, x_enc_coarse_mask, enc_boundary, enc_zero_layer_boundary,
               x_tra_fine, x_tra_coarse,
               x_dec_fine, x_dec_fine_mask, x_dec_coarse, x_dec_coarse_mask, dec_boundary,
               y_enc_fine_cat, y_enc_fine_num, y_enc_coarse_cat, y_enc_coarse_num,
               y_tra_fine, y_tra_coarse,
               y_dec_fine_cat, y_dec_fine_num, y_dec_coarse_cat, y_dec_coarse_num, dec_boundary]
    return tensors, scalers, weights


def pull_parts(y, convert_nans=True):
    y_cat, y_num = y[..., 0], y[..., -1:]
    if convert_nans:
        y_cat = nan_to_value(y_cat, value=-1.0).astype(np.int64)
        y_num = nan_to_value(y_num, value=-1.0)
    return y_cat, y_num


def feed_model_data(model, data, **kwargs):
    (x_enc_fine, x_enc_fine_mask, x_enc_coarse, x_enc_coarse_mask, enc_boundary, enc_zero_layer_boundary,
     x_tra_fine, x_tra_coarse,
     x_dec_fine, x_dec_fine_mask, x_dec_coarse, x_dec_coarse_mask, dec_boundary) = data
    x_enc = x_enc_fine, x_enc_fine_mask, x_enc_coarse, x_enc_coarse_mask
    x_tra = x_tra_fine, x_tra_coarse
    x_dec = x_dec_fine, x_dec_fine_mask, x_dec_coarse, x_dec_coarse_mask
    dx_enc = [enc_boundary, enc_zero_layer_boundary]
    output = model(x_enc, x_tra, x_dec, dx_enc=dx_enc, dx_dec=dec_boundary, hx=None,
                   teacher_prob=kwargs.get('teacher_prob'), scaling_values=kwargs.get('scaling_values'))
    return output


def add_remaining_params(params, model, transition_learning_rate, share_embeddings, share_encoder_decoder,
                         share_predictions, weight_decay, weight_decay_decoder_only):
    if not share_embeddings:
        params += [{'params': model.transition_net.parameters(), 'lr': transition_learning_rate}]
        if weight_decay_decoder_only:
            params += [{'params': model.decoder_net.embedding_layer.parameters(), 'weight_decay': weight_decay}]
        else:
            params += [{'params': model.decoder_net.embedding_layer.parameters()}]
    else:
        params += [{'params': model.transition_net.embedding_layer.embeddings[0].num_embedding.parameters(),
                    'lr': transition_learning_rate},
                   {'params': model.transition_net.embedding_layer.embeddings[0].joint_embedding.parameters(),
                    'lr': transition_learning_rate}
                   ]
        params += [{'params': model.transition_net.embedding_layer.embeddings[1].num_embedding.parameters(),
                    'lr': transition_learning_rate},
                   {'params': model.transition_net.embedding_layer.embeddings[1].joint_embedding.parameters(),
                    'lr': transition_learning_rate}
                   ]
        params += [{'params': model.transition_net.coarse_transition_layer.parameters(),
                    'lr': transition_learning_rate},
                   {'params': model.transition_net.coarse_action_len_layer.parameters(),
                    'lr': transition_learning_rate}
                   ]
        params += [{'params': model.transition_net.fine_transition_layer.parameters(),
                    'lr': transition_learning_rate},
                   {'params': model.transition_net.fine_action_len_layer.parameters(),
                    'lr': transition_learning_rate}
                   ]
    if not share_encoder_decoder:
        if weight_decay_decoder_only:
            params += [{'params': model.decoder_net.decoder_hmgru.parameters(), 'weight_decay': weight_decay}]
        else:
            params += [{'params': model.decoder_net.decoder_hmgru.parameters()}]
    if not share_predictions:
        if weight_decay_decoder_only:
            params += [{'params': model.decoder_net.coarse_action_layer.parameters(), 'weight_decay': weight_decay},
                       {'params': model.decoder_net.fine_action_layer.parameters(), 'weight_decay': weight_decay}]
        else:
            params += [{'params': model.decoder_net.coarse_action_layer.parameters()},
                       {'params': model.decoder_net.fine_action_layer.parameters()}]


def compute_weight_tensor(y_dec_cat, num_classes):
    """y_dec_cat is a numpy array of shape (num_examples, output_seq_len)."""
    unique, unique_counts = np.unique(y_dec_cat.reshape(-1), return_counts=True)
    unique, unique_counts = unique[1:], unique_counts[1:]
    total_counts = np.sum(unique_counts)
    normalised_weights = total_counts / unique_counts
    normalised_weights /= np.sum(normalised_weights)
    weights = np.ones(num_classes, dtype=np.float32)
    weights[unique] = normalised_weights
    return weights


def compute_zero_layer_boundary(x_enc_fine):
    """Numpy array of shape (num_examples, input_seq_len, num_actions + 1)."""
    x_enc_fine = x_enc_fine[..., -1]
    enc_zero_layer_boundary = nan_to_value(x_enc_fine, value=0.0)
    enc_zero_layer_boundary[enc_zero_layer_boundary != 0.0] = 1.0
    return enc_zero_layer_boundary


def add_coarse_only_params(model):
    params = [{'params': model.encoder_net.embedding_layer.embeddings[1].parameters()},
              {'params': model.encoder_net.encoder_hmgru.cells[1].parameters()},
              {'params': model.encoder_net.coarse_action_layer.parameters()}]
    params += [{'params': model.transition_net.embedding_layer.embeddings[1].parameters()},
               {'params': model.transition_net.coarse_transition_layer.parameters()},
               {'params': model.transition_net.coarse_action_len_layer.parameters()}]
    params += [{'params': model.decoder_net.embedding_layer.embeddings[1].parameters()},
               {'params': model.decoder_net.decoder_hmgru.cells[1].parameters()},
               {'params': model.decoder_net.coarse_action_layer.parameters()}]
    return params
