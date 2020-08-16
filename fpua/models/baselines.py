from functools import partial
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from fpua.models.fetchers import multiple_input_multiple_output
from fpua.models.hera import pull_parts
from fpua.models.losses import baseline_loss
from fpua.models.misc import MultiTaskLossLearner, ActionLayer
from fpua.utils import numpy_to_torch, train, save_checkpoint, create_alias, nan_to_value, logit2one_hot
from fpua.utils import num_workers_from_batch_size, set_initial_teacher_prob


class InputEmbedding(nn.Module):
    def __init__(self, num_actions, num_numerical_features, embedding_size, nonlinearity='relu', disable_length=False,
                 bias=True):
        super(InputEmbedding, self).__init__()
        self.num_actions = num_actions
        self.nonlinearity = nonlinearity
        self.disable_length = disable_length
        if embedding_size:
            EmbeddingLayer = nn.Linear
        else:
            EmbeddingLayer, nonlinearity = nn.Identity, 'linear'

        self.cat_embedding = EmbeddingLayer(num_actions, embedding_size, bias=bias)
        self.num_embedding = EmbeddingLayer(num_numerical_features, embedding_size, bias=bias)
        self.joint_embedding = EmbeddingLayer(2 * embedding_size, embedding_size, bias=bias)

    def forward(self, x):
        cat_emb = self.cat_embedding(x[..., :self.num_actions])
        cat_emb = self._apply_nonlinearity(cat_emb, nonlinearity=self.nonlinearity)
        if self.disable_length:
            return cat_emb
        num_emb = self.num_embedding(x[..., self.num_actions:])
        num_emb = self._apply_nonlinearity(num_emb, nonlinearity=self.nonlinearity)
        joint_emb = self.joint_embedding(torch.cat([cat_emb, num_emb], dim=-1))
        joint_emb = self._apply_nonlinearity(joint_emb, nonlinearity=self.nonlinearity)
        return joint_emb

    @staticmethod
    def _apply_nonlinearity(x, nonlinearity):
        if nonlinearity == 'relu':
            x = torch.relu(x)
        elif nonlinearity == 'tanh':
            x = torch.tanh(x)
        elif nonlinearity == 'sigmoid':
            x = torch.sigmoid(x)
        return x


class Baseline0(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, embedding_nonlinearity='tanh',
                 length_activation='sigmoid', with_final_action=False, bias=True):
        super(Baseline0, self).__init__()
        self.with_final_action = with_final_action
        num_actions = input_size - 1

        self.embedding = InputEmbedding(num_actions, 1, embedding_size=embedding_size,
                                        nonlinearity=embedding_nonlinearity, bias=bias)
        if embedding_size:
            input_size = embedding_size
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=1, bias=bias, batch_first=True)
        extra_action = 1 if with_final_action else 0
        self.action = ActionLayer(hidden_size, num_actions + extra_action, length_activation=length_activation,
                                  bias=bias)

    def forward(self, x_enc, hx=None, teacher_prob=None, test_mode=False, effective_num_steps=None):
        x_enc_emb = self.embedding(x_enc)
        output, _ = self.rnn(x_enc_emb, hx=hx)
        if not test_mode:
            y_logits, y_lens = self.action(output)
        else:
            y_logits, y_lens = [], []
            batch_size, seq_len = x_enc.size(0), x_enc.size(1)
            for i in range(batch_size):
                y_ex_logits_fine, y_ex_lens_fine = [], []
                steps = int(effective_num_steps[i].item())
                step_hx = output[i:i + 1][:, steps - 1]
                y_step_logits, y_step_lens = self.action(step_hx)
                x_cat = logit2one_hot(y_step_logits.detach())
                if self.with_final_action:
                    x_cat = x_cat[..., :-1]
                x_num = x_enc[i:i + 1, steps - 1, -1:] + y_step_lens.detach()
                x_enc_step = torch.cat([x_cat, x_num], dim=-1)
                for t in range(seq_len):
                    y_step_logits, y_step_lens, step_hx = self.single_step(x_enc_step, hx_step=step_hx)
                    y_ex_logits_fine.append(y_step_logits)
                    y_ex_lens_fine.append(y_step_lens)

                    x_cat = logit2one_hot(y_step_logits.detach())
                    if self.with_final_action:
                        x_cat = x_cat[..., :-1]
                    x_num = x_enc_step[..., -1:] + y_step_lens.detach()
                    x_enc_step = torch.cat([x_cat, x_num], dim=-1)
                y_logits.append(torch.stack(y_ex_logits_fine, dim=1))
                y_lens.append(torch.stack(y_ex_lens_fine, dim=1))
            y_logits = torch.cat(y_logits, dim=0)
            y_lens = torch.cat(y_lens, dim=0)
        return y_logits, y_lens, None

    def single_step(self, x_enc_step, hx_step):
        x_enc_step = self.embedding(x_enc_step)
        output, hx_step = self.rnn(x_enc_step.unsqueeze(1), hx=hx_step.unsqueeze(0))
        output = output[:, 0]
        y_step_logits, y_step_lens = self.action(output)
        return y_step_logits, y_step_lens, hx_step[0]


class Baseline1(nn.Module):
    def __init__(self, input_size, hidden_size, num_fine_actions, num_coarse_actions, embedding_size,
                 embedding_nonlinearity='tanh', length_activation='sigmoid', with_final_action=False, bias=True):
        super(Baseline1, self).__init__()
        self.with_final_action = with_final_action

        self.embedding = InputEmbedding(input_size - 1, 1, embedding_size=embedding_size,
                                        nonlinearity=embedding_nonlinearity, bias=bias)
        if embedding_size:
            input_size = embedding_size
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=1, bias=bias, batch_first=True)
        extra_action = 1 if with_final_action else 0
        self.coarse_action = nn.Linear(hidden_size, num_coarse_actions + extra_action, bias=bias)
        self.fine_action = ActionLayer(hidden_size, num_fine_actions + extra_action,
                                       length_activation=length_activation, bias=bias)

    def forward(self, x_enc_fine, x_enc_coarse, hx=None, teacher_prob=None, test_mode=False, effective_num_steps=None):
        x_enc = torch.cat([x_enc_coarse[..., :-1], x_enc_fine], dim=-1)
        x_enc = self.embedding(x_enc)
        output, _ = self.rnn(x_enc, hx=hx)
        if not test_mode:
            y_logits_coarse = F.log_softmax(self.coarse_action(output), dim=-1)
            y_logits_fine, y_lens_fine = self.fine_action(output)
        else:
            y_logits_coarse, y_logits_fine, y_lens_fine = [], [], []
            batch_size, seq_len = x_enc.size(0), x_enc.size(1)
            for i in range(batch_size):
                y_ex_logits_coarse, y_ex_logits_fine, y_ex_lens_fine = [], [], []
                steps = int(effective_num_steps[i].item())
                step_hx = output[i:i + 1][:, steps - 1]
                y_step_logits_coarse = F.log_softmax(self.coarse_action(step_hx), dim=-1)
                y_step_logits_fine, y_step_lens_fine = self.fine_action(step_hx)
                x_coarse_cat = logit2one_hot(y_step_logits_coarse.detach())
                x_fine_cat = logit2one_hot(y_step_logits_fine.detach())
                if self.with_final_action:
                    x_coarse_cat = x_coarse_cat[..., :-1]
                    x_fine_cat = x_fine_cat[..., :-1]
                x_fine_num = x_enc_fine[i:i + 1, steps - 1, -1:] + y_step_lens_fine.detach()
                x_enc_step = torch.cat([x_coarse_cat, x_fine_cat, x_fine_num], dim=-1)
                for t in range(seq_len):
                    y_step_logits_coarse, y_step_logits_fine, y_step_lens_fine, step_hx = \
                        self.single_step(x_enc_step, hx_step=step_hx)
                    y_ex_logits_coarse.append(y_step_logits_coarse)
                    y_ex_logits_fine.append(y_step_logits_fine)
                    y_ex_lens_fine.append(y_step_lens_fine)

                    x_coarse_cat = logit2one_hot(y_step_logits_coarse.detach())
                    x_fine_cat = logit2one_hot(y_step_logits_fine.detach())
                    if self.with_final_action:
                        x_coarse_cat = x_coarse_cat[..., :-1]
                        x_fine_cat = x_fine_cat[..., :-1]
                    x_fine_num = x_enc_step[..., -1:] + y_step_lens_fine.detach()
                    x_enc_step = torch.cat([x_coarse_cat, x_fine_cat, x_fine_num], dim=-1)
                y_logits_coarse.append(torch.stack(y_ex_logits_coarse, dim=1))
                y_logits_fine.append(torch.stack(y_ex_logits_fine, dim=1))
                y_lens_fine.append(torch.stack(y_ex_lens_fine, dim=1))
            y_logits_coarse = torch.cat(y_logits_coarse, dim=0)
            y_logits_fine = torch.cat(y_logits_fine, dim=0)
            y_lens_fine = torch.cat(y_lens_fine, dim=0)
        return y_logits_fine, y_lens_fine, y_logits_coarse

    def single_step(self, x_enc_step, hx_step):
        x_enc_step = self.embedding(x_enc_step)
        output, hx_step = self.rnn(x_enc_step.unsqueeze(1), hx=hx_step.unsqueeze(0))
        output = output[:, 0]
        y_step_logits_coarse = F.log_softmax(self.coarse_action(output), dim=-1)
        y_step_logits_fine, y_step_lens_fine = self.fine_action(output)
        return y_step_logits_coarse, y_step_logits_fine, y_step_lens_fine, hx_step[0]


class Baseline2(nn.Module):
    def __init__(self, input_size, hidden_size, num_fine_actions, num_coarse_actions, embedding_size,
                 embedding_nonlinearity='tanh', length_activation='sigmoid', with_final_action=False, bias=True):
        super(Baseline2, self).__init__()
        self.with_final_action = with_final_action
        extra_action = 1 if with_final_action else 0
        input_size = list(input_size)

        self.coarse_embedding = InputEmbedding(num_coarse_actions, 1, embedding_size=embedding_size,
                                               nonlinearity=embedding_nonlinearity, bias=bias)
        if embedding_size:
            input_size[1] = embedding_size
        self.coarse_rnn = nn.GRU(input_size[1], hidden_size, num_layers=1, bias=bias, batch_first=True)
        self.coarse_action = nn.Linear(hidden_size, num_coarse_actions + extra_action, bias=bias)

        self.fine_embedding = InputEmbedding(num_fine_actions, 1, embedding_size=embedding_size,
                                             nonlinearity=embedding_nonlinearity, bias=bias)
        if embedding_size:
            input_size[0] = embedding_size
        self.fine_rnn = nn.GRU(input_size[0] + hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=True)
        self.fine_action = ActionLayer(hidden_size, num_fine_actions + extra_action,
                                       length_activation=length_activation, bias=bias)

    def forward(self, x_enc_fine, x_enc_coarse, hx=None, teacher_prob=None, test_mode=False, effective_num_steps=None):
        # Coarse
        x_enc_coarse_emb = self.coarse_embedding(x_enc_coarse)
        coarse_output, _ = self.coarse_rnn(x_enc_coarse_emb, hx=hx)
        # Fine
        x_enc_fine_emb = self.fine_embedding(x_enc_fine)
        x_enc_fine_input = torch.cat([x_enc_fine_emb, coarse_output], dim=-1)
        fine_output, _ = self.fine_rnn(x_enc_fine_input, hx=hx)
        if not test_mode:
            y_logits_coarse = F.log_softmax(self.coarse_action(coarse_output), dim=-1)
            y_logits_fine, y_lens_fine = self.fine_action(fine_output)
        else:
            y_logits_coarse, y_logits_fine, y_lens_fine = [], [], []
            batch_size, seq_len = x_enc_fine.size(0), x_enc_fine.size(1)
            for i in range(batch_size):
                y_ex_logits_coarse, y_ex_logits_fine, y_ex_lens_fine = [], [], []
                steps = int(effective_num_steps[i].item())
                step_hx_coarse = coarse_output[i:i + 1][:, steps - 1]
                step_hx_fine = fine_output[i:i + 1][:, steps - 1]
                y_step_logits_coarse = F.log_softmax(self.coarse_action(step_hx_coarse), dim=-1)
                y_step_logits_fine, y_step_lens_fine = self.fine_action(step_hx_fine)
                x_coarse_cat = logit2one_hot(y_step_logits_coarse.detach())
                if self.with_final_action:
                    x_coarse_cat = x_coarse_cat[..., :-1]
                x_coarse_num = x_enc_coarse[i:i + 1, steps - 1, -1:] + y_step_lens_fine.detach()
                x_enc_coarse_step = torch.cat([x_coarse_cat, x_coarse_num], dim=-1)
                x_fine_cat = logit2one_hot(y_step_logits_fine.detach())
                if self.with_final_action:
                    x_fine_cat = x_fine_cat[..., :-1]
                x_fine_num = x_enc_fine[i:i + 1, steps - 1, -1:] + y_step_lens_fine.detach()
                x_enc_fine_step = torch.cat([x_fine_cat, x_fine_num], dim=-1)
                for t in range(seq_len):
                    y_step_logits_coarse, y_step_logits_fine, y_step_lens_fine, step_hx_fine, step_hx_coarse = \
                        self.single_step(x_enc_fine_step, x_enc_coarse_step,
                                         hx_fine_step=step_hx_fine, hx_coarse_step=step_hx_coarse)
                    y_ex_logits_coarse.append(y_step_logits_coarse)
                    y_ex_logits_fine.append(y_step_logits_fine)
                    y_ex_lens_fine.append(y_step_lens_fine)

                    x_coarse_cat = logit2one_hot(y_step_logits_coarse.detach())
                    if self.with_final_action:
                        x_coarse_cat = x_coarse_cat[..., :-1]
                    x_coarse_num = x_enc_coarse_step[..., -1:] + y_step_lens_fine.detach()
                    x_enc_coarse_step = torch.cat([x_coarse_cat, x_coarse_num], dim=-1)
                    x_fine_cat = logit2one_hot(y_step_logits_fine.detach())
                    if self.with_final_action:
                        x_fine_cat = x_fine_cat[..., :-1]
                    x_fine_num = x_enc_fine_step[..., -1:] + y_step_lens_fine.detach()
                    x_enc_fine_step = torch.cat([x_fine_cat, x_fine_num], dim=-1)
                y_logits_coarse.append(torch.stack(y_ex_logits_coarse, dim=1))
                y_logits_fine.append(torch.stack(y_ex_logits_fine, dim=1))
                y_lens_fine.append(torch.stack(y_ex_lens_fine, dim=1))
            y_logits_coarse = torch.cat(y_logits_coarse, dim=0)
            y_logits_fine = torch.cat(y_logits_fine, dim=0)
            y_lens_fine = torch.cat(y_lens_fine, dim=0)
        return y_logits_fine, y_lens_fine, y_logits_coarse

    def single_step(self, x_enc_fine_step, x_enc_coarse_step, hx_fine_step, hx_coarse_step):
        x_enc_coarse_step = self.coarse_embedding(x_enc_coarse_step)
        output_coarse, hx_coarse_step = self.coarse_rnn(x_enc_coarse_step.unsqueeze(1), hx=hx_coarse_step.unsqueeze(0))
        output_coarse = output_coarse[:, 0]
        y_step_logits_coarse = F.log_softmax(self.coarse_action(output_coarse), dim=-1)
        x_enc_fine_step = self.fine_embedding(x_enc_fine_step)
        x_enc_fine_step = torch.cat([x_enc_fine_step, output_coarse], dim=-1)
        output_fine, hx_fine_step = self.fine_rnn(x_enc_fine_step.unsqueeze(1), hx=hx_fine_step.unsqueeze(0))
        output_fine = output_fine[:, 0]
        y_step_logits_fine, y_step_lens_fine = self.fine_action(output_fine)
        return y_step_logits_coarse, y_step_logits_fine, y_step_lens_fine, hx_fine_step[0], hx_coarse_step[0]


def train_baselines(args):
    training_data = args.training_data
    validation_data = args.validation_data
    baseline_type = args.baseline_type
    action_level = args.action_level
    seq_len = args.seq_len
    hidden_size = args.hidden_size
    embedding_size = args.embedding_size
    embedding_nonlinearity = args.embedding_nonlinearity
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    length_activation = args.length_activation
    multi_task_loss_learner = args.multi_task_loss_learner
    print_raw_losses = args.print_raw_losses
    teacher_schedule = args.teacher_schedule if args.teacher_schedule != 'None' else None
    teacher_prob = set_initial_teacher_prob(teacher_schedule) if teacher_schedule is not None else None
    weight_decay = args.weight_decay
    clip_gradient_at = args.clip_gradient_at
    with_final_action = '_withfinalaction' in training_data
    log_dir = args.log_dir
    # Data
    num_workers = num_workers_from_batch_size(batch_size)
    with np.load(training_data) as data:
        tensors = assemble_tensors(data, seq_len)
    tensors = numpy_to_torch(*tensors)
    dataset = TensorDataset(*tensors)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=False, drop_last=False)
    val_loader = None
    if validation_data is not None:
        with np.load(validation_data) as data:
            val_tensors = assemble_tensors(data, seq_len)
        val_tensors = numpy_to_torch(*val_tensors)
        val_dataset = TensorDataset(*val_tensors)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=False, drop_last=False)
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if baseline_type == 0:
        Baseline = Baseline0
        input_size = tensors[1].size(-1) if action_level == 'coarse' else tensors[0].size(-1)
        specific_args = {'input_size': input_size}
    else:
        num_fine_actions = tensors[0].size(-1) - 1
        num_coarse_actions = tensors[1].size(-1) - 1
        if baseline_type == 1:
            Baseline = Baseline1
            input_size = num_coarse_actions + (num_fine_actions + 1)
        else:
            Baseline = Baseline2
            input_size = [num_fine_actions + 1, num_coarse_actions + 1]
        specific_args = {'input_size': input_size, 'num_fine_actions': num_fine_actions,
                         'num_coarse_actions': num_coarse_actions}
    model_creation_args = {'hidden_size': hidden_size, 'embedding_size': embedding_size,
                           'embedding_nonlinearity': embedding_nonlinearity, 'length_activation': length_activation,
                           'with_final_action': with_final_action, 'bias': True}
    model_creation_args.update(specific_args)
    model = Baseline(**model_creation_args).to(device)
    loss_types = ['softmax', 'mse']
    if baseline_type > 0:
        loss_types.append('softmax')
    mtll_model = MultiTaskLossLearner(loss_types=loss_types).to(device) if multi_task_loss_learner else None
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if mtll_model is not None:
        optimizer.add_param_group({'params': mtll_model.parameters()})
    criterion = partial(baseline_loss, baseline_type=baseline_type, action_level=action_level)
    if baseline_type == 0:
        action_code = 'C' if action_level == 'coarse' else 'F'
        loss_names = [f'NLL {action_code}', f'MSE {action_code}']
    else:
        loss_names = ['NLL F', 'MSE F', 'NLL C']
    fetch_model_data = partial(multiple_input_multiple_output, n=3)

    test_mode = '_isval' in validation_data
    checkpoint_name = create_alias(hidden_size=hidden_size, epochs=epochs, batch_size=batch_size, input_seq_len=seq_len,
                                   output_seq_len=seq_len, length_activation=length_activation,
                                   learning_rate=learning_rate, embedding_size=embedding_size,
                                   teacher_schedule=teacher_schedule, validation_data=validation_data,
                                   l2_reg=weight_decay, multi_task_loss_learner=multi_task_loss_learner,
                                   embedding_nonlinearity=embedding_nonlinearity, clip_gradient_at=clip_gradient_at,
                                   baseline_type=baseline_type, action_level=action_level, test_mode=test_mode,
                                   with_final_action=with_final_action)
    tensorboard_log_dir = os.path.join(log_dir, 'runs', checkpoint_name) if log_dir is not None else None
    feed_model_data_partial = partial(feed_model_data, baseline_type=baseline_type, action_level=action_level)
    checkpoint = train(model, train_loader, optimizer, criterion, epochs, device, clip_gradient_at=clip_gradient_at,
                       fetch_model_data=fetch_model_data, feed_model_data=feed_model_data_partial,
                       loss_names=loss_names, val_loader=val_loader, mtll_model=mtll_model,
                       print_raw_losses=print_raw_losses, tensorboard_log_dir=tensorboard_log_dir,
                       teacher_schedule=teacher_schedule, teacher_prob=teacher_prob, test_mode=test_mode)
    print('Baseline model successfully trained.')
    if log_dir is not None:
        checkpoint['baseline_type'] = baseline_type
        checkpoint['action_level'] = action_level if baseline_type == 0 else None
        checkpoint['seq_len'] = seq_len
        checkpoint['model_creation_args'] = model_creation_args
        save_checkpoint(log_dir, checkpoint, checkpoint_name=checkpoint_name, include_timestamp=False)


def feed_model_data(model, data, baseline_type, action_level, **kwargs):
    x_enc_fine, x_enc_coarse, input_num_steps = data
    test_mode = kwargs.get('test_mode', False)
    if baseline_type == 0:
        x_enc = x_enc_coarse if action_level == 'coarse' else x_enc_fine
        output = model(x_enc, hx=None, test_mode=test_mode, effective_num_steps=input_num_steps)
    else:
        output = model(x_enc_fine, x_enc_coarse, hx=None, test_mode=test_mode, effective_num_steps=input_num_steps)
    return output


def assemble_tensors(data, seq_len):
    x_enc_coarse = data['x_enc_coarse'][:, :seq_len]
    x_enc_coarse = nan_to_value(x_enc_coarse, value=0.0)
    x_enc_fine = data['x_enc_fine'][:, :seq_len]
    x_enc_fine = nan_to_value(x_enc_fine, value=0.0)
    y_enc_fine = data['y_enc_fine'][:, :seq_len]
    y_enc_fine_cat, y_enc_fine_num = pull_parts(y_enc_fine, convert_nans=True)
    y_enc_coarse = data['y_enc_coarse'][:, :seq_len]
    y_enc_coarse_cat, y_enc_coarse_num = pull_parts(y_enc_coarse, convert_nans=True)
    input_num_steps = data['effective_num_steps']
    tensors = [x_enc_fine, x_enc_coarse, input_num_steps,
               y_enc_fine_cat, y_enc_fine_num, y_enc_coarse_cat, y_enc_coarse_num]
    return tensors
