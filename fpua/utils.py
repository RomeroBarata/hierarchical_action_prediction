import collections.abc
from datetime import datetime
import math
import os
from itertools import takewhile

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import torch
from torch.utils.tensorboard import SummaryWriter

from fpua.models.fetchers import single_input_single_output
from fpua.models.forwarders import basic_forward


def train_single_epoch(model, data_loader, optimizer, criterion, device, clip_gradient_at=5.0,
                       fetch_model_data=single_input_single_output, feed_model_data=basic_forward, loss_names=None,
                       log_interval=25, mtll_model=None, num_main_losses=None, **kwargs):
    """General training function to train a PyTorch model for a single epoch.

    Arg(s):
        model - PyTorch model.
        data_loader - Batch generator for model training.
        optimizer - Model optimizer.
        criterion - Specific loss function for the given model. This function receives as input the output of
            model and the ground-truth target, and returns a list of batch losses (for multi-loss models). Even if
            the model has a single loss, the return value of criterion must be a list containing this single loss.
        device - Which device to use for model training. Either cuda or cpu.
        clip_gradient_at - If nonzero clips the norm of the gradient vector at the specified value. The gradient
            vector is a vector obtained by concatenating all parameters of the model.
        fetch_model_data - Function to fetch the input and output tensors for the model.
        feed_model_data - Function to feed the input tensors to the model.
        loss_names - Names for the individual losses output by criterion. If None, the losses are named loss_1,
            loss_2, ....
        log_interval - Print training statistics every log_interval batches.
        **kwargs - Any extra parameter that needs to be passed to the feed_model_data of a model.
    """
    model.train()
    if mtll_model is not None:
        mtll_model.train()
    loss_names = loss_names if loss_names is not None else ['loss_' + str(n) for n in range(1, 101)]
    num_examples = len(data_loader.dataset)
    for batch_idx, dataset in enumerate(data_loader):
        data, target = fetch_model_data(dataset, device=device)
        optimizer.zero_grad()
        output = feed_model_data(model, data, **kwargs)
        losses = criterion(output, target, reduction='mean')
        if mtll_model is not None:
            losses = mtll_model(losses)
        loss = sum(losses)
        loss.backward()
        if clip_gradient_at:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_gradient_at)
        optimizer.step()
        log_now, is_last_batch = (batch_idx % log_interval) == 0, batch_idx == (len(data_loader) - 1)
        if log_now or is_last_batch:
            num_main_losses = num_main_losses if num_main_losses is not None else len(losses)
            loss = sum(losses[-num_main_losses:])
            batch_initial_example_idx = min((batch_idx + 1) * data_loader.batch_size, num_examples)
            epoch_progress = 100 * (batch_idx + 1) / len(data_loader)
            print(f'(Train) Batch [{batch_initial_example_idx:6d}/{num_examples:6d} ({epoch_progress:3.0f}%)] ',
                  f'Loss: {loss.item(): 8.4f}', end='')
            for loss_name, single_loss in zip(loss_names[-num_main_losses:], losses[-num_main_losses:]):
                print(f'  {loss_name}: {single_loss: 6.4f}', end='')
            print()


def train(model, train_loader, optimizer, criterion, epochs, device, clip_gradient_at=5.0,
          fetch_model_data=single_input_single_output, feed_model_data=basic_forward, loss_names=None,
          val_loader=None, early_stopping=False, initial_epoch=1, mtll_model=None, print_raw_losses=False,
          evaluate_train_like_test=False, num_main_losses=None, **kwargs):
    """General training function to train a PyTorch model.

    If validation data is not given, the returned checkpoint is the one obtained after training the model for the
    specified number of epochs, regardless of the final training loss. If validation data is given, the checkpoint
    returned is the one with the lowest validation loss, which could have been obtained in some epoch before the
    last one.
    Arg(s):
        model - PyTorch model.
        train_loader - Batch generator for model training.
        optimizer - Model optimizer.
        criterion - Specific loss function for the given model. This function receives as input the output of
            model and the ground-truth target, and returns a list of batch losses (for multi-loss models). Even if
            the model has a single loss, the return value of criterion must be a list containing this single loss.
        epochs - Maximum number of epochs for model training.
        device - Which device to use for model training. Either cuda or cpu.
        clip_gradient_at - If nonzero clips the norm of the gradient vector at the specified value. The gradient
            vector is a vector obtained by concatenating all parameters of the model.
        fetch_model_data - Function to fetch the input and output tensors for the model.
        feed_model_data - Function to feed the input tensors to the model.
        loss_names - Names for the individual losses output by criterion. If None, the losses are named loss_1,
            loss_2, ....
        val_loader - Batch generator for model validation.
        early_stopping - TO DO.
        **kwargs - Any extra parameters to be passed during training.
    Returns:
        A dictionary containing the history of train losses, the model's weights and associated epoch, and if
        val_loader is specified, the history of validation losses as well.
    """
    log_dir = kwargs.get('tensorboard_log_dir', None)
    writer = SummaryWriter(log_dir) if log_dir is not None else None
    checkpoint = {}
    train_losses, val_losses, train_raw_losses, val_raw_losses = [], [], [], []
    val_loss = float('Inf')
    for epoch in range(initial_epoch, epochs + initial_epoch):
        # Train
        test_mode = kwargs.pop('test_mode', None)
        print(f'\nEpoch: [{epoch:4d}/{epochs + initial_epoch - 1:4d}]')
        train_single_epoch(model, data_loader=train_loader, optimizer=optimizer, criterion=criterion,
                           device=device, clip_gradient_at=clip_gradient_at, fetch_model_data=fetch_model_data,
                           feed_model_data=feed_model_data, loss_names=loss_names, log_interval=25,
                           mtll_model=mtll_model, num_main_losses=num_main_losses, **kwargs)
        if evaluate_train_like_test:
            temperature, slope = kwargs.get('temperature', 1.0), kwargs.get('slope', 1.0)
            teacher_prob = 0.0 if kwargs.get('teacher_prob') is not None else None
            current_train_loss, current_train_losses, current_train_raw_loss, current_train_raw_losses = \
                test(model, data_loader=train_loader, criterion=criterion,
                     device=device, fetch_model_data=fetch_model_data,
                     feed_model_data=feed_model_data, loss_names=loss_names,
                     test_set_name='Train', mtll_model=mtll_model,
                     print_raw_losses=print_raw_losses, num_main_losses=num_main_losses,
                     temperature=temperature, slope=slope, teacher_prob=teacher_prob)
        else:
            current_train_loss, current_train_losses, current_train_raw_loss, current_train_raw_losses = \
                test(model, data_loader=train_loader, criterion=criterion,
                     device=device, fetch_model_data=fetch_model_data,
                     feed_model_data=feed_model_data, loss_names=loss_names,
                     test_set_name='Train', mtll_model=mtll_model,
                     print_raw_losses=print_raw_losses, num_main_losses=num_main_losses, **kwargs)
        train_losses.append([current_train_loss, current_train_losses])
        if mtll_model is not None:
            train_raw_losses.append([current_train_raw_loss, current_train_raw_losses])
        num_main_losses = num_main_losses if num_main_losses is not None else len(current_train_losses)
        if writer is not None:
            base_str = 'Loss/train_mtll/' if mtll_model is not None else 'Loss/train/'
            for loss_name, loss in zip(loss_names[-num_main_losses:], current_train_losses):
                writer.add_scalar(base_str + loss_name, loss, epoch)
            writer.add_scalar(base_str + 'total', current_train_loss, epoch)
            if mtll_model is not None:
                loss_weights = mtll_model.get_weights()
                for loss_name, raw_loss, loss_weight in zip(loss_names[-num_main_losses:],
                                                            current_train_raw_losses, loss_weights):
                    writer.add_scalar('Loss/train/' + loss_name, raw_loss, epoch)
                    writer.add_scalar('Loss/mtll_weight/' + loss_name, loss_weight, epoch)
                writer.add_scalar('Loss/train/total', current_train_raw_loss, epoch)
        # Validate
        kwargs['test_mode'] = test_mode
        if val_loader is not None:
            temperature, slope = kwargs.get('temperature', 1.0), kwargs.get('slope', 1.0)
            teacher_prob = 0.0 if kwargs.get('teacher_prob') is not None else None
            test_mode = kwargs.get('test_mode', False)
            current_val_loss, current_val_losses, current_val_raw_loss, current_val_raw_losses = \
                test(model, data_loader=val_loader, criterion=criterion,
                     device=device, fetch_model_data=fetch_model_data,
                     feed_model_data=feed_model_data, loss_names=loss_names,
                     test_set_name='Validation', mtll_model=mtll_model,
                     print_raw_losses=print_raw_losses, num_main_losses=num_main_losses,
                     temperature=temperature, slope=slope, teacher_prob=teacher_prob, test_mode=test_mode)
            val_losses.append([current_val_loss, current_val_losses])
            if mtll_model is not None:
                val_raw_losses.append([current_val_raw_loss, current_val_raw_losses])
            if writer is not None:
                base_str = 'Loss/val_mtll/' if mtll_model is not None else 'Loss/val/'
                for loss_name, loss in zip(loss_names[-num_main_losses:], current_val_losses):
                    writer.add_scalar(base_str + loss_name, loss, epoch)
                writer.add_scalar(base_str + 'total', current_val_loss, epoch)
                if mtll_model is not None:
                    for loss_name, raw_loss in zip(loss_names[-num_main_losses:], current_val_raw_losses):
                        writer.add_scalar('Loss/val/' + loss_name, raw_loss, epoch)
                    writer.add_scalar('Loss/val/total', current_val_raw_loss, epoch)
            if current_val_loss < val_loss:
                val_loss = current_val_loss
                checkpoint['epoch'] = epoch
                checkpoint['model_state_dict'] = model.state_dict()
                if mtll_model is not None:
                    checkpoint['mtll_model_state_dict'] = mtll_model.state_dict()
        else:
            checkpoint['epoch'] = epoch
            checkpoint['model_state_dict'] = model.state_dict()
            if mtll_model is not None:
                checkpoint['mtll_model_state_dict'] = mtll_model.state_dict()
        _update_kwargs(kwargs, epoch, epochs)
    checkpoint['train_losses'] = train_losses
    checkpoint['val_losses'] = val_losses
    checkpoint['train_raw_losses'] = train_raw_losses
    checkpoint['val_raw_losses'] = val_raw_losses
    if 'temperature' in kwargs:
        checkpoint['temperature'] = kwargs['temperature']
    if 'slope' in kwargs:
        checkpoint['slope'] = kwargs['slope']
    if writer is not None:
        writer.close()
    return checkpoint


def _update_kwargs(kwargs, epoch, epochs):
    if 'temperature' in kwargs:
        kwargs['temperature'] = max(0.1, kwargs['temperature'] * 0.90)
    if 'slope' in kwargs:
        kwargs['slope'] = min(5.0, 1.0 + 0.04 * epoch)
    if kwargs.get('teacher_schedule') is not None:
        # Since we update the teacher_prob after the first epoch, teacher_prob for the first epoch is the one
        # initially passed by the user to train.
        teacher_schedule = kwargs['teacher_schedule']
        if teacher_schedule == 'linear':
            # First half with some teacher help, second half on its own
            kwargs['teacher_prob'] = max(0, 1 - (2 / epochs) * epoch)
        elif teacher_schedule == 'exponential':
            kwargs['teacher_prob'] = 0.9 ** epoch
        elif teacher_schedule == 'inverse_sigmoid':
            kwargs['teacher_prob'] = 10 / (10 + math.exp(3 * epoch / 10))
        elif teacher_schedule == 'random':
            kwargs['teacher_prob'] = 0.5
        elif teacher_schedule == 'always':
            kwargs['teacher_prob'] = 1.0
        else:
            kwargs['teacher_prob'] = 0.0


def test(model, data_loader, criterion, device, fetch_model_data=single_input_single_output,
         feed_model_data=basic_forward, loss_names=None, test_set_name='Test', mtll_model=None,
         print_raw_losses=False, num_main_losses=None, **kwargs):
    """General testing function to test a PyTorch model.

    Arg(s):
        model - PyTorch model.
        data_loader - Batch generator for model testing.
        criterion - Specific loss function for the given model. This function receives as input the output of
            model and the ground-truth target, and returns a list of batch losses (for multi-loss models). Even if
            the model has a single loss, the return value of criterion must be a list containing this single loss.
        device - Which device to use for model testing. Either cuda or cpu.
        fetch_model_data - Function to fetch the input and output tensors for the model.
        feed_model_data - Function to feed the input tensors to the model.
        loss_names - Names for the individual losses output by criterion. If None, the losses are named loss_1,
            loss_2, ....
        test_set_name - Optional name given to the set being evaluated. Useful for logging purposes.
        num_main_losses - The final test loss is the sum of all non-auxiliary losses. Auxiliary losses should be in
            the beginning of the output list.
        **kwargs - Any extra parameters that need to be passed to the feed_model_data function.
    Returns:
        The model loss.
    """
    model.eval()
    if mtll_model is not None:
        mtll_model.eval()
    test_raw_losses = None
    test_losses = None
    with torch.no_grad():
        for dataset in data_loader:
            data, target = fetch_model_data(dataset, device=device)
            output = feed_model_data(model, data, **kwargs)
            raw_losses = criterion(output, target, reduction='mean')
            if mtll_model is not None:
                if test_raw_losses is None:
                    test_raw_losses = [raw_loss.item() for raw_loss in raw_losses]
                else:
                    test_raw_losses = [test_raw_loss + raw_loss.item()
                                       for test_raw_loss, raw_loss in zip(test_raw_losses, raw_losses)]
                losses = mtll_model(raw_losses)
            else:
                losses = raw_losses
            if test_losses is None:
                test_losses = [loss.item() for loss in losses]
            else:
                test_losses = [test_loss + loss.item() for test_loss, loss in zip(test_losses, losses)]
    num_main_losses = num_main_losses if num_main_losses is not None else len(test_losses)
    test_losses = [test_loss / len(data_loader) for test_loss in test_losses][-num_main_losses:]
    total_test_loss = sum(test_losses)

    name_fmt_str = '({})'
    loss_fmt_str = 'Loss: {: 7.4f}'
    print(name_fmt_str.format(test_set_name).rjust(12, ' '), loss_fmt_str.format(total_test_loss), end='')
    loss_names = loss_names[-num_main_losses:] if loss_names is not None else ['loss_' + str(n) for n in range(1, 101)]
    for loss_name, loss in zip(loss_names, test_losses):
        print('  ', loss_name + ':', '{: 6.4f}'.format(loss), end='')
    print()
    total_test_raw_loss = None
    if test_raw_losses is not None:
        test_raw_losses = [test_raw_loss / len(data_loader) for test_raw_loss in test_raw_losses][-num_main_losses:]
        total_test_raw_loss = sum(test_raw_losses)
        if print_raw_losses:
            name_fmt_str = '({})'
            loss_fmt_str = 'Loss: {: 7.4f}'
            print(name_fmt_str.format(test_set_name).rjust(12, ' '), loss_fmt_str.format(total_test_raw_loss), end='')
            loss_names = loss_names[-num_main_losses:] if loss_names is not None else ['loss_' + str(n)
                                                                                       for n in range(1, 101)]
            for loss_name, raw_loss in zip(loss_names, test_raw_losses):
                print('  ', loss_name + ':', '{: 6.4f}'.format(raw_loss), end='')
            print()
    return total_test_loss, test_losses, total_test_raw_loss, test_raw_losses


def normalise(x, strategy='standard', with_centering=True, quantile_range=(25.0, 75.0), scaler=None):
    """Normalises input n-dimensional tensor according to selected strategy.

    Arg(s):
        x - n-dimensional ndarray to normalise.
        strategy - One of standard, min_max, or robust.
        scaler - If given, ignores the selected strategy and uses the given scaler to normalise the input tensor.
            Otherwise, creates a scaler from the selected strategy and normalises the input tensor.
    Returns:
        The normalised input tensor and scaler used.
    """
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1])
    if scaler is not None:
        x = scaler.transform(x)
    else:
        scaler = select_scaler(strategy, with_centering=with_centering, quantile_range=quantile_range)
        x = scaler.fit_transform(x)
    x = x.reshape(*x_shape)
    return x, scaler


def nan_to_value(x, value, inplace=True):
    """Transform any NaN entry in x to value.

    Arg(s):
        x - An ndarray.
        value - A value to substitute NaNs in x for.
        inplace - Whether to perform the substitution in place or not.
    Returns:
        The input ndarray with NaN values transformed into value.
    """
    if not inplace:
        x = np.copy(x)
    x[np.isnan(x)] = value
    return x


def select_scaler(strategy, with_centering=True, quantile_range=(25.0, 75.0)):
    """Returns an instance of a *Scaler, selected according to input strategy.

    Arg(s):
        strategy - One of standard, min_max, or robust.
        with_centering - If True, subtract the mean of the data from the data in case strategy is standard, or
            subtract the median of the data from the data in case strategy is robust.
        quantile_range - In case the strategy is robust, divide the data by the this quantile range.
    Returns:
        An instance of a *Scaler. * is one of Standard, MinMax, or Robust.
    """
    assert strategy in {'standard', 'min_max', 'robust'}, 'strategy must be one of: standard, min_max, or robust.'
    scalers = {'standard': StandardScaler(with_mean=with_centering),
               'min_max': MinMaxScaler(),
               'robust': RobustScaler(with_centering=with_centering, quantile_range=quantile_range),
               }
    scaler = scalers[strategy]
    return scaler


def numpy_to_torch(*arrays, device='cpu'):
    """Convert any number of numpy arrays to PyTorch tensors."""
    return [torch.from_numpy(array).to(device) for array in arrays]


def save_checkpoint(log_dir, checkpoint, checkpoint_name=None, include_timestamp=True):
    """Save model checkpoint.

    Arg(s):
        log_dir - Directory to save checkpoint file.
        checkpoint - A dictionary containing the model checkpoint and other metadata such as data scalers and
            model creation arguments.
        checkpoint_name - If given, use that as the file name to save. Otherwise, the file name is 'checkpoint'.
    """
    file_save_name = checkpoint_name if checkpoint_name is not None else 'checkpoint'
    if include_timestamp:
        time_now = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        file_save_name = time_now + '_' + file_save_name
    file_save_name += '.tar'
    file_save_path = os.path.join(log_dir, file_save_name)
    torch.save(checkpoint, file_save_path)
    print('log files written to %s' % file_save_path)


def grab_subset(*args, n=5):
    """Grab n examples from tensors in a list.

    We assume that the first dimension of the tensor is the batch dimension.

    Arg(s):
        args - List of tensors to grab a subset of each of them.
        n - Number of examples to grab from the tensors.
    Returns:
        A list of tensors, where each tensor contains only a number of examples from the original tensor.
    """
    return [tensor[:n] for tensor in args]


def create_alias(hidden_size, epochs, batch_size, input_seq_len, output_seq_len, length_activation,
                 learning_rate=None, transition_learning_rate=None, nc=None, embedding_size=None,
                 loss_weights=None, teacher_schedule=None, validation_data=None, l2_reg=0.0,
                 multi_task_loss_learner=False, num_layers=1, normalisation=None, quantile_range=None,
                 optimizer=None, input_normalisation=None, obs_at_least_k_percent=None, share_encoder_decoder=False,
                 share_embeddings=False, share_predictions=False, disable_parent_input=None, disable_encoder_loss=None,
                 embedding_nonlinearity=None, mask_softmax=False, positional_embedding=False,
                 add_skip_connection=False, weight_initialisation='pytorch', clip_gradient_at=5.0,
                 use_plain_gru_cell=None, disable_transition_layer=False, use_hmgruv2_cell=False,
                 disable_gradient_from_child=False, use_lstm_cell=False, weight_decay_decoder_only=False,
                 pretrain_coarse=0, model_v2=False, model_v3=False, do_not_reset_after_flush=False,
                 always_include_parent_state=False, with_final_action=False,
                 baseline_type=None, action_level=None, test_mode=False, input_soft_parent=False):
    checkpoint_name = 'hs'
    if isinstance(hidden_size, collections.abc.Sequence):
        for hs in hidden_size:
            checkpoint_name += '-' + str(hs)
    else:
        checkpoint_name += str(hidden_size)
    checkpoint_name += '_' + str(epochs) + 'e_' + 'bs' + str(batch_size)
    if isinstance(length_activation, collections.abc.Sequence) and not isinstance(length_activation, str):
        if length_activation[0] == length_activation[1]:
            length_activation = length_activation[:1]
        checkpoint_name += '_act'
        for la in length_activation:
            checkpoint_name += '-' + la
    else:
        checkpoint_name += '_act-' + length_activation
    if pretrain_coarse:
        checkpoint_name += '_pce' + str(pretrain_coarse)
    if num_layers:
        checkpoint_name += '_h' + str(num_layers)
    if learning_rate is not None:
        checkpoint_name += '_lr' + '{:.0e}'.format(learning_rate)
    if transition_learning_rate is not None:
        checkpoint_name += '_tlr' + '{:.0e}'.format(transition_learning_rate)
    if optimizer is not None:
        checkpoint_name += '_opt-' + str(optimizer)
    if share_embeddings:
        checkpoint_name += '_sh-emb'
    if share_encoder_decoder:
        checkpoint_name += '_sh-ed'
    if share_predictions:
        checkpoint_name += '_sh-pred'
    if embedding_size is not None:
        checkpoint_name += '_es'
        if isinstance(embedding_size, collections.abc.Sequence):
            for es in embedding_size:
                if isinstance(es, collections.abc.Sequence):
                    checkpoint_name += '-' + str(max(es))
                else:
                    checkpoint_name += '-' + str(es)
        else:
            checkpoint_name += str(embedding_size)
        if embedding_nonlinearity is not None:
            checkpoint_name += '_' + embedding_nonlinearity
        if positional_embedding:
            checkpoint_name += '_pos-emb'
    if l2_reg:
        checkpoint_name += '_l2r' + '{:.0e}'.format(l2_reg)
        if weight_decay_decoder_only:
            checkpoint_name += '-dec'
    if teacher_schedule is not None:
        checkpoint_name += '_ts' + teacher_schedule
    if multi_task_loss_learner:
        checkpoint_name += '_mtll'
    if input_normalisation is not None:
        checkpoint_name += '_istd-' + str(input_normalisation)
        if input_normalisation == 'robust' and quantile_range is not None:
            for qr in quantile_range:
                checkpoint_name += '-' + str(qr)
    if normalisation is not None:
        if normalisation == 'robust':
            checkpoint_name += '_robust'
            if quantile_range is not None:
                for qr in quantile_range:
                    checkpoint_name += '-' + str(qr)
        elif normalisation == 'min_max':
            checkpoint_name += '_min-max'
        elif normalisation == 'standard':
            checkpoint_name += '_std'
    if disable_parent_input is not None:
        if disable_parent_input:
            checkpoint_name += '_rm-par-inp'
    if disable_encoder_loss is not None:
        if disable_encoder_loss:
            checkpoint_name += '_rm-enc-loss'
    if mask_softmax:
        checkpoint_name += '_mask-sm'
    if add_skip_connection:
        checkpoint_name += '_skip-con'
    if weight_initialisation != 'pytorch':
        checkpoint_name += '_winit-' + weight_initialisation
    if clip_gradient_at != 5.0:
        checkpoint_name += '_gclip-' + str(clip_gradient_at)
    if use_plain_gru_cell is not None and use_plain_gru_cell:
        checkpoint_name += '_use-gru'
    elif use_hmgruv2_cell:
        checkpoint_name += '_use-hmgruv2'
    elif use_lstm_cell:
        checkpoint_name += '_use-lstm'
    if disable_transition_layer:
        checkpoint_name += '_rm-tl'
    if disable_gradient_from_child:
        checkpoint_name += '_rm-childgrad'
    if model_v2:
        checkpoint_name += '_mv2'
    elif model_v3:
        checkpoint_name += '_mv3'
        if input_soft_parent:
            checkpoint_name += '_isp'
    if do_not_reset_after_flush:
        checkpoint_name += '_nrh'
    if always_include_parent_state:
        checkpoint_name += '_aips'
    if with_final_action:
        checkpoint_name += '_wfa'
    if baseline_type is not None:
        checkpoint_name += '_bt' + str(baseline_type)
        if baseline_type == 0 and action_level is not None:
            checkpoint_name += action_level
        if test_mode:
            checkpoint_name += '_tm'
    if loss_weights is not None:
        checkpoint_name += '_lw'
        for loss_w in loss_weights:
            checkpoint_name += '-' + str(loss_w)
    checkpoint_name += '_isq' + str(input_seq_len) + '_osq' + str(output_seq_len)
    if nc is not None:
        checkpoint_name += '_nc' + str(nc)
    if obs_at_least_k_percent is not None:
        checkpoint_name += '_atleast' + str(obs_at_least_k_percent)
    if validation_data is not None:
        checkpoint_name += '_with-val'
    return checkpoint_name


def extract_info_from_str(training_data):
    avg_num_actions_per_video, nc, fa, obs_at_least = None, None, None, None
    avg_idx = training_data.find('avg')
    if avg_idx != -1:
        start_idx = avg_idx + 3
        sub_str = training_data[start_idx:]
        sub_str = list(takewhile(lambda x: x not in {'_', '.'}, sub_str))
        avg_num_actions_per_video = ''.join(sub_str)
    nc_idx = training_data.find('nc')
    if nc_idx != -1:
        start_idx = nc_idx + 2
        sub_str = training_data[start_idx:]
        sub_str = list(takewhile(lambda x: x not in {'_', '.'}, sub_str))
        nc = ''.join(sub_str)
    fa_idx = training_data.find('_fa')
    if fa_idx != -1:
        start_idx = fa_idx + 3
        sub_str = training_data[start_idx:]
        sub_str = list(takewhile(lambda x: x not in {'_', '.'}, sub_str))
        fa = ''.join(sub_str)
    obs_at_least_idx = training_data.find('_atleast')
    if obs_at_least_idx != -1:
        start_idx = obs_at_least_idx + len('_atleast')
        sub_str = training_data[start_idx:]
        sub_str = list(takewhile(lambda x: x not in {'_', '.'}, sub_str))
        obs_at_least = ''.join(sub_str)
    return avg_num_actions_per_video, nc, fa, obs_at_least


def one_hot_to_index(x):
    original_shape = x.shape
    x = x.reshape(-1, original_shape[-1])
    na_indices = np.nansum(x, axis=-1, keepdims=True)
    na_indices[na_indices == 0.0] = np.nan
    x = nan_to_value(x, value=-1.0)
    x_new = np.nanargmax(x, axis=-1).reshape(-1, 1)
    x_new = x_new * na_indices
    x_new = x_new.reshape(*original_shape[:-1], 1)
    return x_new


def num_workers_from_batch_size(batch_size):
    """Select number of workers for PyTorch's DataLoader class according to the batch size.

    Arg(s):
        batch_size - Batch size for model training.
    Returns:
        The number of workers for PyTorch's DataLoader class according to the input batch size.
    """
    num_workers = 0
    if 64 < batch_size <= 512:
        num_workers = 2
    elif 512 < batch_size < 2048:
        num_workers = 4
    elif 2048 <= batch_size:
        num_workers = 8
    return num_workers


def set_initial_teacher_prob(teacher_schedule):
    """Set initial probability for teacher forcing depending on the schedule.

    Arg(s):
        teacher_schedule - String containing the name of the schedule.
    Returns:
        The initial probability according to the input schedule.
    """
    if teacher_schedule == 'inverse_sigmoid':
        return 0.9
    elif teacher_schedule == 'random':
        return 0.5
    return 1.0


def maybe_denormalise(y, scaler):
    y_shape = y.shape
    y = y.reshape(-1, y_shape[-1])
    if scaler is not None:
        y = scaler.inverse_transform(y)
    y = y.reshape(*y_shape)
    return y


def logit2one_hot(y_dec_logits):
    """Translate a tensor of logits to a one-hot representation.

    Given a tensor of shape (batch_size, num_categories) containing the logits of each category, create a one-hot
    tensor of shape (batch_size, num_categories).

    Arg(s):
        y_dec_logits - Tensor of shape (batch_size, num_categories).
    Returns:
        A one-hot representation of y_dec_logits.
    """
    y_dec_cat = torch.zeros_like(y_dec_logits)
    indices = torch.argmax(y_dec_logits, dim=-1).long()
    y_dec_cat[range(y_dec_cat.size(0)), indices] = 1.0
    return y_dec_cat
