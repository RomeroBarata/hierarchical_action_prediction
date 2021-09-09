from collections import defaultdict
from itertools import accumulate, zip_longest
import os

import numpy as np
import torch

from fpua.analysis import analyse_hierarchical_observations_and_predictions, analyse_flushes_hierarchical
from fpua.analysis import compute_metrics
from fpua.analysis import analyse_performance_per_future_action, write_results_per_video, compute_moc
from fpua.data.general import read_action_dictionary, extend_smallest_list, split_observed_actions, actions_from_steps
from fpua.data.general import maybe_rebalance_steps
from fpua.data.general import extend_or_trim_predicted_actions, aggregate_actions_and_lengths
from fpua.data.general import extract_last_action_and_observed_length, extract_last_action_and_full_length
from fpua.data.hera import get_local_acc_len
from fpua.models.misc import next_action_info, compute_steps_to_grab, scale_num_part
from fpua.models.misc import compute_ground_truth_flushes
from fpua.models.hera import HERA, compute_zero_layer_boundary
from fpua.models.rnn import HMLSTM
from fpua.utils import nan_to_value, numpy_to_torch, maybe_denormalise, logit2one_hot


def test_hera(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location=device)
    fine_labels_path = args.fine_labels_path
    coarse_labels_path = args.coarse_labels_path
    fine_action_to_id = read_action_dictionary(args.fine_action_to_id)
    fine_id_to_action = {action_id: action for action, action_id in fine_action_to_id.items()}
    coarse_action_to_id = read_action_dictionary(args.coarse_action_to_id)
    coarse_id_to_action = {action_id: action for action, action_id in coarse_action_to_id.items()}
    fraction_observed = args.observed_fraction
    ignore_silence_action = args.ignore_silence_action
    do_error_analysis = args.do_error_analysis
    do_future_performance_analysis = args.do_future_performance_analysis
    do_flush_analysis = args.do_flush_analysis
    input_seq_len = checkpoint['input_seq_len']
    scalers = checkpoint.get('scalers', None)
    disable_parent_input = checkpoint['disable_parent_input']
    # Load model
    model = HERA(**checkpoint['model_creation_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    observed_fine_actions_per_video, observed_coarse_actions_per_video = [], []
    fine_transition_action_per_video, coarse_transition_action_per_video = [], []
    flushes_per_video, ground_truth_flushes_per_video = [], []
    predicted_fine_actions_per_video, predicted_coarse_actions_per_video = [], []
    predicted_fine_steps_per_video, predicted_coarse_steps_per_video = [], []
    unobserved_fine_actions_per_video, unobserved_coarse_actions_per_video = [], []
    fine_label_files = set(os.listdir(fine_labels_path))
    coarse_label_files = set(os.listdir(coarse_labels_path))
    label_files = sorted(fine_label_files & coarse_label_files)
    for label_file in label_files:
        with open(os.path.join(fine_labels_path, label_file), mode='r') as f:
            fine_actions_per_frame = [line.rstrip() for line in f]
        with open(os.path.join(coarse_labels_path, label_file), mode='r') as f:
            coarse_actions_per_frame = [line.rstrip() for line in f]
        if ignore_silence_action is not None:
            fine_actions_per_frame = [fine_action for fine_action in fine_actions_per_frame
                                      if fine_action != ignore_silence_action]
            coarse_actions_per_frame = [coarse_action for coarse_action in coarse_actions_per_frame
                                        if coarse_action != ignore_silence_action]
        fine_actions_per_frame, coarse_actions_per_frame = \
            extend_smallest_list(fine_actions_per_frame, coarse_actions_per_frame)
        observed_fine_actions, unobserved_fine_actions = split_observed_actions(fine_actions_per_frame,
                                                                                fraction_observed=fraction_observed)
        observed_fine_actions_per_video.append(observed_fine_actions)
        fine_transition_action_per_video.append(observed_fine_actions[-1])
        observed_coarse_actions, unobserved_coarse_actions = split_observed_actions(coarse_actions_per_frame,
                                                                                    fraction_observed=fraction_observed)
        observed_coarse_actions_per_video.append(observed_coarse_actions)
        coarse_transition_action_per_video.append(observed_coarse_actions[-1])
        tensors = generate_test_datum(observed_fine_actions, observed_coarse_actions, input_seq_len=input_seq_len,
                                      fine_action_to_id=fine_action_to_id, coarse_action_to_id=coarse_action_to_id,
                                      disable_parent_input=disable_parent_input,
                                      num_frames=len(fine_actions_per_frame), scalers=scalers, coarse_is_complete=False)
        tensors = [nan_to_value(tensor, value=0.0) for tensor in tensors]
        tensors = numpy_to_torch(*tensors, device=device)
        predicted_actions, predicted_steps, dx_dec_fine = \
            predict_future_actions(model, tensors, fine_id_to_action=fine_id_to_action,
                                   coarse_id_to_action=coarse_id_to_action,
                                   disable_parent_input=disable_parent_input,
                                   num_frames=len(fine_actions_per_frame),
                                   maximum_prediction_length=len(unobserved_fine_actions),
                                   observed_fine_actions=observed_fine_actions,
                                   observed_coarse_actions=observed_coarse_actions,
                                   fine_action_to_id=fine_action_to_id, coarse_action_to_id=coarse_action_to_id,
                                   scalers=scalers)
        flushes_per_video.append(dx_dec_fine)
        ground_truth_flushes = compute_ground_truth_flushes(observed_coarse_actions[-1], observed_fine_actions[-1],
                                                            unobserved_coarse_actions, unobserved_fine_actions)
        ground_truth_flushes_per_video.append(ground_truth_flushes)
        predicted_fine_steps, predicted_coarse_steps = predicted_steps
        predicted_fine_steps_per_video.append(predicted_fine_steps)
        predicted_coarse_steps_per_video.append(predicted_coarse_steps)
        predicted_fine_actions, predicted_coarse_actions = predicted_actions
        if not predicted_fine_actions:
            predicted_fine_actions = ['FAILED_TO_PREDICT']
        predicted_fine_actions = extend_or_trim_predicted_actions(predicted_fine_actions, unobserved_fine_actions)
        predicted_fine_actions = np.array(predicted_fine_actions)
        predicted_fine_actions_per_video.append(predicted_fine_actions)
        unobserved_fine_actions = np.array(unobserved_fine_actions)
        unobserved_fine_actions_per_video.append(unobserved_fine_actions)
        if not predicted_coarse_actions:
            predicted_coarse_actions = ['FAILED_TO_PREDICT']
        predicted_coarse_actions = extend_or_trim_predicted_actions(predicted_coarse_actions, unobserved_coarse_actions)
        predicted_coarse_actions = np.array(predicted_coarse_actions)
        predicted_coarse_actions_per_video.append(predicted_coarse_actions)
        unobserved_coarse_actions = np.array(unobserved_coarse_actions)
        unobserved_coarse_actions_per_video.append(unobserved_coarse_actions)
    # Performance and Error Analysis
    f1_results_dict = {}
    moc_results_dict = {}
    unobserved_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
    unobserved_fractions = [unobserved_fraction for unobserved_fraction in unobserved_fractions
                            if fraction_observed + unobserved_fraction <= 1.0]
    for unobserved_fraction in unobserved_fractions:
        save_analysis_path = os.path.join(args.checkpoint[:-4], str(fraction_observed) + '_' + str(unobserved_fraction))
        global_fraction_unobserved = 1.0 - fraction_observed
        predicted_fine_actions_per_video_sub, unobserved_fine_actions_per_video_sub = [], []
        predicted_coarse_actions_per_video_sub, unobserved_coarse_actions_per_video_sub = [], []
        f1_per_video = []  # file_name, coarse_0.5_f1, fine_0.5_f1
        for i, (predicted_fine_actions, unobserved_fine_actions, predicted_coarse_actions, unobserved_coarse_actions) in \
                enumerate(zip(predicted_fine_actions_per_video, unobserved_fine_actions_per_video,
                              predicted_coarse_actions_per_video, unobserved_coarse_actions_per_video)):
            num_frames_to_grab = (len(unobserved_fine_actions) / global_fraction_unobserved) * unobserved_fraction
            num_frames_to_grab = round(num_frames_to_grab)
            predicted_fine_actions_sub = predicted_fine_actions[:num_frames_to_grab]
            predicted_fine_actions_per_video_sub.append(predicted_fine_actions_sub)
            unobserved_fine_actions_sub = unobserved_fine_actions[:num_frames_to_grab]
            unobserved_fine_actions_per_video_sub.append(unobserved_fine_actions_sub)
            predicted_coarse_actions_sub = predicted_coarse_actions[:num_frames_to_grab]
            predicted_coarse_actions_per_video_sub.append(predicted_coarse_actions_sub)
            unobserved_coarse_actions_sub = unobserved_coarse_actions[:num_frames_to_grab]
            unobserved_coarse_actions_per_video_sub.append(unobserved_coarse_actions_sub)
            if do_error_analysis:
                predicted_fine_steps = predicted_fine_steps_per_video[i]
                steps_to_grab = compute_steps_to_grab(predicted_fine_steps, num_frames_to_grab)
                predicted_fine_steps = predicted_fine_steps[:steps_to_grab]
                predicted_coarse_steps = predicted_coarse_steps_per_video[i][:steps_to_grab]
                coarse_actions_per_frame = (observed_coarse_actions_per_video[i] +
                                            unobserved_coarse_actions_per_video[i].tolist())
                analyse_hierarchical_observations_and_predictions(predicted_fine_steps,
                                                                  predicted_coarse_steps,
                                                                  observed_fine_actions_per_video[i],
                                                                  observed_coarse_actions_per_video[i],
                                                                  unobserved_fine_actions_sub,
                                                                  unobserved_coarse_actions_sub,
                                                                  coarse_actions_per_frame_full=coarse_actions_per_frame,
                                                                  save_path=save_analysis_path,
                                                                  save_file_name=label_files[i])
                _, f1_fine_scores = compute_metrics([predicted_fine_actions_sub],
                                                    [unobserved_fine_actions_sub],
                                                    action_to_id=fine_action_to_id)
                _, f1_coarse_scores = compute_metrics([predicted_coarse_actions_sub],
                                                      [unobserved_coarse_actions_sub],
                                                      action_to_id=coarse_action_to_id)
                f1_per_video.append([label_files[i], f1_coarse_scores[-1], f1_fine_scores[-1]])
        if do_error_analysis:
            write_results_per_video(f1_per_video, order_by='coarse', metric_name='f1-0.5', save_path=save_analysis_path)
            write_results_per_video(f1_per_video, order_by='fine', metric_name='f1-0.5', save_path=save_analysis_path)
        if do_future_performance_analysis:
            analyse_performance_per_future_action(predicted_coarse_actions_per_video_sub,
                                                  unobserved_coarse_actions_per_video_sub,
                                                  transition_action_per_video=coarse_transition_action_per_video,
                                                  save_path=save_analysis_path, extra_str='Coarse')
            analyse_performance_per_future_action(predicted_fine_actions_per_video_sub,
                                                  unobserved_fine_actions_per_video_sub,
                                                  transition_action_per_video=fine_transition_action_per_video,
                                                  save_path=save_analysis_path, mode='a', extra_str='Fine')
        print('\nObserved fraction: %.2f | Unobserved fraction: %.2f' % (fraction_observed, unobserved_fraction))
        print('-> Fine')
        overlaps, f1_overlap_scores = compute_metrics(predicted_fine_actions_per_video_sub,
                                                      unobserved_fine_actions_per_video_sub,
                                                      action_to_id=fine_action_to_id)
        for overlap, overlap_f1_score in zip(overlaps, f1_overlap_scores):
            print('F1@%.2f: %.4f' % (overlap, overlap_f1_score))
            f1_results_dict[f'fine-{fraction_observed}_{unobserved_fraction}_{overlap}'] = overlap_f1_score
        fine_moc, _, _ = compute_moc(np.concatenate(predicted_fine_actions_per_video_sub),
                                     np.concatenate(unobserved_fine_actions_per_video_sub),
                                     action_to_id=fine_action_to_id)
        print(f'MoC: {fine_moc:.4f}')
        moc_results_dict[f'fine-moc-{fraction_observed}_{unobserved_fraction}'] = fine_moc
        print('-> Coarse')
        overlaps, f1_overlap_scores = compute_metrics(predicted_coarse_actions_per_video_sub,
                                                      unobserved_coarse_actions_per_video_sub,
                                                      action_to_id=coarse_action_to_id)
        for overlap, overlap_f1_score in zip(overlaps, f1_overlap_scores):
            print('F1@%.2f: %.4f' % (overlap, overlap_f1_score))
            f1_results_dict[f'coarse-{fraction_observed}_{unobserved_fraction}_{overlap}'] = overlap_f1_score
        coarse_moc, _, _ = compute_moc(np.concatenate(predicted_coarse_actions_per_video_sub),
                                       np.concatenate(unobserved_coarse_actions_per_video_sub),
                                       action_to_id=coarse_action_to_id)
        print(f'MoC: {coarse_moc:.4f}')
        moc_results_dict[f'coarse-moc-{fraction_observed}_{unobserved_fraction}'] = coarse_moc
    if do_flush_analysis:
        analyse_flushes_hierarchical(flushes_per_video, ground_truth_flushes_per_video,
                                     label_files, model.decoder_net.output_seq_len,
                                     save_path=args.checkpoint[:-4], encoder=False)
    results_dict = {**f1_results_dict, **moc_results_dict}
    return results_dict


def generate_test_datum(observed_fine_actions, observed_coarse_actions, input_seq_len, fine_action_to_id,
                        coarse_action_to_id, disable_parent_input, num_frames, scalers, coarse_is_complete=False):
    actions, lengths = aggregate_actions_and_lengths(list(zip(observed_coarse_actions, observed_fine_actions)))
    acc_lengths = list(accumulate(lengths))
    num_fine_actions, num_coarse_actions = len(fine_action_to_id), len(coarse_action_to_id)
    # Transition
    if disable_parent_input:
        x_tra_fine = np.full([1, num_fine_actions + 1 + 1], fill_value=np.nan, dtype=np.float32)
    else:
        x_tra_fine = np.full([1, num_coarse_actions + num_fine_actions + 1 + 1], fill_value=np.nan, dtype=np.float32)
    x_tra_coarse = np.full([1, num_coarse_actions + 1 + 1], fill_value=np.nan, dtype=np.float32)

    coarse_tra_action = actions[-1][0]
    coarse_tra_action_id = coarse_action_to_id[coarse_tra_action]
    x_tra_coarse[0, coarse_tra_action_id] = 1.0
    x_tra_coarse[0, -2] = acc_lengths[-1] / num_frames
    last_obs_frame = acc_lengths[-1] - 1
    coarse_tra_action_obs_len = extract_last_action_and_observed_length(observed_coarse_actions, last_obs_frame)[1]
    x_tra_coarse[0, -1] = coarse_tra_action_obs_len / num_frames

    fine_tra_action = actions[-1][1]
    fine_tra_action_id = fine_action_to_id[fine_tra_action]
    if disable_parent_input:
        x_tra_fine[0, fine_tra_action_id] = 1.0
    else:
        x_tra_fine[0, coarse_tra_action_id] = 1.0
        x_tra_fine[0, num_coarse_actions + fine_tra_action_id] = 1.0
    ext_actions, ext_lengths = aggregate_actions_and_lengths(list(zip_longest(observed_coarse_actions,
                                                                              observed_fine_actions,
                                                                              fillvalue='FAKE_ACTION')))
    if coarse_is_complete:
        coarse_tra_action_len = extract_last_action_and_full_length(observed_coarse_actions, last_obs_frame)[1]
        x_tra_fine[0, -2] = get_local_acc_len(ext_actions, ext_lengths, len(ext_lengths) - 2) / coarse_tra_action_len
        fine_tra_action_obs_len = lengths[-1]
        x_tra_fine[0, -1] = fine_tra_action_obs_len / coarse_tra_action_len
    # Encoder
    if disable_parent_input:
        x_enc_fine = np.full([1, input_seq_len, num_fine_actions + 1], fill_value=np.nan, dtype=np.float32)
    else:
        x_enc_fine = np.full([1, input_seq_len, num_coarse_actions + num_fine_actions + 1],
                             fill_value=np.nan, dtype=np.float32)
    x_enc_coarse = np.full([1, input_seq_len, num_coarse_actions + 1], fill_value=np.nan, dtype=np.float32)
    enc_boundaries = np.full([1, 2, input_seq_len], fill_value=np.nan, dtype=np.float32)
    for j, m in zip(range(len(actions) - 2, -1, -1), range(-1, -input_seq_len - 1, -1)):
        past_coarse_action, current_coarse_action = actions[j][0], actions[j + 1][0]
        past_coarse_action_id = coarse_action_to_id[past_coarse_action]
        if past_coarse_action != current_coarse_action:
            x_enc_coarse[0, m, past_coarse_action_id] = 1.0
            x_enc_coarse[0, m, -1] = acc_lengths[j] / num_frames
            enc_boundaries[0, 0, m] = 1.0
        else:
            enc_boundaries[0, 0, m] = 0.0

        past_fine_action = actions[j][1]
        past_fine_action_id = fine_action_to_id[past_fine_action]
        if disable_parent_input:
            x_enc_fine[0, m, past_fine_action_id] = 1.0
        else:
            x_enc_fine[0, m, past_coarse_action_id] = 1.0
            x_enc_fine[0, m, num_coarse_actions + past_fine_action_id] = 1.0
        past_coarse_action_len = extract_last_action_and_full_length(observed_coarse_actions, acc_lengths[j] - 1)[1]
        local_acc_len = get_local_acc_len(ext_actions, ext_lengths, j)
        x_enc_fine[0, m, -1] = local_acc_len / past_coarse_action_len
    enc_layer_zero_boundaries = compute_zero_layer_boundary(np.copy(x_enc_fine))
    if scalers is not None:
        x_tra_fine = scale_num_part(x_tra_fine, scalers, scaler_name='x_tra_fine_scaler')
        x_tra_coarse = scale_num_part(x_tra_coarse, scalers, scaler_name='x_tra_coarse_scaler')
    return x_enc_fine, x_enc_coarse, enc_boundaries, enc_layer_zero_boundaries, x_tra_fine, x_tra_coarse


def predict_future_actions(model, input_tensors, fine_id_to_action, coarse_id_to_action, disable_parent_input,
                           num_frames, maximum_prediction_length, observed_fine_actions, observed_coarse_actions,
                           fine_action_to_id, coarse_action_to_id, scalers=None):
    x_enc_fine, x_enc_coarse, dx_enc, dx_enc_layer_zero, x_tra_fine, x_tra_coarse = input_tensors
    dx = [dx_enc, dx_enc_layer_zero]
    with torch.no_grad():
        _, hx = model.encoder_net(x_enc_fine, x_enc_coarse, dx=dx, hx=None)
        hx_tra = [hl[0] for hl in hx] if isinstance(model.encoder_net.encoder_hmgru, HMLSTM) else hx
        (_, y_tra_coarse_rem_prop), _ = model.transition_net(x_tra_fine, x_tra_coarse, hx=hx_tra)
    coarse_la_id = torch.argmax(x_tra_coarse[..., :-2], dim=-1).item()
    coarse_la_label = coarse_id_to_action[coarse_la_id]
    y_tra_coarse_rem_prop = maybe_denormalise(y_tra_coarse_rem_prop.cpu().numpy(),
                                              scaler=scalers.get('y_tra_coarse_scaler'))
    coarse_la_rem_len = round(y_tra_coarse_rem_prop.item() * num_frames)
    predicted_coarse_actions = [coarse_la_label] * coarse_la_rem_len
    predicted_coarse_steps = [(coarse_la_label, coarse_la_rem_len)]

    # Generate input tensors again.
    new_observed_coarse_actions = observed_coarse_actions + predicted_coarse_actions
    input_seq_len = x_enc_fine.size(1)
    input_tensors = generate_test_datum(observed_fine_actions, new_observed_coarse_actions, input_seq_len=input_seq_len,
                                        fine_action_to_id=fine_action_to_id, coarse_action_to_id=coarse_action_to_id,
                                        disable_parent_input=disable_parent_input,
                                        num_frames=num_frames, scalers=scalers, coarse_is_complete=True)
    input_tensors = [nan_to_value(tensor, value=0.0) for tensor in input_tensors]
    input_tensors = numpy_to_torch(*input_tensors, device=x_enc_fine.device)
    x_enc_fine, x_enc_coarse, dx_enc, dx_enc_layer_zero, x_tra_fine, x_tra_coarse = input_tensors
    dx = [dx_enc, dx_enc_layer_zero]
    with torch.no_grad():
        _, hx, hxs = model.encoder_net(x_enc_fine, x_enc_coarse, dx=dx, hx=None, return_all_hidden_states=True)
        hx_tra = [hl[0] for hl in hx] if isinstance(model.encoder_net.encoder_hmgru, HMLSTM) else hx
        (y_tra_fine_rem_rel_prop, _), hx_tra = model.transition_net(x_tra_fine, x_tra_coarse, hx=hx_tra)
        try:
            if not model.disable_transition_layer:
                if isinstance(model.encoder_net.encoder_hmgru, HMLSTM):
                    for i, hl in enumerate(hx_tra):
                        hx[i][0] = hl
                else:
                    hx = hx_tra
                hxs[0] = torch.cat([hxs[0], hx_tra[0].unsqueeze(1)], dim=1)
                hxs[1] = torch.cat([hxs[1], hx_tra[1].unsqueeze(1)], dim=1)
        except AttributeError:
            if isinstance(model.encoder_net.encoder_hmgru, HMLSTM):
                for i, hl in enumerate(hx_tra):
                    hx[i][0] = hl
            else:
                hx = hx_tra
            hxs[0] = torch.cat([hxs[0], hx_tra[0].unsqueeze(1)], dim=1)
            hxs[1] = torch.cat([hxs[1], hx_tra[1].unsqueeze(1)], dim=1)

    num_coarse_actions = len(coarse_action_to_id)
    if disable_parent_input:
        fine_la_id = torch.argmax(x_tra_fine[..., :-2], dim=-1).item()
    else:
        fine_la_id = torch.argmax(x_tra_fine[..., num_coarse_actions:-2], dim=-1).item()
    fine_la_label = fine_id_to_action[fine_la_id]
    y_tra_fine_rem_rel_prop = maybe_denormalise(y_tra_fine_rem_rel_prop.cpu().numpy(),
                                                scaler=scalers.get('y_tra_fine_scaler'))
    coarse_tra_len_prop = x_tra_coarse[..., -1].item() + y_tra_coarse_rem_prop.item()
    fine_la_rem_len = round(y_tra_fine_rem_rel_prop.item() * coarse_tra_len_prop * num_frames)
    predicted_fine_actions = [fine_la_label] * fine_la_rem_len
    predicted_fine_steps = [(fine_la_label, fine_la_rem_len)]
    # Decoder
    dtype, device = x_enc_fine.dtype, x_enc_fine.device
    x_dec_cat_coarse = x_tra_coarse[..., :-2]
    x_dec_num_coarse = x_tra_coarse[..., -2:-1] + torch.tensor(y_tra_coarse_rem_prop, dtype=dtype, device=device)
    x_dec_coarse = torch.cat([x_dec_cat_coarse, x_dec_num_coarse], dim=-1)

    x_dec_cat_fine = x_tra_fine[..., :-2]
    acc_fine_proportion = x_tra_fine[..., -2:-1] + torch.tensor(y_tra_fine_rem_rel_prop, dtype=dtype, device=device)
    acc_fine_proportion = acc_fine_proportion.item()
    x_dec_num_fine = torch.tensor([[acc_fine_proportion]], dtype=dtype, device=device)
    x_dec_fine = torch.cat([x_dec_cat_fine, x_dec_num_fine], dim=-1)

    coarse_la_obs_prop = maybe_denormalise(x_tra_coarse[..., -1:].cpu().numpy(),
                                           scaler=scalers.get('x_tra_coarse_scaler'))
    coarse_la_prop = coarse_la_obs_prop.item() + y_tra_coarse_rem_prop.item()
    d_fine, d_fines = 0.0, []
    decoder_net, output_seq_len = model.decoder_net, model.decoder_net.output_seq_len
    coarse_exceed_first_time, total_coarse_length = True, 0
    with torch.no_grad():
        for t in range(output_seq_len):
            # Predict
            if model.model_v2:
                x_dec_fine_ = x_dec_fine[0]
                hx_ = [hx[0][0], hx[1][0]]
                y_dec_fine_logits, y_dec_fine_rel_prop, hx_fine = decoder_net.single_step_fine(x_dec_fine_, d_fine,
                                                                                               hx_)
                hx[0] = hx_fine.unsqueeze(0)
            else:
                y_dec_fine_logits, y_dec_fine_rel_prop, hx[0] = \
                    decoder_net.single_step_fine(x_dec_fine, d_fine, hx)
            # Process Prediction
            fine_na_label, _ = next_action_info(y_dec_fine_logits, y_dec_fine_rel_prop, fine_id_to_action, num_frames)
            if acc_fine_proportion >= 1.0 or fine_na_label is None:
                acc_fine_proportion, d_fine = 0.0, 1.0
                if model.model_v2:
                    x_dec_coarse_ = x_dec_coarse[0]
                    hx_ = [hx[0][0], hx[1][0]]
                    y_dec_coarse_logits, y_dec_coarse_prop, hx_coarse = \
                        decoder_net.single_step_coarse(x_dec_coarse_, hx_)
                    hx[1] = hx_coarse.unsqueeze(0)
                else:
                    y_dec_coarse_logits, y_dec_coarse_prop, hx[1] = \
                        decoder_net.single_step_coarse(x_dec_coarse, d_fine, hx)
                y_dec_coarse_prop = maybe_denormalise(y_dec_coarse_prop.cpu().numpy(),
                                                      scaler=scalers.get('y_dec_coarse_scaler'))
                coarse_na_label, coarse_na_len = next_action_info(y_dec_coarse_logits, y_dec_coarse_prop,
                                                                  coarse_id_to_action, num_frames)
                if coarse_na_label is None:
                    break
                predicted_coarse_actions += [coarse_na_label] * coarse_na_len
                predicted_coarse_steps.append((coarse_na_label, coarse_na_len))
                coarse_la_prop = y_dec_coarse_prop.item()
                x_dec_cat_coarse = logit2one_hot(y_dec_coarse_logits)
                if model.with_final_action:
                    x_dec_cat_coarse = x_dec_cat_coarse[..., :-1]
                x_dec_coarse[..., :-1] = x_dec_cat_coarse
                x_dec_coarse[..., -1] += coarse_la_prop
                predicted_fine_steps.append((None, None))
                if model.model_v3 and decoder_net.input_soft_parent:  # Prepare x_dec_cat_coarse for fine steps
                    if model.with_final_action:
                        x_dec_cat_coarse = torch.softmax(y_dec_coarse_logits[..., :-1], dim=-1)
                    else:
                        x_dec_cat_coarse = torch.softmax(y_dec_coarse_logits, dim=-1)
            else:
                y_dec_fine_rel_prop = maybe_denormalise(y_dec_fine_rel_prop.cpu().numpy(),
                                                        scaler=scalers.get('y_dec_fine_scaler'))
                excess = 0.0
                fine_na_label, fine_na_len = next_action_info(y_dec_fine_logits, y_dec_fine_rel_prop - excess,
                                                              fine_id_to_action, num_frames,
                                                              parent_la_prop=coarse_la_prop)
                predicted_fine_actions += [fine_na_label] * fine_na_len
                predicted_fine_steps.append((fine_na_label, fine_na_len))
                acc_fine_proportion += y_dec_fine_rel_prop.item()
                predicted_coarse_steps.append((None, None))
                d_fine = 0.0
            # Post-process
            d_fines.append(d_fine)
            if isinstance(model.encoder_net.encoder_hmgru, HMLSTM):
                hxs[0] = torch.cat([hxs[0], hx[0][0].unsqueeze(1)], dim=1)
                hxs[1] = torch.cat([hxs[1], hx[1][0].unsqueeze(1)], dim=1)
            else:
                hxs[0] = torch.cat([hxs[0], hx[0].unsqueeze(1)], dim=1)
                hxs[1] = torch.cat([hxs[1], hx[1].unsqueeze(1)], dim=1)
            x_dec_cat_fine = logit2one_hot(y_dec_fine_logits)
            if model.with_final_action:
                x_dec_cat_fine = x_dec_cat_fine[..., :-1]
            x_dec_cat_fine = x_dec_cat_fine * float(acc_fine_proportion > 0.0)
            if not disable_parent_input:
                x_dec_cat_fine = torch.cat([x_dec_cat_coarse, x_dec_cat_fine], dim=-1)
            x_dec_num_fine = torch.tensor([[acc_fine_proportion]], dtype=dtype, device=device)
            x_dec_fine = torch.cat([x_dec_cat_fine, x_dec_num_fine], dim=-1)
            coarse_exceed = len(predicted_coarse_actions) >= maximum_prediction_length
            fine_exceed = len(predicted_fine_actions) >= maximum_prediction_length
            if coarse_exceed and fine_exceed:
                break
            if coarse_exceed:
                if coarse_exceed_first_time:
                    coarse_exceed_first_time = False
                    total_coarse_length = len(predicted_coarse_actions)
                elif len(predicted_coarse_actions) > total_coarse_length:
                    predicted_coarse_steps = predicted_coarse_steps[:-1]
                    predicted_fine_steps = predicted_fine_steps[:-1]
                    break
    if model.with_final_action:
        fine_steps = [(None, None)] + predicted_fine_steps
        coarse_steps = predicted_coarse_steps[:1] + [(None, None)] + predicted_coarse_steps[1:]
        coarse_steps = maybe_rebalance_steps(coarse_steps, maximum_prediction_length)
        predicted_fine_steps, predicted_coarse_steps = fix_steps(fine_steps, coarse_steps)
        predicted_fine_actions = actions_from_steps(predicted_fine_steps)
        predicted_coarse_actions = actions_from_steps(predicted_coarse_steps)
    predicted_actions = predicted_fine_actions, predicted_coarse_actions
    predicted_steps = predicted_fine_steps, predicted_coarse_steps
    return predicted_actions, predicted_steps, d_fines


def fix_steps(predicted_fine_steps, predicted_coarse_steps):
    new_fine_steps, new_coarse_steps = [], list(predicted_coarse_steps[:1] + predicted_coarse_steps[2:])
    for i, (predicted_coarse_action, predicted_coarse_action_length) in enumerate(predicted_coarse_steps):
        if predicted_coarse_action is None:
            continue
        fine_actions, fine_lengths = [], []
        for predicted_fine_action, predicted_fine_action_length in predicted_fine_steps[i + 1:]:
            if predicted_fine_action is None:
                break
            fine_actions.append(predicted_fine_action)
            fine_lengths.append(predicted_fine_action_length)
        if not fine_lengths:
            new_fine_steps.append((None, None))
            continue
        fine_lengths_sum = sum(fine_lengths)
        if fine_lengths_sum < predicted_coarse_action_length:
            gap = predicted_coarse_action_length - fine_lengths_sum
            fine_lengths = [fine_length + round(gap * (fine_length / fine_lengths_sum))
                            for fine_length in fine_lengths]
        fine_lengths_sum = sum(fine_lengths)
        if fine_lengths_sum > predicted_coarse_action_length:
            excess = fine_lengths_sum - predicted_coarse_action_length
            fine_lengths[-1] -= excess
        elif fine_lengths_sum < predicted_coarse_action_length:
            gap = predicted_coarse_action_length - fine_lengths_sum
            fine_lengths[-1] += gap
        for fine_action, fine_length in zip(fine_actions, fine_lengths):
            new_fine_steps.append((fine_action, fine_length))
        new_fine_steps.append((None, None))
    return new_fine_steps, new_coarse_steps


def test_hera_cv(args):
    pretrained_root = args.pretrained_root
    pretrained_suffix = args.pretrained_suffix
    fine_labels_root_path = args.fine_labels_root_path
    coarse_labels_root_path = args.coarse_labels_root_path

    splits = '01 02 03 04 05'.split(sep=' ')
    split_folders = [split_folder for split_folder in os.listdir(pretrained_root)]
    results = {}
    for split_folder in split_folders:
        checkpoint = os.path.join(pretrained_root, split_folder, pretrained_suffix)
        args.checkpoint = checkpoint
        fine_labels_folder, coarse_labels_folder = None, None
        for split in splits:
            if split not in split_folder:
                fine_labels_folder = f'S{split}'
                coarse_labels_folder = f'S{split}'
                break
        fine_labels_path = os.path.join(fine_labels_root_path, fine_labels_folder)
        args.fine_labels_path = fine_labels_path
        coarse_labels_path = os.path.join(coarse_labels_root_path, coarse_labels_folder)
        args.coarse_labels_path = coarse_labels_path
        print(split_folder)
        split_results = test_hera(args)
        results[split_folder] = split_results
    results_to_average = defaultdict(list)
    for _, result in results.items():
        for key, score in result.items():
            results_to_average[key].append(score)
    print('\nCross-validated results:')
    for key, result in results_to_average.items():
        print(f'{key:<21}: {np.array(result).mean().item():.4f} +/- {np.array(result).std().item():.4f}')
