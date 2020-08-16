import bisect
from collections import defaultdict
from itertools import accumulate
import os

import numpy as np
import torch

from fpua.analysis import analyse_single_level_observations_and_predictions_per_step, write_results_per_video
from fpua.analysis import compute_metrics
from fpua.analysis import action_sequence_metrics, write_sequence_results_per_video, compute_moc
from fpua.data.general import read_action_dictionary, extend_smallest_list, clean_directory, maybe_rebalance_steps
from fpua.data.general import aggregate_actions_and_lengths, extend_or_trim_predicted_actions
from fpua.models.baselines import Baseline0, Baseline1, Baseline2
from fpua.models.misc import compute_steps_to_grab
from fpua.utils import nan_to_value, numpy_to_torch


def test_baselines(args):
    checkpoint = torch.load(args.checkpoint)
    fine_labels_path = args.fine_labels_path
    coarse_labels_path = args.coarse_labels_path
    fine_action_to_id = read_action_dictionary(args.fine_action_to_id)
    fine_id_to_action = {action_id: action for action, action_id in fine_action_to_id.items()}
    coarse_action_to_id = read_action_dictionary(args.coarse_action_to_id)
    coarse_id_to_action = {action_id: action for action, action_id in coarse_action_to_id.items()}
    fraction_observed = args.observed_fraction
    ignore_silence_action = args.ignore_silence_action
    do_error_analysis = args.do_error_analysis
    print_coarse_results = args.print_coarse_results
    seq_len = checkpoint['seq_len']
    # Load model
    baseline_type = checkpoint['baseline_type']
    action_level = checkpoint['action_level']
    Baseline = {0: Baseline0, 1: Baseline1, 2: Baseline2}[baseline_type]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Baseline(**checkpoint['model_creation_args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    observed_fine_actions_per_video = []
    unobserved_fine_actions_per_video = []
    predicted_fine_steps_per_video = []
    predicted_fine_actions_per_video = []

    observed_coarse_actions_per_video = []
    unobserved_coarse_actions_per_video = []
    predicted_coarse_steps_per_video = []
    predicted_coarse_actions_per_video = []
    
    num_frames_per_video = []
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
        num_frames = len(fine_actions_per_frame)
        num_frames_per_video.append(num_frames)
        num_frames_to_grab = round(num_frames * fraction_observed)
        observed_fine_actions = fine_actions_per_frame[:num_frames_to_grab]
        observed_fine_actions_per_video.append(observed_fine_actions)
        unobserved_fine_actions = fine_actions_per_frame[num_frames_to_grab:]
        observed_coarse_actions = coarse_actions_per_frame[:num_frames_to_grab]
        observed_coarse_actions_per_video.append(observed_coarse_actions)
        unobserved_coarse_actions = coarse_actions_per_frame[num_frames_to_grab:]

        tensors, steps, last_action_obs_length = generate_test_datum(observed_fine_actions, observed_coarse_actions,
                                                                     seq_len=seq_len,
                                                                     fine_action_to_id=fine_action_to_id,
                                                                     coarse_action_to_id=coarse_action_to_id,
                                                                     num_frames=num_frames)
        tensors = [nan_to_value(tensor, value=0.0) for tensor in tensors]
        tensors = numpy_to_torch(*tensors, device=device)
        steps = torch.tensor([steps], device=device)
        predictions = predict_future_actions(model, tensors, effective_steps=steps,
                                             fine_id_to_action=fine_id_to_action,
                                             coarse_id_to_action=coarse_id_to_action,
                                             num_frames=num_frames,
                                             maximum_prediction_length=len(unobserved_fine_actions),
                                             baseline_type=baseline_type, action_level=action_level,
                                             last_action_obs_length=last_action_obs_length)
        predicted_fine_actions, predicted_fine_steps = predictions
        predicted_fine_actions, predicted_coarse_actions = predicted_fine_actions
        predicted_fine_steps, predicted_coarse_steps = predicted_fine_steps
        predicted_fine_steps_per_video.append(predicted_fine_steps)
        predicted_coarse_steps_per_video.append(predicted_coarse_steps)
        _update_level_predictions(predicted_fine_actions, predicted_fine_actions_per_video, unobserved_fine_actions,
                                  unobserved_fine_actions_per_video)
        _update_level_predictions(predicted_coarse_actions, predicted_coarse_actions_per_video,
                                  unobserved_coarse_actions, unobserved_coarse_actions_per_video)
    # Performance and Error Analysis
    f1_results_dict = {}
    moc_results_dict = {}
    unobserved_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
    unobserved_fractions = [unobserved_fraction for unobserved_fraction in unobserved_fractions
                            if fraction_observed + unobserved_fraction <= 1.0]
    for unobserved_fraction in unobserved_fractions:
        save_analysis_path = os.path.join(args.checkpoint[:-4],
                                          str(fraction_observed) + '_' + str(unobserved_fraction))
        if os.path.exists(save_analysis_path):
            clean_directory(save_analysis_path)
        predicted_fine_actions_per_video_sub, unobserved_fine_actions_per_video_sub = [], []
        predicted_coarse_actions_per_video_sub, unobserved_coarse_actions_per_video_sub = [], []
        f1_per_video_fine = []  # file_name, input-level_0.5_f1
        f1_per_video_coarse = []
        sequence_metrics_per_video_fine = []  # file_name, precision, recall, f1 (regardless of length/class)
        sequence_metrics_per_video_coarse = []
        num_videos = len(predicted_fine_actions_per_video)
        for i in range(num_videos):
            predicted_fine_actions = predicted_fine_actions_per_video[i]
            unobserved_fine_actions = unobserved_fine_actions_per_video[i]
            num_frames_to_grab = num_frames_per_video[i] * unobserved_fraction
            num_frames_to_grab = round(num_frames_to_grab)
            predicted_fine_actions_sub = predicted_fine_actions[:num_frames_to_grab]
            predicted_fine_actions_per_video_sub.append(predicted_fine_actions_sub)
            unobserved_fine_actions_sub = unobserved_fine_actions[:num_frames_to_grab]
            unobserved_fine_actions_per_video_sub.append(unobserved_fine_actions_sub)
            predicted_coarse_actions = predicted_coarse_actions_per_video[i]
            unobserved_coarse_actions = unobserved_coarse_actions_per_video[i]
            predicted_coarse_actions_sub = predicted_coarse_actions[:num_frames_to_grab]
            predicted_coarse_actions_per_video_sub.append(predicted_coarse_actions_sub)
            unobserved_coarse_actions_sub = unobserved_coarse_actions[:num_frames_to_grab]
            unobserved_coarse_actions_per_video_sub.append(unobserved_coarse_actions_sub)
            if do_error_analysis:
                if baseline_type == 0:
                    if action_level == 'coarse':
                        observed_actions = observed_coarse_actions_per_video[i]
                        predicted_steps = predicted_coarse_steps_per_video[i]
                        unobserved_actions = unobserved_coarse_actions_sub.tolist()
                        predicted_actions_sub = predicted_coarse_actions_sub
                        unobserved_actions_sub = unobserved_coarse_actions_sub
                        action_to_id = coarse_action_to_id
                    else:
                        observed_actions = observed_fine_actions_per_video[i]
                        predicted_steps = predicted_fine_steps_per_video[i]
                        unobserved_actions = unobserved_fine_actions_sub.tolist()
                        predicted_actions_sub = predicted_fine_actions_sub
                        unobserved_actions_sub = unobserved_fine_actions_sub
                        action_to_id = fine_action_to_id
                    steps_to_grab = compute_steps_to_grab(predicted_steps, num_frames_to_grab)
                    predicted_steps = predicted_steps[:steps_to_grab]
                    analyse_single_level_observations_and_predictions_per_step(predicted_steps,
                                                                               observed_actions,
                                                                               unobserved_actions,
                                                                               num_frames=num_frames_per_video[i],
                                                                               save_path=save_analysis_path,
                                                                               save_file_name=label_files[i])
                    _, f1_scores = compute_metrics([predicted_actions_sub],
                                                   [unobserved_actions_sub],
                                                   action_to_id=action_to_id)
                    if action_level == 'coarse':
                        f1_per_video_coarse.append([label_files[i], f1_scores[-1]])
                    else:
                        f1_per_video_fine.append([label_files[i], f1_scores[-1]])
                    precision, recall, f1 = \
                        action_sequence_metrics(aggregate_actions_and_lengths(predicted_actions_sub.tolist())[0],
                                                aggregate_actions_and_lengths(unobserved_actions_sub.tolist())[0])
                    if action_level == 'coarse':
                        sequence_metrics_per_video_coarse.append([label_files[i], precision, recall, f1])
                    else:
                        sequence_metrics_per_video_fine.append([label_files[i], precision, recall, f1])
                else:
                    observed_actions = [coarse_action + '/' + fine_action
                                        for coarse_action, fine_action in zip(observed_coarse_actions_per_video[i],
                                                                              observed_fine_actions_per_video[i])]
                    predicted_fine_steps = predicted_fine_steps_per_video[i]
                    steps_to_grab = compute_steps_to_grab(predicted_fine_steps, num_frames_to_grab)
                    predicted_fine_steps = predicted_fine_steps[:steps_to_grab]
                    predicted_coarse_steps = predicted_coarse_steps_per_video[i][:steps_to_grab]
                    predicted_steps = [(coarse_step[0] + '/' + fine_step[0], fine_step[1])
                                       for coarse_step, fine_step in zip(predicted_coarse_steps, predicted_fine_steps)]
                    unobserved_actions = [coarse_action + '/' + fine_action
                                          for coarse_action, fine_action in
                                          zip(unobserved_coarse_actions_sub.tolist(),
                                              unobserved_fine_actions_sub.tolist())]
                    analyse_single_level_observations_and_predictions_per_step(predicted_steps,
                                                                               observed_actions,
                                                                               unobserved_actions,
                                                                               num_frames=num_frames_per_video[i],
                                                                               save_path=save_analysis_path,
                                                                               save_file_name=label_files[i])
                    _, f1_scores_fine = compute_metrics([predicted_fine_actions_sub],
                                                        [unobserved_fine_actions_sub],
                                                        action_to_id=fine_action_to_id)
                    f1_per_video_fine.append([label_files[i], f1_scores_fine[-1]])
                    _, f1_scores_coarse = compute_metrics([predicted_coarse_actions_sub],
                                                          [unobserved_coarse_actions_sub],
                                                          action_to_id=coarse_action_to_id)
                    f1_per_video_coarse.append([label_files[i], f1_scores_coarse[-1]])
                    precision, recall, f1 = \
                        action_sequence_metrics(aggregate_actions_and_lengths(predicted_fine_actions_sub.tolist())[0],
                                                aggregate_actions_and_lengths(unobserved_fine_actions_sub.tolist())[0])
                    sequence_metrics_per_video_fine.append([label_files[i], precision, recall, f1])
                    precision, recall, f1 = \
                        action_sequence_metrics(aggregate_actions_and_lengths(predicted_coarse_actions_sub.tolist())[0],
                                                aggregate_actions_and_lengths(unobserved_coarse_actions_sub.tolist())[0])
                    sequence_metrics_per_video_coarse.append([label_files[i], precision, recall, f1])
        if do_error_analysis:
            if f1_per_video_fine:
                write_results_per_video(f1_per_video_fine, order_by=None, metric_name='f1-0.5-fine',
                                        save_path=save_analysis_path)
                write_sequence_results_per_video(sequence_metrics_per_video_fine, save_analysis_path, level='fine')
            if f1_per_video_coarse:
                write_results_per_video(f1_per_video_coarse, order_by=None, metric_name='f1-0.5-coarse',
                                        save_path=save_analysis_path)
                write_sequence_results_per_video(sequence_metrics_per_video_coarse, save_analysis_path, level='coarse')
        print('\nObserved fraction: %.2f | Unobserved fraction: %.2f' % (fraction_observed, unobserved_fraction))
        if baseline_type == 0 and action_level == 'coarse':
            predicted_actions_per_video_sub = predicted_coarse_actions_per_video_sub
            unobserved_actions_per_video_sub = unobserved_coarse_actions_per_video_sub
            action_to_id = coarse_action_to_id
            print('Coarse')
        else:
            predicted_actions_per_video_sub = predicted_fine_actions_per_video_sub
            unobserved_actions_per_video_sub = unobserved_fine_actions_per_video_sub
            action_to_id = fine_action_to_id
            print('Fine')
        moc, _, _ = compute_moc(np.concatenate(predicted_actions_per_video_sub),
                                np.concatenate(unobserved_actions_per_video_sub),
                                action_to_id=action_to_id)
        if baseline_type == 0 and action_level == 'coarse':
            moc_results_dict[f'coarse-moc-{fraction_observed}_{unobserved_fraction}'] = moc
        else:
            moc_results_dict[f'fine-moc-{fraction_observed}_{unobserved_fraction}'] = moc
        overlaps, f1_overlap_scores = compute_metrics(predicted_actions_per_video_sub,
                                                      unobserved_actions_per_video_sub, action_to_id=action_to_id)
        for overlap, overlap_f1_score in zip(overlaps, f1_overlap_scores):
            print('F1@%.2f: %.4f' % (overlap, overlap_f1_score))
            if baseline_type == 0 and action_level == 'coarse':
                f1_results_dict[f'coarse-{fraction_observed}_{unobserved_fraction}_{overlap}'] = overlap_f1_score
            else:
                f1_results_dict[f'fine-{fraction_observed}_{unobserved_fraction}_{overlap}'] = overlap_f1_score
        if baseline_type > 0:
            moc, _, _ = compute_moc(np.concatenate(predicted_coarse_actions_per_video_sub),
                                    np.concatenate(unobserved_coarse_actions_per_video_sub),
                                    action_to_id=coarse_action_to_id)
            moc_results_dict[f'coarse-moc-{fraction_observed}_{unobserved_fraction}'] = moc
            overlaps, f1_overlap_scores = \
                compute_metrics(predicted_coarse_actions_per_video_sub, unobserved_coarse_actions_per_video_sub,
                                action_to_id=coarse_action_to_id)
            for overlap, overlap_f1_score in zip(overlaps, f1_overlap_scores):
                f1_results_dict[f'coarse-{fraction_observed}_{unobserved_fraction}_{overlap}'] = overlap_f1_score
                if print_coarse_results:
                    print('Coarse')
                    print('F1@%.2f: %.4f' % (overlap, overlap_f1_score))
    results_dict = {**f1_results_dict, **moc_results_dict}
    return results_dict


def _update_level_predictions(predicted_actions, predicted_actions_per_video, unobserved_actions,
                              unobserved_actions_per_video):
    if not predicted_actions:
        predicted_actions = ['FAILED_TO_PREDICT']
    predicted_actions = extend_or_trim_predicted_actions(predicted_actions, unobserved_actions)
    predicted_actions = np.array(predicted_actions)
    predicted_actions_per_video.append(predicted_actions)
    unobserved_actions = np.array(unobserved_actions)
    unobserved_actions_per_video.append(unobserved_actions)


def generate_test_datum(observed_fine_actions, observed_coarse_actions, seq_len, fine_action_to_id,
                        coarse_action_to_id, num_frames):
    actions, lengths = aggregate_actions_and_lengths(list(zip(observed_coarse_actions, observed_fine_actions)))
    acc_lengths = list(accumulate(lengths))
    num_fine_actions, num_coarse_actions = len(fine_action_to_id), len(coarse_action_to_id)
    # Tensors
    x_enc_fine = np.full([1, seq_len, num_fine_actions + 1], fill_value=np.nan, dtype=np.float32)
    x_enc_coarse = np.full([1, seq_len, num_coarse_actions + 1], fill_value=np.nan, dtype=np.float32)

    min_len = min(len(actions), seq_len)
    for i in range(min_len):
        coarse_action, fine_action = actions[i]
        coarse_action_id = coarse_action_to_id[coarse_action]
        x_enc_coarse[0, i, coarse_action_id] = 1.0
        fine_action_id = fine_action_to_id[fine_action]
        x_enc_fine[0, i, fine_action_id] = 1.0
        acc_length = acc_lengths[i - 1] if i > 0 else 0
        x_enc_coarse[0, i, -1] = acc_length / num_frames
        x_enc_fine[0, i, -1] = acc_length / num_frames
    tensors = [x_enc_fine, x_enc_coarse]
    return tensors, min_len, lengths[-1]


def predict_future_actions(model, tensors, effective_steps, fine_id_to_action, coarse_id_to_action, num_frames,
                           maximum_prediction_length, baseline_type, action_level, last_action_obs_length):
    x_enc_fine, x_enc_coarse = tensors
    with torch.no_grad():
        if baseline_type == 0:
            x_enc = x_enc_coarse if action_level == 'coarse' else x_enc_fine
            output = model(x_enc, hx=None, test_mode=True, effective_num_steps=effective_steps)
        else:
            output = model(x_enc_fine, x_enc_coarse, hx=None, test_mode=True, effective_num_steps=effective_steps)
    if baseline_type == 0 and action_level == 'coarse':
        y_logits_coarse, y_lens_fine, y_logits_fine = output
    else:
        y_logits_fine, y_lens_fine, y_logits_coarse = output
    predicted_fine_steps, predicted_coarse_steps = [], []
    lengths = []
    last_action_remlength = round(y_lens_fine[:, 0].item() * num_frames) - last_action_obs_length
    if last_action_remlength < 0:
        # print(f'Last action remaining length was {last_action_remlength}')
        last_action_remlength = 0
    lengths.append(last_action_remlength)
    if y_logits_fine is not None:
        last_fine_action_id = torch.argmax(x_enc_fine[:, int(effective_steps.item()) - 1, :-1], dim=-1).item()
        last_fine_action = fine_id_to_action[last_fine_action_id]
        predicted_fine_steps.append((last_fine_action, last_action_remlength))
    if y_logits_coarse is not None:
        last_coarse_action_id = torch.argmax(x_enc_coarse[:, int(effective_steps.item()) - 1, :-1], dim=-1).item()
        last_coarse_action = coarse_id_to_action[last_coarse_action_id]
        predicted_coarse_steps.append((last_coarse_action, last_action_remlength))
    seq_len = y_lens_fine.size(1)
    already_appended = False
    for t in range(1, seq_len):
        action_length = round(y_lens_fine[:, t].item() * num_frames)
        if y_logits_fine is not None:
            fine_action_id = torch.argmax(y_logits_fine[:, t - 1], dim=-1).item()
            fine_action = fine_id_to_action.get(fine_action_id)
            if fine_action is None:
                break
            predicted_fine_steps.append((fine_action, action_length))
            lengths.append(action_length)
            already_appended = True
        if y_logits_coarse is not None:
            coarse_action_id = torch.argmax(y_logits_coarse[:, t - 1], dim=-1).item()
            coarse_action = coarse_id_to_action.get(coarse_action_id)
            if coarse_action is None:
                break
            predicted_coarse_steps.append((coarse_action, action_length))
            if not already_appended:
                lengths.append(action_length)
        already_appended = False
    acc_lengths = list(accumulate(lengths))
    if acc_lengths[-1] < maximum_prediction_length:
        if baseline_type == 0 and action_level == 'coarse':
            predicted_coarse_steps = maybe_rebalance_steps(predicted_coarse_steps, maximum_prediction_length)
            steps_to_grab = len(predicted_coarse_steps)
        else:
            predicted_fine_steps = maybe_rebalance_steps(predicted_fine_steps, maximum_prediction_length)
            if baseline_type > 0:
                predicted_coarse_steps = copy_step_lengths(predicted_fine_steps, predicted_coarse_steps)
            steps_to_grab = len(predicted_fine_steps)
    else:
        steps_to_grab = bisect.bisect_left(acc_lengths, maximum_prediction_length) + 1
    predicted_fine_steps = predicted_fine_steps[:steps_to_grab]
    predicted_coarse_steps = predicted_coarse_steps[:steps_to_grab]
    predicted_fine_actions = actions_from_steps(predicted_fine_steps)
    predicted_coarse_actions = actions_from_steps(predicted_coarse_steps)
    predicted_actions = predicted_fine_actions, predicted_coarse_actions
    predicted_steps = predicted_fine_steps, predicted_coarse_steps
    return predicted_actions, predicted_steps


def actions_from_steps(predicted_steps):
    predicted_actions = []
    for predicted_action, predicted_action_length in predicted_steps:
        predicted_actions += [predicted_action] * predicted_action_length
    return predicted_actions


def copy_step_lengths(predicted_fine_steps, predicted_coarse_steps):
    updated_coarse_steps = []
    for predicted_fine_step, predicted_coarse_step in zip(predicted_fine_steps, predicted_coarse_steps):
        updated_coarse_steps.append((predicted_coarse_step[0], predicted_fine_step[1]))
    return updated_coarse_steps


def test_baselines_cv(args):
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
        split_results = test_baselines(args)
        results[split_folder] = split_results
    results_to_average = defaultdict(list)
    for _, result in results.items():
        for key, score in result.items():
            results_to_average[key].append(score)
    print('\nCross-validation results:')
    for key, result in results_to_average.items():
        print(f'{key:<21}: {np.array(result).mean().item():.4f} +/- {np.array(result).std().item():.4f}')
