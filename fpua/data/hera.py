import bisect
from itertools import accumulate
import os
import random

import numpy as np
from sklearn.utils import shuffle

from fpua.data.general import read_action_dictionary, extend_smallest_list, aggregate_actions_and_lengths
from fpua.data.general import extract_last_action_and_observed_length, extract_last_action_and_remlength
from fpua.data.general import extract_last_action_and_full_length


def generate_hera_training_data(args):
    fine_labels_path = args.fine_labels_path
    coarse_labels_path = args.coarse_labels_path
    fine_action_to_id = read_action_dictionary(args.fine_action_to_id)
    coarse_action_to_id = read_action_dictionary(args.coarse_action_to_id)
    input_seq_len = args.input_seq_len
    output_seq_len = args.output_seq_len
    num_cuts = args.num_cuts
    observe_at_least_k_percent = args.observe_at_least_k_percent
    ignore_silence_action = args.ignore_silence_action
    add_final_action = args.add_final_action
    save_path = args.save_path
    save_name = args.save_name

    tensors_dict = _generate_hera_training_data(fine_labels_path, coarse_labels_path,
                                                fine_action_to_id=fine_action_to_id,
                                                coarse_action_to_id=coarse_action_to_id,
                                                input_seq_len=input_seq_len,
                                                output_seq_len=output_seq_len,
                                                num_cuts=num_cuts,
                                                observe_at_least_k_percent=observe_at_least_k_percent,
                                                ignore_silence_action=ignore_silence_action,
                                                add_final_action=add_final_action)
    print('Generated %d training examples.' % len(tensors_dict['x_enc_fine']))
    if save_path is not None:
        file_name = save_name if save_name is not None else 'training_data'
        save_file = os.path.join(save_path, file_name + '.npz')
        np.savez_compressed(save_file, **tensors_dict)
        print('Training data successfully written to %s' % save_file)


def _generate_hera_training_data(fine_labels_path, coarse_labels_path, fine_action_to_id,
                                 coarse_action_to_id, input_seq_len, output_seq_len, num_cuts,
                                 observe_at_least_k_percent, ignore_silence_action,
                                 add_final_action):
    random.seed(42)
    xs_enc_coarse, xs_tra_coarse, xs_dec_coarse = [], [], []
    xs_enc_fine, xs_tra_fine, xs_dec_fine = [], [], []
    ys_enc_coarse, ys_tra_coarse, ys_dec_coarse = [], [], []
    ys_enc_fine, ys_tra_fine, ys_dec_fine = [], [], []
    encs_boundary, decs_boundary = [], []
    tensors = [xs_enc_coarse, xs_tra_coarse, xs_dec_coarse, xs_enc_fine, xs_tra_fine, xs_dec_fine,
               ys_enc_coarse, ys_tra_coarse, ys_dec_coarse, ys_enc_fine, ys_tra_fine, ys_dec_fine,
               encs_boundary, decs_boundary]

    fine_label_files = set(os.listdir(fine_labels_path))
    coarse_label_files = set(os.listdir(coarse_labels_path))
    label_files = sorted(fine_label_files & coarse_label_files)
    for label_file in label_files:
        with open(os.path.join(fine_labels_path, label_file), mode='r') as f:
            fine_actions_per_frame = [line.rstrip() for line in f]
        with open(os.path.join(coarse_labels_path, label_file), mode='r') as f:
            coarse_actions_per_frame = [line.rstrip() for line in f]
        fine_actions_per_frame, coarse_actions_per_frame = \
            extend_smallest_list(fine_actions_per_frame, coarse_actions_per_frame)
        single_video_tensors = \
            _generate_hera_training_data_from_single_video(fine_actions_per_frame,
                                                           coarse_actions_per_frame,
                                                           fine_action_to_id=fine_action_to_id,
                                                           coarse_action_to_id=coarse_action_to_id,
                                                           input_seq_len=input_seq_len,
                                                           output_seq_len=output_seq_len,
                                                           num_cuts=num_cuts,
                                                           observe_at_least_k_percent=observe_at_least_k_percent,
                                                           ignore_silence_action=ignore_silence_action,
                                                           add_final_action=add_final_action)
        for tensor_list, single_video_tensor in zip(tensors, single_video_tensors):
            tensor_list.append(single_video_tensor)
    tensors = [np.concatenate(tensor_list, axis=0) for tensor_list in tensors]
    tensors = shuffle(*tensors, random_state=42)
    names = ['x_enc_coarse', 'x_tra_coarse', 'x_dec_coarse', 'x_enc_fine', 'x_tra_fine', 'x_dec_fine',
             'y_enc_coarse', 'y_tra_coarse', 'y_dec_coarse', 'y_enc_fine', 'y_tra_fine', 'y_dec_fine',
             'enc_boundary', 'dec_boundary']
    tensors_dict = dict(zip(names, tensors))
    return tensors_dict


def _generate_hera_training_data_from_single_video(fine_actions_per_frame,
                                                   coarse_actions_per_frame,
                                                   fine_action_to_id, coarse_action_to_id,
                                                   input_seq_len, output_seq_len, num_cuts,
                                                   observe_at_least_k_percent,
                                                   ignore_silence_action,
                                                   add_final_action):
    if ignore_silence_action is not None:
        fine_actions_per_frame = [fine_action for fine_action in fine_actions_per_frame
                                  if fine_action != ignore_silence_action]
        coarse_actions_per_frame = [coarse_action for coarse_action in coarse_actions_per_frame
                                    if coarse_action != ignore_silence_action]
        error_msg = 'Action levels do not match after removing silence action.'
        assert len(fine_actions_per_frame) == len(coarse_actions_per_frame), error_msg
    actions, lengths = aggregate_actions_and_lengths(list(zip(coarse_actions_per_frame, fine_actions_per_frame)))
    num_video_actions = len(actions) - 1  # No need for end of video action since we are dealing with proportions
    num_examples = num_video_actions * num_cuts
    num_fine_actions, num_coarse_actions = len(fine_action_to_id), len(coarse_action_to_id)
    # ENCODER
    # Action + cumulative length proportion in relation to video
    x_enc_coarse = np.full([num_examples, input_seq_len, num_coarse_actions + 1], fill_value=np.nan, dtype=np.float32)
    # Next action + full length proportion in relation to video
    y_enc_coarse = np.full([num_examples, input_seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
    # Parent coarse action + fine action + cumulative length proportion (inclusive) in relation to the parent
    x_enc_fine = np.full([num_examples, input_seq_len, num_coarse_actions + num_fine_actions + 1],
                         fill_value=np.nan, dtype=np.float32)
    # Next fine action and its full length proportion in relation to the parent. Ignore future prediction of last
    # child
    y_enc_fine = np.full([num_examples, input_seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
    # REFRESHER
    # Action + partially accumulated length proportion (inclusive) + observed length
    # proportion (both in relation to the video)
    x_tra_coarse = np.full([num_examples, num_coarse_actions + 2], fill_value=np.nan, dtype=np.float32)
    # Remaining length proportion in relation to the video
    y_tra_coarse = np.full([num_examples, 1], fill_value=np.nan, dtype=np.float32)
    # Parent action + fine action + partially accumulated length proportion in relation to the parent (inclusive) +
    # observed proportion in relation to the parent
    x_tra_fine = np.full([num_examples, num_coarse_actions + num_fine_actions + 2], fill_value=np.nan, dtype=np.float32)
    # Remaining length proportion in relation to the parent
    y_tra_fine = np.full([num_examples, 1], fill_value=np.nan, dtype=np.float32)
    # ANTICIPATOR
    # Previous coarse action and accumulated length proportion in relation to the video
    x_dec_coarse = np.full([num_examples, output_seq_len, num_coarse_actions + 1], fill_value=np.nan, dtype=np.float32)
    # Future action + full length proportion in relation to the video
    y_dec_coarse = np.full([num_examples, output_seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
    # Parent coarse action, previous fine action (or zero in case of first child), and accumulated length proportion
    # from the first child in relation to the parent
    x_dec_fine = np.full([num_examples, output_seq_len, num_coarse_actions + num_fine_actions + 1],
                         fill_value=np.nan, dtype=np.float32)
    # Future action + full length proportion in relation to the video
    y_dec_fine = np.full([num_examples, output_seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
    # BOUNDARIES
    enc_boundary = np.full([num_examples, input_seq_len], fill_value=np.nan, dtype=np.float32)
    dec_boundary = np.full([num_examples, output_seq_len], fill_value=np.nan, dtype=np.float32)

    current_example_idx, num_frames, acc_lengths = 0, len(coarse_actions_per_frame), list(accumulate(lengths))
    cutoff_frame = round((observe_at_least_k_percent / 100) * num_frames)
    cutoff_index = bisect.bisect_left(acc_lengths, cutoff_frame)
    for i, (action, length) in enumerate(zip(actions[:-1], lengths[:-1])):
        if i < cutoff_index:
            continue
        k = min(length, num_cuts)
        cuts = random.sample(range(length), k=k)
        # REFRESHER
        for cut in cuts:
            coarse_tra_action = action[0]
            coarse_tra_action_id = coarse_action_to_id[coarse_tra_action]
            x_tra_coarse[current_example_idx, coarse_tra_action_id] = 1.0
            global_cut = cut if i == 0 else acc_lengths[i - 1] + cut
            x_tra_coarse[current_example_idx, -2] = (global_cut + 1) / num_frames
            coarse_tra_action_obs_len = extract_last_action_and_observed_length(coarse_actions_per_frame, global_cut)[1]
            x_tra_coarse[current_example_idx, -1] = coarse_tra_action_obs_len / num_frames
            coarse_tra_action_rem_len = extract_last_action_and_remlength(coarse_actions_per_frame, global_cut)[1]
            y_tra_coarse[current_example_idx, -1] = coarse_tra_action_rem_len / num_frames

            x_tra_fine[current_example_idx, coarse_tra_action_id] = 1.0
            fine_tra_action = action[1]
            fine_tra_action_id = fine_action_to_id[fine_tra_action]
            x_tra_fine[current_example_idx, num_coarse_actions + fine_tra_action_id] = 1.0
            coarse_tra_action_len = coarse_tra_action_obs_len + coarse_tra_action_rem_len
            local_acc_len, fine_tra_action_obs_len = get_local_acc_len(actions, lengths, i), cut + 1
            local_acc_len = local_acc_len - length + fine_tra_action_obs_len
            x_tra_fine[current_example_idx, -2] = local_acc_len / coarse_tra_action_len
            x_tra_fine[current_example_idx, -1] = fine_tra_action_obs_len / coarse_tra_action_len
            fine_tra_action_rem_len = length - fine_tra_action_obs_len
            y_tra_fine[current_example_idx, -1] = fine_tra_action_rem_len / coarse_tra_action_len
            current_example_idx += 1
        # ENCODER
        x_fine = np.full([input_seq_len, num_coarse_actions + num_fine_actions + 1],
                         fill_value=np.nan, dtype=np.float32)
        x_coarse = np.full([input_seq_len, num_coarse_actions + 1], fill_value=np.nan, dtype=np.float32)
        y_fine = np.full([input_seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
        y_coarse = np.full([input_seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
        boundary = np.full(input_seq_len, fill_value=np.nan, dtype=np.float32)
        for j, m in zip(range(i - 1, -1, -1), range(-1, -input_seq_len - 1, -1)):
            past_coarse_action = actions[j][0]
            past_coarse_action_id = coarse_action_to_id[past_coarse_action]
            x_fine[m, past_coarse_action_id] = 1.0
            past_fine_action = actions[j][1]
            past_fine_action_id = fine_action_to_id[past_fine_action]
            x_fine[m, num_coarse_actions + past_fine_action_id] = 1.0
            local_acc_len = get_local_acc_len(actions, lengths, j)
            past_coarse_action_len = extract_last_action_and_full_length(coarse_actions_per_frame,
                                                                         acc_lengths[j] - 1)[1]
            x_fine[m, -1] = local_acc_len / past_coarse_action_len

            current_coarse_action = actions[j + 1][0]
            if past_coarse_action != current_coarse_action:
                x_coarse[m, past_coarse_action_id] = 1.0
                x_coarse[m, -1] = acc_lengths[j] / num_frames
                current_coarse_action_id = coarse_action_to_id[current_coarse_action]
                y_coarse[m, 0] = current_coarse_action_id
                current_coarse_action_len = extract_last_action_and_full_length(coarse_actions_per_frame,
                                                                                acc_lengths[j])[1]
                y_coarse[m, -1] = current_coarse_action_len / num_frames
                boundary[m] = 1.0
                if add_final_action:
                    y_fine[m, 0] = len(fine_action_to_id)
            else:
                current_fine_action = actions[j + 1][1]
                current_fine_action_id = fine_action_to_id[current_fine_action]
                y_fine[m, 0] = current_fine_action_id
                current_fine_action_len = lengths[j + 1]
                y_fine[m, -1] = current_fine_action_len / past_coarse_action_len
                boundary[m] = 0.0
        x_enc_fine[(current_example_idx - k):current_example_idx] = x_fine
        x_enc_coarse[(current_example_idx - k):current_example_idx] = x_coarse
        y_enc_fine[(current_example_idx - k):current_example_idx] = y_fine
        y_enc_coarse[(current_example_idx - k):current_example_idx] = y_coarse
        enc_boundary[(current_example_idx - k):current_example_idx] = boundary
        # ANTICIPATOR
        decoder_tensors = _generate_decoder_tensors(actions, lengths, fine_action_to_id, coarse_action_to_id,
                                                    coarse_actions_per_frame, i, output_seq_len,
                                                    num_coarse_actions, num_fine_actions, num_frames,
                                                    add_final_action)
        boundary, x_coarse, x_fine, y_coarse, y_fine = decoder_tensors
        x_dec_coarse[(current_example_idx - k):current_example_idx] = x_coarse
        x_dec_fine[(current_example_idx - k):current_example_idx] = x_fine
        y_dec_fine[(current_example_idx - k):current_example_idx] = y_fine
        y_dec_coarse[(current_example_idx - k):current_example_idx] = y_coarse
        dec_boundary[(current_example_idx - k):current_example_idx] = boundary
    tensors = [x_enc_coarse, x_tra_coarse, x_dec_coarse, x_enc_fine, x_tra_fine, x_dec_fine,
               y_enc_coarse, y_tra_coarse, y_dec_coarse, y_enc_fine, y_tra_fine, y_dec_fine,
               enc_boundary, dec_boundary]
    tensors = [tensor[:current_example_idx] for tensor in tensors]
    return tensors


def _generate_decoder_tensors(actions, lengths, fine_action_to_id, coarse_action_to_id, coarse_actions_per_frame, i,
                              output_seq_len, num_coarse_actions, num_fine_actions, num_frames, add_final_action):
    x_coarse = np.full([output_seq_len, num_coarse_actions + 1], fill_value=np.nan, dtype=np.float32)
    x_fine = np.full([output_seq_len, num_coarse_actions + num_fine_actions + 1], fill_value=np.nan, dtype=np.float32)
    y_coarse = np.full([output_seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
    y_fine = np.full([output_seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
    boundary = np.full(output_seq_len, fill_value=np.nan, dtype=np.float32)
    padded_actions, padded_lengths = add_padder_actions_and_lens(actions[i:], lengths[i:])
    new_actions, new_lengths = actions[:i] + padded_actions, lengths[:i] + padded_lengths
    new_acc_lengths = list(accumulate(new_lengths))
    max_future_length = min(len(new_actions[i + 1:]), output_seq_len)
    for j, m in zip(range(i + 1, i + 1 + max_future_length), range(max_future_length)):
        parent_coarse_action = new_actions[j - 1][0]
        parent_coarse_action_id = coarse_action_to_id[parent_coarse_action]
        x_fine[m, parent_coarse_action_id] = 1.0
        previous_fine_action = new_actions[j - 1][1]
        previous_fine_action_id = fine_action_to_id.get(previous_fine_action)
        if previous_fine_action_id is not None:
            x_fine[m, num_coarse_actions + previous_fine_action_id] = 1.0
        acc_local_len = get_local_acc_len(new_actions, new_lengths, j - 1)
        parent_coarse_action_length = extract_last_action_and_full_length(coarse_actions_per_frame,
                                                                          new_acc_lengths[j - 1] - 1)[1]
        x_fine[m, -1] = acc_local_len / parent_coarse_action_length

        future_fine_action_length = new_lengths[j]
        if future_fine_action_length:
            future_fine_action = new_actions[j][1]
            future_fine_action_id = fine_action_to_id[future_fine_action]
            y_fine[m, 0] = future_fine_action_id
            parent_coarse_action_length = extract_last_action_and_full_length(coarse_actions_per_frame,
                                                                              new_acc_lengths[j] - 1)[1]
            y_fine[m, -1] = future_fine_action_length / parent_coarse_action_length
            boundary[m] = 0.0
        else:
            x_coarse[m, parent_coarse_action_id] = 1.0
            x_coarse[m, -1] = new_acc_lengths[j - 1] / num_frames

            future_parent_coarse_action = new_actions[j][0]
            future_parent_coarse_action_id = coarse_action_to_id.get(future_parent_coarse_action)
            if future_parent_coarse_action_id is not None:
                y_coarse[m, 0] = future_parent_coarse_action_id
                future_parent_coarse_action_length = extract_last_action_and_full_length(coarse_actions_per_frame,
                                                                                         new_acc_lengths[j])[1]
                y_coarse[m, -1] = future_parent_coarse_action_length / num_frames
            elif add_final_action:
                y_coarse[m, 0] = len(coarse_action_to_id)
            boundary[m] = 1.0
            if add_final_action:
                y_fine[m, 0] = len(fine_action_to_id)
    return boundary, x_coarse, x_fine, y_coarse, y_fine


def get_local_acc_len(actions, lengths, i):
    """Accumulated length up to and including i-th action."""
    if i < 0:
        return 0
    parent_action = actions[i][0]
    acc_local_len = 0
    for (coarse_action, _), length in zip(actions[i::-1], lengths[i::-1]):
        if parent_action == coarse_action:
            acc_local_len += length
        else:
            break
    return acc_local_len


def add_padder_actions_and_lens(actions, lengths):
    actions_copy, lengths_copy = list(actions), list(lengths)
    already_inserted = 0
    for i in range(1, len(actions)):
        curr_coarse_action, prev_coarse_action = actions[i][0], actions[i - 1][0]
        if curr_coarse_action != prev_coarse_action:
            actions_copy.insert(i + already_inserted, (curr_coarse_action, None))
            lengths_copy.insert(i + already_inserted, 0)
            already_inserted += 1
    actions_copy.append((None, None))
    lengths_copy.append(0)
    return actions_copy, lengths_copy
