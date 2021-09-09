from itertools import accumulate
import os

import numpy as np
from sklearn.utils import shuffle

from fpua.data.general import read_action_dictionary, extend_smallest_list, aggregate_actions_and_lengths
from fpua.data.general import identify_action_initial_frame


def generate_baselines_training_data(args):
    fine_labels_path = args.fine_labels_path
    coarse_labels_path = args.coarse_labels_path
    fine_action_to_id = read_action_dictionary(args.fine_action_to_id)
    coarse_action_to_id = read_action_dictionary(args.coarse_action_to_id)
    seq_len = args.seq_len
    ignore_silence_action = args.ignore_silence_action
    add_final_action = args.add_final_action
    is_validation = args.is_validation
    save_path = args.save_path
    save_name = args.save_name

    tensors_dict = _generate_baselines_training_data(fine_labels_path, coarse_labels_path,
                                                     fine_action_to_id=fine_action_to_id,
                                                     coarse_action_to_id=coarse_action_to_id,
                                                     seq_len=seq_len,
                                                     ignore_silence_action=ignore_silence_action,
                                                     add_final_action=add_final_action,
                                                     is_validation=is_validation)
    print('Generated %d training examples.' % len(tensors_dict['x_enc_fine']))
    if save_path is not None:
        file_name = save_name if save_name is not None else 'training_data'
        save_file = os.path.join(save_path, file_name + '.npz')
        np.savez_compressed(save_file, **tensors_dict)
        print('Training data successfully written to %s' % save_file)


def _generate_baselines_training_data(fine_labels_path, coarse_labels_path, fine_action_to_id, coarse_action_to_id,
                                      seq_len, ignore_silence_action, add_final_action, is_validation):
    xs_enc_fine, xs_enc_coarse = [], []
    ys_enc_fine, ys_enc_coarse = [], []
    effective_num_steps_per_video = []
    tensors = [xs_enc_fine, xs_enc_coarse, ys_enc_fine, ys_enc_coarse, effective_num_steps_per_video]

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
        if ignore_silence_action is not None:
            fine_actions_per_frame = [fine_action for fine_action in fine_actions_per_frame
                                      if fine_action != ignore_silence_action]
            coarse_actions_per_frame = [coarse_action for coarse_action in coarse_actions_per_frame
                                        if coarse_action != ignore_silence_action]
            error_msg = 'Action levels do not match after removing silence action.'
            assert len(fine_actions_per_frame) == len(coarse_actions_per_frame), error_msg
        if is_validation:
            single_video_tensors = \
                _generate_baselines_validation_from_single_video(fine_actions_per_frame,
                                                                 coarse_actions_per_frame,
                                                                 fine_action_to_id=fine_action_to_id,
                                                                 coarse_action_to_id=coarse_action_to_id,
                                                                 seq_len=seq_len,
                                                                 add_final_action=add_final_action)
        else:
            single_video_tensors = \
                _generate_baselines_training_data_from_single_video(fine_actions_per_frame,
                                                                    coarse_actions_per_frame,
                                                                    fine_action_to_id=fine_action_to_id,
                                                                    coarse_action_to_id=coarse_action_to_id,
                                                                    seq_len=seq_len,
                                                                    add_final_action=add_final_action)
        for tensor_list, single_video_tensor in zip(tensors, single_video_tensors):
            tensor_list.append(single_video_tensor)
    tensors = [np.concatenate(tensor_list, axis=0) for tensor_list in tensors]
    tensors = shuffle(*tensors, random_state=42)
    names = ['x_enc_fine', 'x_enc_coarse', 'y_enc_fine', 'y_enc_coarse', 'effective_num_steps']
    tensors_dict = dict(zip(names, tensors))
    return tensors_dict


def _generate_baselines_training_data_from_single_video(fine_actions_per_frame, coarse_actions_per_frame,
                                                        fine_action_to_id, coarse_action_to_id,
                                                        seq_len, add_final_action):
    actions, lengths = aggregate_actions_and_lengths(list(zip(coarse_actions_per_frame, fine_actions_per_frame)))
    acc_lengths = list(accumulate(lengths))
    num_frames = len(fine_actions_per_frame)
    num_fine_actions, num_coarse_actions = len(fine_action_to_id), len(coarse_action_to_id)
    # Tensors
    x_enc_fine = np.full([1, seq_len, num_fine_actions + 1], fill_value=np.nan, dtype=np.float32)
    x_enc_coarse = np.full([1, seq_len, num_coarse_actions + 1], fill_value=np.nan, dtype=np.float32)
    y_enc_fine = np.full([1, seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
    y_enc_coarse = np.full([1, seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
    min_lens = np.full(1, fill_value=np.nan, dtype=np.float32)

    input_actions, input_acc_lengths = actions, acc_lengths
    min_len = min(len(input_actions), seq_len)
    min_lens[0] = min_len
    for i in range(min_len):
        input_coarse_action, input_fine_action = input_actions[i]
        input_coarse_action_id = coarse_action_to_id[input_coarse_action]
        x_enc_coarse[0, i, input_coarse_action_id] = 1.0
        input_fine_action_id = fine_action_to_id[input_fine_action]
        x_enc_fine[0, i, input_fine_action_id] = 1.0
        input_acc_length = input_acc_lengths[i - 1] if i > 0 else 0
        x_enc_coarse[0, i, -1] = input_acc_length / num_frames
        x_enc_fine[0, i, -1] = input_acc_length / num_frames
    output_actions, output_lengths = actions, lengths
    for i in range(min_len):
        try:
            output_coarse_action, output_fine_action = output_actions[i + 1]
        except IndexError:
            output_coarse_action = output_fine_action = None
        output_coarse_action_id = coarse_action_to_id.get(output_coarse_action)
        if output_coarse_action_id is not None:
            y_enc_coarse[0, i, 0] = output_coarse_action_id
        elif add_final_action:
            y_enc_coarse[0, i, 0] = len(coarse_action_to_id)
        y_enc_coarse[0, i, 1] = output_lengths[i] / num_frames
        output_fine_action_id = fine_action_to_id.get(output_fine_action)
        if output_fine_action_id is not None:
            y_enc_fine[0, i, 0] = output_fine_action_id
        elif add_final_action:
            y_enc_fine[0, i, 0] = len(fine_action_to_id)
        y_enc_fine[0, i, 1] = output_lengths[i] / num_frames
    tensors = [x_enc_fine, x_enc_coarse, y_enc_fine, y_enc_coarse, min_lens]
    return tensors


def _generate_baselines_validation_from_single_video(fine_actions_per_frame, coarse_actions_per_frame,
                                                     fine_action_to_id, coarse_action_to_id, seq_len,
                                                     add_final_action):
    observed_fractions = [0.2, 0.3]
    num_frames = len(fine_actions_per_frame)
    num_fine_actions, num_coarse_actions = len(fine_action_to_id), len(coarse_action_to_id)
    num_examples = len(observed_fractions)
    # Tensors
    x_enc_fine = np.full([num_examples, seq_len, num_fine_actions + 1], fill_value=np.nan, dtype=np.float32)
    x_enc_coarse = np.full([num_examples, seq_len, num_coarse_actions + 1], fill_value=np.nan, dtype=np.float32)
    y_enc_fine = np.full([num_examples, seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
    y_enc_coarse = np.full([num_examples, seq_len, 1 + 1], fill_value=np.nan, dtype=np.float32)
    min_lens = np.full(num_examples, fill_value=np.nan, dtype=np.float32)

    for i, observed_fraction in enumerate(observed_fractions):
        # Observed
        num_frames_to_grab = round(num_frames * observed_fraction)
        obs_fine_actions_per_frame = fine_actions_per_frame[:num_frames_to_grab]
        obs_coarse_actions_per_frame = coarse_actions_per_frame[:num_frames_to_grab]
        actions, lengths = aggregate_actions_and_lengths(list(zip(obs_coarse_actions_per_frame,
                                                                  obs_fine_actions_per_frame)))
        acc_lengths = list(accumulate(lengths))
        min_len = min(len(actions), seq_len)
        min_lens[i] = min_len
        for t in range(min_len):
            input_coarse_action, input_fine_action = actions[t]
            input_coarse_action_id = coarse_action_to_id[input_coarse_action]
            x_enc_coarse[i, t, input_coarse_action_id] = 1.0
            input_fine_action_id = fine_action_to_id[input_fine_action]
            x_enc_fine[i, t, input_fine_action_id] = 1.0
            input_acc_length = acc_lengths[t - 1] if t > 0 else 0
            x_enc_coarse[i, t, -1] = input_acc_length / num_frames
            x_enc_fine[i, t, -1] = input_acc_length / num_frames
        # Unobserved
        initial_frame = identify_action_initial_frame(fine_actions_per_frame, num_frames_to_grab - 1)
        unobs_fine_actions_per_frame = fine_actions_per_frame[initial_frame:]
        unobs_coarse_actions_per_frame = coarse_actions_per_frame[initial_frame:]
        actions, lengths = aggregate_actions_and_lengths(list(zip(unobs_coarse_actions_per_frame,
                                                                  unobs_fine_actions_per_frame)))
        min_len = min(len(actions), seq_len)
        for t in range(min_len):
            try:
                output_coarse_action, output_fine_action = actions[t + 1]
            except IndexError:
                output_coarse_action = output_fine_action = None
            output_coarse_action_id = coarse_action_to_id.get(output_coarse_action)
            if output_coarse_action_id is not None:
                y_enc_coarse[i, t, 0] = output_coarse_action_id
            elif add_final_action:
                y_enc_coarse[i, t, 0] = len(coarse_action_to_id)
            y_enc_coarse[i, t, 1] = lengths[t] / num_frames
            output_fine_action_id = fine_action_to_id.get(output_fine_action)
            if output_fine_action_id is not None:
                y_enc_fine[i, t, 0] = output_fine_action_id
            elif add_final_action:
                y_enc_fine[i, t, 0] = len(fine_action_to_id)
            y_enc_fine[i, t, 1] = lengths[t] / num_frames
    tensors = [x_enc_fine, x_enc_coarse, y_enc_fine, y_enc_coarse, min_lens]
    return tensors
