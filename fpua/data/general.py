from itertools import accumulate
import math
import os

import numpy as np


def merge_arrays(args):
    """Command-line interface for function to merge arrays across .npz files into a single .npz file.

    See the file process_data.py for more information about the attributes of args.
    """
    arrays_paths = args.arrays
    save_path = args.save_path
    save_name = args.save_name

    arrays = []
    for array_path in arrays_paths:
        arrays.append({})
        with np.load(array_path) as data:
            for name in data.files:
                arrays[-1][name] = data[name]
    merged_arrays = _merge_arrays(arrays)

    if save_path is not None:
        file_name = save_name if save_name is not None else 'training_data'
        save_file = os.path.join(save_path, file_name)
        np.savez_compressed(save_file, **merged_arrays)
        print('Merged arrays successfully written to %s' % (save_file + '.npz'))


def _merge_arrays(arrays):
    """Merge a list of arrays mapped across multiple dictionaries.

    Arg(s):
        arrays - A list of dictionaries, where each individual dictionary contains a number of arrays.
    Returns:
        A dictionary mapping each array name to the concatenation of the respective array across the dictionaries.
        The concatenation is done on the first dimension.
    """
    names = arrays[0].keys()
    result = {}
    for name in names:
        for array in arrays:
            result.setdefault(name, []).append(array[name])
    result = {k: np.concatenate(v, axis=0) for k, v in result.items()}
    return result


def extend_smallest_list(a, b, extension_val=None):
    """Extend the smallest list to match the length of the longest list.

    If extension_val is None, the extension is done by repeating the last element of the list. Otherwise, use
    extension_val.

    Arg(s):
        a - A list.
        b - A list.
        extension_val - Extension value.
    Returns:
        The input lists with the smallest list extended to match the size of the longest list.
    """
    gap = abs(len(a) - len(b))
    if len(a) > len(b):
        extension_val = extension_val if extension_val is not None else b[-1]
        b += [extension_val] * gap
    else:
        extension_val = extension_val if extension_val is not None else a[-1]
        a += [extension_val] * gap
    return a, b


def convert_segmentation_to_frame(segmentation, silence_action='SILENCE'):
    """Convert segmentation information to frame-wise information.

    Gaps between the actions are filled with the silence_action given. Also, if the end frame of an action coincides
    with the start frame of the next action, the start of the next action is increased by 1 frame.

    Arg(s):
        segmentation - A list of segmented actions in a video. Each element in the list is a str in the format
            'A-B ACTION', where A is the starting frame, B is the ending frame, and ACTION is the action conducted
            during the segment A-B. It is assumed that the actions are sorted by the starting frame.
        silence_action - In case there are gaps between the actions, these gaps are filled with the action
            specified in silence_action.
    Returns:
        A list of actions, where each original action in a segment A-B appears repeated B - A + 1 times. If the
        start of an action coincides with the end of the previous action, the action appears repeated B - (A + 1) + 1
        = B - A times.
    """
    actions_per_frame = []
    for segment in segmentation:
        frames, action = segment.split(sep=' ')
        start_frame, end_frame = frames.split(sep='-')
        start_frame, end_frame = int(start_frame), int(end_frame)
        if start_frame < len(actions_per_frame):
            if end_frame > len(actions_per_frame):
                start_frame = len(actions_per_frame) + 1
            else:
                continue
        if start_frame == len(actions_per_frame):
            num_frames = end_frame - start_frame
        else:
            gap = start_frame - len(actions_per_frame) - 1
            actions_per_frame.extend([silence_action] * gap)
            num_frames = end_frame - start_frame + 1
        actions_per_frame.extend([action] * num_frames)
    return actions_per_frame


def convert_segmentation_files(path, silence_action='SILENCE', save_path=None):
    """Convert all segmentation files in a directory to the frame-wise representation.

    Read all segmentation .txt files in a given directory and convert them to the frame-wise representation. The
    converted files can optionally be saved to another directory.

    Arg(s):
        path - str representing the directory containing the segmentation .txt files. Files without a .txt suffix
            are ignored.
        silence_action - In case there are gaps between the actions, these gaps are filled with the action
            specified in silence_action.
        save_path - Directory to save the converted representations (optional).
    Returns:
        A list of lists, where each element list contains the actions that happened frame-wise.
    """
    actions = []
    segmentation_files = (file for file in os.listdir(path) if file.endswith('.txt'))
    for segmentation_file in segmentation_files:
        with open(os.path.join(path, segmentation_file), mode='r') as f:
            segmentation = [line.rstrip() for line in f]
        actions_per_frame = convert_segmentation_to_frame(segmentation, silence_action=silence_action)
        actions.append(actions_per_frame)
        if save_path is not None:
            with open(os.path.join(save_path, segmentation_file), mode='w') as f:
                f.writelines(action + '\n' for action in actions_per_frame)
    return actions


def _generate_action_dictionary(path, ignore=None):
    """Crawl through all action files in a given directory and create a dictionary mapping each action to a unique ID.

    The .txt files are assumed to be in the frame-wise representation.

    Arg(s):
        path - A directory containing all actions .txt files for creation of the mapping dictionary.
        ignore - Set containing actions to be ignored. If None, include every action found.
    Returns:
        A dictionary mapping each action to a unique ID.
    """
    if ignore is None:
        ignore = set()
    action_dict, unique_id = {}, 0
    files = (file for file in os.listdir(path) if file.endswith('.txt'))
    for file in files:
        with open(os.path.join(path, file), mode='r') as f:
            for line in f:
                line = line.rstrip()
                if line not in action_dict and line not in ignore:
                    action_dict[line] = unique_id
                    unique_id += 1
    return action_dict


def generate_action_dictionary(args):
    """Command-line interface to generate and save mapping of actions to unique IDs.

    See the file process_data.py for an explanation of the attributes in args.
    """
    path = args.path
    ignore = set(args.ignore) if args.ignore is not None else None
    save_file = args.save_file

    action_dict = _generate_action_dictionary(path, ignore=ignore)

    if save_file is not None:
        actions_sorted_by_id = sorted(action_dict.items(), key=lambda kv: kv[1])
        with open(save_file, mode='w') as f:
            for action, action_id in actions_sorted_by_id:
                f.write(action + ' ' + str(action_id) + '\n')
        print(f'Action to ID mapping successfully written to {save_file}')


def reverse_action_dictionary(args):
    dict_file = args.dict_file
    save_file = args.save_file

    rev_dict = {}
    with open(dict_file, mode='r') as f:
        for line in f:
            a, b = line.rstrip().split(sep=' ')
            rev_dict[b] = a
    if save_file is not None:
        with open(save_file, mode='w') as f:
            for b, a in rev_dict.items():
                f.write(f'{b} {a}\n')
        print(f'Reverse dictionary successfully written to {save_file}')


def aggregate_actions_and_lengths(actions_per_frame):
    """Identify the actions in a video and count how many frames they last for.

    Given a list of actions (e.g. ['a', 'a', 'a', 'b', 'b']) summarise the actions and count their lifespan. For
    the example input just given, the function returns (['a', 'b'], [3, 2]).

    Arg(s):
        actions_per_frame - list containing the actions per frame.
    Returns:
        A tuple containing two lists. The first list contains the actions that happened throughout the video in
        sequence, whereas the second list contains the count of frames for which each action lived for.
    """
    actions, lengths = [], []
    if not actions_per_frame:
        return actions, lengths
    start_idx = 0
    for i, action in enumerate(actions_per_frame):
        if action != actions_per_frame[start_idx]:
            actions.append(actions_per_frame[start_idx])
            lengths.append(i - start_idx)
            start_idx = i
    actions.append(actions_per_frame[start_idx])
    lengths.append(len(actions_per_frame) - start_idx)
    return actions, lengths


def read_action_dictionary(file_path):
    """Create dictionary from input file.

    Each line of the input file contains an action followed by its ID.

    Arg(s):
        file_path - Path to .txt file containing mapping between actions and IDs.
    Returns:
        A dictionary mapping an action to an unique ID.
    """
    action_dict = {}
    with open(file_path, mode='r') as f:
        for line in f:
            line = line.rstrip()
            action, action_id = line.split()
            action_id = int(action_id)
            action_dict[action] = action_id
    return action_dict


def split_observed_actions(actions_per_frame, fraction_observed=0.2, fraction_unobserved=None):
    """Given frame-wise actions in a video, split them into observed and unobserved actions.

    Arg(s):
        actions_per_frame - A list containing the actions in the video per frame.
        fraction_observed - Fraction of the video that was observed.
        fraction_unobserved - If not specified, defaults to 1 - fraction_observed.
    Returns:
        Two lists. The first list contains the observed actions and the second list contains the unobserved actions.
    """
    fraction_unobserved = fraction_unobserved if fraction_unobserved is not None else 1 - fraction_observed
    assert fraction_observed + fraction_unobserved <= 1.0

    num_frames = len(actions_per_frame)
    num_observed_frames = max(1, math.floor(fraction_observed * num_frames))
    observed_actions_per_frame = actions_per_frame[:num_observed_frames]
    num_unobserved_frames = min(round(fraction_unobserved * num_frames), num_frames - num_observed_frames)
    unobserved_actions_per_frame = actions_per_frame[num_observed_frames:num_observed_frames + num_unobserved_frames]
    return observed_actions_per_frame, unobserved_actions_per_frame


def extend_or_trim_predicted_actions(predicted_actions, unobserved_actions):
    """Extend or trim predicted actions to match the number of unobserved actions.

    If the list of predicted actions is smaller than the list of unobserved actions, the last action in the list
    of predicted actions is repeated until the length of the predicted actions match the length of the unobserved
    actions. If the list of predicted actions is longer than the list of unobserved actions, return the
    first predictions that match the length of the list of unobserved actions.

    Arg(s):
        predicted_actions - List containing the predicted actions.
        unobserved_actions - List containing the unobserved actions.
    Returns:
        The list of predicted actions either extended or trimmed to match the length of the unobserved actions.
    """
    if len(predicted_actions) < len(unobserved_actions):
        predicted_actions, _ = extend_smallest_list(predicted_actions, unobserved_actions)
    else:
        predicted_actions = predicted_actions[:len(unobserved_actions)]
    return predicted_actions


def extract_last_action_and_observed_length(actions_per_frame, cut):
    """Extract the last observed action and its observed length.

    Arg(s):
        actions_per_frame - List containing the actions in a video frame-wise.
        cut - The last observed frame.
    Returns:
        The last observed action and its observed length.
    """
    last_action = actions_per_frame[cut]
    last_action_observed_length = 0
    for action in actions_per_frame[cut::-1]:
        if action == last_action:
            last_action_observed_length += 1
        else:
            break
    return last_action, last_action_observed_length


def extract_last_action_and_remlength(actions_per_frame, cut):
    """Extract the last observed action and the length of the unobserved part.

    Arg(s):
        actions_per_frame - List containing the actions in a video frame-wise.
        cut - The last observed frame.
    Returns:
        The last observed action and its remaining length.
    """
    last_action = actions_per_frame[cut]
    last_action_rem_length = 0
    for action in actions_per_frame[cut + 1:]:
        if action == last_action:
            last_action_rem_length += 1
        else:
            break
    return last_action, last_action_rem_length


def extract_last_action_and_full_length(actions_per_frame, cut):
    """Extract the last observed action and its complete length (observed + unobserved).

    Arg(s):
        actions_per_frame - List containing the actions in a video frame-wise.
        cut - The last observed frame.
    Returns:
        The last observed action and its complete length.
    """
    action, observed_length = extract_last_action_and_observed_length(actions_per_frame, cut)
    _, unobserved_length = extract_last_action_and_remlength(actions_per_frame, cut)
    return action, observed_length + unobserved_length


def extract_observed_actions_and_lengths(actions_per_frame, cut):
    """Extract observed actions and their observed lengths.

    Arg(s):
        actions_per_frame - List containing the actions in a video frame-wise.
        cut = The last observed frame.
    Returns:
        A list containing all observed actions and another list containing their observed lengths.
    """
    actions_per_frame = actions_per_frame[:cut + 1]
    actions, lengths = aggregate_actions_and_lengths(actions_per_frame)
    return actions, lengths


def extract_next_action_and_length(actions_per_frame, cut):
    """Extract the next action and its length.

    Arg(s):
        actions_per_frame - List containing the actions in a video frame-wise.
        cut - The last observed frame.
    Returns:
        The next action and its length.
    """
    last_action = actions_per_frame[cut]
    next_action, next_action_length, has_seen_next_action = None, 0, False
    for action in actions_per_frame[cut + 1:]:
        if action != last_action:
            if has_seen_next_action:
                break
            next_action = action
            has_seen_next_action = True
            last_action = action
        if next_action is not None:
            next_action_length += 1
    return next_action, next_action_length


def extract_next_actions_and_lengths(actions_per_frame, cut, return_zipped_list=True):
    """Extract all future actions and their lengths.

    Arg(s):
        actions_per_frame - List containing actions per frame.
        cut - Last observed frame.
        return_zipped_list - If True, return a list of action/length pairs. Else, return tow lists, one with the
            actions and the other with the lengths
    Returns:
        If return_zipped_list is True, return a list of tuples, where each tuple is an action/length pair. Else, 
        return two lists, one with the actions and the other with their lengths.
    """
    actions_per_frame = remove_observed_actions(actions_per_frame, cut)
    if not actions_per_frame:
        if return_zipped_list:
            return []
        else:
            return [], []
    actions, lengths = aggregate_actions_and_lengths(actions_per_frame)
    if return_zipped_list:
        return list(zip(actions, lengths))
    else:
        return actions, lengths


def remove_observed_actions(actions_per_frame, cut):
    """Remove all frames (including unobserved ones) for all observed actions.

    Arg(s):
        actions_per_frame - List containing actions per frame.
        cut - Last observed frame.
    Returns:
        A list with the subset of frames containing only unobserved actions.
    """
    last_action = actions_per_frame[cut]
    unobserved_actions = actions_per_frame[cut + 1:]
    for idx, action in enumerate(unobserved_actions):
        if action != last_action:
            return unobserved_actions[idx:]
    return []


def identify_action_initial_frame(actions_per_frame, cut):
    """Identify the initial frame of an action based on the last observed frame.

    Arg(s):
        actions_per_frame - List containing actions per frame.
        cut - Last observed frame.
    Returns:
        The initial frame of the action that is happening at cut frame.
    """
    last_action = actions_per_frame[cut]
    initial_frame = cut
    for frame in range(cut - 1, -1, -1):
        if actions_per_frame[frame] != last_action:
            break
        initial_frame -= 1
    return initial_frame


def identify_action_final_frame(actions_per_frame, cut):
    """Identify the final frame of an action based on the last observed frame.

    Arg(s):
        actions_per_frame - List containing actions per frame.
        cut - Last observed frame.
    Returns:
        The final frame of the action that is happening at cut frame.
    """
    last_action = actions_per_frame[cut]
    final_frame = cut
    for frame in range(cut + 1, len(actions_per_frame)):
        if actions_per_frame[frame] != last_action:
            break
        final_frame += 1
    return final_frame


def check_file_existence(path, file_name):
    files = os.listdir(path)
    return file_name in files


def maybe_read_action_file(file_name):
    """Read a file containing the frame-wise labels of a video if it exists. Returns None if the file does not exist."""
    try:
        with open(file_name, mode='r') as f:
            actions_per_frame = [line.rstrip() for line in f]
    except FileNotFoundError:
        actions_per_frame = None
    return actions_per_frame


def convert_action_names_to_ids(next_actions_and_lengths, action_to_id):
    """Translate action names to action IDs.

    Arg(s):
        next_actions_and_lengths - A list of tuples, where each tuple is an action/length pair.
        action_to_id - Dictionary mapping actions to IDs.
    Returns:
        A list of tuples identical to the input one, but with the action names translated to action IDs.
    """
    next_actions_and_lengths = [(action_to_id[action], length) for action, length in next_actions_and_lengths]
    return next_actions_and_lengths


def count_num_actions(actions_per_frame):
    """Count the number of actions in a video."""
    actions, _ = aggregate_actions_and_lengths(actions_per_frame)
    return len(actions)


def reconstruct_sequence_from_segmentation(actions, lengths):
    """Reconstruct sequence of actions from segmentation information.

    Given a list of actions and list of their lengths, create a list of actions where the i-th action appears
    lengths[i] times.
    """
    actions_per_frame = []
    for action, length in zip(actions, lengths):
        actions_per_frame += [action] * length
    return actions_per_frame


def hierarchical_segmentation_from_single_annotations(args):
    """Given frame-wise annotations of coarse and fine levels, write out the hierarchical segmentation."""
    coarse_path = args.coarse_path
    fine_path = args.fine_path
    save_path = args.save_path

    _write_hierarchical_segmentation_from_annotations(coarse_path, fine_path, save_path)


def _write_hierarchical_segmentation_from_annotations(coarse_path, fine_path, save_path):
    coarse_labels = set(os.listdir(coarse_path))
    fine_labels = set(os.listdir(fine_path))
    labels = sorted(coarse_labels & fine_labels)

    for label in labels:
        with open(os.path.join(coarse_path, label), mode='r') as f:
            coarse_actions_per_frame = [line.rstrip() for line in f]
        with open(os.path.join(fine_path, label), mode='r') as f:
            fine_actions_per_frame = [line.rstrip() for line in f]
        actions, lengths = aggregate_actions_and_lengths(list(zip(coarse_actions_per_frame, fine_actions_per_frame)))
        acc_lengths = list(accumulate(lengths))
        with open(os.path.join(save_path, label), mode='w') as f:
            previous_coarse_action = None
            for i, ((coarse_action, fine_action), length) in enumerate(zip(actions, lengths)):
                if coarse_action != previous_coarse_action:
                    previous_coarse_action = coarse_action
                    start_frame, end_frame = extract_action_segment(coarse_actions_per_frame, acc_lengths[i] - 1)
                    f.write(f'{start_frame + 1}-{end_frame + 1}\t{coarse_action}\n')
                if i:
                    start_frame = acc_lengths[i - 1] + 1
                else:
                    start_frame = 1
                end_frame = acc_lengths[i]
                f.write(f'    {start_frame}-{end_frame}\t{fine_action}\n')


def extract_action_segment(actions_per_frame, cut):
    """Return the first and the last frame, both inclusive, of the action at frame cut."""
    start_frame = identify_action_initial_frame(actions_per_frame, cut)
    end_frame = identify_action_final_frame(actions_per_frame, cut)
    return start_frame, end_frame


def write_framewise_actions(actions_per_frame, save_file):
    """Write out to save file the actions in actions per frame."""
    with open(save_file, mode='w') as f:
        f.writelines(action + '\n' for action in actions_per_frame)


def actions_from_steps(predicted_steps):
    predicted_actions = []
    for predicted_action, predicted_action_length in predicted_steps:
        if predicted_action is not None:
            predicted_actions += [predicted_action] * predicted_action_length
    return predicted_actions


def maybe_rebalance_steps(predicted_steps, maximum_prediction_length):
    total_pred_length = 0
    for _, pred_length in predicted_steps:
        if pred_length is not None:
            total_pred_length += pred_length
    if total_pred_length == 0 or total_pred_length >= maximum_prediction_length:
        return predicted_steps
    gap = maximum_prediction_length - total_pred_length
    updated_steps, total_new_length = [], 0
    for action, length in predicted_steps:
        if action is not None:
            new_length = length + round(gap * (length / total_pred_length))
            total_new_length += new_length
        else:
            new_length = length
        updated_steps.append([action, new_length])
    if total_new_length < maximum_prediction_length:  # In case rebalance was not perfect
        gap = maximum_prediction_length - total_new_length
        for i, (_, length) in zip(range(-1, -len(updated_steps) - 1, -1), updated_steps[::-1]):
            if length is not None:
                updated_steps[i][1] += gap
                break
    return updated_steps


def clean_directory(path, extension=None):
    """Delete all files within path."""
    if extension is None:
        files = os.listdir(path)
    else:
        files = [file for file in os.listdir(path) if file.endswith(extension)]
    for file in files:
        os.remove(os.path.join(path, file))
