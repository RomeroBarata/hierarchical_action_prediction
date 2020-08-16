from collections import defaultdict
import os

import numpy as np

from fpua.analysis import compute_moc, compute_segmental_edit_score_multiple_videos, overlap_f1_multiple_videos
from fpua.data.general import split_observed_actions, read_action_dictionary


def test_dummy_model(args):
    labels_path = args.labels_path
    action_to_id = read_action_dictionary(args.action_to_id)
    fraction_observed = args.observed_fraction
    fraction_unobserved = args.unobserved_fraction
    ignore = set(args.ignore) if args.ignore is not None else set()

    predicted_actions_per_video, unobserved_actions_per_video = [], []
    label_files = (file for file in os.listdir(labels_path) if file.endswith('.txt'))
    for label_file in label_files:
        with open(os.path.join(labels_path, label_file), mode='r') as f:
            actions_per_frame = [line.rstrip() for line in f if line.rstrip() not in ignore]
        observed_actions, unobserved_actions = split_observed_actions(actions_per_frame,
                                                                      fraction_observed=fraction_observed,
                                                                      fraction_unobserved=fraction_unobserved)
        predicted_actions = [observed_actions[-1]] * len(unobserved_actions)
        predicted_actions_per_video.append(np.array(predicted_actions))
        unobserved_actions_per_video.append(np.array(unobserved_actions))
    predicted_actions = np.concatenate(predicted_actions_per_video)
    unobserved_actions = np.concatenate(unobserved_actions_per_video)
    print('\nObserved fraction: %.2f | Unobserved fraction: %.2f' % (fraction_observed, fraction_unobserved))
    print('MoF Accuracy: %.4f' % np.mean(predicted_actions == unobserved_actions).item())
    moc, correct_per_class, wrong_per_class = compute_moc(predicted_actions, unobserved_actions, action_to_id)
    moc_dict = {f'moc-{fraction_observed}_{fraction_unobserved}': moc}
    print('MoC Accuracy: %.4f' % moc)
    seg_edit_score = compute_segmental_edit_score_multiple_videos(unobserved_actions_per_video,
                                                                  predicted_actions_per_video)
    print('Segmental Edit Score: %.4f' % seg_edit_score)
    num_classes = len(action_to_id)
    f1_dict = {}
    for overlap in [0.10, 0.25, 0.50]:
        overlap_f1_score = overlap_f1_multiple_videos(unobserved_actions_per_video,
                                                      predicted_actions_per_video, action_to_id=action_to_id,
                                                      num_classes=num_classes, overlap=overlap)
        f1_dict[f'{fraction_observed}_{fraction_unobserved}_{overlap}'] = overlap_f1_score
        print('F1@%.2f: %.4f' % (overlap, overlap_f1_score))
    result_dict = {**f1_dict, **moc_dict}
    return result_dict


def test_dummy_model_cv(args):
    labels_root_path = args.labels_root_path

    splits = '01 02 03 04 05'.split(sep=' ')
    split_folders = ['S' + split for split in splits]
    results = {}
    for split_folder in split_folders:
        labels_path = os.path.join(labels_root_path, split_folder)
        args.labels_path = labels_path
        try:
            split_results = test_dummy_model(args)
        except FileNotFoundError:  # Happens when trying split 05 on Breakfast
            continue
        results[split_folder] = split_results
    results_to_average = defaultdict(list)
    for _, result in results.items():
        for key, score in result.items():
            results_to_average[key].append(score)
    print('\nCross-validated results')
    for key, result in results_to_average.items():
        print(f'{key:<15}: {np.array(result).mean().item():.4f} +/- {np.array(result).std().item():.4f}')
