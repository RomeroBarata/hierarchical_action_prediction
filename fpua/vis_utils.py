from collections import defaultdict
import math
import os
import random

import cv2 as cv
import numpy as np

from fpua.data.general import read_action_dictionary, clean_directory


def generate_colour_dict(args):
    fine_actions = set(read_action_dictionary(args.fine_action_to_id).keys())
    coarse_actions = set(read_action_dictionary(args.coarse_action_to_id).keys())
    actions = sorted(fine_actions | coarse_actions)
    colours = _generate_random_colours(len(actions))
    save_path = args.save_path
    if save_path is not None:
        save_name = args.save_name
        file_name = os.path.join(save_path, save_name + '.txt')
        with open(file_name, mode='w') as f:
            for action, (b, g, r) in zip(actions, colours):
                f.write(f'{action} {b} {g} {r}\n')
        print(f'Colour dictionary successfully written to {file_name}')


def read_colour_dict(colour_dict_path):
    if colour_dict_path is None:
        return {}
    colour_dict = {}
    with open(colour_dict_path, mode='r') as f:
        for line in f:
            line = line.rstrip()
            action, b, g, r = line.split(sep=' ')
            colour_dict[action] = (int(b), int(g), int(r))
    return colour_dict


def draw_visualisation(args):
    """Function to generate visualisation for the predicted hierarchies.

    This function can take an arbitrary number of directories to process, and will include the input and output of
    such models in a single image.
    """
    random.seed(42)
    files_paths = args.files_paths
    colour_dict = read_colour_dict(args.colour_path)
    draw_only_first_ground_truth = args.draw_only_first_ground_truth
    box_height = args.box_height
    box_width_scale = args.box_width_scale
    split_long_words = args.split_long_words
    font_scale = args.font_scale
    write_label_ids = args.write_label_ids
    coarse_action_to_label_id = {}
    fine_action_to_label_id = {}
    save_path = args.save_path

    if save_path is not None:
        clean_directory(save_path, extension='.png')
    img_height = _img_height_from_box_height(box_height)
    if len(files_paths) == 1:  # Only one method to create visualisation.
        files_paths = files_paths[0]
        files = (file for file in os.listdir(files_paths) if _file_name_is_okay(file))
        for file in files:
            _draw_visualisation_from_txt_file(os.path.join(files_paths, file),
                                              colour_dict=colour_dict, save_path=save_path,
                                              img_height=img_height, vertical_gap=box_height,
                                              box_width_scale=box_width_scale, split_long_words=split_long_words,
                                              font_scale=font_scale,
                                              coarse_action_to_label_id=coarse_action_to_label_id,
                                              fine_action_to_label_id=fine_action_to_label_id,
                                              write_label_ids=write_label_ids)
            if save_path is not None:
                save_file = os.path.join(save_path, file.split(sep='.')[0] + '.png')
                print(f'Action segmentation for {file} saved to {save_file}')
    else:  # Multiple methods to visualise in the same image.
        # Only draw visualisation for the videos that all models generated predictions.
        files = {file for file in os.listdir(files_paths[0]) if _file_name_is_okay(file)}
        for file_path in files_paths[1:]:
            files.intersection_update({file for file in os.listdir(file_path) if _file_name_is_okay(file)})
        img_height = img_height * len(files_paths)  # Height dependent on the number of methods to draw.
        vertical_gap = box_height
        for file in sorted(files):
            output = None, 0  # current image, current height to start drawing
            skip_ground_truth = False
            for i, file_path in enumerate(files_paths):
                # If drawing the last method, do not return the image and write it out to the disk
                return_img = False if i == len(files_paths) - 1 else True
                img, top_margin = output
                output = _draw_visualisation_from_txt_file(os.path.join(file_path, file),
                                                           colour_dict=colour_dict, save_path=save_path,
                                                           initial_img=img, img_height=img_height,
                                                           top_margin=top_margin, vertical_gap=vertical_gap,
                                                           return_img=return_img, skip_ground_truth=skip_ground_truth,
                                                           box_width_scale=box_width_scale,
                                                           split_long_words=split_long_words,
                                                           font_scale=font_scale,
                                                           coarse_action_to_label_id=coarse_action_to_label_id,
                                                           fine_action_to_label_id=fine_action_to_label_id,
                                                           write_label_ids=write_label_ids)
                skip_ground_truth = draw_only_first_ground_truth
            if save_path is not None:
                # Output message for user. The image is actually written out inside _draw_visualisation_from_txt_file
                save_file = os.path.join(save_path, file.split(sep='.')[0] + '.png')
                print(f'Action segmentation for {file} saved to {save_file}')


def _draw_visualisation_from_txt_file(file, colour_dict, save_path=None, initial_img=None,
                                      img_height=600, top_margin=0, vertical_gap=100, return_img=False,
                                      skip_ground_truth=False, skip_prediction=False, box_width_scale=1,
                                      split_long_words=False, font_scale=0.35, coarse_action_to_label_id=None,
                                      fine_action_to_label_id=None, write_label_ids=False):
    """Draw observed, unobserved, and predicted hierarchies from txt file.

    This function works by first inspecting the input txt file and creating a dictionary structure that represents it.
    Each key in the dictionary is related to a level (coarse or fine) and a section (observed, unobserved, predicted).
    In the case of a hierarchical model, the dictionary will have 6 keys (2 * 3); in the case of a single level
    to single level model, the dictionary will have 3 keys (1 * 3).

    After building the dictionary with the video sections, the dictionary and current image is given as input
    to draw_*_visualisation to do the actual drawing.

    Args:
        file: Path to file containing observed/unobserved/predicted structure of the video.
        colour_dict: Dictionary mapping actions to their BGR colour tuples.
        save_path: If None, display result image on screen. Else, save to specified path.
    """
    with open(file, mode='r') as f:
        lines = [line.rstrip() for line in f]
    gt_hierarchy = defaultdict(list)  # coarse parent to list of fine children (only for gt section)
    actions_per_section = defaultdict(list)
    title, last_coarse_action = '', None
    for line in lines:
        line_parts = line.split(sep='\t')
        if len(line_parts) == 1 and line_parts[0]:
            title = line_parts[0]
            continue
        if len(line_parts) == 3:
            coarse_action, coarse_action_length = line_parts[1:]
            coarse_action = coarse_action.strip()
            coarse_action_length = int(int(coarse_action_length.strip().split(sep=' ')[0]) * box_width_scale)
            if '/' in coarse_action:
                coarse_action, fine_action = coarse_action.split(sep='/')
                if title in {'Observed', 'Unobserved'}:
                    gt_hierarchy[coarse_action].append(fine_action)
                actions_per_section[title + '-Fine'].append((fine_action, coarse_action_length))
                if actions_per_section[title + '-Coarse']:
                    last_coarse_action = actions_per_section[title + '-Coarse'][-1][0]
                    if last_coarse_action == coarse_action:
                        actions_per_section[title + '-Coarse'][-1][1] += coarse_action_length
                    else:
                        actions_per_section[title + '-Coarse'].append([coarse_action, coarse_action_length])
                else:
                    actions_per_section[title + '-Coarse'].append([coarse_action, coarse_action_length])
            else:
                last_coarse_action = coarse_action
                if title in {'Observed', 'Unobserved'}:
                    gt_hierarchy[coarse_action]
                actions_per_section[title + '-Coarse'].append((coarse_action, coarse_action_length))
        elif len(line_parts) == 4:
            fine_action, fine_action_length = line_parts[2:]
            fine_action = fine_action.strip()
            if title in {'Observed', 'Unobserved'}:
                gt_hierarchy[last_coarse_action].append(fine_action)
            fine_action_length = int(int(fine_action_length.strip().split(sep=' ')[0]) * box_width_scale)
            actions_per_section[title + '-Fine'].append((fine_action, fine_action_length))
    colour_dict = _generate_colour_dict_from_hierarchy(gt_hierarchy, colour_dict)
    coarse_action_to_label_id = _update_coarse_action_to_label_id(coarse_action_to_label_id, gt_hierarchy)
    fine_action_to_label_id = _update_fine_action_to_label_id(fine_action_to_label_id, gt_hierarchy)
    if len(actions_per_section) == 6:
        output = _draw_hierarchical_visualisation(actions_per_section, colour_dict, save_path, file,
                                                  initial_img=initial_img, img_height=img_height,
                                                  top_margin=top_margin, vertical_gap=vertical_gap,
                                                  return_img=return_img, skip_ground_truth=skip_ground_truth,
                                                  skip_prediction=skip_prediction, split_long_words=split_long_words,
                                                  font_scale=font_scale,
                                                  coarse_action_to_label_id=coarse_action_to_label_id,
                                                  fine_action_to_label_id=fine_action_to_label_id,
                                                  write_label_ids=write_label_ids)
    elif len(actions_per_section) == 3:
        output = _draw_single_level_visualisation(actions_per_section, colour_dict, save_path, file,
                                                  initial_img=initial_img, img_height=img_height,
                                                  top_margin=top_margin, vertical_gap=vertical_gap,
                                                  return_img=return_img, skip_ground_truth=skip_ground_truth,
                                                  skip_prediction=skip_prediction, split_long_words=split_long_words)
    else:
        raise RuntimeError(f'Something went wrong with file {file}')
    return output


def _draw_hierarchical_visualisation(actions_per_section, colour_dict, save_path, file, initial_img=None,
                                     img_height=600, top_margin=0, vertical_gap=100, return_img=False,
                                     skip_ground_truth=False, skip_prediction=False, split_long_words=False,
                                     font_scale=0.35, coarse_action_to_label_id=None, fine_action_to_label_id=None,
                                     write_label_ids=False):
    """Draw hierarchical visualisation.

    This function is not the smartest but does the job. It first draws the observed hierarchy (coarse and fine levels),
    then draws the unobserved hierarchy, and finally draws the predicted hierarchy.
    """
    video_length = _video_length_from_actions(actions_per_section['Observed-Coarse'] +
                                              actions_per_section['Unobserved-Coarse'])
    img_width = video_length + 60  # Add a little gap between the left and right margins and the draw boxes.
    if initial_img is None:
        img = np.full([img_height, img_width, 3], fill_value=255, dtype=np.uint8)
    else:
        img = initial_img
    distance_from_left_margin = 20
    distance_from_top_margin = 60 + top_margin if top_margin == 0 else top_margin
    top_left_corner = {'Observed-Coarse': [distance_from_left_margin, distance_from_top_margin],
                       'Observed-Fine': [distance_from_left_margin, distance_from_top_margin + vertical_gap]}
    for section in ['Observed-Coarse', 'Observed-Fine']:
        actions = actions_per_section[section]
        section_top_left_corner = top_left_corner[section]
        if section == 'Observed-Coarse':
            action_to_label_id = coarse_action_to_label_id
        else:
            action_to_label_id = fine_action_to_label_id
        for action, length in actions:
            if not skip_ground_truth:
                _draw_action(img, action, length, section_top_left_corner, vertical_gap, color=colour_dict[action],
                             split_long_words=split_long_words, font_scale=font_scale, write_label_ids=write_label_ids,
                             action_to_label_id=action_to_label_id)
            section_top_left_corner[0] += length
    observed_cutoff_x = top_left_corner['Observed-Coarse'][0]
    for section in ['Unobserved-Coarse', 'Unobserved-Fine']:
        actions = actions_per_section[section]
        if section == 'Unobserved-Coarse':
            section_top_left_corner = list(top_left_corner['Observed-Coarse'])
            action_to_label_id = coarse_action_to_label_id
        else:
            section_top_left_corner = list(top_left_corner['Observed-Fine'])
            action_to_label_id = fine_action_to_label_id
        ignore_text = True
        for action, length in actions:
            if not skip_ground_truth:
                _draw_action(img, action, length, section_top_left_corner, vertical_gap, color=colour_dict[action],
                             ignore_text=ignore_text, split_long_words=split_long_words, font_scale=font_scale,
                             write_label_ids=write_label_ids, action_to_label_id=action_to_label_id)
            ignore_text = False
            section_top_left_corner[0] += length
        maximum_length = section_top_left_corner[0]
    for section in ['Predicted-Coarse', 'Predicted-Fine']:
        actions = actions_per_section[section]
        if section == 'Predicted-Coarse':
            section_top_left_corner = list(top_left_corner['Observed-Coarse'])
            action_to_label_id = coarse_action_to_label_id
        else:
            section_top_left_corner = list(top_left_corner['Observed-Fine'])
            action_to_label_id = fine_action_to_label_id
        if not skip_ground_truth:
            section_top_left_corner[1] += round(2.5 * vertical_gap)
        for action, length in actions:
            if not skip_prediction:
                color = colour_dict.get(action)
                if color is None:
                    _update_colour_dict_with_random_colour(colour_dict, action)
                    color = colour_dict[action]
                if action not in action_to_label_id:
                    _update_action_to_label_id(action_to_label_id, action)
                _draw_action(img, action, length, section_top_left_corner, vertical_gap,
                             color=color, maximum_length=maximum_length, split_long_words=split_long_words,
                             font_scale=font_scale, write_label_ids=write_label_ids,
                             action_to_label_id=action_to_label_id)
            section_top_left_corner[0] += length
    _draw_vertical_line(img, x=observed_cutoff_x)
    if return_img:
        if skip_ground_truth:
            new_top_margin = distance_from_top_margin + round(2.5 * vertical_gap)
        else:
            new_top_margin = distance_from_top_margin + 5 * vertical_gap
        return img, new_top_margin
    else:
        _draw_t_star(img, observed_cutoff_x)
    if save_path is None:
        cv.imshow('image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        file_name = os.path.basename(file).split(sep='.')[0]
        file_name = os.path.join(save_path, file_name)
        cv.imwrite(f'{file_name}.png', img)


def _draw_single_level_visualisation(actions_per_section, colour_dict, save_path, file, initial_img=None,
                                     img_height=600, top_margin=0, vertical_gap=100, return_img=False,
                                     skip_ground_truth=False, skip_prediction=False, split_long_words=False):
    """Draw single level visualisation.

    This function works similarly to _draw_hierarchical_visualisation but is adapted/simplified for single level input
    to single level output.
    """
    video_length = _video_length_from_actions(actions_per_section['Observed-Coarse'] +
                                              actions_per_section['Unobserved-Coarse'])
    img_width = video_length + 60
    if initial_img is None:
        img = np.full([img_height, img_width, 3], fill_value=255, dtype=np.uint8)
    else:
        img = initial_img
    distance_from_left_margin, distance_from_top_margin = 20, 60 + top_margin
    distance_from_top_margin = 60 + top_margin if top_margin == 0 else top_margin
    top_left_corner = {'Observed-Coarse': [distance_from_left_margin, distance_from_top_margin]}

    actions = actions_per_section['Observed-Coarse']
    section_top_left_corner = top_left_corner['Observed-Coarse']
    for action, length in actions:
        if not skip_ground_truth:
            _draw_action(img, action, length, section_top_left_corner, vertical_gap, color=colour_dict[action],
                         split_long_words=split_long_words)
        section_top_left_corner[0] += length
    observed_cutoff_x = top_left_corner['Observed-Coarse'][0]

    actions = actions_per_section['Unobserved-Coarse']
    section_top_left_corner = list(top_left_corner['Observed-Coarse'])
    for action, length in actions:
        if not skip_ground_truth:
            _draw_action(img, action, length, section_top_left_corner, vertical_gap, color=colour_dict[action],
                         split_long_words=split_long_words)
        section_top_left_corner[0] += length
    maximum_length = section_top_left_corner[0]

    actions = actions_per_section['Predicted-Coarse']
    section_top_left_corner = list(top_left_corner['Observed-Coarse'])
    if not skip_ground_truth:
        section_top_left_corner[1] += round(1.5 * vertical_gap)
    for action, length in actions:
        if not skip_prediction:
            _draw_action(img, action, length, section_top_left_corner, vertical_gap,
                         color=colour_dict[action], maximum_length=maximum_length, split_long_words=split_long_words)
        section_top_left_corner[0] += length
    _draw_vertical_line(img, x=observed_cutoff_x)
    if return_img:
        new_top_margin = distance_from_top_margin + 3 * vertical_gap
        return img, new_top_margin
    else:
        _draw_t_star(img, observed_cutoff_x)
    if save_path is None:
        cv.imshow('image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        file_name = os.path.basename(file).split(sep='.')[0]
        file_name = os.path.join(save_path, file_name)
        cv.imwrite(f'{file_name}.png', img)


def _draw_vertical_line(img, x):
    height = img.shape[0]
    draw_line(img, (x, 0), (x, height - 1), (128, 128, 128), thickness=1, line_type=cv.LINE_AA, style='dashed', gap=4)


def _draw_action(img, action, length, top_left_corner, vertical_gap, color, maximum_length=None, ignore_text=False,
                 split_long_words=False, font_scale=0.35, write_label_ids=False, action_to_label_id=None):
    if maximum_length is None:
        maximum_length = 200000
    # Draw main box
    bottom_right_corner = [min(top_left_corner[0] + length - 1, maximum_length),
                           top_left_corner[1] + vertical_gap - 1]
    cv.rectangle(img, tuple(top_left_corner), tuple(bottom_right_corner), color, -1)
    # Draw lines around the box for better visual separation of adjacent actions
    cv.line(img, tuple(top_left_corner), (top_left_corner[0], bottom_right_corner[1]),
            (0, 0, 0), thickness=1, lineType=cv.LINE_AA)
    cv.line(img, tuple(top_left_corner), (bottom_right_corner[0], top_left_corner[1]),
            (0, 0, 0), thickness=1, lineType=cv.LINE_AA)
    cv.line(img, (bottom_right_corner[0], top_left_corner[1]), tuple(bottom_right_corner),
            (0, 0, 0), thickness=1, lineType=cv.LINE_AA)
    cv.line(img, (top_left_corner[0], bottom_right_corner[1]), tuple(bottom_right_corner),
            (0, 0, 0), thickness=1, lineType=cv.LINE_AA)
    if ignore_text:
        return
    # Draw action name inside box. The .translate call is to remove _ and - from the action name.
    thickness, font_face = 0, cv.FONT_HERSHEY_SIMPLEX
    if write_label_ids:
        bottom_left_anchor = (round(0.55 * top_left_corner[0] + 0.45 * bottom_right_corner[0]),
                              round(top_left_corner[1] + 0.65 * vertical_gap))
        label_id = action_to_label_id[action]
        cv.putText(img, label_id, bottom_left_anchor, fontFace=font_face, fontScale=font_scale,
                   color=(255, 255, 255), thickness=thickness, lineType=cv.LINE_AA)
    else:
        bottom_left_anchor = (round(0.9 * top_left_corner[0] + 0.1 * bottom_right_corner[0]),
                              round(top_left_corner[1] + 0.65 * vertical_gap))
        action = action.translate({45: ' ', 95: ' '})
        action_words = action.split(sep=' ')
        if len(action_words) <= 2 or not split_long_words:
            cv.putText(img, action, bottom_left_anchor, fontFace=font_face, fontScale=font_scale,
                       color=(255, 255, 255), thickness=thickness, lineType=cv.LINE_AA)
        else:
            action_a = action_words[0] + ' ' + action_words[1]
            bottom_left_anchor = list(bottom_left_anchor)
            bottom_left_anchor[1] -= round(0.1 * vertical_gap)
            cv.putText(img, action_a, tuple(bottom_left_anchor), fontFace=font_face, fontScale=font_scale,
                       color=(255, 255, 255), thickness=thickness, lineType=cv.LINE_AA)
            action_b = action_words[2]
            bottom_left_anchor = list(bottom_left_anchor)
            bottom_left_anchor[1] += round(0.40 * vertical_gap)
            cv.putText(img, action_b, tuple(bottom_left_anchor), fontFace=font_face, fontScale=font_scale,
                       color=(255, 255, 255), thickness=thickness, lineType=cv.LINE_AA)


def _draw_t_star(img, observed_cutoff_x):
    font_scale, thickness, font_face = 0.75, 0, cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, 't*', (observed_cutoff_x - 25, 45), fontFace=font_face, fontScale=font_scale,
               color=(128, 128, 128), thickness=thickness, lineType=cv.LINE_AA)


def _generate_random_colours(n, fix_seed=True, max_value=256):
    """This function generates n unique random colour codes."""
    if fix_seed:
        random.seed(42)
    colours = []
    for _ in range(n):
        b = int(random.random() * max_value)
        r = int(random.random() * max_value)
        g = int(random.random() * max_value)
        colours.append((b, g, r))
    return colours


def _video_length_from_actions(actions):
    video_length = 0
    for _, length in actions:
        video_length += length
    return video_length


def _file_name_is_okay(file_name):
    if file_name.startswith('metric') or file_name.startswith('mfap'):
        return False
    if not file_name.endswith('.txt'):
        return False
    return True


def _img_height_from_box_height(box_height):
    return 150 * math.ceil(box_height / 30)


def draw_line(img, pt1, pt2, color, thickness=1, line_type=None, style='dotted', gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv.circle(img, p, thickness, color, -1)
    else:
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv.line(img, s, e, color, thickness, line_type)
            i += 1


def _generate_colour_dict_from_hierarchy(hierarchy, colour_dict=None):
    if colour_dict is None:
        colour_dict = {}
    seen_fine_actions = set()
    for coarse_action, fine_actions in hierarchy.items():
        coarse_colour = _generate_random_colours(1, fix_seed=False, max_value=128)[0]
        colour_dict.setdefault(coarse_action, coarse_colour)
        fine_colour = list(coarse_colour)
        for fine_action in fine_actions:
            if fine_action in seen_fine_actions:
                continue
            seen_fine_actions.add(fine_action)
            fine_colour = [(value + 25) % 256 for value in fine_colour]
            colour_dict.setdefault(fine_action, tuple(fine_colour))
    return colour_dict


def _update_colour_dict_with_random_colour(colour_dict, action):
    colour_dict[action] = _generate_random_colours(1, fix_seed=False)[0]


def _update_coarse_action_to_label_id(coarse_action_to_label_id, gt_hierarchy):
    for coarse_action in gt_hierarchy.keys():
        current_id = len(coarse_action_to_label_id) + 1
        coarse_action_to_label_id.setdefault(coarse_action, f'C{current_id}')
    return coarse_action_to_label_id


def _update_fine_action_to_label_id(fine_action_to_label_id, gt_hierarchy):
    for fine_actions in gt_hierarchy.values():
        for fine_action in fine_actions:
            current_id = len(fine_action_to_label_id) + 1
            fine_action_to_label_id.setdefault(fine_action, f'F{current_id}')
    return fine_action_to_label_id


def _update_action_to_label_id(action_to_label_id, action):
    random_label_id = list(action_to_label_id.values())[0]
    label_code = random_label_id[0]
    current_id = len(action_to_label_id) + 1
    action_to_label_id.setdefault(action, f'{label_code}{current_id}')
