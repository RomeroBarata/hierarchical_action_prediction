import argparse

from fpua.vis_utils import draw_visualisation, generate_colour_dict


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Visualisation Functions.')
    subparsers = parser.add_subparsers(title='sub-commands', description='Valid sub-commands.')
    # Draw action segmentation from .txt files containing the single level or hierarchical segmentation.
    parser_visualise_actions_segmentation = subparsers.add_parser('visualise_actions_segmentation',
                                                                  help='Given a directory with .txt files containing '
                                                                       'single level or hierarchical observed, '
                                                                       'unobserved, and predicted actions of a '
                                                                       'model, generate a .png file depicting '
                                                                       'the segmentation.')

    parser_visualise_actions_segmentation.add_argument('--files_paths', nargs='+', type=str, required=True,
                                                       help='Path to a directory containing output segmentation of a '
                                                            'model. If multiple paths are specified to compare '
                                                            'multiple models, draw a single image containing all '
                                                            'models inputs and outputs for matching files.')
    parser_visualise_actions_segmentation.add_argument('--colour_path', type=str,
                                                       help='Path to .txt file containing a mapping between '
                                                            'actions and colours.')
    parser_visualise_actions_segmentation.add_argument('--draw_only_first_ground_truth', action='store_true',
                                                       help='If specified, only draw the ground truth for the '
                                                            'first specified method.')
    parser_visualise_actions_segmentation.add_argument('--box_height', default=20, type=int,
                                                       help='Height of drawn boxes.')
    parser_visualise_actions_segmentation.add_argument('--box_width_scale', default=1, type=float,
                                                       help='Width of action boxes are increased by specified value.')
    parser_visualise_actions_segmentation.add_argument('--split_long_words', action='store_true',
                                                       help='If specified, split 3-word actions into 2 lines.')
    parser_visualise_actions_segmentation.add_argument('--font_scale', default=0.35, type=float,
                                                       help='Size of the written text font.')
    parser_visualise_actions_segmentation.add_argument('--write_label_ids', action='store_true',
                                                       help='Write label IDs for the actions instead of action names.')
    parser_visualise_actions_segmentation.add_argument('--save_path', type=str,
                                                       help='Path to a directory to save output .png files. If not '
                                                            'specified, the output drawings are displayed on '
                                                            'screen, which is useful for debugging purposes. Press '
                                                            'any button to close the displayed drawing.')
    parser_visualise_actions_segmentation.set_defaults(func=draw_visualisation)

    # Generate a colour dictionary from actions
    parser_generate_action_dict = subparsers.add_parser('generate_colour_dict',
                                                        help='Given fine and coarse action to id dictionaries, '
                                                             'generate a dictionary mapping each action to a unique '
                                                             'colour.')

    parser_generate_action_dict.add_argument('--fine_action_to_id', type=str, required=True,
                                             help='Path to .txt file mapping fine actions to unique IDs.')
    parser_generate_action_dict.add_argument('--coarse_action_to_id', type=str, required=True,
                                             help='Path to .txt file mapping coarse actions to unique IDs.')
    parser_generate_action_dict.add_argument('--save_path', type=str,
                                             help='Path to directory to save the action to BGR colour mapping.')
    parser_generate_action_dict.add_argument('--save_name', type=str,
                                             help='File name to save.')
    parser_generate_action_dict.set_defaults(func=generate_colour_dict)
    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
