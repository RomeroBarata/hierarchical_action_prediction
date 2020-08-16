import argparse

from fpua.data.baselines import generate_baselines_training_data
from fpua.data.general import merge_arrays, generate_action_dictionary, reverse_action_dictionary
from fpua.data.hera import generate_hera_training_data


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Data Pre-processing Functions.')
    subparsers = parser.add_subparsers(title='sub-commands', description='Valid sub-commands.')

    # Generate action dictionary
    parser_dict = subparsers.add_parser('generate_action_dict',
                                        help='Generate an Action Dictionary from Unique Actions in the Data Set.')
    parser_dict.add_argument('--path', type=str, required=True,
                             help='Directory containing all action .txt files of the data set.')
    parser_dict.add_argument('--ignore', nargs='+', type=str,
                             help='List of actions to ignore during construction of the dictionary.')
    parser_dict.add_argument('--save_file', type=str,
                             help='A .txt file to write out the mapping between actions and IDs.')
    parser_dict.set_defaults(func=generate_action_dictionary)

    # Reverse an action dictionary
    parser_rev_dict = subparsers.add_parser('reverse_action_dict',
                                            help='Given a .txt file containing a mapping between actions and ids, '
                                                 'reverse their order.')
    parser_rev_dict.add_argument('--dict_file', type=str, required=True,
                                 help='.txt file containing the mapping between actions and ids.')
    parser_rev_dict.add_argument('--save_file', type=str,
                                 help='A .txt file to write out the result.')
    parser_rev_dict.set_defaults(func=reverse_action_dictionary)

    # Merge arrays
    parser_merge_arrays = subparsers.add_parser('merge_arrays',
                                                help='Merge .npz Files Containing Input Output Data.')
    parser_merge_arrays.add_argument('--arrays', nargs='+', required=True,
                                     help='Arrays to merge. Each specified array is an .npz file '
                                          'containing the matching arrays.')
    parser_merge_arrays.add_argument('--save_path', type=str, help='Directory to save merged arrays.')
    parser_merge_arrays.add_argument('--save_name', type=str,
                                     help='Optional name to save the merged arrays. Only meaningful if '
                                          'save_path is specified.')
    parser_merge_arrays.set_defaults(func=merge_arrays)

    # Generate training data for the Baselines
    parser_baselines = \
        subparsers.add_parser('baselines',
                              help='Generate training data for baselines.')

    parser_baselines.add_argument('--fine_labels_path', type=str, required=True,
                                  help='Directory containing fine level annotations.')
    parser_baselines.add_argument('--coarse_labels_path', type=str, required=True,
                                  help='Directory containing coarse level annotations.')
    parser_baselines.add_argument('--fine_action_to_id', type=str, required=True,
                                  help='.txt file containing mapping from fine action to ID.')
    parser_baselines.add_argument('--coarse_action_to_id', type=str, required=True,
                                  help='.txt file containing mapping from coarse action to ID.')
    parser_baselines.add_argument('--seq_len', default=50, type=int,
                                  help='Maximum number of actions to process.')
    parser_baselines.add_argument('--ignore_silence_action', type=str,
                                  help='Remove from the generated data the specified silence action.')
    parser_baselines.add_argument('--add_final_action', action='store_true',
                                  help='Add a new output action to the coarse and '
                                       'fine levels, signalising the end of the '
                                       'video for the coarse level, and the final child '
                                       'for the fine level.')
    parser_baselines.add_argument('--is_validation', action='store_true',
                                  help='The generated array is for validation.')
    parser_baselines.add_argument('--save_path', type=str,
                                  help='Directory to save the generated .npz file.')
    parser_baselines.add_argument('--save_name', type=str,
                                  help='Name for the .npz file.')
    parser_baselines.set_defaults(func=generate_baselines_training_data)

    # Generate training data for HERA
    parser_hera = \
        subparsers.add_parser('hera',
                              help='Generate training data for a two-level label hierarchy to '
                                   'label hierarchy model where all inputs are accumulated proportions in '
                                   'relation to the parent level, and the outputs are proportions in relation '
                                   'to the parent.')

    parser_hera.add_argument('--fine_labels_path', type=str, required=True,
                             help='Directory containing fine level annotations.')
    parser_hera.add_argument('--coarse_labels_path', type=str, required=True,
                             help='Directory containing coarse level annotations.')
    parser_hera.add_argument('--fine_action_to_id', type=str, required=True,
                             help='.txt file containing mapping from fine action to ID.')
    parser_hera.add_argument('--coarse_action_to_id', type=str, required=True,
                             help='.txt file containing mapping from coarse action to ID.')
    parser_hera.add_argument('--input_seq_len', default=35, type=int,
                             help='Maximum number of observed actions.')
    parser_hera.add_argument('--output_seq_len', default=50, type=int,
                             help='Maximum number of unobserved actions.')
    parser_hera.add_argument('--num_cuts', default=5, type=int,
                             help='When generating training examples, for every potential action to be interrupted, '
                                  'generate num_cuts examples from it.')
    parser_hera.add_argument('--observe_at_least_k_percent', default=20.0, type=float,
                             help='Generated training examples observe at least k percent of the video.')
    parser_hera.add_argument('--ignore_silence_action', type=str,
                             help='Remove from the generated data the specified silence action.')
    parser_hera.add_argument('--add_final_action', action='store_true',
                             help='Add a new output action to the coarse and '
                                  'fine levels, signalising the end of the '
                                  'video for the coarse level, and the final child '
                                  'for the fine level.')
    parser_hera.add_argument('--save_path', type=str, help='Directory to save the generated .npz file.')
    parser_hera.add_argument('--save_name', type=str, help='Name for the .npz file.')
    parser_hera.set_defaults(func=generate_hera_training_data)

    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
