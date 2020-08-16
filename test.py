import argparse

from fpua.models.baselines_test import test_baselines, test_baselines_cv
from fpua.models.dummy import test_dummy_model, test_dummy_model_cv
from fpua.models.hera_test import test_hera, test_hera_cv


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Test Pre-trained Models.')
    subparsers = parser.add_subparsers(title='sub-commands', description='Valid sub-commands.')

    # Dummy model
    parser_dummy = subparsers.add_parser('dummy', help='Dummy model that predicts that the transition will go forever.')

    parser_dummy.add_argument('--labels_path', type=str, required=True,
                              help='Directory containing the framewise annotations')
    parser_dummy.add_argument('--action_to_id', type=str, required=True,
                              help='Path to .txt file containing the action to ID mapping.')
    parser_dummy.add_argument('--observed_fraction', type=float, required=True,
                              help='Observed fraction of the actions in a video. Value  between 0 and 1.')
    parser_dummy.add_argument('--unobserved_fraction', type=float, required=True,
                              help='Unobserved fraction of the actions in a video. Value between 0 and '
                                   '1 - observed_fraction.')
    parser_dummy.add_argument('--ignore', nargs='+', type=str,
                              help='Actions to ignore from the input files. For example, remove '
                                   'silence/background actions.')
    parser_dummy.set_defaults(func=test_dummy_model)

    # Dummy model cross-validation
    parser_dummy_cv = subparsers.add_parser('dummy_cv',
                                            help='Dummy model that predicts that the transition will go forever.')

    parser_dummy_cv.add_argument('--labels_root_path', type=str, required=True,
                                 help='Directory containing the framewise annotations split by cross-validation index.')
    parser_dummy_cv.add_argument('--action_to_id', type=str, required=True,
                                 help='Path to .txt file containing the action to ID mapping.')
    parser_dummy_cv.add_argument('--observed_fraction', type=float, required=True,
                                 help='Observed fraction of the actions in a video. Value  between 0 and 1.')
    parser_dummy_cv.add_argument('--unobserved_fraction', type=float, required=True,
                                 help='Unobserved fraction of the actions in a video. Value between 0 and '
                                      '1 - observed_fraction.')
    parser_dummy_cv.add_argument('--ignore', nargs='+', type=str,
                                 help='Actions to ignore from the input files. For example, remove '
                                      'silence/background actions.')
    parser_dummy_cv.set_defaults(func=test_dummy_model_cv)

    # Baselines
    parser_baselines = \
        subparsers.add_parser('baselines', help='Baselines.')

    parser_baselines.add_argument('--checkpoint', type=str, required=True,
                                  help='Checkpoint file containing pre-trained model and auxiliary information.')
    parser_baselines.add_argument('--fine_labels_path', type=str, required=True,
                                  help='Directory containing framewise annotations of the fine level as .txt files.')
    parser_baselines.add_argument('--coarse_labels_path', type=str, required=True,
                                  help='Directory containing framewise annotations of the coarse level as .txt files.')
    parser_baselines.add_argument('--fine_action_to_id', type=str, required=True,
                                  help='File containing mapping between fine actions and their IDs.')
    parser_baselines.add_argument('--coarse_action_to_id', type=str, required=True,
                                  help='File containing mapping between coarse actions and their IDs.')
    parser_baselines.add_argument('--observed_fraction', default=0.2, type=float,
                                  help='Observed fraction of each test video.')
    parser_baselines.add_argument('--ignore_silence_action', default='silence', type=str,
                                  help='Specify silence/background action to be removed from test data.')
    parser_baselines.add_argument('--do_error_analysis', action='store_true',
                                  help='Write-out error analysis of the mistakes made by the model.')
    parser_baselines.add_argument('--print_coarse_results', action='store_true',
                                  help='For baselines 1 and 2, print coarse results along fine results.')
    parser_baselines.set_defaults(func=test_baselines)

    # Baselines cross-validation
    parser_baselines_cv = \
        subparsers.add_parser('baselines_cv', help='Baselines cross-validation.')

    parser_baselines_cv.add_argument('--pretrained_root', type=str, required=True,
                                     help='Path to directory containing the splits folders.')
    parser_baselines_cv.add_argument('--pretrained_suffix', type=str, required=True,
                                     help='Assuming you are inside the split folder, this is the '
                                          'path to the pretrained model (.tar file).')
    parser_baselines_cv.add_argument('--fine_labels_root_path', type=str, required=True,
                                     help='Directory containing folders with annotations of the splits of fine levels.')
    parser_baselines_cv.add_argument('--coarse_labels_root_path', type=str, required=True,
                                     help='Directory containing folder with annotations of '
                                          'the splits of coarse levels.')
    parser_baselines_cv.add_argument('--fine_action_to_id', type=str, required=True,
                                     help='File containing mapping between fine actions and their IDs.')
    parser_baselines_cv.add_argument('--coarse_action_to_id', type=str, required=True,
                                     help='File containing mapping between coarse actions and their IDs.')
    parser_baselines_cv.add_argument('--observed_fraction', default=0.2, type=float,
                                     help='Observed fraction of each test video.')
    parser_baselines_cv.add_argument('--ignore_silence_action', default='silence', type=str,
                                     help='Specify silence/background action to be removed from test data.')
    parser_baselines_cv.add_argument('--do_error_analysis', action='store_true',
                                     help='Write-out error analysis of the mistakes made by the model.')
    parser_baselines_cv.add_argument('--print_coarse_results', action='store_true',
                                     help='For baselines 1 and 2, print coarse results along fine results.')
    parser_baselines_cv.set_defaults(func=test_baselines_cv)

    # HERA
    parser_hera = \
        subparsers.add_parser('hera',
                              help='2-layer model that receives a label hierarchy as input and outputs a label '
                                   'hierarchy. Instead of letting the children decide when the parent finishes, this '
                                   'model lets the parent decide its own length, and the children decide their '
                                   'proportion in relation to the length of the parent. In addition, inputs are '
                                   'accumulated proportions in relation to the parent.')

    parser_hera.add_argument('--checkpoint', type=str, required=True,
                             help='Checkpoint file (.tar file) containing pre-trained model and auxiliary information.')
    parser_hera.add_argument('--fine_labels_path', type=str, required=True,
                             help='Directory containing framewise annotations of the fine level as .txt files.')
    parser_hera.add_argument('--coarse_labels_path', type=str, required=True,
                             help='Directory containing framewise annotations of the coarse level as .txt files.')
    parser_hera.add_argument('--fine_action_to_id', type=str, required=True,
                             help='File containing mapping between fine actions and their IDs.')
    parser_hera.add_argument('--coarse_action_to_id', type=str, required=True,
                             help='File containing mapping between coarse actions and their IDs.')
    parser_hera.add_argument('--observed_fraction', default=0.2, type=float,
                             help='Observed fraction of each test video. Value between 0 and 1.')
    parser_hera.add_argument('--ignore_silence_action', default='silence', type=str,
                             help='Specify silence/background action to be removed from test data.')
    parser_hera.add_argument('--do_error_analysis', action='store_true',
                             help='Write-out error analysis of the mistakes made by the model.')
    parser_hera.add_argument('--do_future_performance_analysis', action='store_true',
                             help='Write-out model performance broken down by future action.')
    parser_hera.add_argument('--do_flush_analysis', action='store_true',
                             help='Write-out analysis of when the children finishes.')
    parser_hera.set_defaults(func=test_hera)

    # HERA cross-validation
    parser_hera_cv = \
        subparsers.add_parser('hera_cv',
                              help='Cross-validation for pre-trained HERA models.')

    parser_hera_cv.add_argument('--pretrained_root', type=str, required=True,
                                help='Directory containing split folders (e.g. split01-02-03).')
    parser_hera_cv.add_argument('--pretrained_suffix', type=str, required=True,
                                help='.tar file containing pre-trained model but the path should be given relative '
                                     'to the split* folder.')
    parser_hera_cv.add_argument('--fine_labels_root_path', type=str, required=True,
                                help='Directory containing framewise annotations of the fine level as .txt files.')
    parser_hera_cv.add_argument('--coarse_labels_root_path', type=str, required=True,
                                help='Directory containing framewise annotations of the coarse level as .txt files.')
    parser_hera_cv.add_argument('--fine_action_to_id', type=str, required=True,
                                help='File containing mapping between fine actions and their IDs.')
    parser_hera_cv.add_argument('--coarse_action_to_id', type=str, required=True,
                                help='File containing mapping between coarse actions and their IDs.')
    parser_hera_cv.add_argument('--observed_fraction', default=0.2, type=float,
                                help='Observed fraction of each test video.')
    parser_hera_cv.add_argument('--ignore_silence_action', default='silence', type=str,
                                help='Specify silence/background action to be removed from test data.')
    parser_hera_cv.add_argument('--do_error_analysis', action='store_true',
                                help='Write-out error analysis of the mistakes made by the model.')
    parser_hera_cv.add_argument('--do_future_performance_analysis', action='store_true',
                                help='Write-out model performance broken down by future action.')
    parser_hera_cv.add_argument('--do_flush_analysis', action='store_true',
                                help='Write-out analysis of when the children finishes.')
    parser_hera_cv.set_defaults(func=test_hera_cv)

    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
