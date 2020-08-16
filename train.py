import argparse

from fpua.models.baselines import train_baselines
from fpua.models.hera import train_hera


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Model Training Functions.')
    subparsers = parser.add_subparsers(title='sub-commands', description='Valid sub-commands.')
    # Baselines
    parser_baselines = subparsers.add_parser('baselines',
                                             help='Train Baselines.')

    parser_baselines.add_argument('--training_data', type=str, required=True,
                                  help='Path to .npz file containing training data.')
    parser_baselines.add_argument('--validation_data', type=str,
                                  help='Path to .npz file containing validation data.')
    parser_baselines.add_argument('--baseline_type', default=0, type=int, choices=[0, 1, 2],
                                  required=True,
                                  help='Which baseline to train. 0 is the Independent-Single-RNN, 1 is the '
                                       'Joint-Single-RNN, and 2 is the Synced-Pair-RNN.')
    parser_baselines.add_argument('--action_level', default='fine', type=str,
                                  choices=['coarse', 'fine'],
                                  help='Which level to train.')
    parser_baselines.add_argument('--seq_len', default=35, type=int, help='Number of steps for the baseline.')
    parser_baselines.add_argument('--hidden_size', default=16, type=int, help='Hidden size of the rnn(s).')
    parser_baselines.add_argument('--embedding_size', default=16, type=int,
                                  help='Size of the embedding layer. If zero, there is no embedding layer.')
    parser_baselines.add_argument('--embedding_nonlinearity', default='tanh', type=str,
                                  choices=['relu', 'tanh', 'sigmoid'],
                                  help='Nonlinearity applied to the embeddings.')
    parser_baselines.add_argument('--epochs', default=1, type=int, help='Number of training epochs.')
    parser_baselines.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate of the optimiser.')
    parser_baselines.add_argument('--batch_size', default=12, type=int, help='Batch size for model training.')
    parser_baselines.add_argument('--length_activation', default='sigmoid', type=str,
                                  choices=['linear', 'relu', 'elu', 'sigmoid'],
                                  help='Activation function for the layers that predict action lengths.')
    parser_baselines.add_argument('--multi_task_loss_learner', action='store_true',
                                  help='Learn the weights of the individual losses.')
    parser_baselines.add_argument('--print_raw_losses', action='store_true',
                                  help='Print raw losses alongside losses of the '
                                       'multi-task learner. Only meaningful if '
                                       '--multi_task_loss_learner is used.')
    parser_baselines.add_argument('--teacher_schedule', default='None', type=str,
                                  choices=['None', 'always', 'linear', 'exponential',
                                           'inverse_sigmoid', 'random'],
                                  help='Teacher forcing schedule. If None, no teacher '
                                       'forcing is done at all. If always, teacher '
                                       'forcing is done only during testing.')
    parser_baselines.add_argument('--weight_decay', default=0.0, type=float, help='L2 regularisation for the model.')
    parser_baselines.add_argument('--clip_gradient_at', default=5.0, type=float,
                                  help='Clip gradient norm at the specified value. The '
                                       'gradient norm is computed by concatenating all '
                                       'parameters into a single vector.')
    parser_baselines.add_argument('--log_dir', type=str, help='Where to save the checkpoint file.')
    parser_baselines.set_defaults(func=train_baselines)
    # HERA
    parser_hera = \
        subparsers.add_parser('hera',
                              help='2-layer model that receives a label hierarchy as input and outputs a label '
                                   'hierarchy. Instead of letting the children decide when the parent finishes, this '
                                   'model lets the parent decide its own length, and the children decide their '
                                   'proportion in relation to the length of the parent. The inputs are cumulative '
                                   'proportions in relation to the parent.')

    parser_hera.add_argument('--training_data', type=str, required=True,
                             help='Path to .npz file containing training data.')
    parser_hera.add_argument('--validation_data', type=str, help='Path to .npz file containing validation data.')
    parser_hera.add_argument('--input_seq_len', default=35, type=int, help='Number of steps for the encoder.')
    parser_hera.add_argument('--output_seq_len', default=50, type=int, help='Number of steps for the decoder.')
    parser_hera.add_argument('--hidden_size_fine', default=16, type=int,
                             help='Hidden size of the encoder, transition and decoder networks for the fine level.')
    parser_hera.add_argument('--hidden_size_coarse', default=16, type=int,
                             help='Hidden size of the encoder, transition and decoder networks for the coarse level.')
    parser_hera.add_argument('--fine_embedding_size', nargs='+', default=[16, 16, 16], type=int,
                             help='Size of the embedding layer for the fine '
                                  'level. If zero, there is no embedding layer.')
    parser_hera.add_argument('--coarse_embedding_size', nargs='+', default=[16, 16], type=int,
                             help='Size of the embedding layer for the coarse '
                                  'level. If zero, there is no embedding layer.')
    parser_hera.add_argument('--embedding_nonlinearity', default='tanh', type=str,
                             choices=['relu', 'tanh'],
                             help='Nonlinearity applied to the embeddings.')
    parser_hera.add_argument('--epochs', default=1, type=int, help='Number of training epochs.')
    parser_hera.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate of the optimiser.')
    parser_hera.add_argument('--transition_learning_rate', default=1e-3, type=float,
                             help='Learning rate of the Refresher network.')
    parser_hera.add_argument('--batch_size', default=512, type=int, help='Batch size for model training.')
    parser_hera.add_argument('--length_activation', nargs=2,
                             default=['elu', 'sigmoid'], type=str,
                             choices=['linear', 'relu', 'elu', 'sigmoid'],
                             help='Activation function for the layers that predict '
                                  'action lengths. First specified activation is '
                                  'for the fine level and second specified '
                                  'activation is for the coarse level.')
    parser_hera.add_argument('--multi_task_loss_learner', action='store_true',
                             help='Learn the weights of the individual losses.')
    parser_hera.add_argument('--print_raw_losses', action='store_true',
                             help='Print raw losses alongside losses of the '
                                  'multi-task learner. Only meaningful if '
                                  '--multi_task_loss_learner is used.')
    parser_hera.add_argument('--loss_weights', nargs='+', default=None, type=float,
                             help='Loss weights. Please specify 10 weights, even for ignored losses. '
                                  'If --multi_task_loss_learner is specified loss_weights are ignored.')
    parser_hera.add_argument('--normalise_input', action='store_true',
                             help='Normalise input length to the fine and coarse levels.')
    parser_hera.add_argument('--normalise_output', action='store_true',
                             help='Normalise predicted output lengths (remaining and future ones).')
    parser_hera.add_argument('--quantile_upper_bound', default=95.0, type=float,
                             help='Upper quantile for output normalisation.')
    parser_hera.add_argument('--disable_parent_input', action='store_true',
                             help='Parent label is not given as input to the children level.')
    parser_hera.add_argument('--input_soft_parent', action='store_true',
                             help='If specified, input the softmax of the parent '
                                  'output instead of the one-hot representation to '
                                  'the fine level during teacher forcing. Ignored '
                                  'if --disable_parent_input is specified.')
    parser_hera.add_argument('--teacher_schedule', default='always', type=str,
                             choices=['None', 'always', 'linear', 'exponential', 'inverse_sigmoid', 'random'],
                             help='Teacher forcing schedule. If None, no teacher '
                                  'forcing is done at all. If always, teacher '
                                  'forcing is done only during testing.')
    parser_hera.add_argument('--share_embeddings', action='store_true',
                             help='Share the parameters of the embeddings whenever possible across the model.')
    parser_hera.add_argument('--share_encoder_decoder', action='store_true',
                             help='Share the parameters between the encoder and decoder RNNs.')
    parser_hera.add_argument('--share_predictions', action='store_true',
                             help='Share the parameters of the prediction layer between the encoder and decoder.')
    parser_hera.add_argument('--disable_encoder_loss', action='store_true',
                             help='Turn-off the encoder loss. For technical reasons '
                                  'the model still has the prediction layer in the '
                                  'encoder, but it is not trained and should not '
                                  'be used.')
    parser_hera.add_argument('--positional_embedding', action='store_true',
                             help='Instead of learning an embedding for the accumulated time, use a '
                                  'positional embedding on it.')
    parser_hera.add_argument('--mask_softmax', action='store_true',
                             help='Next predicted action cannot be the same as the previous action.')
    parser_hera.add_argument('--add_skip_connection', action='store_true',
                             help='Add skip connection between embedded input and the output of the RNN cell.')
    parser_hera.add_argument('--weight_decay', default=0.0, type=float, help='L2 regularisation for the model.')
    parser_hera.add_argument('--weight_decay_decoder_only', action='store_true',
                             help='Apply weight decay only to the decoder.')
    parser_hera.add_argument('--disable_transition_layer', action='store_true',
                             help='Detach the Refresher network from the model '
                                  'pipeline. The initial hidden state of the Anticipator '
                                  'is now the final hidden state from the Encoder. '
                                  'The Refresher network components are still used '
                                  'for prediction of the remaining lengths.')
    parser_hera.add_argument('--weight_initialisation', default='pytorch', type=str,
                             choices=['pytorch', 'keras'],
                             help='Weight initialisation style for the HMGRU RNNs.')
    parser_hera.add_argument('--clip_gradient_at', default=5.0, type=float,
                             help='Clip gradient norm at the specified value. The '
                                  'gradient norm is computed by concatenating all '
                                  'parameters into a single vector.')
    parser_hera.add_argument('--disable_gradient_from_child', action='store_true',
                             help='Disable gradient coming from children.')
    parser_hera.add_argument('--pretrain_coarse', default=0, type=int,
                             help='If positive, pretrain the coarse level for the specified number of epochs.')
    parser_hera.add_argument('--do_not_reset_after_flush', action='store_true',
                             help='Maintain fine level hidden state after flush.')
    parser_hera.add_argument('--always_include_parent_state', action='store_true',
                             help='Every child receives the parent state, not just the first child.')
    parser_hera.add_argument('--train_on_subset_percentage', type=float,
                             help='Percentage of training data to train model.')
    parser_hera.add_argument('--log_dir', type=str, help='Where to save the checkpoint file.')
    parser_hera.set_defaults(func=train_hera)

    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
