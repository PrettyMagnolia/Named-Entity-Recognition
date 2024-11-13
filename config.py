import argparse


def get_config():
    parser = argparse.ArgumentParser()

    # Path-related parameters
    parser.add_argument('--data_dir', type=str, default="/home/yifei/code/bi-lstm-crf/data", help="Directory for input data")
    parser.add_argument('--output_dir', type=str, default="/home/yifei/code/bi-lstm-crf/my/runs/typelstm_lr0.001_hd128", help="Directory for output model files")

    # Training-related parameters
    parser.add_argument('--num_epoch', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="L2 regularization weight decay")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size for training")
    parser.add_argument('--recovery', action="store_true", help="Continue training from saved model in output_dir")
    parser.add_argument('--save_model', action="store_true", default=True, help="Save the best model based on validation score")

    # Model structure-related parameters
    parser.add_argument('--embedding_dim', type=int, default=100, help="Embedding layer dimension")
    parser.add_argument('--hidden_dim', type=int, default=128, help="RNN hidden layer dimension")
    parser.add_argument('--num_rnn_layers', type=int, default=1, help="Number of RNN layers")
    parser.add_argument('--rnn_type', type=str, default="lstm", choices=['lstm', 'gru'], help="Type of RNN (either 'lstm' or 'gru')")

    # Device-related parameters
    parser.add_argument('--device', type=str, default=None, help="Training device: 'cuda:0', 'cpu'. Default is auto-detected")

    return parser.parse_args()
