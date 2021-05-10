import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='command for evaluate on CUHK-PEDES')
    # Directory
    parser.add_argument('--image_dir', type=str, help='directory to store dataset')
    parser.add_argument('--anno_dir', type=str, help='directory to store anno')
    parser.add_argument('--model_path', type=str, help='directory to load checkpoint')
    parser.add_argument('--log_dir', type=str, help='directory to store log')

    # LSTM setting
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--num_lstm_units', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=12000)
    parser.add_argument('--lstm_dropout_ratio', type=float, default=0.7)
    parser.add_argument('--bidirectional', action='store_true')

    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--feature_size', type=int, default=512)

    parser.add_argument('--image_model', type=str, default='mobilenet_v1')
    parser.add_argument('--cnn_dropout_keep', type=float, default=0.999)
    
    parser.add_argument('--epoch_ema', type=int, default=0)
    
    # Default setting
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    return args



def config():
    args = parse_args()
    return args
