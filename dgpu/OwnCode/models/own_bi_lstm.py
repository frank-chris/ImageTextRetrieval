import torch
import torch.nn as nn
import random

seed_num = 1
torch.manual_seed(seed_num)
random.seed(seed_num)

# We will implement a Bidirectional LSTM

class BiLSTM(nn.Module):
    def __init__(self, hidden_dim, num_recur_layers = 1, vocab_size, embedding_dim, vocab_size):
        super(BiLSTM, self).__init__()

