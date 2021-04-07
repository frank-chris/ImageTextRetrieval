import torch
import torch.nn as nn
import random

#seed_num = 1
#torch.manual_seed(seed_num)
#random.seed(seed_num)

# We will implement a Bidirectional LSTM

class BiLSTM(nn.Module):
    def __init__(self, hidden_dim, num_stacked_layers= 1, vocab_size, embedding_dim, vocab_size):
        super(BiLSTM, self).__init__()
        
        # Create a word embedding object
        # Given some text, we will receive an embedding of required dimensions based on our vocabulary size.
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)

        # Create Bidirectional object
        # One is forward directional, the other is will be reversed.
        self.bi_lstm_forw = (nn.LSTM(embedding_dim, hidden_dim, num_stacked_layers, dropout = 0, bidirectional = False, bias = False)) 
        self.bi_lstm_back = (nn.LSTM(embedding_dim, hidden_dim, num_stacked_layers, dropout = 0, bidirectional = False, bias = False)) 

    def forward(self, txt, txt_len):

        # Extracting the embeddings of the given text
        embeddings = self.embed(txt)
        
        # LSTM output in the forward direction
        forw_out = self.lstm(embeddings, txt_len, True)

        # LSTM output in the backward direction
        # First invert the embeddings and the text length for the other direction of the bilstm
        reversed_indices = torch.LongTensor(list(range(embedding.shape[0] - 1, -1, -1))).cuda()
        reversed_embedding = embed.index_select(0, reversed_indices)
        reversed_text_lengths = txt_len.index_select(0, reversed_indices)
        back_out = self.lstm(reversed_embedding, reversed_text_lengths, False)
        # Again reverse this output, so that it matches the original direction
        back_out_rev = back_out.index_select(0, reversed_indices)
        bilstm_out = torch.cat()

