
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
os.environ['HF_HOME'] = '.transformers/'

from transformers import DynamicCache

import random
import math

from role_assignment_functions import *
import config

# Definitions of all the seq2seq models and the TPDN
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')






# Atomic Pair Encoder
class AtomicPairEncoder(nn.Module):
    def __init__(self, hidden_size=None, n_pairs=None, pair_dim=None, has_linear_layer=True, seq2pairs=None): 
        super(AtomicPairEncoder, self).__init__()

        self.hidden_size = hidden_size

        # Vocab size for the pairs
        self.n_pairs = n_pairs

        # Embedding size for pairs
        self.pair_dim = pair_dim

        # Embeddings for pairs
        # The padding_idx means that the padding token, 0, will be embedded
        # as a vector of all zeroes that get no gradient
        self.pair_embedding = nn.Embedding(self.n_pairs, self.pair_dim, padding_idx=config.PAD_TOKEN)

        # Function that takes in the batch and returns
        # the pair ids for the sequences in the batch
        self.seq2pairs = seq2pairs

        # Final linear layer that takes in the sum of the pair embeddings
        # and outputs an encoding of size self.hidden_size
        self.has_linear_layer = has_linear_layer
        if self.has_linear_layer:
            self.output_layer = nn.Linear(self.pair_dim, self.hidden_size)


    # Function for a forward pass through this layer. Takes a list of pairs 
    # and returns a single vector encoding it.
    def forward(self, batch):

        # batch: dict with keys input_seq, target_seq, input_lengths, target_lengths
        # input_seq is a list of shape = (batch_size, max_seq_length).
        # input_seq is already padded with zeroes to deal with
        #     variable sequence lengths within the batch
        # input_seq is already formatted as numerical indices for the tokens
        # input_lengths is the lengths of all the sequences pre-padding.
        #     It is a list of length batch_size

        pairs = self.seq2pairs.assign_pairs_batch(batch)
        pairs = torch.LongTensor(pairs).to(device=device)

        # Embed the pairs
        pairs = self.pair_embedding(pairs)

        hidden = torch.sum(pairs, dim=1).unsqueeze(0)

        
        # Pass the encoding through the final linear layer
        if self.has_linear_layer:
            hidden = self.output_layer(hidden)


        # hidden is shape (1, batch_size, hidden_size)
        # outputs is shape (seq_len, batch_size, hidden_size)
        return None, hidden





