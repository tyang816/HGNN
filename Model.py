import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GatedGraphConv, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, softmax
from Data import ASTGraphDataset
from Modules import GlobalAttention, Encoder, Decoder
import yaml
import math


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_ensemble = False

    def forward(self, src_batch, src_lengths=None, tgt_batch=None, weights=None):
        tgt_batch = tgt_batch[:, :-1]
        memory, final_state = self.encoder(src_batch, src_lengths)
        logits, attn_history = self.decoder(tgt_batch, final_state, memory, src_lengths, weights)
        return logits, attn_history
