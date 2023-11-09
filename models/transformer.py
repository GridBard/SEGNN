import torch
from torch import nn
from .encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, d_model, n_head, ffn_hidden, n_layers, drop_prob, args, eAtt_d):
        super().__init__()
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               ffn_hidden=ffn_hidden,
                               args=args, eAtt_d=eAtt_d
                               )

    def forward(self, src):
        output = self.encoder(src)
        return output
