
from torch import nn

from .encoder_layer import EncoderLayer



class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob, args, eAtt_d):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob,
                                                  args=args,
                                                  eAtt_d=eAtt_d
                                                  )
                                     for _ in range(n_layers)])

    def forward(self, x):


        for layer in self.layers:
            x = layer(x)

        return x
