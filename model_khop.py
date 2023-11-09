import torch.nn.functional as F
from models.GCN import *
from models.transformer import Transformer
from metrics import *

class SemGcnGat(nn.Module):

    def __init__(self, device, n_layers, n_features, hidden_dim, dropout, n_classes, args,
                 ffn_dim=512, tsf_layer=2, tsf_head=8, tsf_drop=0.1, eAtt_d=64):### n_layer  名字与外层橙色n_layer 字母完全一样
        super(SemGcnGat, self).__init__()

        self.dev = device
        # ! Graph Convolution
        self.GCN = GCN(n_layers=n_layers, n_features=n_features, hidden_dim=hidden_dim, dropout=dropout,
                       n_classes=n_classes, args=args, device=device)
        self.args = args

        if args.model == 'gcn+sem+tsf+ExAtt':
            self.transformer = Transformer(d_model=hidden_dim, n_head=tsf_head, ffn_hidden=ffn_dim, n_layers=tsf_layer,
                                          drop_prob=tsf_drop, args=args, eAtt_d=eAtt_d)
        self.classifier = nn.Linear(hidden_dim, n_classes)


        self.scores=  nn.Parameter(torch.FloatTensor(n_features, 1))
        nn.init.xavier_uniform_(self.scores)
        self.bias = nn.Parameter(torch.FloatTensor(1))
        nn.init.zeros_(self.bias)
    def forward(self, features, adj_ori, adj_khop):

        if self.args.model =='gcn+sem+tsf+ExAtt':
            s_i = torch.sigmoid(features @ self.scores + self.bias)
            h_gcn_adj = self.GCN(features, adj_ori)
            h_gcn_hop = self.GCN(features, adj_khop)
            out = s_i * h_gcn_adj + (1 - s_i) * h_gcn_hop

            h_tsf = out.unsqueeze(0)
            h = self.transformer(h_tsf)
            out = torch.squeeze(h, 0)

            out = self.classifier(out)
            return out





