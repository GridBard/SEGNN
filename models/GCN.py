import torch
from torch import nn

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, device, bias=False):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, inputs, adj):
        support = torch.mm(self.dropout(inputs), self.weight.to(self.device))
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, n_layers, n_features, hidden_dim, dropout, n_classes, args, device):
        super(GCN, self).__init__()
        hid = hidden_dim
        if n_layers == 1:
            self.first_layer = GraphConvolution(n_features, hid, dropout,device)
        else:
            self.first_layer = GraphConvolution(n_features, hidden_dim, dropout, device)
            self.last_layer = GraphConvolution(hidden_dim, hid, dropout,device)
            if n_layers > 2:
                self.gc_layers = nn.ModuleList([
                    GraphConvolution(hidden_dim, hidden_dim, 0,device) for _ in range(n_layers - 2)
                ])
            
        self.n_layers = n_layers
        self.relu = nn.ReLU()
    
    def forward(self, inputs, adj):
        if self.n_layers == 1:
            x = self.first_layer(inputs, adj)
        else:
            x = self.relu(self.first_layer(inputs, adj))
            if self.n_layers > 2:
                for i, layer in enumerate(self.gc_layers):
                    x = self.relu(layer(x, adj))
            x = self.last_layer(x, adj)
        return x
