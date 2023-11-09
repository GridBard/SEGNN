import numpy as np
import torch
from torch import nn
from torch.nn import init



class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model,S,bias=False)
        self.mv = nn.Linear(S,d_model,bias=False)
        self.softmax = nn.Softmax(dim=2)  #
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, q):

        batch_size, head, length, d_tensor = q.size()

        attn = self.mk(q) #bs,n,S
        attn = self.softmax(attn) #bs,n,S
        attn = attn/torch.sum(attn, dim=3, keepdim=True) #bs,n,S
        out = self.mv(attn) #bs,n,d_model

        #out = out.permute(0, 2, 1, 3)   # 这一步在外面 self.out
        #out = out.view(batch_size, length, head*d_tensor) # 这一步在外面 self.out

        return out, attn