import torch.nn as nn
from .EncoderLayer import EncoderLayer

class TrajEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TrajEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, 
                                                  nhead, 
                                                  dim_feedforward, 
                                                  dropout) 
                                     for _ in range(num_layers)])
    
    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src