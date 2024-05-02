import torch.nn as nn
import torch
from .TrajEncoder import TrajEncoder

class Trajectory_Classifier(nn.Module):
    def __init__(self, d_model,dim_feedforward,nhead,
                 n_layers,n_classes,hidden_d, max_len,dropout=0.1):
        super(Trajectory_Classifier, self).__init__()

        self.linear_mapping = nn.Linear(d_model,hidden_d)
        self.class_token = nn.Parameter(torch.rand(1,1,hidden_d))
        self.pos_embedding = nn.Parameter(torch.rand(1,max_len+1,hidden_d))
        self.encoder = TrajEncoder(n_layers, hidden_d, nhead, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_d)
   
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_d),
            nn.Linear(hidden_d,n_classes)
        )   
       
    def forward(self, positions): 
        src = self.linear_mapping(positions)  #(B,max_len, d_model) -> (B,max_len, hidden_d)
        B, L, D = src.shape
        cls_token = self.class_token.expand(B, -1, -1) #(B,1, hidden_d)      
        src = torch.cat((cls_token, src), dim=1) #(B,max_len+1, hidden_d)
        src = src + self.pos_embedding #(B,max_len+1, hidden_d)
        src = self.encoder(src) #(B,max_len+1, hidden_d)
        cls_token = src[:, 0] #(B, hidden_d)
        output = self.mlp_head(cls_token)   #(B, n_classes)
        
        return output