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
        self.max_length = max_len
       
    def forward(self, positions):
        ## new
        # self.max_squ_size = 600
        
        
        # hidden_embed_dim = 10    ## feat dim -> 10
        # input : trajs (600, 4)
        # cls token : (1, hidden_embed_dim)'

        # step
        # trajs (600, 4) -> ???? -> trajs_2 (600, hed)
        # trajs_2 (600, hed) + cls token -> trajs_3  (601, hed)
        # + PE  (601, hed) -> encoder -> MLP -> pred label 

        src = self.linear_mapping(positions)  ##(B,max_len, d_model) -> (B,max_len, hidden_d)
        # print("\nself mapping",src.shape)
        B, L, D = src.shape
        cls_token = self.class_token.expand(B, -1, -1) ##(B,1, hidden_d) -> (B, 1, hidden_d)
        # print("\nclass token",cls_token.shape) 
        
        src = torch.cat((cls_token, src), dim=1) ##(B,max_len+1, hidden_d)
        # print("\nsrc + class token",src.shape)
        src = src + self.pos_embedding ##(B,max_len+1, hidden_d)
        # print("\nsrc + pos embedding",src.shape) 
        src = self.dropout(src)
        
        src = self.encoder(src) ##(B,max_len+1, hidden_d)
        # print("\nencoder ",src.shape)
        src = self.norm(src) ##(B,max_len+1, hidden_d)
        # print("\nnorm",src.shape)
        cls_token = src[:, 0] ##(B, hidden_d)
        # print("\ncls_token",cls_token.shape)
        # print("\ncls_token",cls_token)
        
        output = self.mlp_head(cls_token)   #(B, n_classes)
        # print("\noutput",output.shape)
        # print("\noutput",output)
        return output