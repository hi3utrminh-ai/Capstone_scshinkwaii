import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic
import torch.nn.functional as F
from models.graphcnn_congForSJSSP import GraphCNN
import torch


class ActorCritic(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device
                 ):
        super(ActorCritic, self).__init__()
        self.n_j = n_j
        self.n_m = n_m
        self.device = device

        #NOTE FOR PROFESSOR: CHANGED: action space is "rules" (not operations)
        self.num_rules = 8

        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        
  #NOTE FOR PROFESSOR: CHANGE: Actor head now maps the GLOBAL graph embedding (size = hidden_dim)
        #         directly to logits over “num_rules” (instead of scoring each candidate operation).
        #         Old code used hidden_dim*2 (candidate feature + global feature); not needed anymore.
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim, hidden_dim_actor, self.num_rules).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                candidate,  # 0, no longer used
                mask,       # 0, no longer used
                ):

        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        #NOTE FOR PROFESSOR:# CHANGE: removed per-candidate feature assembly:
        #         - no torch.gather() to pick candidate-node embeddings
        #         - no concatenation with h_pooled
        #         - no mask application on per-candidate scores
        # Now we just score “rules” based on the global embedding.
        pi = F.softmax(rule_scores, dim=-1)
        v = self.critic(h_pooled)
        
        return pi, v


if __name__ == '__main__':

    print('This is the ActorCritic model for RULE SELECTION.')
