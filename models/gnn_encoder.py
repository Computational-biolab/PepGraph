import torch
import torch.nn as nn

from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


class GNNEncoder(nn.Module):

    def __init__(
        self,
        num_tokens=22,
        embedding_dim=64,
        hidden_dim=128,
        heads=4
    ):

        super().__init__()

        # Amino acid embedding
        self.embedding = nn.Embedding(
            num_embeddings=21,
            embedding_dim=embedding_dim
        )

        # Physicochemical feature encoder
        self.feature_encoder = nn.Sequential(

        nn.Linear(404, 256),

        nn.BatchNorm1d(256),

        nn.ReLU(),

        nn.Dropout(0.3),

        nn.Linear(256, 128),

        nn.BatchNorm1d(128),

        nn.ReLU(),

        nn.Dropout(0.3),

        nn.Linear(128, 64),

        nn.ReLU()

    )

        # 64 (embedding) + 16 (projected features)
        self.gat1 = GATConv(
            embedding_dim + 64,
            hidden_dim,
            heads=heads,
            concat=False
        )

        self.gat2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=1,
            concat=False
        )

        self.relu = nn.ReLU()

    # ----------------------------------------------------
    # Forward
    # ----------------------------------------------------

    def forward(self, data):

        # Amino acid embedding
        embedding = self.embedding(
            data.aa_token
        )

        # Physicochemical features
        physchem = self.feature_encoder(
            data.x
        )

        # Concatenate
        x = torch.cat(
            [embedding, physchem],
            dim=1
        )

        # Graph Attention Layer 1
        x = self.gat1(
            x,
            data.edge_index
        )

        x = self.relu(x)

        # Graph Attention Layer 2
        x = self.gat2(
            x,
            data.edge_index
        )

        # Graph-level embedding
        x = global_mean_pool(
            x,
            data.batch
        )

        return x