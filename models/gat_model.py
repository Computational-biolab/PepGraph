import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool


class PepGNN(nn.Module):

    def __init__(self):

        super().__init__()

        # -----------------------------
        # Peptide Graph Encoder
        # -----------------------------

        self.gat1 = GATv2Conv(
            in_channels=4,
            out_channels=32,
            heads=2,
            concat=True
        )

        self.gat2 = GATv2Conv(
            in_channels=64,
            out_channels=32,
            heads=1,
            concat=True
        )

        # -----------------------------
        # Protein Feature Encoder
        # -----------------------------

        self.protein_fc = nn.Sequential(

            nn.Linear(3,16),

            nn.ReLU(),

            nn.Linear(16,16)

        )

        # -----------------------------
        # Classifier
        # -----------------------------

        self.classifier = nn.Sequential(

            nn.Linear(48,32),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(32,16),

            nn.ReLU(),

            nn.Linear(16,1)

        )

    def forward(self, data):

        x = data.x

        edge_index = data.edge_index

        batch = data.batch

        protein = data.protein_features

        #########################

        x = self.gat1(x, edge_index)

        x = torch.relu(x)

        x = self.gat2(x, edge_index)

        x = torch.relu(x)

        #########################

        peptide_embedding = global_mean_pool(
            x,
            batch
        )

        #########################

        protein_embedding = self.protein_fc(
            protein
        )

        #########################

        combined = torch.cat(

            [

                peptide_embedding,

                protein_embedding

            ],

            dim=1

        )

        #########################

        output = self.classifier(

            combined

        )

        return output.squeeze()
