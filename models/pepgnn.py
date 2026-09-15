import torch
import torch.nn as nn

from models.gnn_encoder import GNNEncoder


class PepGNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = GNNEncoder()

              

        # Classification Head
        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, 1)

        )

    def forward(self, protein_graph, peptide_graph):

        protein_embedding = self.encoder(protein_graph)

        peptide_embedding = self.encoder(peptide_graph)

        x = torch.cat(
            [protein_embedding, peptide_embedding],
            dim=1
        )

        x = self.classifier(x)

        return x
