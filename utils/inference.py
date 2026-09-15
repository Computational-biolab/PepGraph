import torch
from torch_geometric.data import Batch

from graph_builder import sequence_to_graph


def build_pair(protein_seq, peptide_seq, device):

    protein_graph = sequence_to_graph(protein_seq)
    peptide_graph = sequence_to_graph(peptide_seq)

    protein_batch = Batch.from_data_list([protein_graph]).to(device)
    peptide_batch = Batch.from_data_list([peptide_graph]).to(device)

    return protein_batch, peptide_batch
