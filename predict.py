import argparse
import torch

from prediction.fasta_reader import (
    read_protein_fasta,
    read_peptide_fasta
)

from prediction.validator import validate_sequence

from utils.inference import build_pair

from prediction.ranking import rank_predictions

from models.pepgnn import PepGNN


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protein",
        required=True
    )

    parser.add_argument(
        "--peptides",
        required=True
    )

    parser.add_argument(
        "--output",
        default="predictions.csv"
    )

    args = parser.parse_args()

    print("Loading model...")

    model = PepGNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    print("Reading FASTA files...")

    protein_id, protein_seq = read_protein_fasta(
        args.protein
    )

    validate_sequence(
        protein_seq,
        "Protein"
    )

    peptides = read_peptide_fasta(
        args.peptides
    )

    results = []

    with torch.no_grad():

        for peptide_id, peptide_seq in peptides:

            validate_sequence(
                peptide_seq,
                "Peptide"
            )

            protein_graph, peptide_graph = build_pair(
                protein_seq,
                peptide_seq,
                DEVICE
            )

            logits = model(
                protein_graph,
                peptide_graph
            )

            probability = torch.sigmoid(
                logits
            ).item()

            prediction = (
                "Binding"
                if probability >= 0.54
                else "Non-binding"
            )

            results.append({

                "Protein_ID": protein_id,

                "Peptide_ID": peptide_id,

                "Peptide_Sequence": peptide_seq,

                "Probability": round(
                    probability,
                    4
                ),

                "Prediction": prediction

            })

    ranked = rank_predictions(results)

    ranked.to_csv(
        args.output,
        index=False
    )

    print()

    print(ranked)

    print()

    print("Saved:", args.output)


if __name__ == "__main__":

    main()
