# prediction/validator.py

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def validate_sequence(sequence, seq_type="Protein"):
    """
    Validate a protein or peptide sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence
    seq_type : str
        Protein or Peptide

    Returns
    -------
    bool
    """

    sequence = sequence.upper()

    if len(sequence) == 0:
        raise ValueError(f"{seq_type} sequence is empty.")

    invalid = set(sequence) - VALID_AA

    if invalid:
        raise ValueError(
            f"{seq_type} contains invalid amino acids: {', '.join(sorted(invalid))}"
        )

    return True
