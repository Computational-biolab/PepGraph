from Bio import SeqIO


def read_protein_fasta(fasta_file):

    records = list(SeqIO.parse(fasta_file, "fasta"))

    if len(records) != 1:
        raise ValueError(
            "Protein FASTA must contain exactly one sequence."
        )

    record = records[0]

    return record.id, str(record.seq).upper()


def read_peptide_fasta(file_path):

    peptides = []

    # Check first line
    with open(file_path) as f:
        first = f.readline().strip()

    # ---------- FASTA ----------
    if first.startswith(">"):

        for record in SeqIO.parse(file_path, "fasta"):

            peptides.append(
                (
                    record.id,
                    str(record.seq).upper()
                )
            )

    # ---------- Plain Text ----------
    else:

        with open(file_path) as f:

            lines = f.readlines()

        idx = 1

        for line in lines:

            seq = line.strip().upper()

            if seq == "":
                continue

            # Skip header
            if seq.lower() == "peptide":
                continue

            peptides.append(
                (
                    f"Pep{idx}",
                    seq
                )
            )

            idx += 1

    return peptides
