import numpy as np
import pathlib
from collections import Counter


ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
ALPHABET_LENGTH = len(ALPHABET)
MAP = {x: i for i, x in enumerate(ALPHABET)}

COL_START, COL_END = 566, 1008
BASE_MSA = """
NIT---NLCPFG---EVFNAT--RFASVYAWNRK
RISNCVADYSVLYNSA-------SFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQ
IAPGQTGKIADYNYKLPDDFTG----C--VIAWNSNNLD------------SKVGG-NY-
--------NY----------------LYRLFRKSNLKP-----FERDISTEIYQAGS---
--------TP-C---------------NGVEGF----------------NCYF-------
-----------PLQ---------------------------SY-----------------
----------------GF---------------------QPT----------------NG
VGYQP---------YRV-VVLSFELLHA--PATVCG------PK-KST
""".replace("\n", "")
BETA_FILE = "data/uniprotkb_Betacoronavirus_AND_gene_s_2026_05_03_aligned.fasta"


def msa_str_to_vec(msa_str: str) -> list[int]:
    vec = []
    for char in msa_str:
        subvec = [0 for _ in range(ALPHABET_LENGTH)]
        subvec[MAP[char]] = 1
        vec += subvec

    return vec


def load_msa() -> np.ndarray:
    """
    Read the .fast MSA file, and convert it to a [N_PROTEINS, (21 * RBD_LENGTH)] NumPy array.
    """
    with open(pathlib.Path(__file__).parent.resolve() / BETA_FILE) as f:
        raw_data = f.read()

    sequences = [
        "".join(x.split("\n")[1:])[COL_START:COL_END] for x in raw_data.split(">")[1:]
    ]
    sequences = [s for s in sequences if "X" not in s and "B" not in s]
    inputs = [msa_str_to_vec(msa_str) for msa_str in sequences]
    print(f"Unique sequences: {len(np.unique(sequences))}")
    inputs = np.array(inputs)
    np.random.shuffle(inputs)
    return inputs


def load_mutants_and_scores() -> tuple[np.ndarray, np.ndarray]:
    """
    Read the .csv DMS file, and return 2 NumPy array:
    * the mutants encoded in the RBM input format
    * the average binding score
    """
    rbd_site_to_msa = {}
    rbd_site = 0
    for col, char in enumerate(BASE_MSA):
        if char == "-":
            continue

        rbd_site += 1
        rbd_site_to_msa[rbd_site] = col

    with open(
        pathlib.Path(__file__).parent.resolve() / "data/single_mut_effects.csv"
    ) as f:
        lines = [x.split(",") for x in f][1:]

    mutants, binding_scores = [], []
    for line in lines:
        rbd_site, new_amino_acid, bind_avg = line[0], line[3], line[8]
        new_amino_acid = new_amino_acid.replace('"', "")
        if new_amino_acid == "*":
            continue

        try:
            bind_avg = float(bind_avg)
        except Exception:
            continue

        msa_col = rbd_site_to_msa[int(rbd_site)]
        mutant_msa_str = "".join(
            new_amino_acid if i == msa_col else x for i, x in enumerate(BASE_MSA)
        )
        mutants.append(msa_str_to_vec(mutant_msa_str))
        binding_scores.append(bind_avg)

    return (np.array(mutants), np.array(binding_scores))
