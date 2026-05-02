import numpy as np
import pathlib

ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
MAP = {x: i for i, x in enumerate(ALPHABET)}
COL_START, COL_END = 375, 595
PATH_TO_FASTA = pathlib(__file__).parent.resolve() / "spike_GISAID_aligned.fasta"


def read_fasta(filename: str) -> np.ndarray:
    with open(filename) as f:
        raw_data = f.read()

    sequences = [
        "".join(x.split("\n")[1:])[COL_START: COL_END]
        for x in raw_data.split(">")[1:]
    ]

    inputs = []
    for sequence in sequences:
        if "X" in sequence:
            continue

        _input = []
        for char in sequence:
            sub_input = [0] * 21
            sub_input[MAP[char]] = 1
            _input += sub_input

        inputs.append(_input)

    return np.array(inputs)


if __name__ == "__main__":
    x = read_fasta(PATH_TO_FASTA)

    rbm = BinaryRestrictedBoltzmannMachine(
        samples, labels, epochs=10, hidden_units=400, optimizer="adamw"
    )