<p align="center"><b><i>
	Restricted Boltzmann Machine from scratch
</b></i></p>

<div align="center">

![Human Written](https://img.shields.io/badge/code-human_written-brightgreen)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

</div>

# RBM

* Implementation of a simple RBM
* Based on NumPy (no PyTorch)
* Implemented ReLU and Sigmoid for the potential function, and AdamW and SGD for the optimizer
* See the `rbm` folder for more details (theory, learning resources, concrete implementation)

# Use cases
## MNIST

Implemented a training script for the MNIST dataset. See the training figures in `use_cases/mnist`.

```shell
pip install -r requirements.txt
python -m use_cases.mnist.train
```

## Sars-CoV-2 Spike RBD

* Trained an RBM on a Multiple Sequence Alignment of the betacoronavirus receptor-binding domain downloaded from UniProt
* Used the Deep Mutational Scanning results of SARS-CoV-2 RBD from Starr et al. to compare the ACE2 binding score of mutants with their RBM energy


See `use_cases/sars_cov_2_spike_rbd` for more details.
