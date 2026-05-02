<p align="center"><b><i>
	Restricted Boltzmann Machine from scratch
</b></i></p>

<div align="center">

![Human Written](https://img.shields.io/badge/code-human_written-brightgreen)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

</div>

# RBM

* Implementation of a simple RBM.
* Based on NumPy (no PyTorch); uses Contrastive Divergence for computing the gradients, and AdamW optimizer for updating the weights.
* Implemented both ReLU and Sigmoid for the potential function.
* Check out the `rbm` folder for more details (theory, learning resources, concrete implementation).

# MNIST

I implemented a training script for the MNIST dataset. Check out the `use_cases/mnist` folder to see the training figures.

```shell
pip install -r requirements.txt
python -m use_cases.mnist.train
```

# Sars-CoV-2 Spike RBD

See `use_cases/sars_cov_2_spike_rbd`.
