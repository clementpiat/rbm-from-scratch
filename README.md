<p align="center"><b><i>
	Restricted Boltzmann Machine from scratch
</b></i></p>

<div align="center">

![Human Written](https://img.shields.io/badge/code-human_written-brightgreen)
![No AI](https://img.shields.io/badge/AI-0%25-black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

</div>

# MNIST

I implemented a minimal RBM training script for MNIST.
It is based on numpy (no PyTorch), uses Contrastive Divergence for computing the gradients, and AdamW optimizer for updating the weights.

```shell
pip install -r requirements.txt
cd mnist
python train.py
```

&rarr; Check the `mnist` folder for more details (theory, learning resources, training figures).

# Genomics data

WIP
