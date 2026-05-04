import matplotlib.pyplot as plt
import numpy as np
import pathlib

from rbm.visualizer import Visualizer
from use_cases.sars_cov_2_spike_rbd.utils import (
    load_msa,
    load_mutants_and_scores,
    ALPHABET_LENGTH,
)
from use_cases.sars_cov_2_spike_rbd.softmax_rbm import SoftmaxRestrictedBoltzmannMachine


BATCH_SIZE = 5


class RBDVisualizer(Visualizer):
    def _plot_correl(self) -> None:
        mutants, scores = load_mutants_and_scores()
        free_energy = self.rbm.free_energy(mutants, return_mean=False)

        ax = self.fig.add_subplot(self.gs[2:, :])
        ax.scatter(free_energy, scores, s=1)
        correl = np.corrcoef(free_energy, scores)[0, 1]  # Pearson correlation
        ax.set_xlabel("Free energy")
        ax.set_ylabel("Binding avg from DMS")
        ax.set_title(f"Pearson Correlation: {'%.2f' % correl}")
        ax.set_ylim(-1, 1)

    def plot(self, output_name: str) -> None:
        self.fig = plt.figure(constrained_layout=True, figsize=[15, 10])
        self.gs = self.fig.add_gridspec(4, 2)

        self._plot_ts()
        self._plot_hidden_units_activations()
        self._plot_correl()

        plt.savefig(pathlib.Path(__file__).parent.resolve() / output_name)


if __name__ == "__main__":
    x = load_msa()

    batches = [x[i : (i + BATCH_SIZE)] for i in range(0, len(x), BATCH_SIZE)]
    batches = batches[:-1]
    assert all(len(batch) == BATCH_SIZE for batch in batches)

    rbm = SoftmaxRestrictedBoltzmannMachine(
        ALPHABET_LENGTH,
        np.array(batches),
        epochs=10,
        hidden_units=1_000,
        activation="relu",
        optimizer="adamw",
    )
    rbm.train()
    visualizer = RBDVisualizer(rbm)
    visualizer.plot("figures/M1000_e10_relu_adamw.png")
