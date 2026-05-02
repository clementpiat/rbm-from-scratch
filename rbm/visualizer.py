import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rbm.rbm import RestrictedBoltzmannMachine


class Visualizer:
    def __init__(self, rbm: RestrictedBoltzmannMachine, output_name: str) -> None:
        assert rbm.trained, "RBM needs to be trained."
        self.rbm = rbm
        self.output_name = output_name

    def _plot_ts(self) -> None:
        ax = self.fig.add_subplot(self.gs[0, 0])
        ax.plot(self.rbm.evaluatation_metrics["free_energy_train"], label="train")
        ax.plot(self.rbm.evaluatation_metrics["free_energy_val"], label="eval")
        ax.legend()
        free_energy_noise = int(
            np.mean(self.rbm.evaluatation_metrics["free_energy_noise"])
        )
        ax.set_title(f"Free energy (noise = {free_energy_noise})")

        ax = self.fig.add_subplot(self.gs[0, 1])
        ax.plot(self.rbm.evaluatation_metrics["reconstruction_error"])
        ax.set_title("Reconstruction error")

        ax = self.fig.add_subplot(self.gs[1, 0])
        ax.plot(self.rbm.evaluatation_metrics["weight_abs"])
        ax.set_title("Weights absolute value")

    def _plot_hidden_units_activations(self) -> None:
        ax = self.fig.add_subplot(self.gs[1, 1])
        h_image = self.rbm.probe_h(self.rbm.training_batches[0])
        h_image = (h_image * 255).astype(np.uint8)
        h_image = Image.fromarray(h_image)
        ax.imshow(h_image, aspect="auto")
        ax.set_title("Hidden units activations")

    def plot(self) -> None:
        self.fig = plt.figure(constrained_layout=True, figsize=[10, 10])
        self.gs = self.fig.add_gridspec(2, 2)

        self._plot_ts()
        self._plot_hidden_units_activations()

        plt.savefig(f"figures/{self.output_name}")
