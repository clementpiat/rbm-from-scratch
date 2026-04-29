import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils import to_image, to_binary_flat
from rbm import BinaryRestrictedBoltzmannMachine


class Visualizer:
    """
    Class to visualize a trained RBM.
    """

    def __init__(self, rbm: BinaryRestrictedBoltzmannMachine, output_name: str) -> None:
        assert rbm.trained, "RBM needs to be trained."
        self.rbm = rbm
        self.output_name = output_name

    def _plot_recon(self) -> None:
        ncol = 4
        gs_recon = self.gs[1:5, :].subgridspec(ncol, ncol)

        v = self.rbm.sample_long_chain_v()
        for k, flat_image in enumerate(v[: (ncol**2)]):
            im = to_image(flat_image)
            ax = self.fig.add_subplot(gs_recon[k])
            ax.imshow(im)

    def _plot_ts(self) -> None:
        ax = self.fig.add_subplot(self.gs[0, 0])
        ax.plot(self.rbm.evaluatation_metrics["free_energy_train"], label="train")
        ax.plot(self.rbm.evaluatation_metrics["free_energy_val"], label="eval")
        ax.set_title("Free energy")

        ax = self.fig.add_subplot(self.gs[0, 1])
        ax.plot(self.rbm.evaluatation_metrics["reconstruction_error"])
        ax.set_title("Reconstruction error")

        ax = self.fig.add_subplot(self.gs[0, 2])
        ax.plot(self.rbm.evaluatation_metrics["weight_abs"])
        ax.set_title("Weights absolute value")

    def _plot_tsne(self) -> None:
        ax = self.fig.add_subplot(self.gs[5:, :])

        samples = np.array([x for l in self.rbm.validation_batches for x in l])
        labels = np.array([i % 10 for i, _ in enumerate(samples)])

        X = self.rbm.probe_h(samples)
        X_embedded = TSNE().fit_transform(X)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap="tab10", s=0.5)

    def plot(self) -> None:
        self.fig = plt.figure(constrained_layout=True, figsize=[10, 16])
        self.gs = self.fig.add_gridspec(8, 3)

        self._plot_ts()
        self._plot_recon()
        self._plot_tsne()

        plt.savefig(f"figures/{self.output_name}")


if __name__ == "__main__":
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset("mnist")
    samples = np.array([to_binary_flat(x["image"]) for x in ds["train"]])
    labels = np.array([x["label"] for x in ds["train"]])

    rbm = BinaryRestrictedBoltzmannMachine(samples, labels, epochs=200)
    rbm.train()
    visualizer = Visualizer(rbm, "M400_e200.png")
    visualizer.plot()
