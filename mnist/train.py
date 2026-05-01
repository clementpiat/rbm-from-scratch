import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image

from utils import to_image, to_binary_flat
from rbm import BinaryRestrictedBoltzmannMachine


class Visualizer:
    def __init__(self, rbm: BinaryRestrictedBoltzmannMachine, output_name: str) -> None:
        assert rbm.trained, "RBM needs to be trained."
        self.rbm = rbm
        self.output_name = output_name

    def _plot_patterns(self) -> None:
        nrows, ncols = 4, 10
        gs_recon = self.gs[2, :].subgridspec(nrows, ncols)
        _min, _max = np.min(self.rbm.w), np.max(self.rbm.w)
        for k, flat_image in enumerate(self.rbm.w.T[: (nrows * ncols)]):
            flat_image = (flat_image - _min) / _max
            im = to_image(flat_image)
            ax = self.fig.add_subplot(gs_recon[k])
            ax.imshow(im)
            ax.axis("off")

    def _plot_reconstruction(self) -> None:
        n_labels = 10
        gs_recon = self.gs[3, :].subgridspec(2, n_labels // 2)

        v = self.rbm.validation_batches[0]
        h = self.rbm.sample_h(v)
        v = self.rbm.sample_v(h)
        for k, flat_image in enumerate(v):
            im = to_image(flat_image)
            ax = self.fig.add_subplot(gs_recon[k])
            ax.imshow(im)

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

    def _plot_tsne(self) -> None:
        ax = self.fig.add_subplot(self.gs[4:, :])

        samples = np.array([x for l in self.rbm.validation_batches for x in l])

        X = self.rbm.probe_h(samples)
        X_embedded = TSNE().fit_transform(X)
        for i in range(10):
            X_i = np.array([x for j, x in enumerate(X_embedded) if j % 10 == i])
            ax.scatter(X_i[:, 0], X_i[:, 1], label=i, s=2)

        ax.legend(markerscale=4.0)
        ax.set_title("TSNE on the hidden units activations on the validation set")

    def plot(self) -> None:
        self.fig = plt.figure(constrained_layout=True, figsize=[10, 16])
        self.gs = self.fig.add_gridspec(6, 2)

        self._plot_ts()
        self._plot_hidden_units_activations()
        self._plot_patterns()
        self._plot_reconstruction()
        self._plot_tsne()

        plt.savefig(f"figures/{self.output_name}")


if __name__ == "__main__":
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset("mnist")
    samples = np.array([to_binary_flat(x["image"]) for x in ds["train"]])
    labels = np.array([x["label"] for x in ds["train"]])

    rbm = BinaryRestrictedBoltzmannMachine(
        samples, labels, epochs=10, hidden_units=400, optimizer="adamw"
    )
    rbm.train()
    visualizer = Visualizer(rbm, "M400_adamw.png")
    visualizer.plot()
