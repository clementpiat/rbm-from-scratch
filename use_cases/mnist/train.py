import matplotlib.pyplot as plt
import numpy as np
import pathlib
from collections import defaultdict
from PIL import Image
from sklearn.manifold import TSNE


from rbm.rbm import RestrictedBoltzmannMachine
from rbm.visualizer import Visualizer


def to_binary_flat(image: Image.Image) -> np.ndarray:
    return (np.array(image).flatten() > 128).astype(np.uint8)


def to_image(arr: np.ndarray) -> Image.Image:
    x = np.expand_dims(arr, axis=0).reshape(28, 28) * 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def get_batches(samples: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_to_images = defaultdict(lambda: [])
    for image, label in zip(samples, labels):
        label_to_images[label].append(image)

    min_count = min(len(x) for x in label_to_images.values())
    batches = []
    for i in range(min_count):
        batches.append([label_to_images[label][i] for label in label_to_images])

    return np.array(batches)


class MNISTVisualizer(Visualizer):
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

        plt.savefig(
            pathlib(__file__).parent.resolve() / "figures" / self.output_name
        )


if __name__ == "__main__":
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset("mnist")
    samples = np.array([to_binary_flat(x["image"]) for x in ds["train"]])
    labels = np.array([x["label"] for x in ds["train"]])
    batches = get_batches(samples, labels)

    rbm = RestrictedBoltzmannMachine(
        batches, epochs=10, hidden_units=400, optimizer="adamw"
    )
    rbm.train()
    visualizer = MNISTVisualizer(rbm, "M400_adamw.png")
    visualizer.plot()
