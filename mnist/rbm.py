import numpy as np
from tqdm import tqdm
from scipy.special import expit as sigmoid

from utils import get_batches


class BinaryRestrictedBoltzmannMachine:
    def __init__(
        self,
        training_samples: np.ndarray,
        training_labels: np.ndarray,
        val_ratio: float = 0.1,
        evaluate_every_k_epochs: int = 1,
        epochs: int = 30,
        hidden_units: int = 400,
        cdn_steps: int = 3,
        temperature: float = 1.0,
        wd: float = 1e-4,
        lr: float = 1e-4,
    ) -> None:
        # Training data
        batches = get_batches(training_samples, training_labels)
        assert 0 < val_ratio < 1, f"Bad value for val_ratio: {val_ratio}"
        thresh = int(len(batches) * (1 - val_ratio))
        self.training_batches = batches[:thresh]
        self.validation_batches = batches[thresh:]

        self.N = training_samples.shape[1]  # input lengths

        # Hyperparams
        self.M = hidden_units  # number of hidden units
        self.n = cdn_steps  # number of alternating Gibbs sampling steps in Contrastive Divergence
        self.T = temperature  # temperature
        self.epochs = epochs  # number of epochs
        self.wd = wd  # weight decay
        self.lr = lr

        # Weights and biases
        np.random.seed(0)
        self.W = np.random.normal(loc=0, scale=0.01, size=(self.N, self.M))
        p = np.clip(np.mean(training_samples, axis=0), 1e-4, 1 - 1e-4)
        self.a = np.log(p / (1 - p))
        self.b = np.zeros(self.M)

        # Evaluation metrics
        self.evaluate_every_k_epochs = evaluate_every_k_epochs
        self.evaluatation_metrics = {
            "free_energy_train": [],
            "free_energy_val": [],
            "reconstruction_loss": [],
            "weight_abs": [],
        }
        self.trained = False

    def probe_h(self, v: np.ndarray) -> np.ndarray:
        return sigmoid((self.b + v @ self.W) / self.T)

    def probe_v(self, h: np.ndarray) -> np.ndarray:
        return sigmoid((self.a + h @ self.W.T) / self.T)

    def sample_h(self, v: np.ndarray) -> np.ndarray:
        return np.random.rand(len(v), self.M) < self.probe_h(v)

    def sample_v(self, h: np.ndarray) -> np.ndarray:
        return np.random.rand(len(h), self.N) < self.probe_v(h)

    def train_batch(self, v: np.ndarray) -> None:
        # "Neurones that fire together, wire together", Hebb.
        h = self.probe_h(v)  # (B, M)
        hebbian_term = v.T @ h  # (N, M)

        # Build the negative term with CD_n
        h2 = np.copy(h)
        for _ in range(self.n):
            v2 = self.probe_v(h2)
            h2 = self.sample_h(v2)

        negative_term = v2.T @ h2
        dw = (hebbian_term - negative_term) / self.T
        self.W = self.W + self.lr * dw - self.wd * self.W

        da = np.mean(v - v2, axis=0)
        self.a = self.a + self.lr * da

        db = np.mean(h - h2, axis=0)
        self.b = self.b + self.lr * db

    def train(self) -> None:
        assert not self.trained, "Cannot train an RBM instance twice."

        self.evaluate()
        for e in tqdm(range(1, self.epochs + 1), desc="Training"):
            for batch in self.training_batches:
                self.train_batch(batch)

            if e % self.evaluate_every_k_epochs == 0:
                self.evaluate()

        self.trained = True

    def free_energy(self, v: np.ndarray) -> np.floating:
        x = (self.b + v @ self.W) / self.T  # B, M
        f = -np.dot(v, self.a) - self.T * np.sum(np.logaddexp(0, x), axis=1)  # B
        return np.mean(f)

    def reconstruction_loss(self, v: np.ndarray) -> np.floating:
        h = self.sample_h(v)
        v2 = self.sample_v(h)
        return np.linalg.norm(v - v2)  # L2 norm

    def get_long_chain_visible_inputs(
        self, chain_length: int = 20, n_images: int = 16
    ) -> np.ndarray:
        v = np.random.randint(0, 2, size=[n_images, self.N])
        for _ in range(chain_length):
            h = self.sample_h(v)
            v = self.sample_v(h)

        return v

    def evaluate(self) -> None:
        free_energy_train = self.free_energy(self.training_batches[0])
        free_energy_val = self.free_energy(self.validation_batches[0])
        reconstruction_loss = self.reconstruction_loss(self.training_batches[0])

        self.evaluatation_metrics["free_energy_train"].append(free_energy_train)
        self.evaluatation_metrics["free_energy_val"].append(free_energy_val)
        self.evaluatation_metrics["reconstruction_loss"].append(reconstruction_loss)
        self.evaluatation_metrics["weight_abs"].append(np.abs(self.W).mean())
