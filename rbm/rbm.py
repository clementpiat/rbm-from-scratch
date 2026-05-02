import numpy as np
from tqdm import tqdm
from typing import Literal

from rbm.adamw import AdamWOptimizer


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return x * (x > 0)



class RestrictedBoltzmannMachine:
    def __init__(
        self,
        batches: list[np.ndarray],
        val_ratio: float = 0.1,
        evaluate_every_k_epochs: int = 1,
        epochs: int = 30,
        hidden_units: int = 400,
        cdn_steps: int = 1,
        temperature: float = 1.0,
        wd: float = 0.01,
        lr: float = 5e-4,
        optimizer: Literal["sgd", "adamw"] = "sgd",
        activation: Literal["sigmoid", "relu"] = "sigmoid",
    ) -> None:
        # Training data
        assert 0 < val_ratio < 1, f"Bad value for val_ratio: {val_ratio}"
        thresh = int(len(batches) * (1 - val_ratio))
        self.training_batches = batches[:thresh]
        self.validation_batches = batches[thresh:]
        self.N = batches[0].shape[1]  # input lengths

        # Hyperparams
        self.M = hidden_units  # number of hidden units
        self.n = cdn_steps  # number of alternating Gibbs sampling steps in Contrastive Divergence
        self.T = temperature  # temperature
        self.epochs = epochs  # number of epochs
        self.wd = wd  # weight decay
        self.lr = lr  # learning rate

        # Weights and biases
        np.random.seed(0)
        self.w = np.random.normal(loc=0, scale=0.01, size=(self.N, self.M))
        p = np.clip(np.mean(self.training_batches[0], axis=0), 1e-4, 1 - 1e-4)
        self.a = np.log(p / (1 - p))
        self.b = np.zeros(self.M)

        # Optimizers
        self.optimizer = optimizer
        if self.optimizer == "adamw":
            self.w_optim = AdamWOptimizer(self.w, lr=self.lr, wd=self.wd)
            self.a_optim = AdamWOptimizer(self.a, lr=self.lr, wd=self.wd)
            self.b_optim = AdamWOptimizer(self.b, lr=self.lr, wd=self.wd)

        # Activation
        self.activation = activation

        # Evaluation
        self.evaluate_every_k_epochs = evaluate_every_k_epochs
        self.evaluatation_metrics = {
            "free_energy_train": [],
            "free_energy_val": [],
            "free_energy_noise": [],
            "reconstruction_error": [],
            "weight_abs": [],
        }
        self.trained = False

    def probe_h(self, v: np.ndarray) -> np.ndarray:
        x = (self.b + v @ self.w) / self.T
        if self.activation == "relu":
            return relu(x)

        return sigmoid(x)

    def probe_v(self, h: np.ndarray) -> np.ndarray:
        return sigmoid((self.a + h @ self.w.T) / self.T)

    def sample_h(self, v: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            x = (self.b + v @ self.w) / self.T
            x += np.random.normal(scale=sigmoid(x))
            return relu(x)

        return (np.random.rand(len(v), self.M) < self.probe_h(v)).astype(np.uint)

    def sample_v(self, h: np.ndarray) -> np.ndarray:
        return (np.random.rand(len(h), self.N) < self.probe_v(h)).astype(np.uint)

    def train_batch(self, v: np.ndarray) -> None:
        # Wake phase
        # "Neurones that fire together, wire together", Hebb.
        h = self.probe_h(v)  # (B, M)
        hebbian_term = v.T @ h  # (N, M)

        # Sleep / Dream / Unlearning phase
        # Thermal equilibrium is approximated via Contrastive Divergence
        h2 = np.copy(h)
        for _ in range(self.n):
            v2 = self.probe_v(h2)
            h2 = self.sample_h(v2)

        h2 = self.probe_h(v2)  # use prob for the gradient
        unlearning_term = v2.T @ h2
        dw = (unlearning_term - hebbian_term) / self.T
        da = np.sum(v2 - v, axis=0)
        db = np.sum(h2 - h, axis=0)

        # Optimization step
        if self.optimizer == "sgd":
            self.w -= self.lr * (dw + 2 * self.wd * self.w)
            self.a -= self.lr * da
            self.b -= self.lr * db
        else:
            self.w_optim.update(dw)
            self.a_optim.update(da)
            self.b_optim.update(db)

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
        x = (self.b + v @ self.w) / self.T  # B, M
        if self.activation == "relu":
            f = -np.dot(v, self.a) - self.T / 2 * np.sum(np.square(relu(x)), axis=1)
        else:
            f = -np.dot(v, self.a) - self.T * np.sum(np.logaddexp(0, x), axis=1)
        return np.mean(f)

    def reconstruction_error(self, v: np.ndarray) -> np.floating:
        h = self.sample_h(v)
        v2 = self.sample_v(h)
        return np.linalg.norm(v - v2) / v.size  # L2 norm by default

    def evaluate(self) -> None:
        free_energy_train = self.free_energy(self.training_batches[0])
        free_energy_val = self.free_energy(self.validation_batches[0])
        noise = np.random.randint(0, 2, size=[len(self.training_batches[0]), self.N])
        free_energy_noise = self.free_energy(noise)
        reconstruction_error = self.reconstruction_error(self.training_batches[0])

        self.evaluatation_metrics["free_energy_train"].append(free_energy_train)
        self.evaluatation_metrics["free_energy_val"].append(free_energy_val)
        self.evaluatation_metrics["free_energy_noise"].append(free_energy_noise)
        self.evaluatation_metrics["reconstruction_error"].append(reconstruction_error)
        self.evaluatation_metrics["weight_abs"].append(np.abs(self.w).mean())
