import numpy as np
from tqdm import tqdm

from rbm.rbm import RestrictedBoltzmannMachine


class SoftmaxRestrictedBoltzmannMachine(RestrictedBoltzmannMachine):
    def __init__(self, categories: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.K = categories

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def probe_v(self, h: np.ndarray) -> np.ndarray:
        x = (self.a + h @ self.w.T) / self.T
        probs = self.softmax(x.reshape(-1, self.K))
        return probs.reshape(len(h), self.N)

    def sample_v(self, h: np.ndarray) -> np.ndarray:
        x = (self.a + h @ self.w.T) / self.T
        flat_x = x.reshape(-1, self.K)
        # use the gambel-max trick to sample
        gumbel_noise = np.random.gumbel(size=flat_x.shape)
        sampled_indices = (flat_x + gumbel_noise).argmax(axis=1)

        v = np.zeros_like(flat_x)
        v[np.arange(len(v)), sampled_indices] = 1
        return v.reshape(len(h), self.N)

    def reconstruction_error(self, v: np.ndarray) -> np.floating:
        h = self.sample_h(v)
        v2 = self.sample_v(h)

        v = v.reshape(-1, self.K).argmax(axis=1)
        v2 = v2.reshape(-1, self.K).argmax(axis=1)
        return 1 - np.equal(v, v2).mean()
