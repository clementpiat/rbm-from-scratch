import numpy as np


class AdamWOptimizer:
    def __init__(
        self,
        w: np.ndarray,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        wd: float = 0.01,
        lr: float = 1e-3,
        epsilon: float = 1e-8,
    ) -> None:
        self.w = w
        self.m = np.zeros(w.shape, dtype=np.float64)
        self.v = np.zeros(w.shape, dtype=np.float64)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.wd = wd
        self.lr = lr
        self.epsilon = epsilon

        self.beta_1_t = 1.0
        self.beta_2_t = 1.0

    def update(self, dw: np.ndarray) -> None:
        # Update moments
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * dw
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(dw)

        # Bias correction
        self.beta_1_t *= self.beta_1
        self.beta_2_t *= self.beta_2
        m_hat = self.m / (1 - self.beta_1_t)
        v_hat = self.v / (1 - self.beta_2_t)

        # Update weights
        self.w += -self.lr * (
            self.wd * self.w + m_hat / (np.sqrt(v_hat) + self.epsilon)
        )
