"""
Deep Learning Optimizers - Custom Implementations from Scratch

Implements gradient-based optimization algorithms for neural network training:
SGD, Momentum, RMSProp, Adam, AdaGrad, and a custom AdamW variant with
warm restarts (cosine annealing).

Author: Gabriel Demetrios Lafis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.step_count = 0
        self.history: List[Dict] = []

    @abstractmethod
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Update parameters given gradients."""
        pass

    def record_step(self, loss: float) -> None:
        """Record optimization step for tracking."""
        self.step_count += 1
        self.history.append({
            "step": self.step_count,
            "loss": float(loss),
            "lr": float(self.learning_rate),
        })


class SGD(Optimizer):
    """
    Stochastic Gradient Descent.

    theta_{t+1} = theta_t - lr * grad_t
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        return params - self.learning_rate * grads


class MomentumSGD(Optimizer):
    """
    SGD with Momentum.

    v_t = beta * v_{t-1} + grad_t
    theta_{t+1} = theta_t - lr * v_t
    """

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.velocity: Optional[np.ndarray] = None

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.beta * self.velocity + grads
        return params - self.learning_rate * self.velocity


class AdaGrad(Optimizer):
    """
    Adaptive Gradient Algorithm.

    Accumulates squared gradients to scale learning rate per-parameter.

    G_t = G_{t-1} + grad_t^2
    theta_{t+1} = theta_t - lr * grad_t / (sqrt(G_t) + eps)
    """

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.accumulated: Optional[np.ndarray] = None

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.accumulated is None:
            self.accumulated = np.zeros_like(params)
        self.accumulated += grads ** 2
        return params - self.learning_rate * grads / (
            np.sqrt(self.accumulated) + self.epsilon
        )


class RMSProp(Optimizer):
    """
    Root Mean Square Propagation.

    Uses exponential moving average of squared gradients.

    E[g^2]_t = beta * E[g^2]_{t-1} + (1-beta) * grad_t^2
    theta_{t+1} = theta_t - lr * grad_t / (sqrt(E[g^2]_t) + eps)
    """

    def __init__(
        self, learning_rate: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.sq_avg: Optional[np.ndarray] = None

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.sq_avg is None:
            self.sq_avg = np.zeros_like(params)
        self.sq_avg = self.beta * self.sq_avg + (1 - self.beta) * grads ** 2
        return params - self.learning_rate * grads / (
            np.sqrt(self.sq_avg) + self.epsilon
        )


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam).

    Combines momentum (first moment) with RMSProp (second moment),
    plus bias correction.

    m_t = beta1 * m_{t-1} + (1-beta1) * grad_t
    v_t = beta2 * v_{t-1} + (1-beta2) * grad_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    theta_{t+1} = theta_t - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t = 0

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class AdamW(Optimizer):
    """
    Adam with Decoupled Weight Decay and Cosine Annealing Warm Restarts.

    Implements weight decay directly on parameters (decoupled from gradient)
    with optional cosine annealing learning rate schedule.

    theta_{t+1} = (1 - wd) * theta_t - lr_t * m_hat / (sqrt(v_hat) + eps)
    lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * T_cur / T_max))
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        cosine_annealing: bool = False,
        T_max: int = 100,
        lr_min: float = 1e-6,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cosine_annealing = cosine_annealing
        self.T_max = T_max
        self.lr_min = lr_min
        self.lr_max = learning_rate
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t = 0

    def _get_lr(self) -> float:
        """Compute current learning rate with cosine annealing."""
        if not self.cosine_annealing:
            return self.learning_rate
        T_cur = self.t % self.T_max
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos(np.pi * T_cur / self.T_max)
        )

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        lr = self._get_lr()

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Decoupled weight decay
        params = (1 - self.weight_decay * lr) * params
        params = params - lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        self.learning_rate = lr  # track for history
        return params


def benchmark_optimizers(
    objective_fn,
    grad_fn,
    initial_params: np.ndarray,
    optimizers: Dict[str, Optimizer],
    n_steps: int = 200,
) -> Dict[str, List[float]]:
    """
    Benchmark multiple optimizers on an objective function.

    Args:
        objective_fn: f(params) -> loss
        grad_fn: f(params) -> gradients
        initial_params: Starting point
        optimizers: Dict of name -> Optimizer
        n_steps: Number of optimization steps

    Returns:
        Dict of name -> list of loss values per step
    """
    results = {}
    for name, opt in optimizers.items():
        params = initial_params.copy()
        losses = []
        for _ in range(n_steps):
            loss = objective_fn(params)
            grads = grad_fn(params)
            params = opt.update(params, grads)
            opt.record_step(loss)
            losses.append(float(loss))
        results[name] = losses
    return results
