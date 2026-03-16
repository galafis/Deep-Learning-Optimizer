"""
Tests for the Deep Learning Optimizers module.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimizers import (
    SGD,
    MomentumSGD,
    AdaGrad,
    RMSProp,
    Adam,
    AdamW,
    benchmark_optimizers,
)


# Quadratic objective: f(x) = 0.5 * x^T x, grad = x
def quadratic(params):
    return 0.5 * np.sum(params ** 2)

def quadratic_grad(params):
    return params.copy()


# Rosenbrock objective (2D): f(x,y) = (1-x)^2 + 100*(y-x^2)^2
def rosenbrock(params):
    x, y = params[0], params[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(params):
    x, y = params[0], params[1]
    dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])


@pytest.fixture
def params_1d():
    return np.array([5.0, -3.0, 2.0, -1.0])


@pytest.fixture
def params_2d():
    return np.array([-1.5, 2.0])


class TestSGD:
    def test_basic_update(self, params_1d):
        opt = SGD(learning_rate=0.1)
        grads = quadratic_grad(params_1d)
        new_params = opt.update(params_1d, grads)
        assert new_params.shape == params_1d.shape
        # Parameters should move toward zero
        assert np.all(np.abs(new_params) < np.abs(params_1d))

    def test_convergence_quadratic(self):
        params = np.array([10.0, -10.0])
        opt = SGD(learning_rate=0.1)
        for _ in range(100):
            grads = quadratic_grad(params)
            params = opt.update(params, grads)
        assert np.allclose(params, 0.0, atol=1e-3)

    def test_zero_gradients(self, params_1d):
        opt = SGD(learning_rate=0.1)
        grads = np.zeros_like(params_1d)
        new_params = opt.update(params_1d, grads)
        np.testing.assert_array_equal(new_params, params_1d)


class TestMomentumSGD:
    def test_basic_update(self, params_1d):
        opt = MomentumSGD(learning_rate=0.01, beta=0.9)
        grads = quadratic_grad(params_1d)
        new_params = opt.update(params_1d, grads)
        assert new_params.shape == params_1d.shape

    def test_momentum_accumulation(self):
        params = np.array([5.0])
        opt = MomentumSGD(learning_rate=0.01, beta=0.9)
        # Multiple steps - velocity should accumulate
        prev_delta = 0
        for _ in range(5):
            grads = quadratic_grad(params)
            old_params = params.copy()
            params = opt.update(params, grads)
            delta = np.abs(old_params - params)[0]
            assert delta >= prev_delta * 0.5  # momentum helps
            prev_delta = delta

    def test_convergence(self):
        params = np.array([10.0, -10.0])
        opt = MomentumSGD(learning_rate=0.01, beta=0.9)
        for _ in range(500):
            grads = quadratic_grad(params)
            params = opt.update(params, grads)
        assert np.allclose(params, 0.0, atol=1e-2)


class TestAdaGrad:
    def test_basic_update(self, params_1d):
        opt = AdaGrad(learning_rate=0.5)
        grads = quadratic_grad(params_1d)
        new_params = opt.update(params_1d, grads)
        assert new_params.shape == params_1d.shape

    def test_adaptive_lr_decay(self):
        """AdaGrad should effectively reduce learning rate over time."""
        params = np.array([5.0])
        opt = AdaGrad(learning_rate=1.0)
        deltas = []
        for _ in range(20):
            grads = quadratic_grad(params)
            old_params = params.copy()
            params = opt.update(params, grads)
            deltas.append(float(np.abs(old_params - params)[0]))
        # Later steps should have smaller updates
        assert deltas[-1] < deltas[0]


class TestRMSProp:
    def test_basic_update(self, params_1d):
        opt = RMSProp(learning_rate=0.01)
        grads = quadratic_grad(params_1d)
        new_params = opt.update(params_1d, grads)
        assert new_params.shape == params_1d.shape

    def test_convergence(self):
        params = np.array([10.0, -10.0])
        opt = RMSProp(learning_rate=0.01, beta=0.9)
        for _ in range(500):
            grads = quadratic_grad(params)
            params = opt.update(params, grads)
        assert np.allclose(params, 0.0, atol=0.1)


class TestAdam:
    def test_basic_update(self, params_1d):
        opt = Adam(learning_rate=0.01)
        grads = quadratic_grad(params_1d)
        new_params = opt.update(params_1d, grads)
        assert new_params.shape == params_1d.shape

    def test_bias_correction(self):
        """First steps should have bias-corrected moments."""
        opt = Adam(learning_rate=0.001)
        params = np.array([1.0])
        grads = np.array([1.0])
        new_params = opt.update(params, grads)
        # With bias correction, first step should be meaningful
        assert new_params[0] < params[0]
        assert opt.t == 1

    def test_convergence_quadratic(self):
        params = np.array([10.0, -10.0, 5.0])
        opt = Adam(learning_rate=0.1)
        for _ in range(500):
            grads = quadratic_grad(params)
            params = opt.update(params, grads)
        assert np.allclose(params, 0.0, atol=0.1)

    def test_convergence_rosenbrock(self, params_2d):
        params = params_2d.copy()
        opt = Adam(learning_rate=0.005)
        for _ in range(5000):
            grads = rosenbrock_grad(params)
            params = opt.update(params, grads)
        # Should approach (1, 1)
        assert quadratic(params - np.array([1.0, 1.0])) < 1.0


class TestAdamW:
    def test_basic_update(self, params_1d):
        opt = AdamW(learning_rate=0.01, weight_decay=0.01)
        grads = quadratic_grad(params_1d)
        new_params = opt.update(params_1d, grads)
        assert new_params.shape == params_1d.shape

    def test_weight_decay_effect(self, params_1d):
        """Weight decay should produce smaller params than plain Adam."""
        opt_plain = Adam(learning_rate=0.01)
        opt_wd = AdamW(learning_rate=0.01, weight_decay=0.1)
        params_p = params_1d.copy()
        params_w = params_1d.copy()
        for _ in range(50):
            grads = quadratic_grad(params_p)
            params_p = opt_plain.update(params_p, grads)
            grads_w = quadratic_grad(params_w)
            params_w = opt_wd.update(params_w, grads_w)
        assert np.sum(params_w ** 2) <= np.sum(params_p ** 2) * 1.1

    def test_cosine_annealing_lr(self):
        opt = AdamW(
            learning_rate=0.01,
            cosine_annealing=True,
            T_max=10,
            lr_min=0.0001,
        )
        lrs = []
        params = np.array([5.0])
        for _ in range(20):
            grads = quadratic_grad(params)
            params = opt.update(params, grads)
            lrs.append(opt.learning_rate)
        # LR should vary cyclically
        assert min(lrs) < max(lrs)
        # At step T_max (index T_max-1), LR should be near lr_min
        assert lrs[9] < 0.001  # close to lr_min at end of first cycle

    def test_convergence(self):
        params = np.array([10.0, -10.0])
        opt = AdamW(learning_rate=0.1, weight_decay=0.01)
        for _ in range(500):
            grads = quadratic_grad(params)
            params = opt.update(params, grads)
        assert np.allclose(params, 0.0, atol=0.1)


class TestBenchmark:
    def test_benchmark_returns_all_optimizers(self):
        initial = np.array([5.0, -3.0])
        optimizers = {
            "sgd": SGD(0.1),
            "adam": Adam(0.01),
        }
        results = benchmark_optimizers(
            quadratic, quadratic_grad, initial, optimizers, n_steps=50
        )
        assert "sgd" in results
        assert "adam" in results
        assert len(results["sgd"]) == 50
        assert len(results["adam"]) == 50

    def test_losses_decrease(self):
        initial = np.array([10.0, -10.0])
        optimizers = {"adam": Adam(0.1)}
        results = benchmark_optimizers(
            quadratic, quadratic_grad, initial, optimizers, n_steps=100
        )
        losses = results["adam"]
        assert losses[-1] < losses[0]

    def test_history_recorded(self):
        opt = Adam(0.1)
        params = np.array([5.0])
        for _ in range(10):
            grads = quadratic_grad(params)
            loss = quadratic(params)
            params = opt.update(params, grads)
            opt.record_step(loss)
        assert len(opt.history) == 10
        assert "step" in opt.history[0]
        assert "loss" in opt.history[0]
        assert "lr" in opt.history[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
