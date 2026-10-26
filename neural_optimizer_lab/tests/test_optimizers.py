import unittest

import numpy as np

from optimizers import Adam, Momentum, RMSProp, SGD, create_optimizer


class OptimizerTests(unittest.TestCase):
    def test_sgd_exact_update(self):
        params = {"w": np.array([1.0])}
        SGD(0.1).update(params, {"w": np.array([0.2])})
        np.testing.assert_allclose(params["w"], [0.98], atol=1e-12)

    def test_momentum_accumulates_velocity(self):
        params = {"w": np.array([1.0])}
        opt = Momentum(0.1, momentum=0.9)
        opt.update(params, {"w": np.array([0.2])})
        np.testing.assert_allclose(params["w"], [0.98], atol=1e-12)
        opt.update(params, {"w": np.array([0.2])})
        np.testing.assert_allclose(params["w"], [0.942], atol=1e-12)

    def test_rmsprop_uses_square_average(self):
        params = {"w": np.array([1.0])}
        opt = RMSProp(0.1, beta=0.9, epsilon=1e-8)
        opt.update(params, {"w": np.array([0.2])})
        expected = 1.0 - 0.1 * 0.2 / (np.sqrt(0.1 * 0.2**2) + 1e-8)
        np.testing.assert_allclose(params["w"], [expected], atol=1e-12)

    def test_adam_bias_correction_first_step(self):
        params = {"w": np.array([1.0])}
        opt = Adam(0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
        opt.update(params, {"w": np.array([0.2])})
        np.testing.assert_allclose(params["w"], [0.9], atol=1e-7)
        self.assertEqual(opt.steps, 1)

    def test_factory_returns_distinct_implementations(self):
        instances = [create_optimizer(name, 0.01) for name in ("SGD", "Momentum", "RMSProp", "Adam")]
        self.assertEqual(len({type(instance) for instance in instances}), 4)

    def test_invalid_learning_rate_is_rejected(self):
        for value in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SGD(value)

    def test_nonfinite_gradient_is_rejected(self):
        params = {"w": np.array([1.0])}
        with self.assertRaises(FloatingPointError):
            SGD(0.1).update(params, {"w": np.array([float("nan")])})


if __name__ == "__main__":
    unittest.main()
