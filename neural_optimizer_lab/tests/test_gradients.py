import unittest

import numpy as np

from neural_network import NeuralNetwork


class GradientCheckTests(unittest.TestCase):
    def test_backpropagation_matches_central_finite_differences(self):
        net = NeuralNetwork(input_dim=2, hidden_dim=4, seed=19)
        # Дані підібрані так, щоб похибка не стояла точно на зламі ReLU.
        X = np.array([[0.13, -0.27], [0.72, 0.19], [-0.44, 0.63], [0.31, 0.91]], dtype=float)
        y = np.array([0, 1, 1, 0])
        _, analytic = net.loss_and_gradients(X, y)
        numeric = {key: np.zeros_like(value) for key, value in net.params.items()}
        eps = 1e-6
        for key, param in net.params.items():
            for index in np.ndindex(param.shape):
                original = param[index]
                param[index] = original + eps
                plus = net.loss(X, y)
                param[index] = original - eps
                minus = net.loss(X, y)
                param[index] = original
                numeric[key][index] = (plus - minus) / (2 * eps)
        max_abs_error = max(float(np.max(np.abs(analytic[k] - numeric[k]))) for k in analytic)
        self.assertLess(max_abs_error, 2e-6, msg=f"Максимальна абсолютна похибка градієнта: {max_abs_error}")


if __name__ == "__main__":
    unittest.main()
