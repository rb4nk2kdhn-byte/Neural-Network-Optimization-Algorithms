import unittest

import numpy as np

from neural_network import NeuralNetwork


class NeuralNetworkTests(unittest.TestCase):
    def test_forward_shapes_and_probability_range(self):
        net = NeuralNetwork(input_dim=2, hidden_dim=5, seed=7)
        X = np.array([[0.2, -0.4], [1.0, 0.5], [-0.1, 0.8]])
        probabilities, cache = net.forward(X)
        self.assertEqual(probabilities.shape, (3, 1))
        self.assertEqual(cache["z1"].shape, (3, 5))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_binary_cross_entropy_is_stable_for_extreme_logits(self):
        logits = np.array([[1000.0], [-1000.0], [1000.0], [-1000.0]])
        y = np.array([1, 0, 0, 1])
        value = NeuralNetwork.binary_cross_entropy_from_logits(logits, y)
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, 0.0)

    def test_same_seed_gives_same_parameters(self):
        first = NeuralNetwork(seed=123)
        second = NeuralNetwork(seed=123)
        for key in first.PARAMETER_NAMES:
            np.testing.assert_array_equal(first.params[key], second.params[key])

    def test_parameter_shape_validation(self):
        net = NeuralNetwork(input_dim=2, hidden_dim=4)
        bad = net.copy_parameters()
        bad["W1"] = np.zeros((3, 4))
        with self.assertRaises(ValueError):
            net.set_parameters(bad)


if __name__ == "__main__":
    unittest.main()
