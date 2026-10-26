"""Невелика MLP з ReLU, sigmoid, BCE та власним backpropagation."""
from __future__ import annotations

from typing import Mapping

import numpy as np

ArrayDict = dict[str, np.ndarray]


class NeuralNetwork:
    """Повнозв'язна мережа input -> ReLU(hidden) -> sigmoid(output)."""

    PARAMETER_NAMES = ("W1", "b1", "W2", "b2")

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, seed: int = 42) -> None:
        if not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError("input_dim має бути додатним цілим числом.")
        if not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim має бути додатним цілим числом.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        # He-ініціалізація для шару ReLU та помірна ініціалізація виходу.
        self.params: ArrayDict = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)),
            "b1": np.zeros((1, hidden_dim), dtype=np.float64),
            "W2": rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, 1)),
            "b2": np.zeros((1, 1), dtype=np.float64),
        }

    @staticmethod
    def _validate_X(X: np.ndarray, input_dim: int) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != input_dim or X.shape[0] == 0:
            raise ValueError(f"X має мати форму (кількість_прикладів, {input_dim}) і не бути порожнім.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X містить NaN або нескінченність.")
        return X

    @staticmethod
    def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if y.shape[0] != n_samples:
            raise ValueError("Кількість міток y має збігатися з кількістю рядків X.")
        if not np.all(np.isfinite(y)) or np.any((y < 0) | (y > 1)):
            raise ValueError("Мітки y мають бути скінченними числами в межах [0, 1].")
        return y

    @staticmethod
    def sigmoid(logits: np.ndarray) -> np.ndarray:
        """Чисельно стабільна sigmoid без переповнення exp для великих модулів."""
        logits = np.asarray(logits, dtype=np.float64)
        out = np.empty_like(logits)
        positive = logits >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_x = np.exp(logits[~positive])
        out[~positive] = exp_x / (1.0 + exp_x)
        return out

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        X = self._validate_X(X, self.input_dim)
        W1, b1, W2, b2 = (self.params[k] for k in self.PARAMETER_NAMES)
        z1 = X @ W1 + b1
        a1 = np.maximum(0.0, z1)
        logits = a1 @ W2 + b2
        probabilities = self.sigmoid(logits)
        cache = {"X": X, "z1": z1, "a1": a1, "logits": logits, "probabilities": probabilities}
        return probabilities, cache

    @staticmethod
    def binary_cross_entropy_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
        """BCE = mean(log(1+exp(z)) - y*z), обчислена через logaddexp."""
        logits = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if logits.shape != y.shape or logits.size == 0:
            raise ValueError("logits та y мають мати однакову непорожню форму.")
        if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(y)):
            raise FloatingPointError("logits або y містять некоректні числові значення.")
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("Мітки для BCE мають бути в межах [0, 1].")
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        if not np.isfinite(loss):
            raise FloatingPointError("Функція втрат стала некінцевою.")
        return float(loss)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        X = self._validate_X(X, self.input_dim)
        y_col = self._validate_y(y, X.shape[0])
        _, cache = self.forward(X)
        return self.binary_cross_entropy_from_logits(cache["logits"], y_col)

    def loss_and_gradients(self, X: np.ndarray, y: np.ndarray) -> tuple[float, ArrayDict]:
        """Повертає BCE та точні градієнти, отримані правилом ланцюга."""
        X = self._validate_X(X, self.input_dim)
        y_col = self._validate_y(y, X.shape[0])
        _, cache = self.forward(X)
        loss = self.binary_cross_entropy_from_logits(cache["logits"], y_col)

        n = X.shape[0]
        # Для sigmoid + BCE похідна за логітами спрощується до (p - y) / n.
        dz2 = (cache["probabilities"] - y_col) / n
        dW2 = cache["a1"].T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (cache["z1"] > 0.0)
        dW1 = cache["X"].T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        gradients = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        if not all(np.all(np.isfinite(g)) for g in gradients.values()):
            raise FloatingPointError("Обчислені градієнти містять NaN або нескінченність.")
        return loss, gradients

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0].reshape(-1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Поріг класифікації має бути строго між 0 та 1.")
        return (self.predict_proba(X) >= threshold).astype(int)

    def copy_parameters(self) -> ArrayDict:
        return {key: value.copy() for key, value in self.params.items()}

    def set_parameters(self, parameters: Mapping[str, np.ndarray]) -> None:
        if set(parameters) != set(self.PARAMETER_NAMES):
            raise ValueError("Набір параметрів не відповідає архітектурі мережі.")
        expected = {
            "W1": (self.input_dim, self.hidden_dim),
            "b1": (1, self.hidden_dim),
            "W2": (self.hidden_dim, 1),
            "b2": (1, 1),
        }
        candidate: ArrayDict = {}
        for key, shape in expected.items():
            value = np.asarray(parameters[key], dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"Параметр {key} має неправильну форму або некоректні значення.")
            candidate[key] = value.copy()
        self.params = candidate
