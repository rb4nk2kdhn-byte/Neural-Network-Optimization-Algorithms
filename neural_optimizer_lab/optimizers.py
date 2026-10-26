"""Власні реалізації SGD, Momentum, RMSProp та Adam."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np


class Optimizer(ABC):
    def __init__(self, learning_rate: float) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Швидкість навчання learning_rate має бути скінченним числом > 0.")
        self.learning_rate = float(learning_rate)
        self.steps = 0

    @abstractmethod
    def update(self, params: dict[str, np.ndarray], gradients: Mapping[str, np.ndarray]) -> None:
        """Оновлює параметри мережі на місці."""

    @staticmethod
    def _validate(params: dict[str, np.ndarray], gradients: Mapping[str, np.ndarray]) -> None:
        if params.keys() != gradients.keys():
            raise ValueError("Ключі параметрів і градієнтів мають збігатися.")
        for key, param in params.items():
            grad = np.asarray(gradients[key])
            if grad.shape != param.shape:
                raise ValueError(f"Градієнт {key} має форму {grad.shape}, а очікувалася {param.shape}.")
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError(f"Градієнт {key} містить NaN або нескінченність.")

    @staticmethod
    def _assert_finite(params: dict[str, np.ndarray]) -> None:
        if any(not np.all(np.isfinite(value)) for value in params.values()):
            raise FloatingPointError("Після оновлення параметри містять NaN або нескінченність.")


class SGD(Optimizer):
    """Звичайний стохастичний градієнтний спуск."""
    def update(self, params: dict[str, np.ndarray], gradients: Mapping[str, np.ndarray]) -> None:
        self._validate(params, gradients)
        for key in params:
            params[key] -= self.learning_rate * gradients[key]
        self.steps += 1
        self._assert_finite(params)


class Momentum(Optimizer):
    """SGD із експоненційно накопичуваною швидкістю градієнта."""
    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        super().__init__(learning_rate)
        if not np.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError("momentum має бути в межах [0, 1).")
        self.momentum = float(momentum)
        self.velocity: dict[str, np.ndarray] = {}

    def update(self, params: dict[str, np.ndarray], gradients: Mapping[str, np.ndarray]) -> None:
        self._validate(params, gradients)
        if not self.velocity:
            self.velocity = {key: np.zeros_like(value) for key, value in params.items()}
        self.steps += 1
        for key in params:
            self.velocity[key] = self.momentum * self.velocity[key] + gradients[key]
            params[key] -= self.learning_rate * self.velocity[key]
        self._assert_finite(params)


class RMSProp(Optimizer):
    """Адаптивний крок за експоненційною оцінкою квадрата градієнта."""
    def __init__(self, learning_rate: float, beta: float = 0.9, epsilon: float = 1e-8) -> None:
        super().__init__(learning_rate)
        if not np.isfinite(beta) or not 0.0 <= beta < 1.0:
            raise ValueError("beta RMSProp має бути в межах [0, 1).")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon має бути скінченним числом > 0.")
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.square_average: dict[str, np.ndarray] = {}

    def update(self, params: dict[str, np.ndarray], gradients: Mapping[str, np.ndarray]) -> None:
        self._validate(params, gradients)
        if not self.square_average:
            self.square_average = {key: np.zeros_like(value) for key, value in params.items()}
        self.steps += 1
        for key in params:
            self.square_average[key] = (
                self.beta * self.square_average[key]
                + (1.0 - self.beta) * np.square(gradients[key])
            )
            params[key] -= self.learning_rate * gradients[key] / (
                np.sqrt(self.square_average[key]) + self.epsilon
            )
        self._assert_finite(params)


class Adam(Optimizer):
    """Adam з поправкою зміщення для першого і другого моментів."""
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        if not np.isfinite(beta1) or not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 має бути в межах [0, 1).")
        if not np.isfinite(beta2) or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 має бути в межах [0, 1).")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon має бути скінченним числом > 0.")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.first_moment: dict[str, np.ndarray] = {}
        self.second_moment: dict[str, np.ndarray] = {}

    def update(self, params: dict[str, np.ndarray], gradients: Mapping[str, np.ndarray]) -> None:
        self._validate(params, gradients)
        if not self.first_moment:
            self.first_moment = {key: np.zeros_like(value) for key, value in params.items()}
            self.second_moment = {key: np.zeros_like(value) for key, value in params.items()}
        self.steps += 1
        for key in params:
            grad = gradients[key]
            self.first_moment[key] = self.beta1 * self.first_moment[key] + (1.0 - self.beta1) * grad
            self.second_moment[key] = self.beta2 * self.second_moment[key] + (1.0 - self.beta2) * np.square(grad)
            m_hat = self.first_moment[key] / (1.0 - self.beta1 ** self.steps)
            v_hat = self.second_moment[key] / (1.0 - self.beta2 ** self.steps)
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self._assert_finite(params)


OPTIMIZER_NAMES = ("SGD", "Momentum", "RMSProp", "Adam")


def create_optimizer(name: str, learning_rate: float) -> Optimizer:
    constructors = {"SGD": SGD, "Momentum": Momentum, "RMSProp": RMSProp, "Adam": Adam}
    if name not in constructors:
        raise ValueError(f"Невідомий оптимізатор {name!r}. Доступні: {', '.join(constructors)}")
    return constructors[name](learning_rate)
