"""Генерація та підготовка нелінійно роздільних наборів даних."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles


DATASET_NAMES = {
    "moons": "Два півмісяці",
    "circles": "Концентричні кола",
    "xor": "XOR-кластери",
}


@dataclass(frozen=True)
class DatasetSplit:
    """Підготовлені дані; y зберігається як одновимірний масив міток 0/1."""

    name: str
    title: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _make_xor(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Створює чотири гаусові кластери з XOR-розміткою."""
    if n_samples < 20:
        raise ValueError("Для набору XOR потрібно щонайменше 20 прикладів.")
    rng = np.random.default_rng(seed)
    labels = np.arange(n_samples) % 2
    # Балансуємо класи та розподіляємо кожен клас по двох протилежних кутах.
    centers = np.empty((n_samples, 2), dtype=float)
    for i, label in enumerate(labels):
        corner = int(rng.integers(0, 2))
        if label == 1:
            center = (-0.75, -0.75) if corner == 0 else (0.75, 0.75)
        else:
            center = (-0.75, 0.75) if corner == 0 else (0.75, -0.75)
        centers[i] = center
    X = centers + rng.normal(0.0, noise, size=(n_samples, 2))
    order = rng.permutation(n_samples)
    return X[order], labels[order].astype(int)


def create_dataset(
    name: str,
    *,
    n_samples: int = 600,
    test_size: float = 0.25,
    noise: float = 0.18,
    seed: int = 42,
) -> DatasetSplit:
    """Створює відтворювані дані, stratified split та масштабує лише за train."""
    if name not in DATASET_NAMES:
        raise ValueError(f"Невідомий набір даних: {name!r}. Доступні: {', '.join(DATASET_NAMES)}")
    if not isinstance(n_samples, int) or n_samples < 40:
        raise ValueError("n_samples має бути цілим числом не меншим за 40.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size має бути в межах від 0.1 до 0.4.")
    if not np.isfinite(noise) or noise < 0 or noise > 0.5:
        raise ValueError("noise має бути числом у межах [0, 0.5].")

    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif name == "circles":
        X, y = make_circles(
            n_samples=n_samples,
            factor=0.42,
            noise=min(noise, 0.25),
            random_state=seed,
        )
    else:
        X, y = _make_xor(n_samples=n_samples, noise=max(noise, 0.05), seed=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.astype(int),
        test_size=test_size,
        random_state=seed + 1,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return DatasetSplit(
        name=name,
        title=DATASET_NAMES[name],
        X_train=np.asarray(X_train, dtype=np.float64),
        y_train=np.asarray(y_train, dtype=np.int64),
        X_test=np.asarray(X_test, dtype=np.float64),
        y_test=np.asarray(y_test, dtype=np.int64),
    )


def available_dataset_names() -> list[str]:
    return list(DATASET_NAMES)
