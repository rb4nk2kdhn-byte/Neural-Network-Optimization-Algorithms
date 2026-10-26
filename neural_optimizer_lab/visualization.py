"""Побудова графіків і збереження їх у PNG."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import DatasetSplit
from neural_network import NeuralNetwork


# Графіки залишаються кольоровими на тлі темної теми інтерфейсу.
# Adam виділено рожевим, інші алгоритми мають яскраві контрастні кольори.
COLORS = {
    "SGD": "#56C8FF",       # яскраво-блакитний
    "Momentum": "#FFC857",  # бурштиновий
    "RMSProp": "#50E3C2",   # бірюзовий
    "Adam": "#FF5DB1",      # рожевий
}
LINESTYLES = {
    "SGD": "-",
    "Momentum": "--",
    "RMSProp": "-.",
    "Adam": ":",
}
DARK_BG = "#000000"
DARK_AXES = "#090909"
TEXT = "#FFFFFF"
GRID = "#454545"
CLASS_COLORS = {0: "#56C8FF", 1: "#FF5DB1"}


def _style_axis(ax) -> None:
    ax.set_facecolor(DARK_AXES)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, alpha=0.25, color=GRID)


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_dataset(data: DatasetSplit, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), facecolor=DARK_BG)
    for label, color, name in ((0, CLASS_COLORS[0], "Клас 0"), (1, CLASS_COLORS[1], "Клас 1")):
        mask = data.y_train == label
        ax.scatter(data.X_train[mask, 0], data.X_train[mask, 1], s=22, alpha=0.78, c=color, label=f"{name} · навчання", edgecolors="none")
        mask_test = data.y_test == label
        ax.scatter(data.X_test[mask_test, 0], data.X_test[mask_test, 1], s=35, alpha=0.9, c=color, marker="x", label=f"{name} · тест")
    ax.set_title(f"Геометрія даних: {data.title}", fontsize=13, pad=12)
    ax.set_xlabel("Ознака 1 (масштабована)")
    ax.set_ylabel("Ознака 2 (масштабована)")
    _style_axis(ax)
    legend = ax.legend(fontsize=8, framealpha=0.95, facecolor=DARK_AXES, edgecolor=GRID)
    if legend:
        for t in legend.get_texts():
            t.set_color(TEXT)
    _save_figure(fig, output_path)


def plot_losses(dataset_name: str, title: str, histories: list[dict[str, Any]], base_lr: float, output_path: Path) -> None:
    frame = pd.DataFrame(histories)
    if frame.empty:
        return
    frame = frame[(frame["dataset"] == dataset_name) & np.isclose(frame["learning_rate"].astype(float), base_lr)]
    if frame.empty:
        return
    grouped = frame.groupby(["optimizer", "epoch"])["train_loss"].agg(mean="mean", std="std").reset_index()
    fig, ax = plt.subplots(figsize=(8.3, 5.2), facecolor=DARK_BG)
    for optimizer, part in grouped.groupby("optimizer"):
        part = part.sort_values("epoch")
        means = part["mean"].to_numpy(dtype=float)
        stds = part["std"].fillna(0.0).to_numpy(dtype=float)
        epochs = part["epoch"].to_numpy(dtype=float)
        color = COLORS.get(optimizer)
        ax.plot(epochs, means, label=optimizer, color=color, linewidth=2.0, linestyle=LINESTYLES.get(optimizer, "-"))
        ax.fill_between(epochs, np.maximum(0.0, means - stds), means + stds, color=color, alpha=0.12, linewidth=0)
    ax.set_title(f"Зміна навчальної BCE · {title} · lr={base_lr:g}", fontsize=12, pad=12)
    ax.set_xlabel("Епоха")
    ax.set_ylabel("Binary Cross-Entropy (менше — краще)")
    _style_axis(ax)
    leg = ax.legend(framealpha=0.95, facecolor=DARK_AXES, edgecolor=GRID)
    if leg:
        for item in leg.get_texts():
            item.set_color(TEXT)
    _save_figure(fig, output_path)


def plot_accuracy(dataset_name: str, title: str, runs: list[dict[str, Any]], base_lr: float, output_path: Path) -> None:
    frame = pd.DataFrame(runs)
    if frame.empty:
        return
    frame = frame[(frame["dataset"] == dataset_name) & (frame["status"] == "completed") & np.isclose(frame["learning_rate"].astype(float), base_lr)]
    if frame.empty:
        return
    grouped = frame.groupby("optimizer")["test_accuracy"].agg(["mean", "std"])
    order = [name for name in ("SGD", "Momentum", "RMSProp", "Adam") if name in grouped.index]
    fig, ax = plt.subplots(figsize=(7.7, 5.0), facecolor=DARK_BG)
    means = grouped.loc[order, "mean"].to_numpy() * 100
    errors = grouped.loc[order, "std"].fillna(0).to_numpy() * 100
    ax.bar(order, means, yerr=errors, capsize=5, color=[COLORS.get(x) for x in order], alpha=0.95, edgecolor=GRID)
    ax.set_ylim(0, 105)
    ax.set_title(f"Точність на тестовій вибірці · {title}", fontsize=12, pad=12)
    ax.set_ylabel("Точність, % (середнє ± стандартне відхилення)")
    ax.set_xlabel(f"Оптимізатор · базова швидкість навчання {base_lr:g}")
    _style_axis(ax)
    _save_figure(fig, output_path)


def plot_convergence(
    dataset_name: str,
    title: str,
    runs: list[dict[str, Any]],
    base_lr: float,
    output_path: Path,
    target_loss: float | None = None,
) -> None:
    """Показує досягнення цільової BCE, не маскуючи невдалі досягнення порога.

    Якщо жоден запуск не досяг порога, кількість епох до збіжності невизначена.
    У такому випадку показуємо частку запусків, що досягли порога, а не порожню
    вісь з оманливим масштабом 0..1.
    """
    frame = pd.DataFrame(runs)
    if frame.empty:
        return
    frame = frame[(frame["dataset"] == dataset_name) & np.isclose(frame["learning_rate"].astype(float), base_lr)]
    if frame.empty:
        return

    # Нормалізуємо булеві значення для надійної роботи як зі словниками,
    # так і з даними, повторно прочитаними з CSV.
    def as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "так"}
        if pd.isna(value):
            return False
        return bool(value)

    frame = frame.copy()
    frame["_target_reached"] = frame["target_reached"].map(as_bool)
    order = [name for name in ("SGD", "Momentum", "RMSProp", "Adam") if name in set(frame["optimizer"])]
    if not order:
        return

    means: list[float] = []
    errors: list[float] = []
    success_rates: list[float] = []
    for name in order:
        part = frame[frame["optimizer"] == name]
        reached = part.loc[part["_target_reached"], "epochs_to_target"].dropna().astype(float)
        means.append(float(reached.mean()) if len(reached) else np.nan)
        errors.append(float(reached.std(ddof=1)) if len(reached) > 1 else 0.0)
        denominator = max(1, len(part))
        success_rates.append(100.0 * float(part["_target_reached"].sum()) / denominator)

    fig, ax = plt.subplots(figsize=(8.1, 5.0), facecolor=DARK_BG)
    positions = np.arange(len(order))
    valid = np.isfinite(means)
    target_text = f"BCE ≤ {target_loss:g}" if target_loss is not None else "цільову BCE"

    if not np.any(valid):
        # Порожній графік epoch-to-target був би математично некоректним:
        # жодного виміряного значення епох до цілі немає. Перемикаємося на
        # частку успішних запусків і явно повідомляємо причину.
        rates = np.asarray(success_rates, dtype=float)
        ax.bar(positions, rates, color=[COLORS.get(name) for name in order],
               alpha=0.88, edgecolor=GRID, width=0.62)
        ax.scatter(positions, rates, color=[COLORS.get(name) for name in order],
                   s=54, marker="D", edgecolors=TEXT, linewidths=0.6, zorder=4, clip_on=False)
        for idx, rate in enumerate(success_rates):
            ax.annotate(f"{rate:.0f}%", (idx, rate), xytext=(0, 9),
                        textcoords="offset points", ha="center", va="bottom",
                        color=TEXT, fontsize=9, fontweight="bold", clip_on=False)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Запусків, що досягли порога, %")
        ax.set_title(f"Частка запусків, що досягли {target_text} · {title}", fontsize=12, pad=12)
        ax.text(
            0.5, 0.78,
            f"Жоден запуск не досяг {target_text}.",
            transform=ax.transAxes, ha="center", va="center", color=TEXT, fontsize=10,
            bbox={"boxstyle": "round,pad=0.65", "facecolor": "#202020", "edgecolor": "#FF5DB1", "linewidth": 1.2},
        )
        ax.set_xlabel("Оптимізатор")
    else:
        valid_positions = positions[valid]
        valid_means = np.asarray(means, dtype=float)[valid]
        valid_errors = np.asarray(errors, dtype=float)[valid]
        ax.bar(
            valid_positions, valid_means, yerr=valid_errors, capsize=4,
            color=[COLORS.get(order[i]) for i in valid_positions], alpha=0.95,
            edgecolor=GRID, width=0.62, label="Середня кількість епох серед досягнень",
        )
        max_height = float(np.max(valid_means + valid_errors)) if len(valid_means) else 1.0
        ax.set_ylim(0, max(1.0, max_height * 1.32))
        for idx, rate in enumerate(success_rates):
            if np.isfinite(means[idx]):
                label_y = means[idx] + errors[idx]
                ax.annotate(f"досягли: {rate:.0f}%", (idx, label_y), xytext=(0, 7),
                            textcoords="offset points", ha="center", va="bottom",
                            color=TEXT, fontsize=8, clip_on=False)
            else:
                ax.annotate("не досягнуто\n0%", (idx, 0), xytext=(0, 9),
                            textcoords="offset points", ha="center", va="bottom",
                            color=TEXT, fontsize=8, clip_on=False)
        ax.set_ylabel("Кількість епох до цільової BCE")
        ax.set_title(f"Епохи до досягнення {target_text} · {title}", fontsize=12, pad=12)
        ax.set_xlabel("Оптимізатор · підпис показує частку запусків, що досягли порога")

    ax.set_xticks(positions, order)
    _style_axis(ax)
    _save_figure(fig, output_path)


def plot_learning_rate(dataset_name: str, title: str, runs: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(runs)
    if frame.empty:
        return
    frame = frame[(frame["dataset"] == dataset_name) & (frame["status"] == "completed")]
    if frame["learning_rate"].nunique() < 2:
        return
    grouped = frame.groupby(["optimizer", "learning_rate"], as_index=False).agg(
        test_accuracy=("test_accuracy", "mean"),
        final_test_loss=("final_test_loss", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8.1, 5.0), facecolor=DARK_BG)
    for optimizer, part in grouped.groupby("optimizer"):
        part = part.sort_values("learning_rate")
        ax.plot(part["learning_rate"], 100 * part["test_accuracy"], marker="o", linewidth=2,
                color=COLORS.get(optimizer), linestyle=LINESTYLES.get(optimizer, "-"), label=optimizer)
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.set_title(f"Вплив швидкості навчання на точність · {title}", fontsize=12, pad=12)
    ax.set_xlabel("Learning rate (логарифмічна шкала)")
    ax.set_ylabel("Середня тестова точність, %")
    _style_axis(ax)
    leg = ax.legend(framealpha=0.95, facecolor=DARK_AXES, edgecolor=GRID)
    if leg:
        for t in leg.get_texts():
            t.set_color(TEXT)
    _save_figure(fig, output_dir / f"learning_rate_accuracy_{dataset_name}.png")

    fig, ax = plt.subplots(figsize=(8.1, 5.0), facecolor=DARK_BG)
    for optimizer, part in grouped.groupby("optimizer"):
        part = part.sort_values("learning_rate")
        ax.plot(part["learning_rate"], part["final_test_loss"], marker="o", linewidth=2,
                color=COLORS.get(optimizer), linestyle=LINESTYLES.get(optimizer, "-"), label=optimizer)
    ax.set_xscale("log")
    ax.set_title(f"Вплив швидкості навчання на тестову BCE · {title}", fontsize=12, pad=12)
    ax.set_xlabel("Learning rate (логарифмічна шкала)")
    ax.set_ylabel("Кінцева тестова BCE")
    _style_axis(ax)
    leg = ax.legend(framealpha=0.95, facecolor=DARK_AXES, edgecolor=GRID)
    if leg:
        for t in leg.get_texts():
            t.set_color(TEXT)
    _save_figure(fig, output_dir / f"learning_rate_loss_{dataset_name}.png")


def plot_decision_boundary(data: DatasetSplit, candidate, output_path: Path) -> None:
    _, _, model, optimizer_name, learning_rate = candidate
    X_all = np.vstack([data.X_train, data.X_test])
    pad = 0.45
    x_min, x_max = X_all[:, 0].min() - pad, X_all[:, 0].max() + pad
    y_min, y_max = X_all[:, 1].min() - pad, X_all[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.predict_proba(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7.3, 5.4), facecolor=DARK_BG)
    ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 12), cmap="coolwarm", alpha=0.42, vmin=0, vmax=1)
    ax.contour(xx, yy, zz, levels=[0.5], colors=["#FFFFFF"], linewidths=1.8)
    for label, color, name in ((0, CLASS_COLORS[0], "Клас 0"), (1, CLASS_COLORS[1], "Клас 1")):
        train_mask = data.y_train == label
        test_mask = data.y_test == label
        ax.scatter(data.X_train[train_mask, 0], data.X_train[train_mask, 1], c=color, s=22, label=f"{name} · train", edgecolors="none")
        ax.scatter(data.X_test[test_mask, 0], data.X_test[test_mask, 1], c=color, s=38, marker="x", label=f"{name} · test")
    ax.set_title(f"Межа рішень · {data.title} · {optimizer_name}, lr={learning_rate:g}", fontsize=11, pad=12)
    ax.set_xlabel("Ознака 1 (масштабована)")
    ax.set_ylabel("Ознака 2 (масштабована)")
    _style_axis(ax)
    leg = ax.legend(fontsize=8, framealpha=0.95, facecolor=DARK_AXES, edgecolor=GRID)
    if leg:
        for t in leg.get_texts():
            t.set_color(TEXT)
    _save_figure(fig, output_path)


def create_all_plots(
    output_dir: str | Path,
    config: dict[str, Any],
    runs: list[dict[str, Any]],
    histories: list[dict[str, Any]],
    datasets: dict[str, DatasetSplit],
    model_registry: dict[str, list[tuple[float, dict[str, Any], NeuralNetwork, str, float]]] | None = None,
) -> None:
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for name, data in datasets.items():
        plot_dataset(data, plot_dir / f"dataset_{name}.png")
        plot_losses(name, data.title, histories, config["learning_rate"], plot_dir / f"loss_{name}.png")
        plot_accuracy(name, data.title, runs, config["learning_rate"], plot_dir / f"accuracy_{name}.png")
        plot_convergence(name, data.title, runs, config["learning_rate"], plot_dir / f"convergence_{name}.png", target_loss=float(config.get("target_loss", 0.05)))
        plot_learning_rate(name, data.title, runs, plot_dir)
        candidates = (model_registry or {}).get(name, [])
        base_candidates = [item for item in candidates if np.isclose(item[4], config["learning_rate"])]
        if base_candidates:
            # Вибір ілюстративної межі робиться за найменшою навчальною втратою, а не тестовою метрикою.
            best_candidate = min(base_candidates, key=lambda item: item[0])
            plot_decision_boundary(data, best_candidate, plot_dir / f"decision_boundary_{name}.png")
