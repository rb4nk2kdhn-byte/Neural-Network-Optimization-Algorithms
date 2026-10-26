"""Керування контрольованими експериментами та збереження результатів."""
from __future__ import annotations

import csv
import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from datasets import DATASET_NAMES, create_dataset
from metrics import classification_metrics
from neural_network import NeuralNetwork
from optimizers import OPTIMIZER_NAMES, create_optimizer

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class TrainingOutcome:
    record: dict[str, Any]
    history: list[dict[str, Any]]
    model: NeuralNetwork | None


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Перевіряє конфігурацію і повертає нормалізовану копію."""
    required = {
        "datasets", "optimizers", "learning_rate", "learning_rate_sweep",
        "max_epochs", "batch_size", "repeats", "target_loss", "patience",
        "min_delta", "seed", "n_samples",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Не задано параметри: {', '.join(sorted(missing))}.")
    normalized = dict(config)
    if not normalized["datasets"] or any(name not in DATASET_NAMES for name in normalized["datasets"]):
        raise ValueError("Виберіть принаймні один відомий набір даних.")
    if not normalized["optimizers"] or any(name not in OPTIMIZER_NAMES for name in normalized["optimizers"]):
        raise ValueError("Виберіть принаймні один відомий оптимізатор.")
    normalized["datasets"] = list(dict.fromkeys(normalized["datasets"]))
    normalized["optimizers"] = list(dict.fromkeys(normalized["optimizers"]))

    lr = float(normalized["learning_rate"])
    if not math.isfinite(lr) or not 1e-6 <= lr <= 1.0:
        raise ValueError("Швидкість навчання має бути в межах від 0.000001 до 1.0.")
    normalized["learning_rate"] = lr

    for key, low, high in (("max_epochs", 1, 5000), ("batch_size", 1, 10000), ("repeats", 1, 30),
                           ("patience", 1, 1000), ("n_samples", 40, 100000)):
        value = normalized[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{key} має бути цілим числом у межах [{low}, {high}].")
        normalized[key] = int(value)
    target_loss = float(normalized["target_loss"])
    min_delta = float(normalized["min_delta"])
    if not math.isfinite(target_loss) or not 0 < target_loss < 1:
        raise ValueError("target_loss має бути числом строго між 0 та 1.")
    if not math.isfinite(min_delta) or min_delta < 0 or min_delta >= 1:
        raise ValueError("min_delta має бути в межах [0, 1).")
    normalized["target_loss"] = target_loss
    normalized["min_delta"] = min_delta
    seed = normalized["seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) <= 2**32 - 1:
        raise ValueError("seed має бути цілим числом від 0 до 4294967295.")
    normalized["seed"] = int(seed)
    normalized["learning_rate_sweep"] = bool(normalized["learning_rate_sweep"])
    return normalized


def _train_one(
    *,
    data,
    optimizer_name: str,
    learning_rate: float,
    repeat: int,
    model_seed: int,
    config: dict[str, Any],
    cancel_event: threading.Event | None = None,
) -> TrainingOutcome:
    model = NeuralNetwork(input_dim=data.X_train.shape[1], hidden_dim=16, seed=model_seed)
    optimizer = create_optimizer(optimizer_name, learning_rate)
    # Окремий генератор із тим самим seed у парних запусках дає однаковий порядок мініпакетів.
    shuffle_rng = np.random.default_rng(model_seed + 100_003)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch: int | None = None
    patience_used = 0
    target_epoch: int | None = None
    stopping_epoch: int | None = None
    stop_reason = "max_epochs"
    status = "completed"
    error_message = ""
    last_stable_params = model.copy_parameters()
    start = time.perf_counter()

    try:
        n_train = len(data.X_train)
        for epoch in range(1, config["max_epochs"] + 1):
            if cancel_event is not None and cancel_event.is_set():
                status = "cancelled"
                stop_reason = "cancelled"
                break
            order = shuffle_rng.permutation(n_train)
            for start_index in range(0, n_train, config["batch_size"]):
                if cancel_event is not None and cancel_event.is_set():
                    status = "cancelled"
                    stop_reason = "cancelled"
                    break
                indices = order[start_index:start_index + config["batch_size"]]
                _, gradients = model.loss_and_gradients(data.X_train[indices], data.y_train[indices])
                optimizer.update(model.params, gradients)
            if status == "cancelled":
                break

            train_loss = model.loss(data.X_train, data.y_train)
            test_loss = model.loss(data.X_test, data.y_test)
            if not np.isfinite(train_loss) or not np.isfinite(test_loss):
                raise FloatingPointError("Функція втрат стала NaN або нескінченністю.")
            train_accuracy = float(np.mean(model.predict(data.X_train) == data.y_train))
            test_accuracy = float(np.mean(model.predict(data.X_test) == data.y_test))
            history.append({
                "dataset": data.name,
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "repeat": repeat,
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            })
            last_stable_params = model.copy_parameters()

            if train_loss <= config["target_loss"]:
                target_epoch = epoch
                stopping_epoch = epoch
                stop_reason = "target_loss"
                break
            if train_loss < best_loss - config["min_delta"]:
                best_loss = train_loss
                best_epoch = epoch
                patience_used = 0
            else:
                patience_used += 1
            if patience_used >= config["patience"]:
                stopping_epoch = epoch
                stop_reason = "plateau_patience"
                break
        else:
            stopping_epoch = config["max_epochs"]

    except FloatingPointError as exc:
        status = "diverged"
        stop_reason = "numerical_error"
        error_message = str(exc)
        # Не записуємо метрики для частково оновленої епохи: повертаємо останні стабільні ваги.
        model.set_parameters(last_stable_params)

    elapsed = time.perf_counter() - start
    valid_model = status == "completed" and bool(history)
    if status == "cancelled":
        valid_model = bool(history) and all(np.all(np.isfinite(v)) for v in model.params.values())
    if history and all(np.all(np.isfinite(v)) for v in model.params.values()):
        final_train_loss = float(history[-1]["train_loss"])
        final_test_loss = float(history[-1]["test_loss"])
        min_train_loss = float(min(row["train_loss"] for row in history))
        train_metrics = classification_metrics(data.y_train, model.predict(data.X_train))
        test_metrics = classification_metrics(data.y_test, model.predict(data.X_test))
        # Результати частково перерваного запуску корисні, але явно позначаються як часткові.
    else:
        final_train_loss = final_test_loss = min_train_loss = float("nan")
        train_metrics = {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "tn": 0, "fp": 0, "fn": 0, "tp": 0}
        test_metrics = dict(train_metrics)
        valid_model = False

    record: dict[str, Any] = {
        "dataset": data.name,
        "dataset_title": data.title,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "repeat": repeat,
        "seed": model_seed,
        "status": status,
        "stopping_reason": stop_reason,
        "target_reached": target_epoch is not None,
        "epochs_to_target": target_epoch if target_epoch is not None else float("nan"),
        "epochs_to_stopping_rule": stopping_epoch if stopping_epoch is not None else (len(history) if history else 0),
        "epochs_run": len(history),
        "final_train_loss": final_train_loss,
        "minimum_train_loss": min_train_loss,
        "final_test_loss": final_test_loss,
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tp": test_metrics["tp"],
        "elapsed_seconds": elapsed,
        "error_message": error_message,
    }
    return TrainingOutcome(record=record, history=history, model=model if valid_model else None)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.8g")


def _build_summary(runs: list[dict[str, Any]]) -> pd.DataFrame:
    if not runs:
        return pd.DataFrame()
    frame = pd.DataFrame(runs)
    usable = frame[frame["status"] == "completed"].copy()
    if usable.empty:
        return pd.DataFrame()
    numeric = [
        "final_train_loss", "minimum_train_loss", "final_test_loss", "train_accuracy", "test_accuracy",
        "test_precision", "test_recall", "test_f1", "epochs_to_target", "epochs_to_stopping_rule", "elapsed_seconds",
    ]
    grouped = usable.groupby(["dataset", "optimizer", "learning_rate"], dropna=False)[numeric].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def _write_report(output_dir: Path, config: dict[str, Any], runs: list[dict[str, Any]], summary: pd.DataFrame) -> None:
    path = output_dir / "report_uk.txt"
    completed = [row for row in runs if row["status"] == "completed"]
    failed = [row for row in runs if row["status"] != "completed"]
    lines = [
        "ЗВІТ ЕКСПЕРИМЕНТАЛЬНОГО ДОСЛІДЖЕННЯ",
        "=" * 44,
        f"Час створення: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Набори даних: {', '.join(config['datasets'])}",
        f"Оптимізатори: {', '.join(config['optimizers'])}",
        f"Базова швидкість навчання: {config['learning_rate']}",
        f"Дослідження learning rate: {'так' if config['learning_rate_sweep'] else 'ні'}",
        f"Максимум епох: {config['max_epochs']}; batch size: {config['batch_size']}; повторів: {config['repeats']}",
        f"Цільова BCE: {config['target_loss']}; терпимість плато: {config['patience']} епох; min_delta: {config['min_delta']}",
        "",
        f"Запусків загалом: {len(runs)}; завершено нормально: {len(completed)}; перервано/з числовою помилкою: {len(failed)}.",
        "",
        "Методика та застереження:",
        "- Навчання використовує лише навчальну вибірку; тестова використовується для оцінювання.",
        "- epochs_to_target заповнюється лише тоді, коли train BCE досягла заданого порога.",
        "- Зупинка через plateau_patience не означає досягнення цільової BCE; ці випадки відрізняються у таблиці.",
        "- Середні значення та стандартні відхилення розраховуються за успішно завершеними повторами для кожної комбінації.",
        "- Час залежить від CPU, версії Python, бібліотек і фонового навантаження системи.",
        "- Невдалі запуски не вилучаються з runs.csv, але не включаються в агреговані середні.",
        "",
        "Зведена таблиця також збережена у summary.csv; повні дані — у runs.csv і history.csv.",
    ]
    if not summary.empty:
        lines.extend(["", "Короткий огляд найкращої середньої тестової BCE в межах кожного набору:"])
        for dataset in summary["dataset"].drop_duplicates().tolist():
            subset = summary[summary["dataset"] == dataset].dropna(subset=["final_test_loss_mean"])
            if not subset.empty:
                row = subset.loc[subset["final_test_loss_mean"].idxmin()]
                lines.append(
                    f"- {dataset}: {row['optimizer']} при lr={row['learning_rate']:.6g}; "
                    f"середня test BCE={row['final_test_loss_mean']:.6f} "
                    f"(std={row['final_test_loss_std'] if pd.notna(row['final_test_loss_std']) else float('nan'):.6f})."
                )
    lines.extend(["", "Цей автоматичний опис узагальнює лише фактично записані запуски та не є доказом універсальної переваги оптимізатора."])
    path.write_text("\n".join(lines), encoding="utf-8")


def _persist(output_dir: Path, config: dict[str, Any], runs: list[dict[str, Any]], histories: list[dict[str, Any]]) -> pd.DataFrame:
    _write_csv(output_dir / "runs.csv", runs)
    _write_csv(output_dir / "history.csv", histories)
    summary = _build_summary(runs)
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    details = {
        "configuration": config,
        "runs": runs,
        "history": histories,
        "summary": summary.to_dict(orient="records") if not summary.empty else [],
    }
    (output_dir / "details.json").write_text(
        json.dumps(_json_safe(details), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_report(output_dir, config, runs, summary)
    return summary


def run_experiments(
    config: dict[str, Any],
    *,
    output_root: str | Path = "results",
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Запускає серію контрольованих експериментів і постійно зберігає прогрес."""
    cfg = validate_config(config)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir = output_root / run_stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # Три рівні швидкості дають порівняння на порядок нижче/на базовому/на порядок вище.
    learning_rates = [cfg["learning_rate"]]
    if cfg["learning_rate_sweep"]:
        learning_rates = sorted({max(1e-6, cfg["learning_rate"] / 10), cfg["learning_rate"], min(1.0, cfg["learning_rate"] * 10)})
    planned = len(cfg["datasets"]) * len(cfg["optimizers"]) * len(learning_rates) * cfg["repeats"]
    completed_count = 0
    runs: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    dataset_cache = {}
    for dataset_index, dataset_name in enumerate(cfg["datasets"]):
        dataset_cache[dataset_name] = create_dataset(
            dataset_name, n_samples=cfg["n_samples"], seed=cfg["seed"] + dataset_index * 1009
        )

    summary = pd.DataFrame()
    model_registry: dict[str, list[tuple[float, dict[str, Any], NeuralNetwork, str, float]]] = {name: [] for name in cfg["datasets"]}
    # Вкладені цикли забезпечують однакову вибірку, архітектуру та ініціалізацію
    # для порівнюваних алгоритмів в межах набору/повтору.
    try:
        for dataset_name in cfg["datasets"]:
            if cancel_event is not None and cancel_event.is_set():
                break
            data = dataset_cache[dataset_name]
            for learning_rate in learning_rates:
                for repeat in range(cfg["repeats"]):
                    model_seed = cfg["seed"] + 17_171 * (cfg["datasets"].index(dataset_name) + 1) + repeat * 97
                    for optimizer_name in cfg["optimizers"]:
                        if cancel_event is not None and cancel_event.is_set():
                            break
                        outcome = _train_one(
                            data=data,
                            optimizer_name=optimizer_name,
                            learning_rate=learning_rate,
                            repeat=repeat + 1,
                            model_seed=model_seed,
                            config=cfg,
                            cancel_event=cancel_event,
                        )
                        runs.append(outcome.record)
                        histories.extend(outcome.history)
                        if outcome.model is not None:
                            model_registry[data.name].append((
                                outcome.record["minimum_train_loss"], outcome.model.copy_parameters(),
                                outcome.model, optimizer_name, learning_rate
                            ))
                        completed_count += 1
                        # Кожен завершений запуск одразу потрапляє на диск.
                        summary = _persist(output_dir, cfg, runs, histories)
                        if progress_callback:
                            msg = (f"{data.title} · {optimizer_name} · lr={learning_rate:g} · "
                                   f"повтор {repeat + 1}/{cfg['repeats']} · {outcome.record['status']}")
                            progress_callback(completed_count, planned, msg)
                    if cancel_event is not None and cancel_event.is_set():
                        break
                if cancel_event is not None and cancel_event.is_set():
                    break
            if cancel_event is not None and cancel_event.is_set():
                break
    finally:
        summary = _persist(output_dir, cfg, runs, histories)

    # Візуалізація імпортується тут, щоб обчислювальне ядро можна було використовувати окремо.
    from visualization import create_all_plots
    create_all_plots(output_dir, cfg, runs, histories, dataset_cache, model_registry)
    _write_report(output_dir, cfg, runs, summary)
    return {
        "output_dir": str(output_dir.resolve()),
        "runs": runs,
        "histories": histories,
        "summary": summary,
        "planned_runs": planned,
        "completed_runs": completed_count,
        "cancelled": cancel_event.is_set() if cancel_event is not None else False,
    }
