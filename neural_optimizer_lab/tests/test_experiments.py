import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import create_dataset
from experiments import run_experiments, validate_config


class DatasetAndExperimentTests(unittest.TestCase):
    def test_dataset_is_reproducible_and_split_is_valid(self):
        first = create_dataset("circles", n_samples=100, seed=3)
        second = create_dataset("circles", n_samples=100, seed=3)
        np.testing.assert_array_equal(first.X_train, second.X_train)
        np.testing.assert_array_equal(first.y_test, second.y_test)
        self.assertEqual(first.X_train.shape[1], 2)
        self.assertEqual(len(first.X_train) + len(first.X_test), 100)

    def test_invalid_config_is_rejected(self):
        config = {
            "datasets": ["moons"], "optimizers": ["SGD"], "learning_rate": -0.1,
            "learning_rate_sweep": False, "max_epochs": 3, "batch_size": 16, "repeats": 1,
            "target_loss": 0.05, "patience": 2, "min_delta": 0.0001, "seed": 1, "n_samples": 100,
        }
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_complete_smoke_experiment_saves_tables_and_png(self):
        config = {
            "datasets": ["moons"],
            "optimizers": ["SGD", "Momentum", "RMSProp", "Adam"],
            "learning_rate": 0.01,
            "learning_rate_sweep": False,
            "max_epochs": 4,
            "batch_size": 24,
            "repeats": 1,
            "target_loss": 0.0001,
            "patience": 4,
            "min_delta": 0.000001,
            "seed": 23,
            "n_samples": 100,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiments(config, output_root=tmp)
            folder = Path(result["output_dir"])
            self.assertEqual(result["completed_runs"], 4)
            for filename in ("config.json", "runs.csv", "history.csv", "summary.csv", "details.json", "report_uk.txt"):
                self.assertTrue((folder / filename).exists(), filename)
            runs = pd.read_csv(folder / "runs.csv")
            self.assertEqual(set(runs["optimizer"]), {"SGD", "Momentum", "RMSProp", "Adam"})
            self.assertTrue((folder / "plots" / "dataset_moons.png").exists())
            self.assertTrue((folder / "plots" / "loss_moons.png").exists())
            self.assertTrue(any((folder / "plots").glob("*.png")))


    def test_training_results_are_reproducible_with_same_seed(self):
        config = {
            "datasets": ["xor"], "optimizers": ["Adam"], "learning_rate": 0.01,
            "learning_rate_sweep": False, "max_epochs": 6, "batch_size": 20, "repeats": 1,
            "target_loss": 0.00001, "patience": 6, "min_delta": 0.000001, "seed": 31, "n_samples": 100,
        }
        with tempfile.TemporaryDirectory() as tmp:
            first = run_experiments(config, output_root=Path(tmp) / "first")
            second = run_experiments(config, output_root=Path(tmp) / "second")
            one = pd.read_csv(Path(first["output_dir"]) / "runs.csv")
            two = pd.read_csv(Path(second["output_dir"]) / "runs.csv")
            for column in ("final_train_loss", "minimum_train_loss", "final_test_loss", "train_accuracy", "test_accuracy"):
                np.testing.assert_allclose(one[column], two[column], rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
