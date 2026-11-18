from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.market_direction import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    chronological_split,
    evaluate_models,
)


class MarketDirectionTests(unittest.TestCase):
    @staticmethod
    def sample_prices(rows: int = 90) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=rows, freq="D")
        close = 100 + np.linspace(0, 12, rows) + np.sin(np.arange(rows) / 2)
        volume = 1_000_000 + (np.arange(rows) % 9) * 25_000
        return pd.DataFrame({"Close": close, "Volume": volume}, index=index)

    def test_target_uses_the_following_close(self) -> None:
        prices = self.sample_prices()
        features = build_features(prices)
        for timestamp, row in features.iterrows():
            position = prices.index.get_loc(timestamp)
            expected = int(prices.iloc[position + 1]["Close"] > prices.iloc[position]["Close"])
            self.assertEqual(row[TARGET_COLUMN], expected)

    def test_split_preserves_time_order(self) -> None:
        features = build_features(self.sample_prices())
        train, test = chronological_split(features, train_fraction=0.7)
        self.assertLess(train.index.max(), test.index.min())
        self.assertEqual(len(train) + len(test), len(features))

    def test_models_return_finite_metrics(self) -> None:
        features = build_features(self.sample_prices(140))
        train, test = chronological_split(features)
        results = evaluate_models(train, test)
        self.assertEqual(set(results), {"majority_baseline", "logistic_regression"})
        for metrics in results.values():
            self.assertIn("accuracy", metrics)
            self.assertIn("f1", metrics)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            self.assertTrue(all(0 <= value <= 1 for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
