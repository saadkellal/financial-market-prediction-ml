"""Small reproducible baseline for next-day market-direction classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "return_1d",
    "ma20_gap",
    "volatility_20",
    "momentum_5",
    "volume_ratio_20",
]
TARGET_COLUMN = "target_up_next_day"


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build past-only features and a next-day binary target.

    The input index must be date-like and the frame must contain `Close` and
    `Volume`. Each feature at time t uses observations available at or before t.
    """

    missing = {"Close", "Volume"}.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = prices.loc[:, ["Close", "Volume"]].copy().sort_index()
    close = pd.to_numeric(frame["Close"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")
    returns = close.pct_change()

    frame["return_1d"] = returns
    frame["ma20_gap"] = close.div(close.rolling(20).mean()).sub(1)
    frame["volatility_20"] = returns.rolling(20).std()
    frame["momentum_5"] = close.pct_change(5)
    frame["volume_ratio_20"] = volume.div(volume.rolling(20).mean())

    future_close = close.shift(-1)
    frame[TARGET_COLUMN] = (future_close > close).where(future_close.notna())
    frame = frame.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    return frame


def chronological_split(
    frame: pd.DataFrame, train_fraction: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return non-overlapping, order-preserving train and test partitions."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if len(frame) < 4:
        raise ValueError("At least four observations are required")

    split_index = int(len(frame) * train_fraction)
    split_index = min(max(split_index, 1), len(frame) - 1)
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def evaluate_models(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Compare a majority baseline with a scaled logistic-regression model."""

    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    x_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    models: dict[str, Any] = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
        ),
    }

    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "f1": float(f1_score(y_test, prediction, zero_division=0)),
        }
        if hasattr(model, "predict_proba") and y_test.nunique() == 2:
            probability = model.predict_proba(x_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, probability))
        results[name] = metrics
    return results


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download price history at runtime; no market dataset is redistributed."""

    import yfinance as yf

    prices = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if prices.empty:
        raise RuntimeError(f"No price data returned for {ticker}")
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    return prices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-01")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prices = download_prices(args.ticker, args.start, args.end)
    frame = build_features(prices)
    train, test = chronological_split(frame)
    results = evaluate_models(train, test)

    print(f"Observations: {len(frame)}")
    print(f"Train: {train.index.min()} -> {train.index.max()} ({len(train)})")
    print(f"Test: {test.index.min()} -> {test.index.max()} ({len(test)})")
    print(pd.DataFrame(results).T.to_string())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output)


if __name__ == "__main__":
    main()
