# Financial Market Direction Classification

Exploratory machine-learning analysis of Apple (AAPL) daily market data, with an honest comparison against a majority-class baseline.

> Educational research only. This repository is not financial advice and does not provide a validated trading strategy.

## Problem

The project asks whether price, volume, momentum and volatility features can classify the direction of AAPL's next trading day. The original notebook covers exploratory analysis, feature engineering, PCA, several classifiers and strategy comparisons. A small leakage-aware baseline module is included so the core workflow can be tested independently of notebook state.

## Repository structure

```text
.
├── notebooks/
│   └── aapl_market_direction_analysis.ipynb
├── src/
│   └── market_direction.py
├── tests/
│   └── test_market_direction.py
├── requirements.txt
└── README.md
```

## Original notebook workflow

- Downloads AAPL history with `yfinance`
- Analyzes price, volume, returns and outliers
- Builds moving-average, volatility, momentum and volume-ratio features
- Applies transformations and PCA
- Defines next-day direction as a binary classification target
- Uses a chronological train/test split
- Compares logistic regression, random forest, SVM, k-nearest neighbors and gradient boosting
- Adds later experiments with XGBoost, LightGBM, technical indicators and a Dash notebook view

The notebook source is based on the newest located variant (`Projet_Analyse_donnees (2).ipynb`, modified 2025-12-22). Duplicate exports remain in the original Downloads folder and were not copied. For publication, stored execution outputs and execution counts were cleared; all code and Markdown cell sources were preserved.

## Recorded results from the notebook

During the local source audit, the original notebook's stored output reported 1,217 modeled observations after feature preparation:

- Training: 839 observations through 2023-05-31
- Test: 378 observations from 2023-06-01 through 2024-11-29
- Majority baseline accuracy: **0.5503**
- Initial logistic-regression result: accuracy **0.5079**, F1 **0.5811**
- Highest initial test accuracy among the five listed models: **0.5238** (KNN)
- Later optimized logistic/SVM output: accuracy **0.5503**, recall **1.0000**, F1 **0.7099**
- Reported optimized strategy return: **+32.45%**, equal to buy-and-hold (**0.00% alpha**)

These values were transcribed from the audited local notebook before its outputs were cleared for publication; they are not newly reproduced test results. The optimized classifier's accuracy equals the majority baseline and its recall of 1.0 indicates an all/near-all positive prediction pattern. It should not be presented as evidence of predictive edge.

## Methodology audit

The notebook contains useful exploratory work, but the following limitations matter:

- PCA and its scaler are fit on the full analysis frame before the temporal train/test split, which leaks test-period distribution information.
- The initial `cross_val_score(..., cv=5)` uses ordinary folds rather than a time-series split.
- Model selection emphasizes F1 on an imbalanced target; the optimized result matches baseline accuracy.
- The strategy comparison reports no improvement over buy-and-hold.
- One chart simulates trade-return distributions with random numbers; it is illustrative, not measured performance.
- Multiple appended notebook stages reuse variable names and make clean, top-to-bottom execution difficult to verify.

The tested `src/market_direction.py` baseline avoids PCA and fits scaling inside a training-only pipeline. It is intentionally modest and does not claim outperformance.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

Run the reproducible baseline. Price data is downloaded at runtime and is not redistributed:

```powershell
python -m src.market_direction --ticker AAPL --start 2020-01-01 --end 2024-12-01
```

Run the network-free unit tests:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Open the output-cleared exploratory notebook only after installing the full requirements:

```powershell
jupyter lab notebooks/aapl_market_direction_analysis.ipynb
```

## Data and licensing

No market dataset is committed. The script uses `yfinance` at runtime; users are responsible for the data provider's terms and for any downstream use. Generated CSV/XLS files and trained models are ignored by default.

## Technical contribution

This portfolio publication preserves the financial-analysis notebook's source cells, documents results observed during the local audit, identifies evaluation risks, and adds a small testable implementation of past-only feature construction, chronological splitting and baseline comparison.

## Next improvements

- Fit every transformation, including PCA, on training folds only.
- Use walk-forward or nested time-series validation for model selection.
- Compare probability calibration and transaction-cost-aware strategies.
- Freeze a clean notebook run and export a compact metrics table.
- Test multiple assets and regimes before making any generalization claim.

## Suggested topics

`python` · `pandas` · `scikit-learn` · `time-series` · `classification` · `feature-engineering` · `model-evaluation` · `financial-data`

## License

Original source code and original project documentation are available under
the [MIT License](LICENSE). The license does not cover third-party market data,
API responses, datasets, trademarks or other third-party material; see
[LICENSE_SCOPE.md](LICENSE_SCOPE.md). No market dataset is distributed here.
