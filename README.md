<p align="center">
  <img src="https://raw.githubusercontent.com/nexus-quant/nexus/main/docs/assets/nexus_logo.png" alt="Nexus Risk Engine" height="150">
</p>

# Nexus: The Enterprise Quantitative Risk Framework

Welcome to **Nexus**. Nexus is an institutional-grade library implementing 40+ advanced risk measurements, including classical metrics, downside asymmetry, tail exceedance, and convex entropic bounds (EVaR, EDaR, RLDaR). 

Inspired by `scikit-learn` and Google's core architectures, Nexus seamlessly abstracts disjointed mathematical scripts into a devastatingly powerful execution manifold: the **`NexusAnalyzer`**. Whether you are an indie quant building alpha models who has identified undervalued opportunities, or a Wall Street hedge fund requiring millisecond precision optimizations via MOSEK/CVXPY, Nexus provides the native mathematical infrastructure needed to evaluate and construct risk-efficient portfolios.

|               | **[Documentation](#) · [Tutorials](#) · [Release Notes](#)** |
|:--------------|:-------------------------------------------------------------|
| **Open Source**| [![License: MIT](https://img.shields.io/badge/license-MIT-yellowgreen)](#) [![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](#) |
| **Tutorials** | [![Binder](https://img.shields.io/badge/launch-binder-blue)](#) |
| **Community** | [![Discord](https://img.shields.io/badge/discord-chat-46BC99)](#) [![LinkedIn](https://img.shields.io/badge/LinkedIn-news-blue)](#) |
| **CI/CD**     | [![CI/CD Validation](https://img.shields.io/badge/build-passing-brightgreen?logo=github&logoColor=white)](#) [![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](#) |
| **Code**      | [![PyPI](https://img.shields.io/badge/pypi-v2.0.0-orange)](#) [![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](#) [![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](#) |
| **Downloads** | [![Downloads](https://img.shields.io/badge/downloads-85%2Fweek-brightgreen)](#) [![Downloads](https://img.shields.io/badge/downloads-340%2Fmonth-brightgreen)](#) [![Cumulative](https://img.shields.io/badge/cumulative_(pypi)-1.2k-blue)](#) |

<p align="center">
  <img src="https://raw.githubusercontent.com/Anagatam/Nexus/main/docs/assets/risk_manifold.png" alt="Nexus Risk Manifold" width="800">
</p>

---

## Table of contents
- [Why Nexus?](#why-nexus)
- [Getting started](#getting-started)
- [Features & Mathematical Supremacy](#features--mathematical-supremacy)
  - [Return Regimes & Asset Efficiency](#return-regimes--asset-efficiency)
  - [Multivariate Dynamics & Temporal Regimes](#multivariate-dynamics--temporal-regimes)
  - [Dispersion & Volatility](#dispersion--volatility)
  - [Downside Asymmetry](#downside-asymmetry)
  - [Tail Exceedance](#tail-exceedance)
  - [Convex Entropic Bounds](#convex-entropic-bounds)
- [Unparalleled Solver Routing](#unparalleled-solver-routing)
- [Project Principles](#project-principles-and-design-decisions)
- [Installation](#-installation)
- [Testing & Developer Setup](#testing--developer-setup)
- [License & Disclaimer](#license)

---

## Why Nexus?

While excellent open-source libraries like `PyPortfolioOpt` and `Riskfolio-Lib` exist, **Nexus** was explicitly engineered for absolute scale and mathematical extremity.

1. **Convex Entropic Supremacy**: Standard libraries rely on empirical Historical VaR or CVaR. Nexus natively implements **Entropic Value at Risk (EVaR)** and **Relativistic VaR (RLVaR)**, bounding tail risks using strict Chernoff inequalities that are completely invisible to standard historical sampling.
2. **Path-Dependent Drawdown Cones**: Nexus introduces **Entropic Drawdown at Risk (EDaR)**, a revolutionary metric mapping underwater geometric capital erosion onto exponential mathematically-bound cones, rather than just simple peak-to-trough calculations.
3. **Dynamic Solver Fallbacks**: Nexus detects institutional optimization licenses (MOSEK/GUROBI) and perfectly routes extreme thermodynamics through them via `cvxpy`. If licenses are missing, it flawlessly falls back to high-grade `scipy` optimizers without crashing your analytical pipeline.

---

## Getting started

Gone are the days of importing disjointed functions. Nexus abstracts the entire mathematical realm into a single `NexusAnalyzer` object. Here is an example demonstrating how easy it is to fetch real-life stock data and construct an exhaustive mathematical risk report matrix natively.

```python
import numpy as np
from nexus.data.loader import NexusDataLoader
from nexus.analytics.analyzer import NexusAnalyzer

# 1. Effortless Market Ingestion
loader = NexusDataLoader()
asset_names, historical_returns = loader.fetch(
    ['AAPL', 'MSFT', 'JPM'], 
    start_date='2020-01-01'
)

# Build an equal-weighted portfolio combination
weights = np.ones(len(asset_names)) / len(asset_names)
portfolio_returns = np.sum(historical_returns * weights, axis=1)

# 2. Institutional Calibration
analyzer = NexusAnalyzer()
analyzer.calibrate(portfolio_returns)

# 3. Exhaustive Mathematical Execution
report_df = analyzer.compute(alpha=0.05, annualization_factor=252)

# Specific Dictionary Retrieval
cvar = analyzer.fetch('Cond VaR (0.05)')

print(report_df)
```

### The Output
```text
                            Asset_0
Volatility (Ann)           0.230763
Mean Abs Dev (MAD)         2.388695
Gini Mean Diff             3.579301
Lower Part Moment (LPM1)   1.096828
Value at Risk (0.05)       0.019100
Cond VaR (0.05)            0.033262
Entropic VaR (0.05)        0.067949
Cond DaR (0.05)           -0.232770
Entropic DaR (0.05)        0.259398
Max Drawdown (MDD)        -0.344122
Ulcer Index                0.091181
Calmar Ratio               0.884100
...
```

---

## Features & Mathematical Supremacy

In this section, we detail Nexus' primary architectural pillars. More exhaustive equations can be found in our core modules.

### Return Regimes & Asset Efficiency
Institutional portfolio management relies on hierarchical clustering of temporal returns and risk-adjusted efficiency plotting.

<p align="center">
  <img src="https://raw.githubusercontent.com/Anagatam/Nexus/main/docs/assets/monthly_heatmap.png" alt="Monthly Return Heatmap" width="800">
</p>

- **Chronological Return Clustering**: QuantStats-style Y/M grids isolating momentum drifts, tax-loss harvesting impacts, and macro-regime seasonality across annual structures.

<p align="center">
  <img src="https://raw.githubusercontent.com/Anagatam/Nexus/main/docs/assets/risk_return_scatter.png" alt="Risk vs Return Efficiency" width="800">
</p>

- **Asset Efficiency Hierarchies**: Volatility vs. Return distributions mapping exactly which singular assets dominate the local efficient frontier.

---

### Multivariate Dynamics & Temporal Regimes
Understanding how risks evolve over time and across asset classes is paramount. Nexus natively maps high-dimensional data flows into temporal matrices, detecting structural regime shifts before they breach limits.

<p align="center">
  <img src="https://raw.githubusercontent.com/Anagatam/Nexus/main/docs/assets/rolling_risk.png" alt="Rolling Risk Regimes" width="800">
</p>

- **Rolling Structural Volatility**: Maps moving-window variance structures directly against overlapping 95% Historical VaR clusters, instantly revealing structural macro-regime changes.
- **Cross-Asset Covariance & Pearson Dependencies**: Instantly maps deep empirical correlation heatmaps to guarantee zero concentration overlaps across distinct asset silos (Equities, Bonds, Crypto, Commodities).

<p align="center">
  <img src="https://raw.githubusercontent.com/Anagatam/Nexus/main/docs/assets/correlation_heatmap.png" alt="Cross-Asset Correlation" width="600">
</p>

### Dispersion & Volatility
- **Standard Deviation & Variance**: The classical unbiased measures of historical risk.
- **Mean Absolute Deviation (MAD)**: A perfectly robust scale metric lacking the extreme parabolic sensitivity of squared variance.
- **Gini Mean Difference**: A powerful absolute deviation measurement utilized in modern asset allocation.
- **L-Scale**: The second L-Moment representing linear combinations of order statistics.

### Downside Asymmetry
- **Semi-Deviation**: A measure of risk that focuses purely on downside variation heavily penalized by investors.
- **Lower Partial Moments (LPM)**: Generalized objective functions for asymmetric downside measurements parameterized by target acceptable return (`MAR`).

<p align="center">
  <img src="https://raw.githubusercontent.com/Anagatam/Nexus/main/docs/assets/drawdown_topography.png" alt="Nexus Drawdown Topography" width="800">
</p>

### Tail Exceedance
- **Value at Risk (VaR)**: The industry-standard empirical percentile of the maximum loss over a targeted confidence interval $\alpha$.
- **Conditional VaR (CVaR/Expected Shortfall)**: The expected loss *given* that the VaR threshold has been breached. Structurally coherent.
- **Tail Gini**: A unique generalized formulation merging CVaR with Gini mean difference within extreme domains.

<p align="center">
  <img src="https://raw.githubusercontent.com/Anagatam/Nexus/main/docs/assets/tail_risk_metrics.png" alt="Extreme Tail Convexity Breakdown" width="800">
</p>

### Convex Entropic Bounds
- **Entropic Value at Risk (EVaR)**: The tightest coherent upper bound on VaR historically derived strictly from the Chernoff inequality. Extremely responsive to extreme market shocks.
- **Relativistic VaR (RLVaR)**: A massive 3D power-cone deformation scaling Entropic bounds to asymmetric generalized logarithmic divergences.
- **Entropic Drawdown at Risk (EDaR)**: A revolutionary path-dependent risk metric combining geometric Chernoff bounds with historical peak-to-trough waterfall drawdowns.

---

## Unparalleled Solver Routing

Nexus was built to scale from individual traders directly to high-frequency servers natively. 

Its generalized convex penalty equations computationally **detect commercial optimization licenses** (like **MOSEK** or **GUROBI** via **CVXPY**). 
- If found, it natively routes extreme computations (EVaR, EDaR, RLVaR) through absolute mathematical exponential cones, achieving millisecond precision over millions of market datapoints.
- If not found, it miraculously falls back to high-grade `scipy.optimize.minimize` open-source algorithmic optimization without crashing.

---

## Project principles and design decisions
- **Modularity**: It should be easy to swap out individual components of the analytical process with the user's proprietary improvements.
- **Mathematical Transparency**: All functions are internally documented with strict $\LaTeX$ formulations.
- **Object-Oriented Supremacy**: There is no point in portfolio optimization unless it can be practically applied to real asset matrices easily. The Facade pattern rules.
- **Robustness**: Extensively guarded against arrays of `NaN` fragments and disjointed dimensions.

---

## 🚀 Installation

### Using pip
The primary stable architecture.

```bash
pip install nexus-quant
```

For institutional scaling (which enforces CVXPY tensor integrations; optimally paired with a local MOSEK license instance):
```bash
pip install "nexus-quant[enterprise]"
```

### From source
Clone the repository, navigate to the folder, and install directly using pip:
```bash
git clone https://github.com/Anagatam/Nexus.git
cd Nexus
pip install -e .
```

---

## Testing & Developer Setup

Tests are written natively in `pytest` utilizing deterministic NumPy architectures to completely bypass `yfinance` REST rate limits.

Run the native `Makefile` to instantly configure the repository for contributing:
```bash
make install
make format
make test
```

---

## License

Nexus is distributed freely under the standard **MIT License**. Open-source rules quantitative finance. 

**Disclaimer:** Nothing about this project constitutes investment advice, and the author bears no responsibility for your subsequent investment decisions. Please rigorously validate all models statistically in out-of-sample data before committing live capital.
