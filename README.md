<p align="center">
  <img src="https://raw.githubusercontent.com/nexus-quant/nexus/main/docs/assets/nexus_logo.png" alt="Nexus Risk Engine" height="150">
</p>

# Nexus: The Enterprise Quantitative Risk Framework

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white" alt="Python Versions">
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey" alt="Platform">
  <a href="https://pypi.org/project/nexus-quant/"><img src="https://img.shields.io/badge/pypi-v2.0.0-orange" alt="PyPI"></a>
  <a href="https://github.com/nexus-quant/nexus/actions"><img src="https://img.shields.io/badge/Upload%20Python%20Package-passing-brightgreen?logo=github&logoColor=white" alt="Upload Python Package"></a>
  <br>
  <a href="https://github.com/nexus-quant/nexus/actions"><img src="https://img.shields.io/badge/Test-no%20status-lightgrey?logo=github&logoColor=white" alt="Test"></a>
  <img src="https://img.shields.io/badge/docs-passing-brightgreen" alt="Docs">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-yellowgreen" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/chat-on%20gitter-46BC99" alt="Chat">
</p>

## Overview

**Nexus** is a mathematically exhaustive, institutional-grade quantitative risk architecture. Built on the elegance of Google's core paradigms and the `scikit-learn` Facade pattern, Nexus replaces convoluted functional scripts with a single, devastatingly powerful execution manifold: the `NexusAnalyzer`.

Nexus calculates **18+ high-level risk measurements** concurrently, featuring everything from fundamental statistical moments to extreme dual-tracked 3D-Power Cone deformations (Relativistic VaR).

### Unparalleled Solver Routing
Nexus was built to scale from indie quants directly to Wall Street hedge funds. Its generalized convex penalty equations natively and opportunistically detect commercial optimization licenses (like **MOSEK** via **CVXPY**). If found, it routes extreme metrics (EVaR, EDaR, RLVaR) through absolute mathematical exponential cones. If not found, it miraculously falls back to high-grade `scipy` open-source algorithms.

---

## Installation

Install Nexus and its core dependencies via `pip`:

```bash
pip install nexus-quant
```

For institutional scaling (requires a local MOSEK or GUROBI license):
```bash
pip install "nexus-quant[enterprise]"
```

---

## ⚡ The Nexus Facade: 3 Lines to Institutional Analysis

Gone are the days of importing disjointed functions. Nexus abstracts the entire mathematical realm into a single `NexusAnalyzer` object.

```python
import numpy as np
from nexus.data.loader import NexusDataLoader
from nexus.analytics.analyzer import NexusAnalyzer

# 1. Effortless Market Ingestion
loader = NexusDataLoader()
asset_names, historical_returns = loader.fetch(['AAPL', 'MSFT', 'JPM'], start_date='2020-01-01')

# Build an equal-weighted portfolio combination
weights = np.ones(len(asset_names)) / len(asset_names)
portfolio_returns = np.sum(historical_returns * weights, axis=1)

# 2. Institutional Calibration
analyzer = NexusAnalyzer()
analyzer.calibrate(portfolio_returns)

# 3. Exhaustive Mathematical Execution
report_df = analyzer.compute(alpha=0.05, annualization_factor=252)

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
...
```

---

## 📐 Mathematical Supremacy

Nexus calculates risk across four architectural pillars:
1. **Dispersion & Volatility:** Standard Deviation, Mean Absolute Deviation, Gini Mean Difference.
2. **Downside Asymmetry:** Semi-Variance, Lower Partial Moments (LPM), Target Semi-Deviation.
3. **Tail Exceedance:** Empirical VaR, Conditional VaR (Expected Shortfall), Tail Gini, Max Loss.
4. **Convex Entropic Bounds:** Entropic VaR (EVaR), Relativistic VaR (RLVaR), Entropic Drawdown at Risk (EDaR).

## Developer Setup
Run the `Makefile` to instantly configure the repository for contributing:
```bash
make install
make format
make test
```

## License
Nexus is distributed freely under the standard **MIT License**. Open-source rules quantitative finance.
