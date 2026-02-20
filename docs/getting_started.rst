.. _getting_started:

Getting Started
===============

Installation
------------

Nexus is deployed to PyPI. To install the production build, use pip:

.. code-block:: bash

   pip install nexus-quant

For users attempting to optimize extreme entropic manifolds (EVaR, EDaR, RLDaR) with precision, you can natively attach CVXPY and your preferred commercial solvers (e.g., MOSEK, GUROBI). Nexus will automatically route extreme calculations through these engines if detected.

.. code-block:: bash

   pip install nexus-quant cvxpy

Quickstart
----------

Nexus utilizes the Facade pattern for execution, eliminating disjointed mathematical scripts and routing all computation through the `NexusAnalyzer`.

.. code-block:: python

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

   print(report_df)
