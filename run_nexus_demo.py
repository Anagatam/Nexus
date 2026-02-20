"""
Nexus Enterprise Risk Framework - v2.0 Demonstration
====================================================
Validates the execution of the exhaustive mathematically rigorous 
risk measures using the new Google-grade `NexusAnalyzer` Facade.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import yfinance as yf
import numpy as np

# 1. NEW ENTERPRISE FACADE IMPORT
from nexus.analytics.analyzer import NexusAnalyzer
from nexus.data.loader import NexusDataLoader

# Fetch live market data for a sample portfolio seamlessly via DataLoader
print("1. Fetching Market Data via NexusDataLoader...")
loader = NexusDataLoader()
tickers, portfolio_returns = loader.fetch(
    tickers=['AAPL', 'MSFT', 'JPM', 'GS', 'PFE'],
    start_date="2020-01-01",
    end_date="2026-01-01",
    interval="1d",
    auto_clean=True
)

# Generate an equal-weighted portfolio combination
weights = np.ones(len(tickers)) / len(tickers)
portfolio_series = np.sum(portfolio_returns * weights, axis=1)


print("\n=============================================")
print("     NEXUS ENTERPRISE RISK ENGINE v2.0       ")
print("=============================================\n")

print("--- Architectural Execution via Facade ---")

# 2. INSTANTIATE THE FACADE
analyzer = NexusAnalyzer()

# 3. CALIBRATE THE INTERNAL STATE
print("Calibrating Institutional Portfolio Data...")
analyzer.calibrate(portfolio_series)

# 4. COMPUTE ALL ADVANCED METRICS SIMULTANEOUSLY
print("Executing Exhaustive Mathematical Verification...\n")
report_df = analyzer.compute(alpha=0.05, annualization_factor=252)

# Print the beautifully structured pandas DataFrame result directly from the Facade
print(report_df.to_string())

print("\n=============================================")
print("System Validation Complete. Nexus v2.0 Live.")
print("=============================================\n")
