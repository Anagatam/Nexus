import numpy as np
import pandas as pd
from nexus.analytics.analyzer import NexusAnalyzer

def test_nexus_analyzer_offline_calibration():
    """Validates entire Nexus pipeline using deterministic numpy arrays."""
    # Create an identical deterministic state
    np.random.seed(42)
    # 3 assets, 252 days
    raw_array = np.random.normal(0.0005, 0.015, (252, 3))
    simulated_returns = pd.DataFrame(raw_array, columns=["SimA", "SimB", "SimC"])
    
    # Instantiate the Enterprise Engine
    analyzer = NexusAnalyzer()
    analyzer.calibrate(simulated_returns)
    
    # Guarantee that the risk manifold calculates without crashing
    report = analyzer.compute(alpha=0.05, annualization_factor=252)
    
    assert isinstance(report, pd.DataFrame), "Nexus did not yield a DataFrame array."
    assert "Volatility (Ann)" in report.index, "Volatility metric missing."
    assert report.shape == (19, 3), f"Matrix shape should be (19, 3), got {report.shape}."

def test_nexus_analyzer_single_asset():
    """Evaluates the structural limits computing a pure 1D array."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 100)
    
    analyzer = NexusAnalyzer()
    analyzer.calibrate(returns)
    report = analyzer.compute()
    
    assert report.shape[1] == 1, "Should gracefully map 1D sequence into 1 column facade"
    # Ensure expected shortfall exists
    assert "Cond VaR (0.05)" in report.index
