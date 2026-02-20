"""
Nexus Enterprise Data Framework
===============================

Module: ``nexus.data.loader``
Description: Institutional data ingestion mechanisms.

Handles strict type checking, forward-filling, and geometric uncompounding
for live market histories.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Union, Tuple, Optional

class NexusDataLoader:
    """
    A unified, institutional-grade data ingestion facade.
    
    Seamlessly manages the extraction and transformation of messy, 
    incomplete empirical market histories into clean geometric 
    return arrays ready for the NexusAnalyzer.
    """

    @staticmethod
    def fetch(
        tickers: Union[str, List[str]], 
        start_date: str, 
        end_date: Optional[str] = None,
        interval: str = '1d',
        auto_clean: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetches historical price data from Yahoo Finance and converts it 
        into discrete, uncompounded returns compatible with Nexus manifolds.

        Parameters
        ----------
        tickers : str or list of str
            Asset tickers (e.g., ['AAPL', 'MSFT']).
        start_date : str
            Start date (YYYY-MM-DD).
        end_date : str, optional
            End date (YYYY-MM-DD). Defaults to None (Today).
        interval : str, default '1d'
            Data interval frequency.
        auto_clean : bool, default True
            If True, automatically forward-fills NaNs and drops unrecoverable rows.

        Returns
        -------
        asset_names : ndarray of str
            The ordered list of asset names.
        asset_returns : ndarray of floats, shape (T, N)
            The discrete, time-series matrix of asset returns. 
            Can be immediately pushed to `analyzer.calibrate()`.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        warnings.filterwarnings('ignore', category=FutureWarning)
        
        try:
            # Download Close Prices
            raw_data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                interval=interval,
                progress=False,
                ignore_tz=True
            )
            
            # Handle Single vs Multiple Tickering output shapes
            if 'Close' in raw_data.columns.levels[0] if isinstance(raw_data.columns, pd.MultiIndex) else False:
                prices = raw_data['Close']
            else:
                prices = raw_data.get('Close', raw_data)

            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])

            if auto_clean:
                prices = prices.ffill().dropna()

            if prices.empty:
                raise ValueError("YFinance returned an empty DataFrame.")

            # Calculate the discrete geometric uncompounded returns
            # R_t = (P_t - P_{t-1}) / P_{t-1}
            price_matrix = prices.values
            asset_returns = np.diff(price_matrix, axis=0) / price_matrix[:-1]

            return np.array(prices.columns), asset_returns

        except Exception as e:
            raise ConnectionError(f"NexusDataLoader failed to fetch or parse {tickers}: {str(e)}")
