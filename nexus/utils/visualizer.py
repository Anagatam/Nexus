import os
import calendar
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

class NexusVisualizer:
    """Institutional Visualization Engine for Nexus Risk Manifolds."""

    @staticmethod
    def render_distribution_manifold(
        returns: np.ndarray, 
        asset_name: str = "Portfolio",
        output_path: str = "docs/assets/risk_manifold.png"
    ):
        """
        Renders a Bloomberg/Aladdin style dark-themed distribution 
        curve comparing empirical returns to a theoretical Normal curve.
        """
        # Set Institutional Dark Theme
        sns.set_theme(style="darkgrid")
        plt.style.use("dark_background")
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        fig.patch.set_facecolor('#0d1117') # GitHub Dark Mode Background
        ax.set_facecolor('#0d1117')

        # Flatten the returns array if necessary
        clean_returns = returns.flatten()
        mu, std = np.mean(clean_returns), np.std(clean_returns)

        # 1. Plot Empirical KDE Distribution (Teal)
        sns.kdeplot(
            clean_returns, 
            color="#00f2fe", 
            fill=True, 
            alpha=0.4, 
            linewidth=2, 
            label=f"{asset_name} Empirical KDE", 
            ax=ax
        )

        # 2. Plot Theoretical Normal Distribution (Pink/Red)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2, color="#fe0060", linestyle="--", label="Theoretical Normal (Parametric)")

        # 3. Mark the Mean and extreme tails (95% VaR)
        var_95 = np.percentile(clean_returns, 5)
        ax.axvline(var_95, color="#ffb703", linestyle=":", linewidth=2, label=f"95% Value-at-Risk ({var_95:.2%})")
        ax.axvline(mu, color="#ffffff", linestyle="-", linewidth=1, alpha=0.5)

        # 4. Institutional Formatting
        ax.set_title(f"Nexus Exceedance Manifold: {asset_name}", fontsize=16, fontweight='bold', color='white', pad=15)
        ax.set_xlabel("Geometric Returns", fontsize=12, color='white')
        ax.set_ylabel("Probability Density", fontsize=12, color='white')
        
        # Grid and borders
        ax.grid(color='#30363d', linestyle='-', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

        plt.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
        
        plt.tight_layout()
        
        # Ensure directory exists and save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print(f"Institutional graph saved to: {output_path}")
    @staticmethod
    def render_drawdown_underwater(
        returns: np.ndarray, 
        asset_name: str = "Portfolio",
        output_path: str = "docs/assets/drawdown_topography.png"
    ):
        """
        Renders a Bloomberg-style cumulative return line paired with an 
        underwater drawdown waterfall filled area.
        """
        sns.set_theme(style="darkgrid")
        plt.style.use("dark_background")
        
        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
        fig.patch.set_facecolor('#0d1117')
        ax1.set_facecolor('#0d1117')

        # Calculate geometric compounding and drawdowns
        clean_returns = returns.flatten()
        cum_returns = np.cumprod(1 + clean_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max

        # Plot Cumulative Returns
        ax1.plot(cum_returns, color="#00f2fe", linewidth=2, label=f"Cumulative {asset_name}")
        ax1.set_ylabel("Wealth Index", color="#00f2fe", fontsize=12)
        ax1.tick_params(axis='y', labelcolor="#00f2fe")

        # Plot Drawdown Underwater
        ax2 = ax1.twinx()
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color="#fe0060", alpha=0.3, label="Drawdown Waterfall")
        ax2.plot(drawdown, color="#fe0060", linewidth=1.5)
        ax2.set_ylabel("Drawdown (%)", color="#fe0060", fontsize=12)
        ax2.tick_params(axis='y', labelcolor="#fe0060")

        # Formatting
        plt.title(f"Drawdown Topography: {asset_name}", fontsize=16, fontweight='bold', color='white', pad=15)
        ax1.grid(color='#30363d', linestyle='-', linewidth=0.5)
        
        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print(f"Institutional drawdown graph saved to: {output_path}")
        plt.close()

    @staticmethod
    def render_tail_risk_comparison(
        metrics_dict: dict,
        output_path: str = "docs/assets/tail_risk_metrics.png"
    ):
        """
        Renders a bar chart comparing Empirical VaR vs CVaR vs Convex EVaR.
        """
        sns.set_theme(style="darkgrid")
        plt.style.use("dark_background")
        
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')

        labels = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        # Color gradient for increasing extremity
        colors = ["#ffb703", "#fb8500", "#fe0060"]

        bars = ax.bar(labels, values, color=colors, alpha=0.9, width=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', color='white', fontweight='bold')

        ax.set_title("Extreme Tail Convexity Breakdown (95%)", fontsize=16, fontweight='bold', color='white', pad=15)
        ax.set_ylabel("Maximum Capital Loss Exposure", fontsize=12, color='white')
        ax.grid(color='#30363d', linestyle='-', linewidth=0.5, axis='y')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print(f"Tail risk comparison graph saved to: {output_path}")
        plt.close()


    @staticmethod
    def render_correlation_heatmap(
        returns_df: pd.DataFrame, 
        output_path: str = "docs/assets/correlation_heatmap.png"
    ):
        """
        Renders an institutional cross-asset correlation matrix heatmap.
        """
        sns.set_theme(style="darkgrid")
        plt.style.use("dark_background")
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        
        # Calculate Pearson Correlation
        corr = returns_df.corr()
        
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, cmap="mako", annot=True, fmt=".2f", 
                    linewidths=0.5, linecolor='#30363d', cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title("Cross-Asset Correlation Matrix", fontsize=16, fontweight='bold', color='white', pad=15)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print(f"Correlation Heatmap saved to: {output_path}")
        plt.close()

    @staticmethod
    def render_rolling_risk(
        returns: pd.Series, 
        window: int = 63,
        asset_name: str = "Portfolio",
        output_path: str = "docs/assets/rolling_risk.png"
    ):
        """
        Renders rolling risk regimes contrasting Volatility against Empirical VaR over time.
        """
        sns.set_theme(style="darkgrid")
        plt.style.use("dark_background")
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')

        # Calculate Rolling Annualized Volatility (assuming daily)
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        ax.plot(rolling_vol.index, rolling_vol, color="#00f2fe", linewidth=2, label=f"{window}-Day Volatility (Ann.)")
        ax.fill_between(rolling_vol.index, rolling_vol, 0, color="#00f2fe", alpha=0.1)
        
        # Calculate Rolling Empirical VaR directly
        rolling_var = returns.rolling(window=window).quantile(0.05) * -np.sqrt(252)
        ax.plot(rolling_var.index, rolling_var, color="#fe0060", linewidth=1.5, linestyle="--", label=f"{window}-Day VaR (95%)")

        ax.set_title(f"Dynamic Risk Regimes: {asset_name}", fontsize=16, fontweight='bold', color='white', pad=15)
        ax.set_ylabel("Annualized Risk Equivalent", fontsize=12, color='white')
        ax.tick_params(axis='y', labelcolor="white")
        ax.tick_params(axis='x', labelcolor="white")
        ax.grid(color='#30363d', linestyle='-', linewidth=0.5)
        
        plt.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white', loc='upper left')
        
        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print(f"Rolling Risk graph saved to: {output_path}")
        plt.close()

    @staticmethod
    def render_monthly_returns_heatmap(
        returns: pd.Series,
        asset_name: str = "Portfolio",
        output_path: str = "docs/assets/monthly_heatmap.png"
    ):
        """
        Renders an institutional monthly returns heatmap (QuantStats style).
        """
        sns.set_theme(style="darkgrid")
        plt.style.use("dark_background")
        
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.date_range(end=datetime.date.today(), periods=len(returns), freq='B')
            
        monthly_ret = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        heatmap_data = monthly_ret.groupby([monthly_ret.index.year, monthly_ret.index.month]).sum().unstack()
        heatmap_data.columns = [calendar.month_abbr[i] for i in heatmap_data.columns]
        heatmap_data.index.name = "Year"
        
        yearly_ret = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        yearly_ret.index = yearly_ret.index.year
        heatmap_data['YTD'] = yearly_ret
        
        fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap_data) * 0.8 + 2)), dpi=300)
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        
        from matplotlib.colors import LinearSegmentedColormap
        # 5-stop high-saturation custom violet palette to eliminate dull mid-tones
        colors = ["#4d0099", "#26004d", "#0d1117", "#b333ff", "#e699ff"]
        cmap = LinearSegmentedColormap.from_list("vibrant_violet", colors)
        
        sns.heatmap(heatmap_data, annot=True, fmt=".1%", cmap=cmap, center=0.0,
                    linewidths=1.5, linecolor='#0d1117', cbar=False, ax=ax,
                    mask=heatmap_data.isnull(),
                    annot_kws={"weight": "bold", "size": 10})
        
        ax.set_title(f"Monthly Returns: {asset_name}", fontsize=16, fontweight='bold', color='white', pad=15)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white', rotation=0)
        ax.set_ylabel("")
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print(f"Monthly Heatmap saved to: {output_path}")
        plt.close()

    @staticmethod
    def render_risk_return_scatter(
        returns_df: pd.DataFrame,
        output_path: str = "docs/assets/risk_return_scatter.png"
    ):
        """
        Renders a risk (volatility) vs. return scatter plot for multiple assets.
        """
        sns.set_theme(style="darkgrid")
        plt.style.use("dark_background")
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        
        ann_return = (1 + returns_df.mean())**252 - 1
        ann_vol = returns_df.std() * np.sqrt(252)
        
        ax.scatter(ann_vol, ann_return, color="#00f2fe", edgecolors="white", s=100, alpha=0.9, zorder=5)
        
        for i, txt in enumerate(returns_df.columns):
            ax.annotate(txt, (ann_vol.iloc[i], ann_return.iloc[i]), 
                       xytext=(8, 8), textcoords='offset points', 
                       color='white', fontsize=11, weight='bold')
                       
        if len(returns_df.columns) > 1:
            z = np.polyfit(ann_vol, ann_return, 1)
            p = np.poly1d(z)
            ax.plot(ann_vol, p(ann_vol), color="#fe0060", linestyle="--", alpha=0.5, linewidth=2, zorder=4)
        
        ax.set_title("Asset Efficiency: Risk vs. Return", fontsize=16, fontweight='bold', color='white', pad=15)
        ax.set_xlabel("Annualized Risk (Volatility)", fontsize=12, color='white')
        ax.set_ylabel("Annualized Return", fontsize=12, color='white')
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        ax.tick_params(colors='white')
        ax.grid(color='#30363d', linestyle='-', linewidth=0.5, zorder=0)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print(f"Risk-Return Scatter saved to: {output_path}")
        plt.close()


if __name__ == "__main__":
    from nexus.analytics.analyzer import NexusAnalyzer
    
    np.random.seed(42)
    normal_days = np.random.normal(0.0005, 0.01, 1000)
    crash_days = np.random.normal(-0.02, 0.04, 50)
    boom_days = np.random.normal(0.02, 0.03, 30)
    
    simulated_returns = np.concatenate([normal_days, crash_days, boom_days])
    
    # 1. Manifold Plot
    NexusVisualizer.render_distribution_manifold(simulated_returns, asset_name="Simulated Global Equities")
    
    # 2. Drawdown Plot
    NexusVisualizer.render_drawdown_underwater(simulated_returns, asset_name="Simulated Global Equities")
    
    # 3. Tail Risk Comparison via Analyzer
    analyzer = NexusAnalyzer()
    analyzer.calibrate(simulated_returns)
    analyzer.compute(alpha=0.05)
    
    tail_metrics = {
        "Empirical VaR": analyzer.fetch('Value at Risk (0.05)')[0],
        "Conditional VaR": analyzer.fetch('Cond VaR (0.05)')[0],
        "Entropic VaR": analyzer.fetch('Entropic VaR (0.05)')[0]
    }
    
    NexusVisualizer.render_tail_risk_comparison(tail_metrics)
    
    # 4. Correlation Heatmap
    dates = pd.date_range(start="2020-01-01", periods=1080, freq="B")
    df_sim = pd.DataFrame({
        "Equities": np.random.normal(0.0005, 0.012, 1080),
        "Bonds": np.random.normal(0.0001, 0.004, 1080),
        "Commodities": np.random.normal(0.0003, 0.015, 1080),
        "Crypto": np.random.normal(0.0020, 0.040, 1080)
    }, index=dates)
    
    # Induce artificial correlation
    df_sim["Equities"] = df_sim["Equities"] * 0.7 + df_sim["Crypto"] * 0.3
    
    NexusVisualizer.render_correlation_heatmap(df_sim)
    
    # 5. Rolling Risk
    portfolio_sim = df_sim.sum(axis=1) / 4
    NexusVisualizer.render_rolling_risk(portfolio_sim, asset_name="Multi-Asset Portfolio")
    
    # 6. Monthly Returns Heatmap
    NexusVisualizer.render_monthly_returns_heatmap(portfolio_sim, asset_name="Multi-Asset Portfolio")
    
    # 7. Risk-Return Scatter
    NexusVisualizer.render_risk_return_scatter(df_sim)
