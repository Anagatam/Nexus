import os
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
        plt.close()

if __name__ == "__main__":
    # Simulate a stylized leptokurtic return stream (fat tails)
    np.random.seed(42)
    # Mixture of two normals to create a heavy-tailed distribution matching real markets
    normal_days = np.random.normal(0.0005, 0.01, 1000)
    crash_days = np.random.normal(-0.02, 0.04, 50)
    boom_days = np.random.normal(0.02, 0.03, 30)
    
    simulated_returns = np.concatenate([normal_days, crash_days, boom_days])
    
    NexusVisualizer.render_distribution_manifold(simulated_returns, asset_name="Simulated Global Equities")
