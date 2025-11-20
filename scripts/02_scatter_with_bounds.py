"""
Scatter plot of TURC estimated vs observed discharge with ±20% bounds.

Points are colored by year type (dry/normal/wet) to visualize
performance differences across hydrologic conditions.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Create scatter plot with ±20% performance bounds."""

    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "outputs" / "analysis_data.parquet"
    output_path = project_root / "outputs" / "figures" / "scatter_20pct_bounds.png"

    # Load data
    logger.info("Loading analysis data...")
    data = pl.read_parquet(data_path)
    logger.info(f"✓ Loaded {len(data):,} basin-years")

    # Convert to pandas for seaborn
    df = data.select([
        "Q_observed_mm",
        "Q_turc_mm",
        "year_type",
        "within_20pct",
    ]).to_pandas()

    # Set plot style
    sns.set_context("paper", font_scale=1.3)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define colors for year types
    palette = {
        "dry": "#d62728",      # Red
        "normal": "#7f7f7f",   # Gray
        "wet": "#1f77b4",      # Blue
    }

    # Scatter plot with hue
    sns.scatterplot(
        data=df,
        x="Q_observed_mm",
        y="Q_turc_mm",
        hue="year_type",
        hue_order=["dry", "normal", "wet"],
        palette=palette,
        alpha=0.6,
        s=30,
        ax=ax,
        legend=True,
    )

    # Get axis limits for reference lines
    max_val = max(df["Q_observed_mm"].max(), df["Q_turc_mm"].max())
    min_val = 0

    # 1:1 line (perfect prediction)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='1:1 (perfect)', zorder=1)

    # ±20% bounds
    x_range = [min_val, max_val]
    ax.plot(x_range, [x * 1.2 for x in x_range], 'k:', linewidth=1, alpha=0.7, label='±20%', zorder=1)
    ax.plot(x_range, [x * 0.8 for x in x_range], 'k:', linewidth=1, alpha=0.7, zorder=1)

    # Labels and title
    ax.set_xlabel('Observed Discharge (mm/year)')
    ax.set_ylabel('TURC Estimated Discharge (mm/year)')
    ax.set_title('TURC Performance by Year Type')

    # Grid
    ax.grid(color='lightgrey', alpha=0.5, zorder=0)

    # Move legend outside plot
    ax.legend(loc='upper left', frameon=True, fontsize=10)

    # Equal aspect ratio to emphasize deviations
    ax.set_aspect('equal', adjustable='box')

    # Remove top and right spines
    sns.despine()

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved scatter plot to {output_path}")

    # Print statistics by year type
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE BY YEAR TYPE")
    logger.info("=" * 60)

    for year_type in ["dry", "normal", "wet"]:
        subset = df[df["year_type"] == year_type]
        n = len(subset)
        within_20 = subset["within_20pct"].sum()
        pct_within = within_20 / n * 100 if n > 0 else 0

        # Count points in different regions
        overestimate_big = (subset["Q_turc_mm"] > subset["Q_observed_mm"] * 1.2).sum()
        underestimate_big = (subset["Q_turc_mm"] < subset["Q_observed_mm"] * 0.8).sum()

        logger.info(f"\n{year_type.upper()} years (n={n}):")
        logger.info(f"  Within ±20%:        {within_20:4d} ({pct_within:5.1f}%)")
        logger.info(f"  Overestimate >20%:  {overestimate_big:4d} ({overestimate_big/n*100:5.1f}%)")
        logger.info(f"  Underestimate >20%: {underestimate_big:4d} ({underestimate_big/n*100:5.1f}%)")

    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Figure saved: {output_path.name}")
    logger.info("=" * 60)

    plt.close()


if __name__ == "__main__":
    main()
