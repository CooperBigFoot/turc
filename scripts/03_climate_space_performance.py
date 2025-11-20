"""
Visualize TURC performance across climate space.

Plots basin mean temperature vs precipitation with points colored by
median percent error. Includes correlation analysis and binned statistics.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Create climate space performance plot and statistics."""

    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "outputs" / "analysis_data.parquet"
    output_path = project_root / "outputs" / "figures" / "climate_space_performance.png"

    # Load data
    logger.info("Loading analysis data...")
    data = pl.read_parquet(data_path)
    logger.info(f"✓ Loaded {len(data):,} basin-years")

    # Aggregate to basin level
    logger.info("\nAggregating to basin level...")
    basin_stats = data.group_by("gauge_id").agg([
        pl.col("basin_mean_P").first(),
        pl.col("basin_mean_T").first(),
        pl.col("percent_error").median().alias("median_percent_error"),
        pl.col("within_20pct").mean().alias("pct_within_20"),
        pl.len().alias("n_years"),
    ])

    logger.info(f"✓ Aggregated to {len(basin_stats)} basins")

    # Convert to pandas for plotting
    df = basin_stats.to_pandas()

    # Compute correlations
    logger.info("\n" + "=" * 60)
    logger.info("CORRELATION ANALYSIS")
    logger.info("=" * 60)

    corr_P = np.corrcoef(df["basin_mean_P"], df["median_percent_error"])[0, 1]
    corr_T = np.corrcoef(df["basin_mean_T"], df["median_percent_error"])[0, 1]

    logger.info(f"\nCorrelation with median percent error:")
    logger.info(f"  Basin mean precipitation: r = {corr_P:+.3f}")
    logger.info(f"  Basin mean temperature:   r = {corr_T:+.3f}")

    # Binned analysis (split at median)
    logger.info("\n" + "=" * 60)
    logger.info("BINNED ANALYSIS (split at median)")
    logger.info("=" * 60)

    median_P = df["basin_mean_P"].median()
    median_T = df["basin_mean_T"].median()

    logger.info(f"\nMedian precipitation: {median_P:.1f} mm/year")
    logger.info(f"Median temperature:   {median_T:.1f} °C")

    bins = [
        ("Low P, Low T", (df["basin_mean_P"] <= median_P) & (df["basin_mean_T"] <= median_T)),
        ("Low P, High T", (df["basin_mean_P"] <= median_P) & (df["basin_mean_T"] > median_T)),
        ("High P, Low T", (df["basin_mean_P"] > median_P) & (df["basin_mean_T"] <= median_T)),
        ("High P, High T", (df["basin_mean_P"] > median_P) & (df["basin_mean_T"] > median_T)),
    ]

    logger.info(f"\nPerformance by climate bin:")
    for bin_name, mask in bins:
        subset = df[mask]
        n_basins = len(subset)
        median_err = subset["median_percent_error"].median()
        mean_within_20 = subset["pct_within_20"].mean() * 100

        logger.info(f"  {bin_name:15s} (n={n_basins:2d}): median error = {median_err:+6.1f}%, avg within ±20% = {mean_within_20:4.1f}%")

    # Create plot
    logger.info("\n" + "=" * 60)
    logger.info("CREATING PLOTS")
    logger.info("=" * 60)

    sns.set_context("paper", font_scale=1.3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Error vs Precipitation
    ax1 = axes[0]
    ax1.scatter(
        df["basin_mean_P"],
        df["median_percent_error"],
        s=60,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
    )
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='No bias')
    ax1.axhline(y=10, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax1.axhline(y=-10, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Basin Mean Precipitation (mm/year)')
    ax1.set_ylabel('Median % Error')
    ax1.set_title(f'Error vs Precipitation (r = {corr_P:+.3f})')
    ax1.grid(color='lightgrey', alpha=0.5, zorder=0)
    ax1.legend(loc='best', fontsize=10)

    # Right plot: Error vs Temperature
    ax2 = axes[1]
    ax2.scatter(
        df["basin_mean_T"],
        df["median_percent_error"],
        s=60,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
    )
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='No bias')
    ax2.axhline(y=10, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.axhline(y=-10, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Basin Median Temperature (°C)')
    ax2.set_ylabel('Median % Error')
    ax2.set_title(f'Error vs Temperature (r = {corr_T:+.3f})')
    ax2.grid(color='lightgrey', alpha=0.5, zorder=0)
    ax2.legend(loc='best', fontsize=10)

    # Remove top and right spines
    sns.despine()

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot to {output_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Figure saved: {output_path.name}")
    logger.info("=" * 60)

    plt.close()


if __name__ == "__main__":
    main()
