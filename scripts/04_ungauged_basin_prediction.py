"""
Predict TURC performance for ungauged Madagascar basin.

Loads climate data for Madagascar basin, classifies it into climate bins
based on the benchmark analysis, and predicts whether TURC will over or
underestimate discharge.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.data_interface import CaravanDataSource

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_basin_climate(data_path: str, basin_name: str) -> dict[str, float]:
    """
    Load basin climate from static attributes.

    Args:
        data_path: Path to CARAVAN-formatted data
        basin_name: Name for logging

    Returns:
        Dict with basin_mean_P and basin_mean_T
    """
    logger.info(f"\nLoading {basin_name} data from {data_path}...")
    ds = CaravanDataSource(base_path=data_path)

    # Get available gauge IDs (should be just one)
    gauge_ids = ds.list_gauge_ids()
    logger.info(f"  Found {len(gauge_ids)} gauge(s): {gauge_ids}")

    if len(gauge_ids) == 0:
        raise ValueError(f"No gauges found in {data_path}")

    gauge_id = gauge_ids[0]

    # Load static attributes
    lazy_attrs = ds.get_static_attributes(
        gauge_ids=[gauge_id],
        columns=["tmp_dc_syr", "p_mean"],
    )
    attrs = lazy_attrs.collect()

    # Extract and convert units
    # tmp_dc_syr: °C × 10
    # p_mean: mm/day
    basin_mean_T = attrs["tmp_dc_syr"].item() / 10.0
    basin_mean_P = attrs["p_mean"].item() * 365.0

    logger.info(f"  Basin mean P: {basin_mean_P:.1f} mm/year (from p_mean static attribute)")
    logger.info(f"  Basin mean T: {basin_mean_T:.1f} °C (from tmp_dc_syr static attribute)")

    return {
        "gauge_id": gauge_id,
        "basin_mean_P": basin_mean_P,
        "basin_mean_T": basin_mean_T,
    }


def classify_basin(basin_mean_P: float, basin_mean_T: float, median_P: float, median_T: float) -> tuple[str, str]:
    """
    Classify basin into climate bin and predict TURC bias.

    Args:
        basin_mean_P: Basin mean precipitation
        basin_mean_T: Basin mean temperature
        median_P: Median precipitation from benchmark
        median_T: Median temperature from benchmark

    Returns:
        Tuple of (climate_bin, prediction)
    """
    if basin_mean_P <= median_P and basin_mean_T <= median_T:
        return "Low P, Low T", "likely OVERESTIMATES"
    elif basin_mean_P <= median_P and basin_mean_T > median_T:
        return "Low P, High T", "likely UNDERESTIMATES (severe)"
    elif basin_mean_P > median_P and basin_mean_T <= median_T:
        return "High P, Low T", "likely UNDERESTIMATES (moderate)"
    else:  # High P, High T
        return "High P, High T", "likely UNDERESTIMATES (moderate)"


def main() -> None:
    """Predict TURC performance for Madagascar ungauged basin."""

    # Paths
    project_root = Path(__file__).parent.parent
    madagascar_path = "/Users/nicolaslazaro/Desktop/carvanify_madagascar"
    benchmark_data_path = project_root / "outputs" / "analysis_data.parquet"
    output_path = project_root / "outputs" / "figures" / "climate_space_with_ungauged.png"

    # Load benchmark data to get median P and T
    logger.info("=" * 60)
    logger.info("LOADING BENCHMARK DATA")
    logger.info("=" * 60)
    data = pl.read_parquet(benchmark_data_path)
    basin_stats = data.group_by("gauge_id").agg([
        pl.col("basin_mean_P").first(),
        pl.col("basin_mean_T").first(),
        pl.col("percent_error").median().alias("median_percent_error"),
    ])

    median_P = basin_stats["basin_mean_P"].median()
    median_T = basin_stats["basin_mean_T"].median()

    logger.info(f"Benchmark median P: {median_P:.1f} mm/year")
    logger.info(f"Benchmark median T: {median_T:.1f} °C")

    # Load ungauged basin
    logger.info("\n" + "=" * 60)
    logger.info("LOADING UNGAUGED BASIN")
    logger.info("=" * 60)

    madagascar = load_basin_climate(madagascar_path, "Madagascar")

    # Classify basin
    logger.info("\n" + "=" * 60)
    logger.info("TURC PERFORMANCE PREDICTION")
    logger.info("=" * 60)

    madagascar_bin, madagascar_pred = classify_basin(
        madagascar["basin_mean_P"], madagascar["basin_mean_T"], median_P, median_T
    )

    logger.info(f"\nMadagascar Basin:")
    logger.info(f"  Mean P: {madagascar['basin_mean_P']:.1f} mm/year")
    logger.info(f"  Mean T: {madagascar['basin_mean_T']:.1f} °C")
    logger.info(f"  Climate bin: {madagascar_bin}")
    logger.info(f"  Prediction: TURC {madagascar_pred} discharge")

    # Create visualization
    logger.info("\n" + "=" * 60)
    logger.info("CREATING VISUALIZATION")
    logger.info("=" * 60)

    # Convert to pandas for plotting
    df = basin_stats.to_pandas()

    # Compute correlations
    corr_P = np.corrcoef(df["basin_mean_P"], df["median_percent_error"])[0, 1]
    corr_T = np.corrcoef(df["basin_mean_T"], df["median_percent_error"])[0, 1]

    # Create plot
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
        label='Benchmark basins',
    )
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='No bias')
    ax1.axhline(y=10, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax1.axhline(y=-10, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    # Add vertical line for Madagascar
    ax1.axvline(x=madagascar["basin_mean_P"], color='red', linestyle='-', linewidth=2,
                alpha=0.8, label='Madagascar')

    ax1.set_xlabel('Basin Mean Precipitation (mm/year)')
    ax1.set_ylabel('Median % Error')
    ax1.set_title(f'Error vs Precipitation (r = {corr_P:+.3f})')
    ax1.grid(color='lightgrey', alpha=0.5, zorder=0)
    ax1.legend(loc='best', fontsize=9)

    # Right plot: Error vs Temperature
    ax2 = axes[1]
    ax2.scatter(
        df["basin_mean_T"],
        df["median_percent_error"],
        s=60,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
        label='Benchmark basins',
    )
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='No bias')
    ax2.axhline(y=10, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.axhline(y=-10, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    # Add vertical line for Madagascar
    ax2.axvline(x=madagascar["basin_mean_T"], color='red', linestyle='-', linewidth=2,
                alpha=0.8, label='Madagascar')

    ax2.set_xlabel('Basin Mean Temperature (°C)')
    ax2.set_ylabel('Median % Error')
    ax2.set_title(f'Error vs Temperature (r = {corr_T:+.3f})')
    ax2.grid(color='lightgrey', alpha=0.5, zorder=0)
    ax2.legend(loc='best', fontsize=9)

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
