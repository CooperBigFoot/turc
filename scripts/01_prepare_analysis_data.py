"""
Prepare analysis dataset for TURC evaluation.

Loads annual TURC results, joins with CARAVAN static attributes,
computes derived metrics (errors, wet/dry flags, etc.), and saves
a combined dataset for downstream analysis.
"""

import logging
import sys
from pathlib import Path

import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.data_interface import CaravanDataSource

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Prepare analysis dataset."""

    # Paths
    project_root = Path(__file__).parent.parent
    annual_path = project_root / "outputs" / "annual_raw.parquet"
    output_path = project_root / "outputs" / "analysis_data.parquet"
    caravan_path = "/Users/nicolaslazaro/Desktop/CARAVAN_CLEAN_prod/train"

    # Load annual TURC results
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    annual = pl.read_parquet(annual_path)
    logger.info(f"✓ Loaded {len(annual):,} basin-years from {annual_path.name}")
    logger.info(f"  Basins: {annual['gauge_id'].n_unique()}")
    logger.info(f"  Year range: {annual['year'].min()}-{annual['year'].max()}")

    # Load static attributes
    logger.info("\nLoading CARAVAN static attributes...")
    gauge_ids = annual["gauge_id"].unique().to_list()
    ds = CaravanDataSource(base_path=caravan_path)

    static_columns = [
        "gauge_lat",
        "gauge_lon",
        "area",
        "ele_mt_sav",
        "slp_dg_sav",
        "for_pc_sse",
        "crp_pc_sse",
        "urb_pc_sse",
        "p_mean",
        "tmp_dc_syr",
        "aridity_ERA5_LAND",
    ]

    attrs_lazy = ds.get_static_attributes(gauge_ids=gauge_ids, columns=static_columns)
    attrs = attrs_lazy.collect()
    logger.info(f"✓ Loaded {len(static_columns)} static attributes for {len(attrs)} basins")

    # Join annual data with static attributes
    logger.info("\nJoining annual data with static attributes...")
    data = annual.join(attrs, on="gauge_id", how="left")
    logger.info(f"✓ Joined dataset: {len(data):,} rows × {len(data.columns)} columns")

    # Compute error metrics
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING DERIVED METRICS")
    logger.info("=" * 60)

    data = data.with_columns(
        [
            # Error metrics
            (pl.col("Q_turc_mm") - pl.col("Q_observed_mm")).alias("error_mm"),
            ((pl.col("Q_turc_mm") - pl.col("Q_observed_mm")) / pl.col("Q_observed_mm") * 100).alias("percent_error"),
            (pl.col("Q_turc_mm") / pl.col("Q_observed_mm")).alias("ratio_turc_obs"),
            # Performance flag
            (((pl.col("Q_turc_mm") - pl.col("Q_observed_mm")) / pl.col("Q_observed_mm") * 100).abs() <= 20).alias(
                "within_20pct"
            ),
        ]
    )

    logger.info("✓ Computed error metrics (error_mm, percent_error, ratio_turc_obs, within_20pct)")

    # Compute basin climatology (mean P and median T per basin across all years)
    logger.info("\nComputing basin climatology...")
    basin_climatology = data.group_by("gauge_id").agg(
        [
            pl.col("annual_precip_mm").mean().alias("basin_mean_P"),
            pl.col("annual_temp_c").median().alias("basin_mean_T"),
        ]
    )

    data = data.join(basin_climatology, on="gauge_id", how="left")
    logger.info("✓ Added basin climatology (basin_mean_P, basin_median_T)")

    # Classify wet/dry years (20% threshold)
    logger.info("\nClassifying wet/dry years (threshold: ±20% from basin mean)...")
    data = data.with_columns(
        [
            pl.when(pl.col("annual_precip_mm") > pl.col("basin_mean_P") * 1.2)
            .then(pl.lit("wet"))
            .when(pl.col("annual_precip_mm") < pl.col("basin_mean_P") * 0.8)
            .then(pl.lit("dry"))
            .otherwise(pl.lit("normal"))
            .alias("year_type"),
            # Deviation from basin climatology (%)
            ((pl.col("annual_precip_mm") - pl.col("basin_mean_P")) / pl.col("basin_mean_P") * 100).alias(
                "precip_anomaly_pct"
            ),
        ]
    )

    year_type_counts = data["year_type"].value_counts().sort("year_type")
    logger.info("✓ Year classification:")
    for row in year_type_counts.iter_rows(named=True):
        logger.info(f"    {row['year_type']:8s}: {row['count']:4d} years ({row['count'] / len(data) * 100:5.1f}%)")

    # Save prepared dataset
    logger.info("\n" + "=" * 60)
    logger.info("SAVING PREPARED DATASET")
    logger.info("=" * 60)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_path)
    logger.info(f"✓ Saved to {output_path}")
    logger.info(f"  Rows: {len(data):,}")
    logger.info(f"  Columns: {len(data.columns)}")

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)

    # Overall performance
    within_20 = data["within_20pct"].sum()
    total = len(data)
    logger.info(f"\nOverall Performance:")
    logger.info(f"  Within ±20%: {within_20:,} / {total:,} ({within_20 / total * 100:.1f}%)")

    median_error = data["percent_error"].median()
    logger.info(f"  Median percent error: {median_error:.1f}%")

    # Error distribution
    logger.info(f"\nError Distribution (percentiles):")
    percentiles = data.select(
        [
            pl.col("percent_error").quantile(0.10).alias("p10"),
            pl.col("percent_error").quantile(0.25).alias("p25"),
            pl.col("percent_error").quantile(0.50).alias("p50"),
            pl.col("percent_error").quantile(0.75).alias("p75"),
            pl.col("percent_error").quantile(0.90).alias("p90"),
        ]
    )

    for col in ["p10", "p25", "p50", "p75", "p90"]:
        val = percentiles[col].item()
        logger.info(f"  {col.upper():4s}: {val:7.1f}%")

    # Over/underestimation
    overestimate = (data["percent_error"] > 0).sum()
    underestimate = (data["percent_error"] < 0).sum()
    logger.info(f"\nBias Direction:")
    logger.info(f"  Overestimates (>0%):  {overestimate:,} ({overestimate / total * 100:.1f}%)")
    logger.info(f"  Underestimates (<0%): {underestimate:,} ({underestimate / total * 100:.1f}%)")

    # Performance by year type
    logger.info(f"\nPerformance by Year Type:")
    for year_type in ["dry", "normal", "wet"]:
        subset = data.filter(pl.col("year_type") == year_type)
        if len(subset) > 0:
            within = subset["within_20pct"].sum()
            n = len(subset)
            med_err = subset["percent_error"].median()
            logger.info(
                f"  {year_type.capitalize():8s}: {within:4d}/{n:4d} within ±20% ({within / n * 100:5.1f}%), median error: {med_err:6.1f}%"
            )

    # Basin-level summary
    logger.info(f"\nBasin-Level Statistics:")
    basin_stats = data.group_by("gauge_id").agg(
        [
            pl.col("within_20pct").mean().alias("pct_within_20"),
            pl.col("percent_error").median().alias("median_error"),
        ]
    )

    best_basins = basin_stats.sort("pct_within_20", descending=True).head(3)
    worst_basins = basin_stats.sort("pct_within_20", descending=False).head(3)

    logger.info(f"  Best 3 basins (highest % within ±20%):")
    for row in best_basins.iter_rows(named=True):
        logger.info(
            f"    {row['gauge_id']:20s}: {row['pct_within_20'] * 100:5.1f}% (median error: {row['median_error']:6.1f}%)"
        )

    logger.info(f"  Worst 3 basins (lowest % within ±20%):")
    for row in worst_basins.iter_rows(named=True):
        logger.info(
            f"    {row['gauge_id']:20s}: {row['pct_within_20'] * 100:5.1f}% (median error: {row['median_error']:6.1f}%)"
        )

    logger.info("\n" + "=" * 60)
    logger.info("✓ PREPARATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
