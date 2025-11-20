"""
Comprehensive analysis of deep learning model predictions.

Performs the same analysis as TURC scripts 01-05, but for DL model predictions.
Filters to the exact basin-year pairs that TURC used for fair comparison.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.data_interface import CaravanDataSource

# Set random seed for reproducibility
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_and_filter_dl_predictions(
    predictions_path: Path, turc_basin_years: pl.DataFrame
) -> tuple[pl.DataFrame, str]:
    """
    Load DL predictions and filter to TURC's basin-year pairs.

    Args:
        predictions_path: Path to predictions parquet file
        turc_basin_years: DataFrame with gauge_id and year columns from TURC

    Returns:
        Tuple of (filtered daily predictions, model_name)
    """
    logger.info("=" * 60)
    logger.info("LOADING DL PREDICTIONS")
    logger.info("=" * 60)

    # Load predictions
    daily_preds = pl.read_parquet(predictions_path)
    logger.info(f"✓ Loaded {len(daily_preds):,} daily predictions")

    # Extract model name
    model_name = daily_preds["model_name"][0]
    logger.info(f"  Model: {model_name}")

    # Extract year from date
    daily_preds = daily_preds.with_columns(
        pl.col("issue_date").dt.year().alias("year")
    )

    # Get unique gauge_ids from TURC
    turc_gauges = turc_basin_years["gauge_id"].unique().to_list()
    logger.info(f"\n✓ TURC analysis used {len(turc_gauges)} basins")

    # Filter to TURC gauges
    daily_preds = daily_preds.filter(
        pl.col("group_identifier").is_in(turc_gauges)
    )

    # Rename group_identifier to gauge_id for consistency
    daily_preds = daily_preds.rename({"group_identifier": "gauge_id"})

    logger.info(f"✓ Filtered to {daily_preds['gauge_id'].n_unique()} matching basins")
    logger.info(f"  Date range: {daily_preds['issue_date'].min()} to {daily_preds['issue_date'].max()}")

    return daily_preds, model_name


def aggregate_to_annual(daily_preds: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate daily predictions to annual totals.

    Args:
        daily_preds: Daily predictions in mm/d

    Returns:
        Annual aggregated data
    """
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATING TO ANNUAL")
    logger.info("=" * 60)

    # Filter out filled observations
    daily_preds = daily_preds.filter(pl.col("observation_was_filled") == 0)
    logger.info(f"✓ Filtered out filled observations: {len(daily_preds):,} observations remaining")

    # Aggregate to annual
    annual = daily_preds.group_by(["gauge_id", "year"]).agg([
        pl.col("prediction").sum().alias("predicted_Q_mm"),
        pl.col("observation").sum().alias("observed_Q_mm"),
        pl.col("prediction").count().alias("n_days"),
    ])

    # Filter complete years only
    annual = annual.filter(pl.col("n_days") >= 365)
    logger.info(f"✓ Aggregated to annual: {len(annual):,} basin-years (≥365 days)")
    logger.info(f"  Year range: {annual['year'].min()}-{annual['year'].max()}")

    return annual


def get_caravan_annual_data(
    annual_dl: pl.DataFrame, caravan_path: str
) -> pl.DataFrame:
    """
    Get annual precipitation and temperature from CARAVAN.

    Args:
        annual_dl: Annual DL predictions with gauge_id and year
        caravan_path: Path to CARAVAN data

    Returns:
        Annual P and T data
    """
    logger.info("\n" + "=" * 60)
    logger.info("LOADING CARAVAN TIMESERIES DATA")
    logger.info("=" * 60)

    ds = CaravanDataSource(base_path=caravan_path)
    gauge_ids = annual_dl["gauge_id"].unique().to_list()
    years = annual_dl["year"].unique().to_list()

    # Get date range
    min_year = min(years)
    max_year = max(years)

    logger.info(f"  Loading P and T for {len(gauge_ids)} basins, {min_year}-{max_year}")

    # Load timeseries
    ts_lazy = ds.get_timeseries(
        gauge_ids=gauge_ids,
        columns=["total_precipitation_sum", "temperature_2m_mean"],
        date_range=(f"{min_year}-01-01", f"{max_year}-12-31"),
    )

    ts = ts_lazy.collect()
    logger.info(f"✓ Loaded {len(ts):,} daily observations")

    # Extract year and aggregate
    ts = ts.with_columns(pl.col("date").dt.year().alias("year"))

    annual_caravan = ts.group_by(["gauge_id", "year"]).agg([
        pl.col("total_precipitation_sum").sum().alias("annual_P_mm"),
        pl.col("temperature_2m_mean").median().alias("annual_T_degC"),
        pl.col("total_precipitation_sum").count().alias("n_days_caravan"),
    ])

    # Filter complete years
    annual_caravan = annual_caravan.filter(pl.col("n_days_caravan") >= 365)
    logger.info(f"✓ Aggregated to {len(annual_caravan):,} annual observations")

    return annual_caravan


def get_static_attributes(gauge_ids: list[str], caravan_path: str) -> pl.DataFrame:
    """
    Load static attributes from CARAVAN.

    Args:
        gauge_ids: List of gauge IDs
        caravan_path: Path to CARAVAN data

    Returns:
        Static attributes DataFrame
    """
    logger.info("\n" + "=" * 60)
    logger.info("LOADING STATIC ATTRIBUTES")
    logger.info("=" * 60)

    ds = CaravanDataSource(base_path=caravan_path)

    # Same attributes as TURC
    attr_cols = [
        "area",
        "ele_mt_sav",
        "slp_dg_sav",
        "tmp_dc_syr",
        "p_mean",
        "aridity_ERA5_LAND",
        "frac_snow",
        "glc_cl_smj",
        "for_pc_sse",
        "cmi_ix_syr",
        "rdd_mk_sav",
    ]

    attrs_lazy = ds.get_static_attributes(gauge_ids=gauge_ids, columns=attr_cols)
    attrs = attrs_lazy.collect()

    logger.info(f"✓ Loaded {len(attr_cols)} attributes for {len(attrs)} basins")

    return attrs


def compute_metrics_and_classify(data: pl.DataFrame) -> pl.DataFrame:
    """
    Compute error metrics and classify wet/dry years.

    Args:
        data: Combined annual data

    Returns:
        Data with metrics and classifications
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING METRICS")
    logger.info("=" * 60)

    # Filter Q > P (physically impossible)
    n_before = len(data)
    data = data.filter(pl.col("observed_Q_mm") <= pl.col("annual_P_mm"))
    n_removed = n_before - len(data)
    if n_removed > 0:
        logger.info(f"✓ Removed {n_removed} basin-years where Q > P")

    # Compute error metrics
    data = data.with_columns([
        (pl.col("predicted_Q_mm") - pl.col("observed_Q_mm")).alias("error_mm"),
        ((pl.col("predicted_Q_mm") - pl.col("observed_Q_mm")) / pl.col("observed_Q_mm") * 100).alias("percent_error"),
        (pl.col("predicted_Q_mm") / pl.col("observed_Q_mm")).alias("ratio_pred_obs"),
    ])

    data = data.with_columns(
        (pl.col("percent_error").abs() <= 20).alias("within_20pct")
    )

    logger.info("✓ Computed error metrics")

    # Compute basin climatology
    basin_clim = data.group_by("gauge_id").agg([
        pl.col("annual_P_mm").mean().alias("basin_mean_P"),
        pl.col("annual_T_degC").median().alias("basin_median_T"),
    ])

    data = data.join(basin_clim, on="gauge_id", how="left")
    logger.info("✓ Added basin climatology")

    # Classify wet/dry years (±20% from basin mean P)
    data = data.with_columns(
        pl.when(pl.col("annual_P_mm") < pl.col("basin_mean_P") * 0.8)
        .then(pl.lit("dry"))
        .when(pl.col("annual_P_mm") > pl.col("basin_mean_P") * 1.2)
        .then(pl.lit("wet"))
        .otherwise(pl.lit("normal"))
        .alias("year_type")
    )

    year_counts = data["year_type"].value_counts().sort("year_type")
    logger.info("\n✓ Year classification:")
    for row in year_counts.iter_rows(named=True):
        logger.info(f"    {row['year_type']:8s}: {row['count']:4d} years ({row['count']/len(data)*100:5.1f}%)")

    return data


def print_performance_summary(data: pl.DataFrame, model_name: str) -> None:
    """Print performance summary statistics."""
    logger.info("\n" + "=" * 60)
    logger.info(f"PERFORMANCE SUMMARY: {model_name}")
    logger.info("=" * 60)

    # Overall performance
    n_within = data["within_20pct"].sum()
    total = len(data)
    pct_within = n_within / total * 100

    logger.info(f"\nOverall Performance:")
    logger.info(f"  Within ±20%: {n_within} / {total} ({pct_within:.1f}%)")
    logger.info(f"  Median percent error: {data['percent_error'].median():.1f}%")

    # Error distribution
    logger.info(f"\nError Distribution (percentiles):")
    logger.info(f"  P10 : {data['percent_error'].quantile(0.10):7.1f}%")
    logger.info(f"  P25 : {data['percent_error'].quantile(0.25):7.1f}%")
    logger.info(f"  P50 : {data['percent_error'].quantile(0.50):7.1f}%")
    logger.info(f"  P75 : {data['percent_error'].quantile(0.75):7.1f}%")
    logger.info(f"  P90 : {data['percent_error'].quantile(0.90):7.1f}%")

    # Bias direction
    n_over = (data["percent_error"] > 0).sum()
    n_under = (data["percent_error"] < 0).sum()
    logger.info(f"\nBias Direction:")
    logger.info(f"  Overestimates (>0%):  {n_over:4d} ({n_over/total*100:.1f}%)")
    logger.info(f"  Underestimates (<0%): {n_under:4d} ({n_under/total*100:.1f}%)")

    # Performance by year type
    logger.info(f"\nPerformance by Year Type:")
    for year_type in ["dry", "normal", "wet"]:
        subset = data.filter(pl.col("year_type") == year_type)
        n_total_type = len(subset)
        n_within_type = subset["within_20pct"].sum()
        pct = n_within_type / n_total_type * 100 if n_total_type > 0 else 0
        median_err = subset["percent_error"].median()
        logger.info(
            f"  {year_type.capitalize():8s}: {n_within_type:4d}/{n_total_type:4d} within ±20% "
            f"({pct:5.1f}%), median error: {median_err:7.1f}%"
        )

    # Basin-level stats
    basin_stats = data.group_by("gauge_id").agg([
        pl.col("within_20pct").mean().alias("pct_within"),
        pl.col("percent_error").median().alias("median_error"),
    ])

    basin_stats = basin_stats.sort("pct_within", descending=True)

    logger.info(f"\nBest 3 basins (highest % within ±20%):")
    for row in basin_stats.head(3).iter_rows(named=True):
        logger.info(
            f"  {row['gauge_id']:20s}: {row['pct_within']*100:5.1f}% "
            f"(median error: {row['median_error']:6.1f}%)"
        )

    logger.info(f"\nWorst 3 basins (lowest % within ±20%):")
    for row in basin_stats.tail(3).iter_rows(named=True):
        logger.info(
            f"  {row['gauge_id']:20s}: {row['pct_within']*100:5.1f}% "
            f"(median error: {row['median_error']:6.1f}%)"
        )


def create_scatter_plot(data: pl.DataFrame, output_dir: Path, model_name: str) -> None:
    """Create scatter plot with ±20% bounds."""
    logger.info("\n" + "=" * 60)
    logger.info("CREATING SCATTER PLOT")
    logger.info("=" * 60)

    # Convert to pandas for plotting
    df = data.select([
        "observed_Q_mm",
        "predicted_Q_mm",
        "year_type",
        "within_20pct",
    ]).to_pandas()

    # Set plot style
    sns.set_context("paper", font_scale=1.3)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define colors for year types (same as TURC)
    palette = {
        "dry": "#d62728",      # Red
        "normal": "#7f7f7f",   # Gray
        "wet": "#1f77b4",      # Blue
    }

    # Scatter plot with hue
    sns.scatterplot(
        data=df,
        x="observed_Q_mm",
        y="predicted_Q_mm",
        hue="year_type",
        hue_order=["dry", "normal", "wet"],
        palette=palette,
        alpha=0.6,
        s=30,
        ax=ax,
        legend=True,
    )

    # Get axis limits for reference lines
    max_val = max(df["observed_Q_mm"].max(), df["predicted_Q_mm"].max())
    min_val = 0

    # 1:1 line (perfect prediction)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='1:1 (perfect)', zorder=1)

    # ±20% bounds
    x_range = [min_val, max_val]
    ax.plot(x_range, [x * 1.2 for x in x_range], 'k:', linewidth=1, alpha=0.7, label='±20%', zorder=1)
    ax.plot(x_range, [x * 0.8 for x in x_range], 'k:', linewidth=1, alpha=0.7, zorder=1)

    # Labels and title
    ax.set_xlabel('Observed Discharge (mm/year)')
    ax.set_ylabel('Deep Learning Predicted Discharge (mm/year)')
    ax.set_title(f'{model_name} Performance by Year Type')

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
    output_path = output_dir / "scatter_20pct_bounds.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved scatter plot to {output_path}")


def create_climate_space_plot(data: pl.DataFrame, output_dir: Path, model_name: str) -> None:
    """Create climate space performance plots."""
    logger.info("\n" + "=" * 60)
    logger.info("CREATING CLIMATE SPACE PLOT")
    logger.info("=" * 60)

    # Aggregate to basin level
    basin_stats = data.group_by("gauge_id").agg([
        pl.col("percent_error").median().alias("median_percent_error"),
        pl.col("within_20pct").mean().alias("pct_within_20"),
        pl.col("basin_mean_P").first(),
        pl.col("basin_median_T").first(),
    ])

    df = basin_stats.to_pandas()

    # Correlations
    corr_p = np.corrcoef(df["basin_mean_P"], df["median_percent_error"])[0, 1]
    corr_t = np.corrcoef(df["basin_median_T"], df["median_percent_error"])[0, 1]

    logger.info(f"\nCorrelation with median percent error:")
    logger.info(f"  Basin mean precipitation: r = {corr_p:+.3f}")
    logger.info(f"  Basin median temperature: r = {corr_t:+.3f}")

    # Climate bins (split at median)
    median_p = df["basin_mean_P"].median()
    median_t = df["basin_median_T"].median()

    logger.info(f"\nMedian precipitation: {median_p:.1f} mm/year")
    logger.info(f"Median temperature:   {median_t:.1f} °C")

    bins = [
        ("Low P, Low T", (df["basin_mean_P"] <= median_p) & (df["basin_median_T"] <= median_t)),
        ("Low P, High T", (df["basin_mean_P"] <= median_p) & (df["basin_median_T"] > median_t)),
        ("High P, Low T", (df["basin_mean_P"] > median_p) & (df["basin_median_T"] <= median_t)),
        ("High P, High T", (df["basin_mean_P"] > median_p) & (df["basin_median_T"] > median_t)),
    ]

    logger.info(f"\nPerformance by climate bin:")
    for bin_name, mask in bins:
        subset = df[mask]
        n_basins = len(subset)
        median_err = subset["median_percent_error"].median()
        mean_within_20 = subset["pct_within_20"].mean() * 100
        logger.info(f"  {bin_name:15s} (n={n_basins:2d}): median error = {median_err:+6.1f}%, avg within ±20% = {mean_within_20:4.1f}%")

    # Set plot style
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
    ax1.set_title(f'Error vs Precipitation (r = {corr_p:+.3f})')
    ax1.grid(color='lightgrey', alpha=0.5, zorder=0)
    ax1.legend(loc='best', fontsize=10)

    # Right plot: Error vs Temperature
    ax2 = axes[1]
    ax2.scatter(
        df["basin_median_T"],
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
    ax2.set_title(f'Error vs Temperature (r = {corr_t:+.3f})')
    ax2.grid(color='lightgrey', alpha=0.5, zorder=0)
    ax2.legend(loc='best', fontsize=10)

    # Remove top and right spines
    sns.despine()

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = output_dir / "climate_space_performance.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved climate space plot to {output_path}")


def train_classifier_and_predict(
    data: pl.DataFrame, output_dir: Path, model_name: str, caravan_path: str
) -> None:
    """Train RF classifier and predict Madagascar performance."""
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING CLASSIFIER")
    logger.info("=" * 60)

    # Aggregate to basin level and classify
    basin_stats = data.group_by("gauge_id").agg([
        pl.col("percent_error").median().alias("median_percent_error"),
    ])

    basin_stats = basin_stats.with_columns([
        pl.when(pl.col("median_percent_error") > 20)
        .then(pl.lit("over"))
        .when(pl.col("median_percent_error") < -20)
        .then(pl.lit("under"))
        .otherwise(pl.lit("good"))
        .alias("class")
    ])

    # Class distribution
    class_counts = basin_stats["class"].value_counts().sort("class")
    logger.info("\nClass distribution:")
    for row in class_counts.iter_rows(named=True):
        logger.info(f"  {row['class']:6s}: {row['count']:3d} basins ({row['count']/len(basin_stats)*100:5.1f}%)")

    # Load static attributes
    gauge_ids = basin_stats["gauge_id"].to_list()
    feature_cols = [
        "tmp_dc_syr", "p_mean", "area", "ele_mt_sav", "high_prec_dur",
        "frac_snow", "high_prec_freq", "slp_dg_sav", "cly_pc_sav",
        "aridity_ERA5_LAND", "aridity_FAO_PM", "low_prec_dur", "gauge_lat",
        "snd_pc_sav", "pet_mean_ERA5_LAND", "gauge_lon", "slt_pc_sav",
        "low_prec_freq", "glc_cl_smj", "seasonality_ERA5_LAND",
        "cmi_ix_syr", "rdd_mk_sav", "for_pc_sse",
    ]

    ds = CaravanDataSource(base_path=caravan_path)
    attrs_lazy = ds.get_static_attributes(gauge_ids=gauge_ids, columns=feature_cols)
    attrs = attrs_lazy.collect()

    basin_data = basin_stats.join(attrs, on="gauge_id", how="left")
    df = basin_data.to_pandas()

    # Handle missing values
    for col in feature_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    X = df[feature_cols].values
    y = df["class"].values

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X, y)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    logger.info(f"\nCross-validation (5-fold):")
    logger.info(f"  Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Training predictions
    y_pred = rf.predict(X)

    # Confusion matrix
    logger.info("\nTraining Confusion Matrix:")
    cm = confusion_matrix(y, y_pred, labels=["good", "over", "under"])
    logger.info("                 Predicted")
    logger.info("               good  over  under")
    for i, label in enumerate(["good", "over", "under"]):
        logger.info(f"  Actual {label:6s} {cm[i][0]:4d}  {cm[i][1]:4d}  {cm[i][2]:5d}")

    # Feature importance
    logger.info("\nTop 10 Features:")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(10, len(feature_cols))):
        idx = indices[i]
        logger.info(f"  {i+1:2d}. {feature_cols[idx]:25s} {importances[idx]:.4f}")

    logger.info(f"\n✓ Classifier training complete for {model_name}")


def main() -> None:
    """Main analysis pipeline."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Analyze DL model predictions")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions parquet file",
    )
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    turc_data_path = project_root / "outputs" / "analysis_data.parquet"
    caravan_path = "/Users/nicolaslazaro/Desktop/CARAVAN_CLEAN_prod/train"

    # Load TURC data to get basin-year pairs
    logger.info("=" * 60)
    logger.info("LOADING TURC REFERENCE DATA")
    logger.info("=" * 60)
    turc_data = pl.read_parquet(turc_data_path)
    turc_basin_years = turc_data.select(["gauge_id", "year"])
    logger.info(f"✓ TURC used {len(turc_basin_years):,} basin-years from {turc_data['gauge_id'].n_unique()} basins")

    # Load and filter DL predictions
    daily_preds, model_name = load_and_filter_dl_predictions(args.predictions, turc_basin_years)

    # Create output directory
    output_dir = project_root / "outputs" / f"dl_{model_name}"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n✓ Output directory: {output_dir}")

    # Aggregate to annual
    annual_dl = aggregate_to_annual(daily_preds)

    # Filter to exact TURC basin-year pairs
    logger.info("\n" + "=" * 60)
    logger.info("FILTERING TO TURC BASIN-YEAR PAIRS")
    logger.info("=" * 60)
    annual_dl = annual_dl.join(
        turc_basin_years.with_columns(pl.lit(True).alias("in_turc")),
        on=["gauge_id", "year"],
        how="inner"
    )
    logger.info(f"✓ Matched {len(annual_dl):,} basin-years with TURC")

    # Get CARAVAN annual data
    annual_caravan = get_caravan_annual_data(annual_dl, caravan_path)

    # Join DL with CARAVAN
    annual_data = annual_dl.join(annual_caravan, on=["gauge_id", "year"], how="left")

    # Get static attributes
    gauge_ids = annual_data["gauge_id"].unique().to_list()
    static_attrs = get_static_attributes(gauge_ids, caravan_path)

    # Join with static attributes
    annual_data = annual_data.join(static_attrs, on="gauge_id", how="left")

    # Compute metrics and classify
    annual_data = compute_metrics_and_classify(annual_data)

    # Save analysis data
    output_path = output_dir / "analysis_data.parquet"
    annual_data.write_parquet(output_path)
    logger.info(f"\n✓ Saved analysis data to {output_path}")
    logger.info(f"  Rows: {len(annual_data):,}")
    logger.info(f"  Columns: {len(annual_data.columns)}")

    # Print performance summary
    print_performance_summary(annual_data, model_name)

    # Create visualizations
    create_scatter_plot(annual_data, figures_dir, model_name)
    create_climate_space_plot(annual_data, figures_dir, model_name)

    # Train classifier
    train_classifier_and_predict(annual_data, figures_dir, model_name, caravan_path)

    logger.info("\n" + "=" * 60)
    logger.info("✓ ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
