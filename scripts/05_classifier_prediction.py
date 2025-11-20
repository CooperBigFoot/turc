"""
Train classifier to predict TURC over/underestimation.

Uses static basin attributes to classify basins into three categories:
- over: TURC overestimates by >20%
- good: TURC within ±20%
- under: TURC underestimates by >20%
"""

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.data_interface import CaravanDataSource

# Set random seed for reproducibility
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Train classifier and predict Madagascar performance."""

    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "outputs" / "analysis_data.parquet"
    caravan_path = "/Users/nicolaslazaro/Desktop/CARAVAN_CLEAN_prod/train"
    madagascar_path = "/Users/nicolaslazaro/Desktop/carvanify_madagascar"

    # Feature list
    feature_cols = [
        "tmp_dc_syr",
        "p_mean",
        "area",
        "ele_mt_sav",
        "high_prec_dur",
        "frac_snow",
        "high_prec_freq",
        "slp_dg_sav",
        "cly_pc_sav",
        "aridity_ERA5_LAND",
        "aridity_FAO_PM",
        "low_prec_dur",
        "gauge_lat",
        "snd_pc_sav",
        "pet_mean_ERA5_LAND",
        "gauge_lon",
        "slt_pc_sav",
        "low_prec_freq",
        "glc_cl_smj",
        "seasonality_ERA5_LAND",
        "cmi_ix_syr",
        "rdd_mk_sav",
        "for_pc_sse",
    ]

    # Load and aggregate to basin level
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)

    data = pl.read_parquet(data_path)
    logger.info(f"✓ Loaded {len(data):,} basin-years")

    # Aggregate to basin level
    basin_stats = data.group_by("gauge_id").agg([
        pl.col("percent_error").median().alias("median_percent_error"),
    ])

    logger.info(f"✓ Aggregated to {len(basin_stats)} basins")

    # Classify basins into 3 categories (±20% threshold)
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

    # Load static attributes for training basins
    logger.info("\n" + "=" * 60)
    logger.info("LOADING STATIC ATTRIBUTES")
    logger.info("=" * 60)

    gauge_ids = basin_stats["gauge_id"].to_list()
    ds = CaravanDataSource(base_path=caravan_path)

    attrs_lazy = ds.get_static_attributes(gauge_ids=gauge_ids, columns=feature_cols)
    attrs = attrs_lazy.collect()

    # Join with basin stats
    basin_data = basin_stats.join(attrs, on="gauge_id", how="left")

    logger.info(f"✓ Loaded {len(feature_cols)} features for {len(basin_data)} basins")

    # Convert to pandas for sklearn
    df = basin_data.to_pandas()

    # Handle missing values
    missing_counts = df[feature_cols].isnull().sum()
    if missing_counts.sum() > 0:
        logger.info("\nMissing values detected:")
        for col in feature_cols:
            if missing_counts[col] > 0:
                logger.info(f"  {col}: {missing_counts[col]} missing")
        logger.info("\nFilling missing values with median...")
        for col in feature_cols:
            if missing_counts[col] > 0:
                df[col].fillna(df[col].median(), inplace=True)

    # Prepare features and target
    X = df[feature_cols].values
    y = df["class"].values

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RANDOM FOREST CLASSIFIER")
    logger.info("=" * 60)

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
    logger.info("\nCross-validation (5-fold):")
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    logger.info(f"  Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Training predictions
    y_pred = rf.predict(X)

    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred, labels=["good", "over", "under"])
    logger.info("                 Predicted")
    logger.info("               good  over  under")
    class_labels = ["good", "over", "under"]
    for i, label in enumerate(class_labels):
        logger.info(f"  Actual {label:6s} {cm[i][0]:4d}  {cm[i][1]:4d}  {cm[i][2]:5d}")

    # Classification report
    logger.info("\nClassification Report:")
    report = classification_report(y, y_pred, target_names=["good", "over", "under"], zero_division=0)
    for line in report.split('\n'):
        if line.strip():
            logger.info(f"  {line}")

    # Feature importance
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE IMPORTANCE (Top 10)")
    logger.info("=" * 60)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    logger.info("")
    for i in range(min(10, len(feature_cols))):
        idx = indices[i]
        logger.info(f"  {i+1:2d}. {feature_cols[idx]:25s} {importances[idx]:.4f}")

    # Predict for Madagascar
    logger.info("\n" + "=" * 60)
    logger.info("MADAGASCAR PREDICTION")
    logger.info("=" * 60)

    # Load Madagascar static attributes
    ds_mad = CaravanDataSource(base_path=madagascar_path)
    madagascar_gauge_ids = ds_mad.list_gauge_ids()

    attrs_mad_lazy = ds_mad.get_static_attributes(
        gauge_ids=madagascar_gauge_ids,
        columns=feature_cols
    )
    attrs_mad = attrs_mad_lazy.collect()

    # Convert to pandas and prepare features
    mad_df = attrs_mad.to_pandas()

    # Check which features are available in Madagascar
    available_features = [col for col in feature_cols if col in mad_df.columns]
    missing_features = [col for col in feature_cols if col not in mad_df.columns]

    if missing_features:
        logger.info(f"\nMissing features in Madagascar data: {missing_features}")
        logger.info(f"Using training median for missing features...")

        # Add missing features with training median values
        for col in missing_features:
            mad_df[col] = df[col].median()

    # Fill missing values with training median
    for col in feature_cols:
        if mad_df[col].isnull().any():
            mad_df[col].fillna(df[col].median(), inplace=True)

    X_mad = mad_df[feature_cols].values

    # Predict
    mad_pred = rf.predict(X_mad)[0]
    mad_proba = rf.predict_proba(X_mad)[0]

    # Get class order
    classes = rf.classes_

    logger.info(f"\nMadagascar Basin ({madagascar_gauge_ids[0]}):")
    logger.info(f"  Predicted class: {mad_pred}")
    logger.info(f"\n  Class probabilities:")
    for i, cls in enumerate(classes):
        logger.info(f"    {cls:6s}: {mad_proba[i]:.3f}")

    # Interpretation
    logger.info(f"\n  Interpretation:")
    if mad_pred == "good":
        logger.info(f"    TURC estimates are likely within ±20% for Madagascar")
    elif mad_pred == "over":
        logger.info(f"    TURC likely OVERESTIMATES discharge by >20% for Madagascar")
    else:
        logger.info(f"    TURC likely UNDERESTIMATES discharge by >20% for Madagascar")

    logger.info("\n" + "=" * 60)
    logger.info("✓ CLASSIFICATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
