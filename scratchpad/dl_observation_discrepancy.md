# DL Model Observation Discrepancy Investigation

**Date:** 2025-11-07
**Issue:** DL model observations do not match CARAVAN streamflow values

---

## Problem Statement

When analyzing DL model predictions against TURC baseline, scatter plots showed observed discharge only reaching ~450 mm/year for DL models vs ~1500 mm/year for TURC, despite both methods supposedly using the same CARAVAN observations.

---

## Investigation

### Data Sources Compared

1. **DL Predictions Parquet:**
   - Path: `/Users/nicolaslazaro/Desktop/work/wb-project/model-training/results/evaluation/eval_2025-11-07_123625/model_name=ealstm_275k_params_all_0_casted/seed=42/predictions.parquet`
   - Contains `observation` column (daily streamflow)
   - Model: `ealstm_275k_params_all_0_casted`

2. **CARAVAN Dataset:**
   - Path: `/Users/nicolaslazaro/Desktop/CARAVAN_CLEAN_prod/train`
   - Source of truth for TURC analysis
   - Contains `streamflow` column (daily discharge)

### Methodology

1. Filtered both datasets to non-filled observations only
2. Selected 4 sample gauges across different regions
3. Compared daily values for year 1985
4. Computed ratios: `DL observation / CARAVAN streamflow`

---

## Key Findings

### 1. Values Do Not Match

**DL observations ≠ CARAVAN streamflow**, despite representing the same physical quantity (observed discharge).

### 2. Systematic Ratios by Region

| Gauge ID | Region | DL/CARAVAN Ratio | Interpretation |
|----------|--------|------------------|----------------|
| `camels_08176900` | CAMELS (US) | 95.5% | Close but not identical |
| `hysets_08189500` | HYSETS | 96.5% | Close but not identical |
| `camelsbr_54165000` | CAMELS-BR | **84.3%** | Significant discrepancy |
| `camelsaus_121002A` | CAMELS-AUS | 89.2% | Moderate discrepancy |

**Brazilian basins show worst mismatch (~84%)**, which explains why annual sums are so different.

### 3. Concrete Example: Daily Comparison

**Basin:** `camelsbr_54165000` (CAMELS-BR)
**Date range:** January 1-5, 1985

| Date | CARAVAN (mm/d) | DL observation (mm/d) | Ratio |
|------|----------------|----------------------|-------|
| 1985-01-01 | 0.51 | 0.41211 | 0.808 |
| 1985-01-02 | 0.48 | 0.392042 | 0.817 |
| 1985-01-03 | 0.52 | 0.41871 | 0.805 |
| 1985-01-04 | 2.77 | 1.327075 | **0.479** |
| 1985-01-05 | 2.35 | 1.20896 | **0.514** |

High flow events show even larger discrepancies (50-80% of CARAVAN values).

### 4. Annual Aggregation Impact

**For same basin-years:**

| Metric | DL Observations | CARAVAN (TURC) | DL as % of CARAVAN |
|--------|----------------|----------------|-------------------|
| Median annual discharge | 137.56 mm/yr | 235.02 mm/yr | **58.5%** |
| Mean annual discharge | 170.92 mm/yr | 286.55 mm/yr | **59.7%** |
| Max annual discharge | 1858.18 mm/yr | 1519.12 mm/yr | 122.3% |

**DL observations are systematically ~40% lower** than CARAVAN values when aggregated annually.

---

## Hypotheses

### What Could Cause This?

1. **Different CARAVAN Version**
   - DL model trained on earlier/later version of CARAVAN
   - Preprocessing or corrections applied to one dataset but not the other

2. **Different Unit Conversions**
   - Catchment area normalization applied differently
   - However, ratios don't align with area ratios

3. **Different Quality Control**
   - DL dataset has stricter filtering/correction procedures
   - Some high flows flagged or adjusted in DL but not CARAVAN

4. **Preprocessing Pipeline Differences**
   - DL model uses custom preprocessing before saving predictions
   - Normalization, scaling, or transformation applied to observations

5. **Data Source Mismatch**
   - DL model's "observations" come from different source entirely
   - Not actually from CARAVAN despite using CARAVAN gauge IDs

---

## Implications

### For Current Analysis

1. **Unfair Comparison:** Comparing DL vs TURC using different ground truth observations is invalid
2. **Misleading Performance:** DL model performance metrics are against different observations than TURC
3. **Plot Issues:** Scatter plots show different observation ranges (450 vs 1500 mm/yr)

### For Model Evaluation

**Cannot determine if DL outperforms TURC** because they're evaluated against different targets:
- TURC: Evaluated against current CARAVAN streamflow
- DL: Evaluated against mystery observations (~60% of CARAVAN values)

---

## Recommendations

### Option 1: Use DL Observations As-Is
**Pros:** Reflects how model was actually trained
**Cons:** Not comparable to TURC, unclear what observations represent

### Option 2: Re-fetch Observations from CARAVAN
**Pros:** Fair comparison, both methods against same ground truth
**Cons:** DL model was NOT trained on these observations

### Option 3: Investigate Data Provenance
**Pros:** Understand root cause, potentially fix upstream
**Cons:** Time-consuming, may require retraining models

---

## Action Items

- [ ] Confirm with model training code what observations were used
- [ ] Check if DL predictions parquet was created with different CARAVAN version
- [ ] Verify if any preprocessing/normalization was applied to observations
- [ ] Decide on approach for fair TURC vs DL comparison
- [ ] Document observation source in model evaluation methodology

---

## Files Generated During Investigation

- Script: `scripts/analyze_dl_model.py`
- Output: `outputs/dl_ealstm_275k_params_all_0_casted/analysis_data.parquet`
- Figures: `outputs/dl_ealstm_275k_params_all_0_casted/figures/`

**Note:** Current analysis uses DL parquet observations, which do not match CARAVAN values.
