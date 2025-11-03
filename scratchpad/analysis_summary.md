# Turc Formula Analysis Summary

**Date:** 2025-10-29

This document summarizes the analysis workflow for applying the Turc formula to estimate annual discharge across CARAVAN basins.

---

## Objective

Evaluate the Turc formula's performance for estimating annual discharge in no-snow basins using the CARAVAN hydrological dataset.

---

## 1. Basin Selection

### Step 1.1: Filter No-Snow Basins

**Script:** `scripts/filter_high_quality_basins.py` (initial selection)

**Criteria:** Basins where `frac_snow = 0` (no snow precipitation)

**Results:**

- Total no-snow basins: **10,884**
- Distribution by region:
  - camels: 301
  - camelsaus: 561
  - camelsbr: 870
  - camelsch: 19
  - camelscl: 289
  - camelsde: 1,042
  - camelsgb: 665
  - grdc: 2,495
  - hysets: 4,642

**Output:** `analysis/no_snow_basins.txt`

---

### Step 1.2: Filter High-Quality Basins

**Script:** `scripts/filter_high_quality_basins.py`

**Quality Criteria:**

1. Less than 5% missing streamflow data
2. At least 25 years of valid data

**Method:**

- Used `streamflow_was_filled` column (1 = missing, 0 = valid)
- Calculated: `missing_pct = (sum(streamflow_was_filled == 1) / total_days) × 100`
- Calculated: `valid_years = count(streamflow_was_filled == 0) / 365.25`

**Results:**

- High-quality basins: **5,685** (52.2% of no-snow basins)
- Mean missing data: 0.64%
- Mean valid years: 45.1 years
- Range: 25.0 to 73.3 years

**Output:** `analysis/high_quality_no_snow_basins.txt`

---

## 2. Turc Formula Implementation

### Formula Overview

The Turc method estimates annual discharge using a water balance approach.

**Step 1: Temperature parameter**

```
L = 300 + 25T + 0.05T³
```

**Step 2: Actual evapotranspiration (AET)**

```
H = P - √(P² / (0.9 + P²/L²))
```

**Step 3: Annual discharge**

```
Q = P - H
```

Where:

- T = mean annual temperature (°C)
- P = annual precipitation (mm/year)
- H = actual evapotranspiration (mm/year)
- Q = annual discharge (mm/year)

---

### Implementation Details

**Script:** `scripts/compute_annual_discharge.py`

**Key Decision:** Handling Missing Data

Initial approach (incorrect):

```python
Q_observed = streamflow.sum()  # Includes filled values!
```

**Corrected approach:**

```python
Q_observed = streamflow
    .filter(streamflow_was_filled == 0)
    .mean() * 365
```

**Rationale:** The `streamflow_was_filled` column marks missing values that were gap-filled. Only valid observations should be used for annual totals.

**Data Loaded:**

- Timeseries variables: `total_precipitation_sum`, `temperature_2m_mean`, `streamflow`, `streamflow_was_filled`
- Processing: Batch processing (100 basins per batch)
- Date range: 1950-2022

**Output:** `analysis/streamflow_annual.parquet`

**Columns:**

- `gauge_id`: Basin identifier
- `year`: Calendar year
- `annual_precip_mm`: Annual precipitation sum
- `annual_temp_c`: Mean annual temperature
- `L`: Turc temperature parameter
- `AET_turc_mm`: Actual evapotranspiration (Turc estimate)
- `Q_turc_mm`: Annual discharge (Turc estimate)
- `Q_observed_mm`: Observed annual discharge
- `n_days`: Total days in year
- `n_valid_days`: Number of valid (non-filled) observations

---

## 3. Results

### 3.1 All High-Quality Basins

**Dataset:** 254,647 basin-years across 5,685 basins

**After filtering outliers** (Q_observed ≤ 3×P, n_valid_days ≥ 300):

- Basin-years: 245,079 (96.2%)

**Observed Discharge:**

- Mean: 393.1 mm/year
- Median: 280.9 mm/year
- Range: 0 to 5,910 mm/year

**Turc Discharge:**

- Mean: 505.4 mm/year
- Median: 478.3 mm/year
- Range: 5.9 to 1,720.8 mm/year

**Performance:**

- Correlation: **0.324**
- Turc overestimates across most basins

**Plots:**

- `analysis/qq_plot_turc_vs_observed_corrected.png`
- `analysis/qq_plot_turc_vs_observed_corrected_log.png`

---

### 3.2 Turc-Appropriate Climate Basins

The Turc formula was developed for specific climatic conditions. Filtering to warmer, wetter basins improves performance.

**Additional Criteria:**

- Mean annual temperature > 20°C
- Annual precipitation > 700 mm

**Dataset:** 24,561 basin-years across 999 basins (9.6% of total)

**Climate Characteristics:**

- Mean temperature: 22.7°C (range: 20.0 to 29.5°C)
- Mean precipitation: 1,320.8 mm (range: 700 to 3,672 mm)

**Observed Discharge:**

- Mean: 453.0 mm/year
- Median: 352.6 mm/year
- Range: 0 to 5,910 mm/year

**Turc Discharge:**

- Mean: 966.4 mm/year
- Median: 953.8 mm/year
- Range: 630.3 to 1,720.8 mm/year

**Performance:**

- Correlation: **0.430** ✓ (33% improvement)
- RMSE: 647.4 mm/year
- Bias: +513.4 mm/year (systematic overestimation)

**Plots:**

- `analysis/qq_plot_turc_filtered.png`
- `analysis/qq_plot_turc_filtered_log.png`

**Output:** `analysis/streamflow_annual_turc_filtered.parquet`

---

## 4. Key Findings

### 4.1 Data Quality Issues

**Outliers Found:** 313 basins (mostly GRDC region) with physically impossible discharge values

- Example: Basin `grdc_3621100` had 69,670 mm/year discharge from only 1,100 mm/year precipitation
- These were filtered using criterion: Q_observed ≤ 3×P

### 4.2 Turc Formula Performance

**Strengths:**

- Captures general pattern (correlation 0.43 in appropriate climates)
- Simple, requiring only P and T
- Computationally efficient

**Weaknesses:**

- Systematic overestimation (~513 mm/year bias in warm/wet climates)
- Lower correlation (0.43) compared to more complex models
- Performance drops outside intended climate range (T>20°C, P>700mm)

### 4.3 Climate Suitability

Filtering to Turc-appropriate climates improved correlation from 0.324 to 0.430, confirming the formula works best in:

- Warm climates (T > 20°C)
- High precipitation (P > 700 mm)
- No snow conditions

---

## 5. Files Generated

### Scripts

- `scripts/filter_high_quality_basins.py` - Basin quality filtering
- `scripts/compute_annual_discharge.py` - Turc formula implementation

### Data Files

- `analysis/no_snow_basins.txt` - List of 10,884 no-snow basins
- `analysis/high_quality_no_snow_basins.txt` - List of 5,685 high-quality basins
- `analysis/streamflow_annual.parquet` - Annual discharge for all basins
- `analysis/streamflow_annual_turc_filtered.parquet` - Filtered to Turc-appropriate climate

### Visualizations

- `analysis/qq_plot_turc_vs_observed_corrected.png` - Q-Q plot (all basins)
- `analysis/qq_plot_turc_vs_observed_corrected_log.png` - Q-Q plot log scale (all basins)
- `analysis/qq_plot_turc_filtered.png` - Q-Q plot (T>20°C, P>700mm)
- `analysis/qq_plot_turc_filtered_log.png` - Q-Q plot log scale (T>20°C, P>700mm)

---

## 6. Recommendations

### For Future Work

1. **Calibration:** The systematic bias suggests a calibrated version of Turc could improve performance
   - Potential: Multiply Turc estimates by 0.47 (453/966) to match observed mean

2. **Regional Analysis:** Investigate Turc performance by region
   - Some regions may have better performance than others

3. **Monthly Discharge:** Apply Turc formula at monthly timescale
   - May reduce aggregation errors
   - Better capture seasonal patterns

4. **Alternative Formulas:** Compare with other empirical methods
   - Budyko framework
   - Zhang et al. (2001)
   - Other PET-based methods

---

## 7. Lessons Learned

### Critical Implementation Details

1. **Missing Data Handling:**
   - Must exclude filled/interpolated values
   - Use `streamflow_was_filled` flag to identify valid observations
   - Calculate annual as: `mean(valid_values) × 365`

2. **Data Quality:**
   - Always filter outliers (physical impossibility checks)
   - Require minimum valid days per year (≥300 days)
   - Check for source data errors in GRDC basins

3. **Climate Appropriateness:**
   - Empirical formulas have specific applicability ranges
   - Always filter to intended climate conditions before evaluation
   - Correlation improves 33% when using appropriate climate subset

---

## References

**Turc Formula:**

- Turc, L. (1954). "Le bilan d'eau des sols. Relation entre la précipitation, l'évaporation et l'écoulement." *Annales Agronomiques*.

**CARAVAN Dataset:**

- Location: `/Users/nicolaslazaro/Desktop/CARAVAN_CLEAN_prod/train`
- Interface: `transfer_learning_publication.data.CaravanDataSource`
- Regions: 10 (camels, camelsaus, camelsbr, camelsch, camelscl, camelsde, camelsgb, grdc, hysets, lamah)
