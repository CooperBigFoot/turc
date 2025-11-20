# TURC Method Benchmark Results

## Dataset

- **Basins**: 97 tropical basins from CARAVAN dataset
- **Basin-years**: 3,771 annual observations
- **Period**: 1951-2021
- **Temperature range**: 21-27°C (mean annual temperature)
- **Regions**: CAMELS-BR, CAMELS (US), HydroSETS, GRDC

**Filtering applied**:

- Negative TURC discharge estimates set to zero
- Years where observed discharge exceeded annual precipitation removed
- Only complete years (≥365 days) retained

---

## Overall Performance

**Accuracy within ±20% of observed discharge**: 20.5% (772 / 3,771 basin-years)

**Error distribution**:

| Metric | Value |
|--------|-------|
| Median percent error | -20.7% |
| 10th percentile | -75.8% |
| 25th percentile | -54.4% |
| 75th percentile | +33.1% |
| 90th percentile | +124.8% |

**Bias direction**:

- Underestimates (error < 0%): 63.0% of cases
- Overestimates (error > 0%): 37.0% of cases

---

## Performance by Hydrologic Condition

Year classification based on ±20% deviation from basin-specific mean annual precipitation.

| Year Type | Count | % of Total | Within ±20% | Median Error |
|-----------|-------|------------|-------------|--------------|
| Dry       | 686   | 18.2%      | 7.3%        | -64.3%       |
| Normal    | 2,429 | 64.4%      | 23.3%       | -16.9%       |
| Wet       | 656   | 17.4%      | 23.8%       | +3.0%        |

**Dry year performance**:

- Within ±20%: 7.3% (50 / 686)
- Underestimate >20%: 79.0% (542 / 686)
- Overestimate >20%: 13.7% (94 / 686)

**Normal year performance**:

- Within ±20%: 23.3% (566 / 2,429)
- Underestimate >20%: 46.9% (1,138 / 2,429)
- Overestimate >20%: 29.8% (725 / 2,429)

**Wet year performance**:

- Within ±20%: 23.8% (156 / 656)
- Underestimate >20%: 34.5% (226 / 656)
- Overestimate >20%: 41.8% (274 / 656)

---

## Performance by Climate

**Climate bin analysis** (split at median precipitation: 1035.3 mm/year, median temperature: 22.4°C):

| Climate Bin | Basins | Median Error | Avg Within ±20% |
|-------------|--------|--------------|-----------------|
| Low P, Low T | 27 | +7.4% | 18.1% |
| Low P, High T | 22 | -57.3% | 10.9% |
| High P, Low T | 22 | -12.0% | 26.9% |
| High P, High T | 26 | -16.0% | 26.3% |

**Climate correlations with error**:

- Temperature (median): r = -0.345 (moderate negative correlation)
- Precipitation (mean): r = -0.107 (weak negative correlation)

Higher temperature basins show stronger underestimation bias.

---

## Basin-Level Variation

**Best performing basins** (% of years within ±20%):

| Basin ID | % Within ±20% | Median Error |
|----------|---------------|--------------|
| camelsbr_42840000 | 47.4% | -19.7% |
| camelsbr_60381000 | 47.4% | -23.0% |
| camelsbr_60110000 | 47.4% | -15.6% |

**Worst performing basins** (% of years within ±20%):

| Basin ID | % Within ±20% | Median Error |
|----------|---------------|--------------|
| grdc_3635350 | 0.0% | -91.8% |
| grdc_3637815 | 0.0% | -89.4% |
| grdc_3633100 | 0.0% | -88.4% |

---

## Key Findings

1. **Performance degrades in dry years**: Accuracy within ±20% drops from 23.8% (wet) to 7.3% (dry).

2. **Systematic underestimation**: Median error is negative across all year types except wet years. 63% of basin-years show underestimation.

3. **Error magnitude increases with dryness**: Median error ranges from +3.0% (wet) to -64.3% (dry).

4. **Temperature is key predictor**: Basins with higher temperatures show worse performance (r = -0.345). Hot, dry basins (Low P, High T) perform worst with median error of -57.3%.

5. **Basin-specific variation**: Performance ranges from 0% to 47.4% of years within ±20%, indicating basin characteristics strongly influence accuracy.

---

## Machine Learning Classification

A Random Forest classifier was trained on 97 benchmark basins to predict TURC performance category using 23 basin characteristics (temperature, elevation, slope, precipitation, land cover, aridity, etc.).

**Model performance** (random_state=42):

- Training accuracy: 94%
- Cross-validation accuracy: 65% ± 7%

**Class distribution**:

- Good (within ±20%): 30 basins (30.9%)
- Overestimate (>20%): 23 basins (23.7%)
- Underestimate (<20%): 44 basins (45.4%)

**Top predictive features** (feature importance):

1. Slope (7.9%)
2. Temperature (7.4%)
3. Longitude (7.2%)
4. Elevation (7.1%)
5. Forest cover (7.0%)

---

## Madagascar Basin Assessment

**Basin characteristics**:

- Mean precipitation: 900.3 mm/year
- Mean temperature: 21.4°C
- Mean elevation: 602 m

**Climate bin classification**: Low P, Low T

**Simple climate-based prediction**: TURC likely overestimates discharge (based on +7.4% median error in Low P, Low T bin).

**Random Forest prediction** (random_state=42): Good performance (within ±20%)

- Probability good: 45.6%
- Probability overestimate: 27.3%
- Probability underestimate: 27.1%

The machine learning model, considering multiple basin characteristics beyond temperature and precipitation, predicts Madagascar will fall within the acceptable ±20% range.

---

## Implications for Deep Learning Benchmark

TURC provides a baseline that:

- Achieves 20.5% accuracy within ±20% error bounds
- Shows systematic bias dependent on hydrologic conditions (dry vs wet years)
- Performs poorly in water-scarce conditions (only 7.3% within ±20% in dry years)
- Shows temperature-dependent performance (higher T → worse underestimation)

Deep learning models should target:

- **Primary goal**: >20.5% of predictions within ±20% of observed discharge
- **Robustness**: Consistent performance across dry/normal/wet years (especially dry years)
- **Bias reduction**: Minimize systematic under/overestimation
- **Basin transferability**: Performance less sensitive to basin characteristics

---

## Methods Summary

**TURC formula**:

```
L = 300 + 25T + 0.05T³
AET = P - √(P² / (0.9 + P²/L²))
Q = P - AET
```

Where:

- L = temperature-dependent parameter
- T = mean annual temperature (°C)
- P = annual precipitation (mm)
- AET = actual evapotranspiration (mm)
- Q = discharge (mm)

**Data sources**:

- Precipitation: `total_precipitation_sum` (daily, aggregated annually)
- Temperature: `temperature_2m_mean` (daily, median annually for benchmark, mean from static attributes for Madagascar)
- Observed discharge: `streamflow` (daily mean × 365, excluding filled values)
- Static attributes: CARAVAN basin characteristics

**Analysis outputs**:

- `outputs/analysis_data.parquet`: Full dataset with derived metrics
- `outputs/figures/scatter_20pct_bounds.png`: Performance visualization with ±20% bounds
- `outputs/figures/climate_space_performance.png`: Error vs temperature and precipitation
- `outputs/figures/climate_space_with_ungauged.png`: Madagascar basin context
- Scripts: `scripts/01_prepare_analysis_data.py` through `scripts/05_classifier_prediction.py`
