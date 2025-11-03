"""Turc formula for annual discharge estimation.

This package provides tools to estimate annual discharge using the Turc water
balance formula. The main interface is the TurcEstimator class, which handles
integration with the CARAVAN hydrological dataset.

Example:
    >>> from turc import TurcEstimator
    >>> estimator = TurcEstimator(
    ...     caravan_path="/path/to/CARAVAN",
    ...     basin_list_path="basins.txt"
    ... )
    >>> results = estimator.compute_annual()
    >>> estimator.to_parquet("output.parquet")
"""

from .compute import compute_turc_discharge
from .estimator import TurcEstimator

__all__ = ["TurcEstimator", "compute_turc_discharge"]
