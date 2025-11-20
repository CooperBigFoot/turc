"""TurcEstimator class for computing annual discharge from CARAVAN basins."""

import logging
from pathlib import Path

import polars as pl
from transfer_learning_publication.data import CaravanDataSource

from .compute import compute_turc_discharge

logger = logging.getLogger(__name__)


class TurcEstimator:
    """Estimate annual discharge using the Turc formula for CARAVAN basins.

    This class handles loading basin IDs, fetching timeseries data from CARAVAN,
    computing annual aggregates, and applying the Turc formula to estimate discharge.

    Attributes:
        caravan_path: Path to CARAVAN dataset
        basin_list_path: Path to text file with basin IDs (one per line)
        batch_size: Number of basins to process in each batch
        results: Computed results as polars DataFrame (None until compute_annual() is called)

    Example:
        >>> estimator = TurcEstimator(
        ...     caravan_path="/path/to/CARAVAN",
        ...     basin_list_path="basins.txt"
        ... )
        >>> results = estimator.compute_annual()
        >>> estimator.to_parquet("output.parquet")
    """

    def __init__(
        self,
        caravan_path: str,
        basin_list_path: str,
        batch_size: int = 100,
        use_pet: bool = False,
        pet_column: str = "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
    ) -> None:
        """Initialize TurcEstimator.

        Args:
            caravan_path: Path to CARAVAN dataset directory
            basin_list_path: Path to text file containing basin IDs (one per line)
            batch_size: Number of basins to process per batch (default: 100)
            use_pet: Whether to use PET instead of temperature-derived L (default: False)
            pet_column: Which PET column to use from CARAVAN (default: FAO Penman-Monteith)

        Raises:
            FileNotFoundError: If basin_list_path does not exist
            ValueError: If caravan_path is invalid
        """
        self.caravan_path = Path(caravan_path)
        self.basin_list_path = Path(basin_list_path)
        self.batch_size = batch_size
        self.use_pet = use_pet
        self.pet_column = pet_column
        self.results: pl.DataFrame | None = None

        # Validate paths
        if not self.basin_list_path.exists():
            raise FileNotFoundError(f"Basin list file not found: {basin_list_path}")

        if not self.caravan_path.exists():
            raise ValueError(f"CARAVAN path does not exist: {caravan_path}")

        # Load basin IDs
        self._gauge_ids = self._load_basin_ids()

        # Initialize data source
        self._data_source = CaravanDataSource(str(self.caravan_path))

    def _load_basin_ids(self) -> list[str]:
        """Load basin IDs from text file.

        Returns:
            List of gauge IDs (one per line, stripped of whitespace)

        Raises:
            ValueError: If file is empty or contains no valid IDs
        """
        with open(self.basin_list_path) as f:
            gauge_ids = [line.strip() for line in f if line.strip()]

        if not gauge_ids:
            raise ValueError(f"No basin IDs found in {self.basin_list_path}")

        logger.info(f"Loaded {len(gauge_ids)} basin IDs from {self.basin_list_path}")
        return gauge_ids

    def _process_basin(self, gauge_id: str, df: pl.DataFrame, use_pet: bool = False) -> pl.DataFrame:
        """Process a single basin to compute annual discharge.

        Args:
            gauge_id: Gauge identifier
            df: DataFrame with daily timeseries data for this basin
            use_pet: Whether to use PET instead of temperature-derived L

        Returns:
            DataFrame with annual values and Turc estimates
        """
        # Filter to this basin
        basin_df = df.filter(pl.col("gauge_id") == gauge_id)

        # Extract year from date
        basin_df = basin_df.with_columns(pl.col("date").dt.year().alias("year"))

        # Group by year and compute annual aggregates
        # For streamflow, only use valid observations (not filled)
        agg_exprs = [
            pl.col("total_precipitation_sum").sum().alias("annual_precip_mm"),
            pl.col("temperature_2m_mean").mean().alias("annual_temp_c"),
            # Calculate mean of valid streamflow values and multiply by 365
            pl.col("streamflow")
            .filter(pl.col("streamflow_was_filled") == 0)
            .mean()
            .mul(365)
            .alias("Q_observed_mm"),
            # Count valid streamflow days
            (pl.col("streamflow_was_filled") == 0).sum().alias("n_valid_days"),
            pl.len().alias("n_days"),
        ]

        # Add PET aggregation if needed
        if use_pet:
            agg_exprs.append(pl.col("PM_PET").sum().alias("annual_pet_mm"))

        annual = basin_df.group_by("year").agg(agg_exprs)

        # Filter to complete years only (365 or 366 days)
        annual = annual.filter(pl.col("n_days") >= 365)

        # Apply Turc formula to each year
        results = []
        for row in annual.iter_rows(named=True):
            # Pass PET if available
            pet_value = row.get("annual_pet_mm") if use_pet else None
            turc_values = compute_turc_discharge(
                row["annual_precip_mm"], row["annual_temp_c"], annual_pet_mm=pet_value
            )

            result_dict = {
                "gauge_id": gauge_id,
                "year": row["year"],
                "annual_precip_mm": row["annual_precip_mm"],
                "annual_temp_c": row["annual_temp_c"],
                "L": turc_values["L"],
                "AET_turc_mm": turc_values["AET_mm"],
                "Q_turc_mm": turc_values["Q_mm"],
                "Q_observed_mm": row["Q_observed_mm"],
                "n_days": row["n_days"],
                "n_valid_days": row["n_valid_days"],
            }

            # Add PET to output if used
            if use_pet:
                result_dict["annual_pet_mm"] = row["annual_pet_mm"]

            results.append(result_dict)

        return pl.DataFrame(results)

    def _process_batch(self, gauge_ids: list[str], use_pet: bool | None = None) -> pl.DataFrame:
        """Process a batch of basins.

        Args:
            gauge_ids: List of gauge IDs to process
            use_pet: Override class-level use_pet setting (optional)

        Returns:
            DataFrame with annual discharge estimates for all basins in batch
        """
        # Use provided use_pet or fall back to class attribute
        _use_pet = use_pet if use_pet is not None else self.use_pet

        # Build column list
        columns = [
            "total_precipitation_sum",
            "temperature_2m_mean",
            "streamflow",
            "streamflow_was_filled",
        ]

        if _use_pet:
            columns.append(self.pet_column)

        # Load timeseries for batch
        lazy_ts = self._data_source.get_timeseries(
            gauge_ids=gauge_ids,
            columns=columns,
        )

        # Rename PET column to PM_PET for clarity
        if _use_pet:
            df = lazy_ts.rename({self.pet_column: "PM_PET"}).collect()
        else:
            df = lazy_ts.collect()

        # Process each basin
        batch_results = []
        for gauge_id in gauge_ids:
            basin_annual = self._process_basin(gauge_id, df, use_pet=_use_pet)
            if len(basin_annual) > 0:
                batch_results.append(basin_annual)

        # Combine all basins
        if batch_results:
            return pl.concat(batch_results)
        else:
            return pl.DataFrame()

    def compute_annual(self, use_pet: bool | None = None) -> pl.DataFrame:
        """Compute annual discharge for all basins using the Turc formula.

        This method loads timeseries data in batches, aggregates to annual values,
        and applies the Turc formula. Results are stored in self.results and returned.

        Args:
            use_pet: Override class-level use_pet setting (optional)

        Returns:
            DataFrame with columns:
                - gauge_id: Basin identifier
                - year: Calendar year
                - annual_precip_mm: Annual precipitation sum (mm)
                - annual_temp_c: Mean annual temperature (°C)
                - L: Turc temperature parameter (or PET if use_pet=True)
                - AET_turc_mm: Actual evapotranspiration (Turc estimate, mm)
                - Q_turc_mm: Annual discharge (Turc estimate, mm)
                - Q_observed_mm: Observed annual discharge (mm)
                - n_days: Total days in year
                - n_valid_days: Number of valid (non-filled) observations
                - annual_pet_mm: Annual PET sum (only if use_pet=True)
        """
        logger.info(f"Computing annual discharge for {len(self._gauge_ids)} basins")

        all_results = []

        for i in range(0, len(self._gauge_ids), self.batch_size):
            batch = self._gauge_ids[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(self._gauge_ids) + self.batch_size - 1) // self.batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches}: "
                f"gauges {i} to {i + len(batch)} of {len(self._gauge_ids)}"
            )

            batch_results = self._process_batch(batch, use_pet=use_pet)
            if len(batch_results) > 0:
                all_results.append(batch_results)

        # Combine all results
        logger.info("Combining results from all batches")
        self.results = pl.concat(all_results)

        # Log summary
        logger.info(f"Computed {len(self.results)} basin-years across {self.results['gauge_id'].n_unique()} basins")
        logger.info(f"Year range: {self.results['year'].min()} to {self.results['year'].max()}")

        return self.results

    def to_parquet(self, path: str) -> None:
        """Save computed results to parquet file.

        Args:
            path: Output path for parquet file

        Raises:
            RuntimeError: If compute_annual() has not been called yet
        """
        if self.results is None:
            raise RuntimeError(
                "No results to save. Call compute_annual() first."
            )

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(self.results)} rows to {output_path}")
        self.results.write_parquet(output_path)
        logger.info("Save complete")
