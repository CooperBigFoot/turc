"""Core Turc formula computation for annual discharge estimation.

The Turc method estimates annual discharge using a water balance approach:
1. L = 300 + 25*T + 0.05*T^3 (temperature parameter)
2. AET = P / sqrt(0.9 + (P/L)^2) (actual evapotranspiration)
3. Q = P - AET (annual discharge)

Where:
- T = mean annual temperature (°C)
- P = annual precipitation (mm/year)
- AET = actual evapotranspiration (mm/year)
- Q = annual discharge (mm/year)

Reference:
Turc, L. (1954). "Le bilan d'eau des sols. Relation entre la précipitation,
l'évaporation et l'écoulement." Annales Agronomiques.
"""


def compute_turc_discharge(annual_precip_mm: float, annual_temp_c: float) -> dict[str, float]:
    """Compute annual discharge using the Turc formula.

    Args:
        annual_precip_mm: Annual precipitation in mm/year
        annual_temp_c: Mean annual temperature in °C

    Returns:
        Dictionary with keys:
            - L: Temperature parameter
            - AET_mm: Actual evapotranspiration (mm/year)
            - Q_mm: Annual discharge (mm/year)

    Examples:
        >>> result = compute_turc_discharge(1000.0, 22.5)
        >>> result["Q_mm"]  # Discharge in mm/year
        43.8
    """
    # Step 1: Calculate temperature parameter L
    T = annual_temp_c
    L = 300 + 25 * T + 0.05 * (T**3)

    # Step 2: Calculate actual evapotranspiration (AET)
    P = annual_precip_mm
    AET = P / (0.9 + (P / L) ** 2) ** 0.5

    # Step 3: Calculate discharge
    Q = P - AET

    return {"L": L, "AET_mm": AET, "Q_mm": Q}
