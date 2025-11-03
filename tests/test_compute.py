"""Unit tests for Turc formula computation."""

import pytest

from turc.compute import compute_turc_discharge


class TestComputeTurcDischarge:
    """Tests for compute_turc_discharge function."""

    def test_typical_warm_wet_climate(self) -> None:
        """Test with typical values for warm, wet climate."""
        result = compute_turc_discharge(annual_precip_mm=1000.0, annual_temp_c=22.5)

        # Check all keys present
        assert "L" in result
        assert "AET_mm" in result
        assert "Q_mm" in result

        # Check reasonable ranges
        assert result["L"] > 0
        assert result["AET_mm"] > 0
        assert result["Q_mm"] >= 0
        assert result["Q_mm"] < result["AET_mm"] + result["Q_mm"]  # Q < P

    def test_cold_climate_low_precipitation(self) -> None:
        """Test with cold climate and low precipitation."""
        result = compute_turc_discharge(annual_precip_mm=300.0, annual_temp_c=5.0)

        assert result["L"] > 0
        assert result["AET_mm"] > 0
        assert result["Q_mm"] >= 0

    def test_very_high_precipitation(self) -> None:
        """Test with very high precipitation."""
        result = compute_turc_discharge(annual_precip_mm=3000.0, annual_temp_c=25.0)

        assert result["L"] > 0
        assert result["AET_mm"] > 0
        assert result["Q_mm"] >= 0
        # In very wet climates, Q should be substantial
        assert result["Q_mm"] > 0

    def test_very_high_temperature(self) -> None:
        """Test with very high temperature."""
        result = compute_turc_discharge(annual_precip_mm=1500.0, annual_temp_c=30.0)

        # L should be large for high temperature (300 + 25*30 + 0.05*30^3)
        assert result["L"] > 1000
        assert result["AET_mm"] > 0
        assert result["Q_mm"] >= 0

    def test_zero_temperature(self) -> None:
        """Test with zero temperature."""
        result = compute_turc_discharge(annual_precip_mm=800.0, annual_temp_c=0.0)

        # L = 300 when T = 0
        assert result["L"] == 300.0
        assert result["AET_mm"] > 0
        assert result["Q_mm"] >= 0

    def test_zero_precipitation(self) -> None:
        """Test with zero precipitation."""
        result = compute_turc_discharge(annual_precip_mm=0.0, annual_temp_c=20.0)

        # If P = 0, then AET = 0 and Q = 0
        assert result["AET_mm"] == 0.0
        assert result["Q_mm"] == 0.0

    def test_negative_temperature(self) -> None:
        """Test with negative temperature."""
        result = compute_turc_discharge(annual_precip_mm=600.0, annual_temp_c=-5.0)

        # L should still be positive (300 + 25*(-5) + 0.05*(-5)^3)
        # L = 300 - 125 - 6.25 = 168.75
        assert result["L"] == pytest.approx(168.75)
        assert result["AET_mm"] > 0
        assert result["Q_mm"] >= 0

    def test_discharge_always_less_than_precipitation(self) -> None:
        """Test that Q ≤ P for various inputs."""
        test_cases = [
            (500.0, 10.0),
            (1000.0, 20.0),
            (1500.0, 25.0),
            (2000.0, 30.0),
            (300.0, 5.0),
        ]

        for precip, temp in test_cases:
            result = compute_turc_discharge(annual_precip_mm=precip, annual_temp_c=temp)
            # Q = P - AET, so Q should always be ≤ P
            assert result["Q_mm"] <= precip, f"Q ({result['Q_mm']}) > P ({precip}) for T={temp}"

    def test_water_balance_closure(self) -> None:
        """Test that P = AET + Q (water balance closure)."""
        P = 1200.0
        T = 22.0
        result = compute_turc_discharge(annual_precip_mm=P, annual_temp_c=T)

        # P = AET + Q (within floating point precision)
        assert P == pytest.approx(result["AET_mm"] + result["Q_mm"], rel=1e-9)

    def test_aet_increases_with_temperature(self) -> None:
        """Test that AET increases with temperature (holding P constant)."""
        P = 1000.0

        result_cold = compute_turc_discharge(annual_precip_mm=P, annual_temp_c=10.0)
        result_warm = compute_turc_discharge(annual_precip_mm=P, annual_temp_c=25.0)

        # Higher temperature should lead to higher AET
        assert result_warm["AET_mm"] > result_cold["AET_mm"]

    def test_discharge_increases_with_precipitation(self) -> None:
        """Test that Q increases with precipitation (holding T constant)."""
        T = 20.0

        result_dry = compute_turc_discharge(annual_precip_mm=500.0, annual_temp_c=T)
        result_wet = compute_turc_discharge(annual_precip_mm=2000.0, annual_temp_c=T)

        # Higher precipitation should lead to higher discharge
        assert result_wet["Q_mm"] > result_dry["Q_mm"]

    def test_return_type(self) -> None:
        """Test that function returns correct dictionary structure."""
        result = compute_turc_discharge(annual_precip_mm=1000.0, annual_temp_c=20.0)

        assert isinstance(result, dict)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result.values())

    def test_known_values(self) -> None:
        """Test with known reference values."""
        # Test case from the original script
        P = 1000.0
        T = 20.0

        result = compute_turc_discharge(annual_precip_mm=P, annual_temp_c=T)

        # L = 300 + 25*20 + 0.05*20^3 = 300 + 500 + 400 = 1200
        assert result["L"] == pytest.approx(1200.0)

        # AET = P / sqrt(0.9 + (P/L)^2)
        # AET = 1000 / sqrt(0.9 + (1000/1200)^2)
        # AET = 1000 / sqrt(0.9 + 0.6944) = 1000 / sqrt(1.5944) = 1000 / 1.2627 ≈ 792.2
        expected_aet = P / (0.9 + (P / 1200.0) ** 2) ** 0.5
        assert result["AET_mm"] == pytest.approx(expected_aet, rel=1e-6)

        # Q = P - AET
        expected_q = P - expected_aet
        assert result["Q_mm"] == pytest.approx(expected_q, rel=1e-6)
