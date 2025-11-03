# Turc Formula for Annual Discharge Estimation

## Overview

The Turc method estimates annual discharge using a water balance approach: precipitation minus actual evapotranspiration.

## Formula

**Step 1: Calculate the temperature parameter L**

$$L = 300 + 25T + 0.05T^3$$

Where:

- T = mean annual temperature (°C)

**Step 2: Calculate actual evapotranspiration (AET)**

$$H = P - \sqrt{\frac{P^2}{0.9 + \frac{P^2}{L^2}}}$$

Where:

- H = actual evapotranspiration (mm/year)
- P = annual precipitation (mm/year)
- L = parameter from Step 1

**Step 3: Calculate discharge**

$$Q = P - H$$

Where:

- Q = annual discharge (mm/year)
- P = annual precipitation (mm/year)
- H = actual evapotranspiration from Step 2

## Units

- All values in mm/year
- To convert to volumetric discharge: multiply by basin area and convert units appropriately
