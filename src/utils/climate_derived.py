"""
climate_derived.py
------------------
Derived climate variables, shared by every step that computes them from the monthly
climate (training set, figures, projections), so all use the same formulas.

month_lengths(year)                     days in each month of the year
precip_mm(pr, days)                     precipitation, kg m-2 s-1 -> mm month-1
pet_priestley_taylor(tas, rsds, days)   potential evapotranspiration, mm month-1
climatic_water_balance(pr, pet, days)   P - PET, mm month-1

Potential evapotranspiration, Priestley–Taylor:
    PET = alpha * Delta / (Delta + gamma) * Rn / lambda * days
    Delta = 4098 * e_s(T) / (T + 237.3)^2,  e_s(T) = 0.6108 * exp(17.27 T / (T + 237.3))
with T = tas in degC, alpha = 1.26, gamma = 0.066 kPa degC-1, lambda = 2.45 MJ kg-1 and
Rn = (1 - 0.23) * rsds (net shortwave radiation), converted from W m-2 to MJ m-2 day-1.

Inputs are monthly means in the units of the climate files (tas K, rsds W m-2,
pr kg m-2 s-1); `days` is the number of days in each value's month.
"""

import calendar

import numpy as np

ALPHA = 1.26            # Priestley–Taylor coefficient
GAMMA = 0.066           # psychrometric constant, kPa degC-1
LAMBDA = 2.45           # latent heat of vaporisation, MJ kg-1
ALBEDO = 0.23
W_M2_TO_MJ_DAY = 0.0864
SECONDS_PER_DAY = 86400


def month_lengths(year):
    """Days in each month of the year (index 0 = January)."""
    return np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])


def precip_mm(pr, days):
    """Monthly precipitation (mm month-1) from the mean rate pr (kg m-2 s-1)."""
    return pr * SECONDS_PER_DAY * days


def pet_priestley_taylor(tas, rsds, days):
    """Monthly potential evapotranspiration (mm month-1)."""
    t = tas - 273.15
    e_s = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    delta = 4098 * e_s / (t + 237.3) ** 2
    rn = (1 - ALBEDO) * rsds * W_M2_TO_MJ_DAY
    return ALPHA * delta / (delta + GAMMA) * rn / LAMBDA * days


def climatic_water_balance(pr, pet, days):
    """Monthly climatic water balance P - PET (mm month-1)."""
    return precip_mm(pr, days) - pet
