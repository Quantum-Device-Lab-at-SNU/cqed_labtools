"""Small unit helpers. Frequencies are Hz unless names say angular/rad/s."""

import math
import numpy as np


def hz_to_angular(freq_hz: float) -> float:
    return 2.0 * np.pi * freq_hz


def angular_to_hz(omega: float) -> float:
    return omega / (2.0 * np.pi)


def watt_to_dbm(power_w: float) -> float:
    if power_w <= 0:
        raise ValueError("power_w must be positive")
    return 10.0 * np.log10(power_w / 1e-3)


def dbm_to_watt(power_dbm: float) -> float:
    return 1e-3 * 10.0 ** (power_dbm / 10.0)


def db_to_linear_power(db: float) -> float:
    return 10.0 ** (db / 10.0)


def linear_power_to_db(linear: float) -> float:
    if linear <= 0:
        raise ValueError("linear factor must be positive")
    return 10.0 * math.log10(linear)
