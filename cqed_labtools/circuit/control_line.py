"""Control-line models for transmon circuits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import E_CHARGE
from .transmon import Transmon

@dataclass(frozen=True)
class ChargeDriveLine:
    """Capacitive charge/XY drive line attached to the transmon.

    Parameters
    ----------
    coupling_capacitance:
        Coupling capacitance to the transmon island in F.
    attenuation_db:
        Power attenuation from room temperature to chip in dB.
    impedance:
        Line impedance in ohms.
    """

    coupling_capacitance: float
    attenuation_db: float = 0.0
    impedance: float = 50.0

    def __post_init__(self) -> None:
        if self.coupling_capacitance <= 0:
            raise ValueError("coupling_capacitance must be positive")
        if self.attenuation_db < 0:
            raise ValueError("attenuation_db must be non-negative")
        if self.impedance <= 0:
            raise ValueError("impedance must be positive")

    @property
    def voltage_attenuation_factor(self) -> float:
        """Voltage amplitude factor V_chip / V_room."""
        return 10.0 ** (-self.attenuation_db / 20.0)

    @property
    def power_attenuation_factor(self) -> float:
        """Power factor P_chip / P_room."""
        return 10.0 ** (-self.attenuation_db / 10.0)

    def voltage_at_chip(self, voltage_room: float) -> float:
        """Convert room-temperature voltage amplitude to chip voltage amplitude."""
        return voltage_room * self.voltage_attenuation_factor

    def power_at_chip(self, power_room_w: float) -> float:
        """Convert room-temperature power to chip power in W."""
        return power_room_w * self.power_attenuation_factor

    def chip_voltage_from_power(self, power_w: float, rms: bool = True) -> float:
        """Voltage amplitude at chip from available line power.

        If rms=True, returns Vrms = sqrt(P Z). If rms=False, returns the peak
        sinusoidal voltage sqrt(2 P Z).
        """
        if power_w < 0:
            raise ValueError("power_w must be non-negative")
        v_rms = np.sqrt(power_w * self.impedance)
        return v_rms if rms else np.sqrt(2.0) * v_rms

    def drive_charge_amplitude(self, voltage: float, voltage_is_at_chip: bool = True) -> float:
        """Dimensionless AC gate charge amplitude n_g,ac = C_d V / 2e."""
        if not voltage_is_at_chip:
            voltage = self.voltage_at_chip(voltage)
        return self.coupling_capacitance * voltage / (2.0 * E_CHARGE)

    def rabi_frequency_from_voltage(
        self,
        transmon: Transmon,
        voltage: float,
        voltage_is_at_chip: bool = True,
    ) -> float:
        """Estimate resonant Rabi frequency f_R in Hz from voltage amplitude.

        Uses H_drive/h = -8 EC/h n_g,ac n, so f_R = 8 EC/h |<0|n|1>| n_g,ac.
        """
        n_g_ac = self.drive_charge_amplitude(voltage=voltage, voltage_is_at_chip=voltage_is_at_chip)
        return 8.0 * transmon.EC_over_h * abs(transmon.n_matrix_element(0, 1)) * n_g_ac

    def rabi_frequency_from_power(
        self,
        transmon: Transmon,
        power_w: float,
        power_is_at_chip: bool = True,
        rms_voltage: bool = False,
    ) -> float:
        """Estimate Rabi frequency f_R in Hz from microwave power.

        By default rms_voltage=False, meaning the drive amplitude is interpreted
        as peak sinusoidal voltage. Set rms_voltage=True if your convention uses Vrms.
        """
        if not power_is_at_chip:
            power_w = self.power_at_chip(power_w)
        voltage = self.chip_voltage_from_power(power_w, rms=rms_voltage)
        return self.rabi_frequency_from_voltage(transmon, voltage=voltage, voltage_is_at_chip=True)