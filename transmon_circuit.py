from dataclasses import dataclass
import numpy as np
from control_line import ChargeDriveLine
from readout_resonator import ReadoutResonator
from transmon import Transmon
    
@dataclass
class TransmonCircuit:
    """Single-transmon circuit composed of optional attached components."""

    transmon: Transmon
    readout_resonator: ReadoutResonator | None = None
    charge_drive_line: ChargeDriveLine | None = None

    def _require_readout(self) -> ReadoutResonator:
        if self.readout_resonator is None:
            raise ValueError("No readout_resonator is attached")
        return self.readout_resonator

    def _require_drive_line(self) -> ChargeDriveLine:
        if self.charge_drive_line is None:
            raise ValueError("No charge_drive_line is attached")
        return self.charge_drive_line

    @property
    def f_qubit_bare(self) -> float:
        return self.transmon.f01

    @property
    def anharmonicity(self) -> float:
        return self.transmon.anharmonicity

    @property
    def f_resonator_bare(self) -> float:
        return self._require_readout().bare_frequency

    @property
    def g_over_2pi(self) -> float:
        resonator = self._require_readout()
        if resonator.g_over_2pi is None:
            raise ValueError("readout_resonator.g_over_2pi is required for dispersive properties")
        return resonator.g_over_2pi

    @property
    def qr_detuning(self) -> float:
        """Qubit-readout detuning Delta/2pi = f_q - f_r in Hz."""
        return self.f_qubit_bare - self.f_resonator_bare

    @property
    def qr_straddling_detuning(self) -> float:
        return self.qr_detuning + self.anharmonicity

    def _check_dispersive_denominators(self) -> None:
        if np.isclose(self.qr_detuning, 0.0):
            raise ValueError("Qubit and readout resonator are too close to resonance")
        if np.isclose(self.readout_straddling_detuning, 0.0):
            raise ValueError("Delta + anharmonicity is too close to zero")

    @property
    def chi_readout_over_2pi(self) -> float:
        """Transmon dispersive shift chi/2pi in Hz."""
        self._check_dispersive_denominators()
        return (
            self.g_over_2pi**2 * self.anharmonicity
            / (self.qr_detuning * self.qr_straddling_detuning)
        )

    @property
    def readout_state_separation(self) -> float:
        """Resonator frequency separation f_e - f_g = 2 chi/2pi in Hz."""
        return 2.0 * self.chi_readout_over_2pi

    @property
    def f_resonator_dressed(self) -> float:
        """Reference dressed readout frequency before state-dependent +/- chi."""
        self._check_dispersive_denominators()
        return self.f_resonator_bare - self.g_over_2pi**2 / self.qr_straddling_detuning

    @property
    def f_resonator_ground(self) -> float:
        """Resonator frequency conditioned on transmon |g>."""
        return self.f_resonator_dressed - self.chi_readout_over_2pi

    @property
    def f_resonator_excited(self) -> float:
        """Resonator frequency conditioned on transmon |e>."""
        return self.f_resonator_dressed + self.chi_readout_over_2pi

    @property
    def f_qubit_dressed(self) -> float:
        """Approximate Lamb-shifted qubit frequency."""
        self._check_dispersive_denominators()
        return self.f_qubit + self.g_over_2pi**2 / self.qr_detuning 

    @property
    def critical_photon_number(self) -> float:
        return self.qr_detuning**2 / (4.0 * self.g_over_2pi**2)

    @property
    def purcell_decay_rate_over_2pi(self) -> float:
        """Purcell decay rate Gamma_P/2pi in Hz."""
        readout = self._require_readout()
        return (self.g_over_2pi / self.qr_detuning) ** 2 * readout.kappa_over_2pi

    @property
    def purcell_decay_rate(self) -> float:
        """Purcell decay rate in rad/s."""
        return 2.0 * np.pi * self.purcell_decay_rate_over_2pi

    @property
    def purcell_limited_t1(self) -> float:
        """Purcell-limited T1 in seconds."""
        return 1.0 / self.purcell_decay_rate

    @property
    def dispersive_validity_ratio(self) -> float:
        return abs(self.g_over_2pi / self.qr_detuning)

    @property
    def straddling_validity_ratio(self) -> float:
        return abs(self.g_over_2pi / self.qr_straddling_detuning)

    def _readout_with_frequency(self, frequency: float) -> ReadoutResonator:
        r = self._require_readout()
        return ReadoutResonator(
            bare_frequency=frequency,
            kappa_internal_over_2pi=r.kappa_internal_over_2pi,
            coupling_geometry=r.coupling_geometry,
            kappa_input_over_2pi=r.kappa_input_over_2pi,
            kappa_output_over_2pi=r.kappa_output_over_2pi,
            g_over_2pi=r.g_over_2pi,
        )

    @property
    def readout_resonator_ground(self) -> ReadoutResonator:
        return self._readout_with_frequency(self.f_readout_ground)

    @property
    def readout_resonator_excited(self) -> ReadoutResonator:
        return self._readout_with_frequency(self.f_readout_excited)

    def ground_state_field(self, f_drive: float, input_field: complex) -> complex:
        return self.readout_resonator_ground.intracavity_field(f_drive, input_field)

    def excited_state_field(self, f_drive: float, input_field: complex) -> complex:
        return self.readout_resonator_excited.intracavity_field(f_drive, input_field)

    def measurement_response_separation(self, f_drive: float, input_field: complex) -> complex:
        return self.excited_state_field(f_drive, input_field) - self.ground_state_field(f_drive, input_field)

    def measurement_response_separation_abs(self, f_drive: float, input_field: complex) -> float:
        return abs(self.measurement_response_separation(f_drive, input_field))

    def ground_state_n_photon_from_input_power(self, f_drive: float, input_power: float) -> float:
        return self.readout_resonator_ground.n_photon_from_input_power(f_drive, input_power)

    def excited_state_n_photon_from_input_power(self, f_drive: float, input_power: float) -> float:
        return self.readout_resonator_excited.n_photon_from_input_power(f_drive, input_power)

    def input_power_for_ground_state_n_photon(self, f_drive: float, n_photon: float) -> float:
        return self.readout_resonator_ground.input_power_for_n_photon(f_drive, n_photon)

    def input_power_for_excited_state_n_photon(self, f_drive: float, n_photon: float) -> float:
        return self.readout_resonator_excited.input_power_for_n_photon(f_drive, n_photon)

    def rabi_frequency_from_charge_voltage(self, voltage: float, voltage_is_at_chip: bool = True) -> float:
        """Estimate f_R in Hz from the attached charge-drive line."""
        return self._require_drive_line().rabi_frequency_from_voltage(
            transmon=self.transmon,
            voltage=voltage,
            voltage_is_at_chip=voltage_is_at_chip,
        )

    def rabi_frequency_from_charge_power(
        self,
        power_w: float,
        power_is_at_chip: bool = True,
        rms_voltage: bool = False,
    ) -> float:
        """Estimate f_R in Hz from microwave power on the attached charge-drive line."""
        return self._require_drive_line().rabi_frequency_from_power(
            transmon=self.transmon,
            power_w=power_w,
            power_is_at_chip=power_is_at_chip,
            rms_voltage=rms_voltage,
        )

    @classmethod
    def from_f01_anharmonicity(
        cls,
        f01: float,
        anharmonicity: float,
        readout_resonator: ReadoutResonator | None = None,
        charge_drive_line: ChargeDriveLine | None = None,
        ng: float = 0.0,
        n_cutoff: int = 30,
    ) -> "TransmonCircuit":
        transmon = Transmon.from_f01_anharmonicity(
            f01=f01,
            anharmonicity=anharmonicity,
            ng=ng,
            n_cutoff=n_cutoff,
        )
        return cls(
            transmon=transmon,
            readout_resonator=readout_resonator,
            charge_drive_line=charge_drive_line,
        )
