from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from circuit.transmon import BareTransmon

@dataclass(frozen=True)
class TransmonDispersiveReadout:
    """Dispersive readout parameters for a transmon-resonator system.

    All frequency-like quantities are expressed in Hz, not angular frequency.

    Parameters
    ----------
    transmon: BareTransmon
        The transmon qubit for which to model dispersive readout.
    fr_bare:
        Bare readout resonator frequency fr (corresponds to the resonator frequency when the qubit is punched out).
    g_over_2pi:
        Qubit-resonator coupling strength g / 2pi.
    kappa_over_2pi:
        Resonator energy decay rate kappa / 2pi.
    """
    transmon: BareTransmon
    fr_bare: float
    g_over_2pi: float
    kappa_over_2pi: float

    def __post_init__(self) -> None:
        if self.fq_bare <= 0:
            raise ValueError("fq_bare must be positive")
        if self.fr_bare <= 0:
            raise ValueError("fr_bare must be positive")
        if self.g_over_2pi <= 0:
            raise ValueError("g_over_2pi must be positive")
        if self.kappa_over_2pi <= 0:
            raise ValueError("kappa_over_2pi must be positive")
        if self.anharmonicity >= 0:
            raise ValueError("anharmonicity should be negative for a transmon")
        if np.isclose(self.detuning, 0.0):
            raise ValueError("qubit and resonator are too close to resonance")
        if np.isclose(self.detuning_hz + self.anharmonicity_hz, 0.0):
            raise ValueError("detuning_hz + anharmonicity_hz is too close to zero")

    @classmethod
    def from_dressed(
        cls,
        fq_dressed: float,
        anharmonicity: float,
        fr_dressed: float,
        chi_over_2pi: float,
        kappa_over_2pi: float,
    ) -> "TransmonDispersiveReadout":
        """Construct from measured dressed quantities.

            Parameters
            ----------
            dressed_qubit_frequency_hz:
                Measured Lamb-shifted qubit frequency.
            anharmonicity_hz:
                Measured or estimated transmon anharmonicity.
            dressed_resonator_frequency_hz:
                Measured dressed readout resonator frequency.
            dispersive_shift_hz:
                Transmon dispersive shift chi.

                This assumes the convention

                    f_r,|1> - f_r,|0> = 2 chi.

                If your measured resonator separation is 2 chi, pass half of it.
            resonator_kappa_hz:
                Resonator linewidth / energy decay rate in Hz.

            Returns
            -------
            TransmonDispersiveReadout
                Object initialized with inferred bare qubit frequency,
                bare resonator frequency, and coupling.
        """
        if fq_dressed <= 0:
            raise ValueError("fq_dressed must be positive")
        if fr_dressed <= 0:
            raise ValueError("fr_dressed must be positive")
        if anharmonicity >= 0:
            raise ValueError("anharmonicity should be negative for a transmon")
        if kappa_over_2pi <= 0:
            raise ValueError("kappa_over_2pi must be positive")
        if chi_over_2pi == 0:
            raise ValueError("chi_over_2pi must be nonzero")

        alpha = anharmonicity * 2 * np.pi
        chi = chi_over_2pi * 2 * np.pi

        dressed_detuning = fq_dressed - fr_dressed

        denominator = 1.0 + 2.0 * chi_over_2pi / alpha_over_2pi
        if np.isclose(denominator, 0.0):
            raise ValueError("Cannot infer bare detuning: singular denominator")

        detuning_hz = (dressed_detuning - chi_over_2pi) / denominator
        coupling_squared_hz2 = (
            chi_over_2pi
            * detuning_hz
            * (detuning_hz + alpha_over_2pi)
            / alpha_over_2pi
        )

        if coupling_squared_hz2 <= 0:
            raise ValueError(
                "Inferred coupling_hz^2 is non-positive. "
                "Check the sign convention of dispersive_shift_hz."
            )

        coupling_hz = np.sqrt(coupling_squared_hz2)

        bare_qubit_frequency_hz = (
            dressed_qubit_frequency_hz
            - coupling_squared_hz2 / detuning_hz
        )

        bare_resonator_frequency_hz = (
            dressed_resonator_frequency_hz
            + coupling_squared_hz2 / (detuning_hz + alpha_over_2pi)
        )

        return cls(
            fq_bare=bare_qubit_frequency_hz,
            fr_bare=bare_resonator_frequency_hz,
            g_over_2pi=coupling_hz,
            kappa_over_2pi=resonator_kappa_hz,
            anharmonicity=alpha_over_2pi,
        )


    @property
    def detuning(self) -> float:
        """Qubit-resonator bare detuning f_q - f_r."""
        return self.fq_bare - self.fr_bare

    @property
    def straddling_detuning(self) -> float:
        """Delta + alpha.

        This denominator appears in the transmon dispersive shift.
        """
        return self.detuning + self.anharmonicity

    @property
    def tau_r(self) -> float:
        """Resonator energy ringdown time, 1 / kappa."""
        return 1.0 / (2 * np.pi * self.kappa_over_2pi)

    @property
    def fr_dressed(self) -> float:
        """Approximate dressed resonator frequency.

        Uses the perturbative expression

            omega_r,dressed = omega_r,bare - g^2 / (Delta + alpha)
        """
        return (
            self.fr_bare - self.g_over_2pi**2 / self.straddling_detuning
        )

    @property
    def fq_dressed(self) -> float:
        """Approximate dressed qubit frequency.

        Uses the perturbative expression

            omega_q,dressed = omega_q,bare + g^2 / Delta
        """
        return (
            self.fq_bare
            + self.g_over_2pi**2 / self.detuning
        )

    @property
    def chi_over_2pi(self) -> float:
        """Transmon dispersive shift chi / 2pi.

        Uses

            chi = g^2 alpha / [Delta (Delta + alpha)]

        All quantities are in Hz, so the output is also in Hz.
        """
        return (
            self.g_over_2pi**2
            * self.anharmonicity
            / (self.detuning * self.straddling_detuning)
        )

    @property
    def fr_qubit_g(self) -> float:
        """Readout resonator frequency conditioned on qubit state |0>."""
        return self.fr_dressed - self.chi_over_2pi

    @property
    def fr_qubit_e(self) -> float:
        """Readout resonator frequency conditioned on qubit state |1>."""
        return self.fr_dressed + self.chi_over_2pi

    @property
    def fr_qubit_f(self) -> float:
        """Readout resonator frequency conditioned on qubit state |2>."""
        return self.fr_dressed + 2 * self.chi_over_2pi


    @property
    def resonator_state_separation_hz(self) -> float:
        """Frequency separation between the |0> and |1> resonator responses.

        This is 2 chi.
        """
        return 2.0 * self.chi_over_2pi

    @property
    def critical_photon_number(self) -> float:
        """Critical photon number Delta^2 / (4 g^2)."""
        return self.detuning**2 / (4.0 * self.g_over_2pi**2)

    @property
    def purcell_decay_rate_hz(self) -> float:
        """Approximate Purcell decay rate.

        Uses

            Gamma_P = (g / Delta)^2 kappa
        """
        return (
            (self.g_over_2pi / self.detuning) ** 2
            * self.resonator_kappa_hz
        )

    @property
    def purcell_limited_t1_s(self) -> float:
        """Purcell-limited qubit lifetime."""
        return 1.0 / (2 * np.pi * self.purcell_decay_rate_hz)

    @property
    def dispersive_validity_ratio(self) -> float:
        """Simple dispersive validity ratio |g / Delta|."""
        return abs(self.g_over_2pi / self.detuning)

    @property
    def straddling_validity_ratio(self) -> float:
        """Validity ratio |g / (Delta + alpha)|."""
        return abs(self.g_over_2pi / self.straddling_detuning)

    def ground_state_field(
        self,
        drive_frequency_hz: float,
        drive_amplitude: complex,
    ) -> complex:
        """Steady-state resonator field amplitude for qubit state |0>.

        Parameters
        ----------
        drive_frequency_hz:
            Readout drive frequency.
        drive_amplitude:
            Effective resonator drive amplitude epsilon.

        Returns
        -------
        complex
            Steady-state coherent amplitude alpha_0.
        """
        detuning = self.dressed_resonator_frequency_hz - drive_frequency_hz

        return -drive_amplitude / (
            detuning
            - self.chi_over_2pi
            - 1j * self.resonator_kappa_hz / 2.0
        )

    def excited_state_field(
        self,
        drive_frequency_hz: float,
        drive_amplitude: complex,
    ) -> complex:
        """Steady-state resonator field amplitude for qubit state |1>."""
        detuning = self.dressed_resonator_frequency_hz - drive_frequency_hz

        return -drive_amplitude / (
            detuning
            + self.chi_over_2pi
            - 1j * self.resonator_kappa_hz / 2.0
        )

    def ground_state_photon_number(
        self,
        drive_frequency_hz: float,
        drive_amplitude: complex,
    ) -> float:
        """Steady-state resonator photon number for qubit state |0>."""
        return abs(self.ground_state_field(drive_frequency_hz, drive_amplitude)) ** 2

    def excited_state_photon_number(
        self,
        drive_frequency_hz: float,
        drive_amplitude: complex,
    ) -> float:
        """Steady-state resonator photon number for qubit state |1>."""
        return abs(self.excited_state_field(drive_frequency_hz, drive_amplitude)) ** 2

    def measurement_response_separation(
        self,
        drive_frequency_hz: float,
        drive_amplitude: complex,
    ) -> complex:
        """Difference between excited- and ground-state resonator fields."""
        return (
            self.excited_state_field(drive_frequency_hz, drive_amplitude)
            - self.ground_state_field(drive_frequency_hz, drive_amplitude)
        )

    def measurement_response_separation_abs(
        self,
        drive_frequency_hz: float,
        drive_amplitude: complex,
    ) -> float:
        """Magnitude of the IQ-plane separation between qubit states."""
        return abs(
            self.measurement_response_separation(
                drive_frequency_hz,
                drive_amplitude,
            )
        )

    @classmethod
    def from_bare_transmon(
        cls,
        transmon: BareTransmon,
        resonator_frequency_hz: float,
        coupling_hz: float,
        resonator_kappa_hz: float,
    ) -> "TransmonDispersiveReadout":
        """Construct readout parameters from a BareTransmon-like object.

        The object must expose `f01_hz` and `alpha_hz`.
        """
        return cls(
            qubit_frequency_hz=transmon.f01_hz,
            anharmonicity_hz=transmon.alpha_hz,
            resonator_frequency_hz=resonator_frequency_hz,
            coupling_hz=coupling_hz,
            resonator_kappa_hz=resonator_kappa_hz,
        )
