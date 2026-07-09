"""Approximate transmon parameter extraction."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from functools import cached_property

@dataclass(frozen=True)
class BareTransmon:
    """Bare Transmon parameters stored as frequencies EJ/h and EC/h in Hz."""

    EJ_over_h_hz: float
    EC_over_h_hz: float
    n_cutoff: int = 30


    def transmon_hamiltonian(
        self, EJ_over_h_hz: float, EC_over_h_hz: float, ng: float = 0.0, n_cutoff: int = 30
    ) -> np.ndarray:
        """
        Build the charge-basis transmon Hamiltonian.

        Parameters
        ----------
        EJ_over_h_hz : float
            Josephson energy EJ/h, in Hz.
        EC_over_h_hz : float
            Charging energy EC/h, in Hz.
        ng : float
            Offset charge in units of Cooper-pair charge 2e.
        n_cutoff : int
            Charge states from -n_cutoff to +n_cutoff are included.

        Returns
        -------
        H : ndarray
            Hamiltonian matrix with shape (2*n_cutoff + 1, 2*n_cutoff + 1).
        """
        n_vals = np.arange(-n_cutoff, n_cutoff + 1, dtype=float)
        dim = len(n_vals)

        H = np.zeros((dim, dim), dtype=float)

        # Charging term: 4 EC (n - ng)^2
        np.fill_diagonal(H, 4.0 * EC_over_h_hz * (n_vals - ng) ** 2)

        # Josephson term: -EJ/2 coupling between neighboring charge states
        offdiag = -0.5 * EJ_over_h_hz * np.ones(dim - 1)
        H += np.diag(offdiag, k=1)
        H += np.diag(offdiag, k=-1)

        return H

    @cached_property
    def spectrum_hz(self) -> np.ndarray:
        """transmon bound energy spectrum in Hz, sorted in ascending order."""
        H = self.transmon_hamiltonian(self.EJ_over_h_hz, self.EC_over_h_hz, n_cutoff=self.n_cutoff)
        energies = np.linalg.eigvalsh(H)
        # Bound states only
        return energies[energies < self.EJ_over_h_hz]

    @property
    def n_levels(self) -> int:
        """Return the number of bound energy levels."""
        return len(self.spectrum_hz)

    @property
    def f01_hz(self) -> float:
        """Return the f01 transition frequency in Hz."""
        spectrum = self.spectrum_hz
        return spectrum[1] - spectrum[0]
    
    @property
    def f12_hz(self) -> float:
        spectrum = self.spectrum_hz
        return spectrum[2] - spectrum[1]

    @property
    def anharmonicity_hz(self) -> float:
        """Return the anharmonicity alpha/2pi = f12 - f01 in Hz."""
        spectrum = self.spectrum_hz
        return spectrum[2] - 2 * spectrum[1] + spectrum[0]

    @classmethod
    def from_f01_alpha(cls, f01_hz: float, anharmonicity_hz: float, n_cutoff: int = 30) -> "BareTransmon":
        if f01_hz <= 0:
            raise ValueError("f01_hz must be positive")
        if anharmonicity_hz >= 0:
            raise ValueError("anharmonicity_hz should be negative")
        target_f01 = f01_hz
        target_f12 = f01_hz + anharmonicity_hz

        EC0_h = -anharmonicity_hz
        EJ0_h = (f01_hz + EC0_h) ** 2 / (8 * EC0_h)

        def residual(log_params):
            EJ_h, EC_h = np.exp(log_params)
            t = cls(EJ_over_h_hz=EJ_h, EC_over_h_hz=EC_h, n_cutoff=n_cutoff)

            H = t.transmon_hamiltonian(EJ_h, EC_h, n_cutoff=n_cutoff)
            e = np.linalg.eigvalsh(H)[:3]

            f01 = e[1] - e[0]
            f12 = e[2] - e[1]

            return np.array([
                (f01 - target_f01) / target_f01,
                (f12 - target_f12) / abs(anharmonicity_hz),
            ])

        result = least_squares(
            residual,
            x0=np.log([EJ0_h, EC0_h]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )

        EJ_h_fit, EC_h_fit = np.exp(result.x)
        return cls(EJ_over_h_hz=EJ_h_fit, EC_over_h_hz=EC_h_fit)

    @property
    def EJ_over_EC(self) -> float:
        return self.EJ_over_h_hz / self.EC_over_h_hz

    @property
    def f01_hz_approx(self) -> float:
        """Approximate f01 transition frequency in Hz using the transmon limit formula."""
        return np.sqrt(8 * self.EJ_over_h_hz * self.EC_over_h_hz) - self.EC_over_h_hz

    @property
    def anharmonicity_hz_approx(self) -> float:
        """Approximate anharmonicity alpha/2pi in Hz using the transmon limit formula."""
        return -self.EC_over_h_hz



@dataclass(frozen=True)
class TransmonDispersiveReadout:
    """Dispersive readout parameters for a transmon-resonator system.

    All frequency-like quantities are expressed in Hz, not angular frequency.

    Parameters
    ----------
    qubit_frequency_hz:
        Bare qubit transition frequency f01.
    resonator_frequency_hz:
        Bare readout resonator frequency fr.
    coupling_hz:
        Qubit-resonator coupling strength g / 2pi, expressed in Hz.
    resonator_kappa_hz:
        Resonator energy decay rate kappa / 2pi, expressed in Hz.
    anharmonicity_hz:
        Transmon anharmonicity alpha / 2pi = f12 - f01. Usually negative.
    """

    bare_qubit_frequency_hz: float
    bare_resonator_frequency_hz: float
    coupling_hz: float
    resonator_kappa_hz: float
    anharmonicity_hz: float

    def __post_init__(self) -> None:
        if self.bare_qubit_frequency_hz <= 0:
            raise ValueError("bare_qubit_frequency_hz must be positive")
        if self.bare_resonator_frequency_hz <= 0:
            raise ValueError("bare_resonator_frequency_hz must be positive")
        if self.coupling_hz <= 0:
            raise ValueError("coupling_hz must be positive")
        if self.resonator_kappa_hz <= 0:
            raise ValueError("resonator_kappa_hz must be positive")
        if self.anharmonicity_hz >= 0:
            raise ValueError("anharmonicity_hz should be negative for a transmon")
        if np.isclose(self.detuning_hz, 0.0):
            raise ValueError("qubit and resonator are too close to resonance")
        if np.isclose(self.detuning_hz + self.anharmonicity_hz, 0.0):
            raise ValueError("detuning_hz + anharmonicity_hz is too close to zero")

    @classmethod
    def from_dressed(
        cls,
        dressed_qubit_frequency_hz: float,
        anharmonicity_hz: float,
        dressed_resonator_frequency_hz: float,
        dispersive_shift_hz: float,
        resonator_kappa_hz: float,
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
        if dressed_qubit_frequency_hz <= 0:
            raise ValueError("dressed_qubit_frequency_hz must be positive")
        if dressed_resonator_frequency_hz <= 0:
            raise ValueError("dressed_resonator_frequency_hz must be positive")
        if anharmonicity_hz >= 0:
            raise ValueError("anharmonicity_hz should be negative for a transmon")
        if resonator_kappa_hz <= 0:
            raise ValueError("resonator_kappa_hz must be positive")
        if dispersive_shift_hz == 0:
            raise ValueError("dispersive_shift_hz must be nonzero")

        alpha_over_2pi = anharmonicity_hz
        chi_over_2pi = dispersive_shift_hz

        dressed_detuning_hz = (
            dressed_qubit_frequency_hz
            - dressed_resonator_frequency_hz
        )

        denominator = 1.0 + 2.0 * chi_over_2pi / alpha_over_2pi
        if np.isclose(denominator, 0.0):
            raise ValueError("Cannot infer bare detuning: singular denominator")

        detuning_hz = (dressed_detuning_hz - chi_over_2pi) / denominator
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
            bare_qubit_frequency_hz=bare_qubit_frequency_hz,
            bare_resonator_frequency_hz=bare_resonator_frequency_hz,
            coupling_hz=coupling_hz,
            resonator_kappa_hz=resonator_kappa_hz,
            anharmonicity_hz=alpha_over_2pi,
        )

    @classmethod
    def from_dressed_with_resonator_separation(
        cls,
        dressed_qubit_frequency_hz: float,
        anharmonicity_hz: float,
        dressed_resonator_frequency_hz: float,
        resonator_state_separation_hz: float,
        resonator_kappa_hz: float,
    ) -> "TransmonDispersiveReadout":
        """Construct from measured resonator separation f_r,|1> - f_r,|0> = 2 chi / 2pi."""
        return cls.from_dressed(
            dressed_qubit_frequency_hz=dressed_qubit_frequency_hz,
            anharmonicity_hz=anharmonicity_hz,
            dressed_resonator_frequency_hz=dressed_resonator_frequency_hz,
            dispersive_shift_hz=resonator_state_separation_hz / 2.0,
            resonator_kappa_hz=resonator_kappa_hz,
        )

    @property
    def detuning_hz(self) -> float:
        """Qubit-resonator detuning f_q - f_r."""
        return self.bare_qubit_frequency_hz - self.bare_resonator_frequency_hz

    @property
    def straddling_detuning_hz(self) -> float:
        """Delta + alpha.

        This denominator appears in the transmon dispersive shift.
        """
        return self.detuning_hz + self.anharmonicity_hz

    @property
    def resonator_ringdown_time_s(self) -> float:
        """Resonator energy ringdown time, 1 / kappa."""
        return 1.0 / self.resonator_kappa_hz

    @property
    def dressed_resonator_frequency_hz(self) -> float:
        """Approximate dressed resonator frequency.

        Uses the perturbative expression

            f_r,dressed = f_r - g^2 / (Delta + alpha)
        """
        return (
            self.bare_resonator_frequency_hz
            - self.coupling_hz**2 / self.straddling_detuning_hz
        )

    @property
    def dressed_qubit_frequency_hz(self) -> float:
        """Approximate dressed qubit frequency.

        Uses the perturbative expression

            f_q,dressed = f_q + g^2 / Delta
        """
        return (
            self.bare_qubit_frequency_hz
            + self.coupling_hz**2 / self.detuning_hz
        )

    @property
    def dispersive_shift_hz(self) -> float:
        """Transmon dispersive shift chi / 2pi.

        Uses

            chi = g^2 alpha / [Delta (Delta + alpha)]

        All quantities are in Hz, so the output is also in Hz.
        """
        return (
            self.coupling_hz**2
            * self.anharmonicity_hz
            / (self.detuning_hz * self.straddling_detuning_hz)
        )

    @property
    def resonator_frequency_ground_hz(self) -> float:
        """Readout resonator frequency conditioned on qubit state |0>."""
        return self.dressed_resonator_frequency_hz - self.dispersive_shift_hz

    @property
    def resonator_frequency_excited_hz(self) -> float:
        """Readout resonator frequency conditioned on qubit state |1>."""
        return self.dressed_resonator_frequency_hz + self.dispersive_shift_hz

    @property
    def resonator_state_separation_hz(self) -> float:
        """Frequency separation between the |0> and |1> resonator responses.

        This is 2 chi.
        """
        return 2.0 * self.dispersive_shift_hz

    @property
    def critical_photon_number(self) -> float:
        """Critical photon number Delta^2 / (4 g^2)."""
        return self.detuning_hz**2 / (4.0 * self.coupling_hz**2)

    @property
    def purcell_decay_rate_hz(self) -> float:
        """Approximate Purcell decay rate.

        Uses

            Gamma_P = (g / Delta)^2 kappa
        """
        return (
            (self.coupling_hz / self.detuning_hz) ** 2
            * self.resonator_kappa_hz
        )

    @property
    def purcell_limited_t1_s(self) -> float:
        """Purcell-limited qubit lifetime."""
        return 1.0 / (2 * np.pi * self.purcell_decay_rate_hz)

    @property
    def dispersive_validity_ratio(self) -> float:
        """Simple dispersive validity ratio |g / Delta|."""
        return abs(self.coupling_hz / self.detuning_hz)

    @property
    def straddling_validity_ratio(self) -> float:
        """Validity ratio |g / (Delta + alpha)|."""
        return abs(self.coupling_hz / self.straddling_detuning_hz)

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
            - self.dispersive_shift_hz
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
            + self.dispersive_shift_hz
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
        transmon,
        resonator_frequency_hz: float,
        coupling_hz: float,
        resonator_kappa_hz: float,
    ) -> "TransmonReadout":
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