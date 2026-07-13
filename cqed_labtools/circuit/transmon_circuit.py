"""Composite transmon circuit model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.optimize import least_squares

from .control_line import ChargeDriveLine
from .readout_resonator import ReadoutResonator
from .transmon import Transmon
    
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
    def f_qubit_bare(self, method: str = "exact") -> float:
        return self.transmon.f01(method = method)

    @property
    def f_qubit_dressed(self, method: str = "exact") -> float:
        if method == "exact":
            return self.dressed_transition_frequency(initial = (0, 0), final = (1, 0))
        elif method == "approx":
            return self.f_qubit_bare + self.g_over_2pi ** 2 / self.qr_detuning
        else:
            raise ValueError("Invalid method. Choose 'exact' or 'approx'.")

    @property
    def anharmonicity_bare(self, method: str = "exact") -> float:
        return self.transmon.anharmonicity(method = method)
    
    @property
    def anharmonicity_dressed(self, method: str = "exact") -> float:
        if method == "exact":
            f01 = self.dressed_transition_frequency(initial=(0, 0), final=(1, 0))
            f12 = self.dressed_transition_frequency(initial=(1, 0), final=(2, 0))
            return f12 - f01

    @property
    def f_resonator_bare(self) -> float:
        return self._require_readout().frequency

    @property
    def f_resonator_dressed(self, method: str = "exact") -> float:
        """Reference dressed readout frequency before state-dependent shift +/- chi."""
        if method == "exact":
            pass
        elif method == "approx":
            self._check_dispersive_denominators()
            return self.f_resonator_bare - self.g_over_2pi**2 / self.qr_straddling_detuning
        else:
            raise ValueError("Invalid method. Choose 'exact' or 'approx'.")

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
        return self.qr_detuning + self.anharmonicity_bare(method = 'approx')

    def _check_dispersive_denominators(self) -> None:
        if np.isclose(self.qr_detuning, 0.0):
            raise ValueError("Qubit and readout resonator are too close to resonance")
        if np.isclose(self.qr_straddling_detuning, 0.0):
            raise ValueError("Delta + anharmonicity is too close to zero")
    @property
    def f_resonator_for_transmon_level(self, m: int):
        """"""
        return self.dressed_transition_frequency(initial=(m, 0), final=(m, 1))

    @property
    def chi_readout_over_2pi(self, method: str = "exact") -> float:
        """Transmon dispersive shift chi/2pi = (f_r,|e> - f_r,|g>) / 2."""
        if method == "exact":
            return (
                self.f_resonator_for_transmon_level(1)
                - self.f_resonator_for_transmon_level(0)
             ) / 2
        elif method == "approx":
            self._check_dispersive_denominators()
            return (
                self.g_over_2pi**2 * self.anharmonicity
                / (self.qr_detuning * self.qr_straddling_detuning)
            )
        else:
            raise ValueError("Invalid method. Choose 'exact' or 'approx'.")

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

    # TODO: add property for photon-shot-noise dephasing rate Gamma_phi,PSN/2pi = 8 chi^2 n_bar / kappa

    def full_hamiltonian(self) -> np.ndarray:
        """Full transmon-readout Hamiltonian in charge x photon basis.

        Returns H/h in Hz.
        """
        readout = self._require_readout()

        Hq = self.transmon.hamiltonian().astype(complex)
        Hr = readout.hamiltonian().astype(complex)

        n_op = self.transmon.n_operator.astype(complex)
        n_g = self.transmon.ng
        x_r = readout.quadrature_operator(angle=0.0).astype(complex)

        # We choose the prefactor so that the matrix element between |e,0> and |g,1> is g.
        n01 = self.transmon.n_matrix_element(0, 1)
        charge_resonator_coupling_prefactor  = self.g / np.abs(n01)

        dim_q = Hq.shape[0]
        dim_r = Hr.shape[0]

        Iq = np.eye(dim_q, dtype=complex)
        Ir = np.eye(dim_r, dtype=complex)

        H = (
            np.kron(Hq, Ir)
            + np.kron(Iq, Hr)
            + np.kron(
                charge_resonator_coupling_prefactor * (n_op - n_g * Iq),
                x_r,
            )
        )

        return H

    @cached_property
    def dressed_eigensystem_default(self) -> tuple[np.ndarray, np.ndarray]:
        """Default dressed eigensystem with resonator_cutoff=5."""
        H = self.full_hamiltonian()
        return np.linalg.eigh(H)

    def bare_product_state(self, transmon_level: int, photon_number: int) -> np.ndarray:
        """Bare product state |transmon_level, photon_number>.

        The transmon part is represented in the charge basis.
        """
        q_state = self.transmon.eigenstates

        r_state = np.zeros(self.readout_resonator.photon_cutoff + 1, dtype=complex)
        r_state[photon_number] = 1.0

        return np.kron(q_state, r_state)

    def dressed_energy(self, transmon_level: int, photon_number: int) -> float:
        """Find dressed energy adiabatically connected to |level, photon_number>."""
        evals, evecs = self.dressed_eigensystem()
        target = self.bare_product_state(transmon_level=transmon_level, photon_number=photon_number)

        overlaps = np.abs(evecs.conj().T @ target) ** 2
        idx = int(np.argmax(overlaps))

        return float(evals[idx])

    def dressed_transition_energy(self, initial: tuple[int, int], final: tuple[int, int]) -> float:
        """Dressed transition energy between two dressed product-like states."""
        Ei = self.dressed_energy(*initial)
        Ef = self.dressed_energy(*final)
        return Ef - Ei

    def dressed_transition_frequency(self, initial: tuple[int, int], final: tuple[int, int]) -> float:
        """Dressed transition frequency between two dressed product-like states."""
        return self.dressed_transition_energy(initial=initial, final=final) / H_PLANCK

    @property
    def dispersive_validity_ratio(self) -> float:
        """ Dispersive validity ratio |g/Delta|.
        Should be much less than 1 for the dispersive approximation to hold.
        """
        return abs(self.g_over_2pi / self.qr_detuning)

    @property
    def straddling_validity_ratio(self) -> float:
        """ Straddling validity ratio |g/(Delta + alpha)|.
        Should be much less than 1 for the dispersive approximation to hold.
        """
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
        """Resonator conditioned on transmon |g>."""
        return self._readout_with_frequency(self.f_resonator_ground)

    @property
    def readout_resonator_excited(self) -> ReadoutResonator:
        """Resonator conditioned on transmon |e>."""
        return self._readout_with_frequency(self.f_resonator_excited)

    # def ground_state_field(self, f_drive: float, input_field: complex) -> complex:
    #     return self.readout_resonator_ground.intracavity_field(f_drive, input_field)

    # def excited_state_field(self, f_drive: float, input_field: complex) -> complex:
    #     return self.readout_resonator_excited.intracavity_field(f_drive, input_field)

    # def measurement_response_separation(self, f_drive: float, input_field: complex) -> complex:
    #     return self.excited_state_field(f_drive, input_field) - self.ground_state_field(f_drive, input_field)

    # def measurement_response_separation_abs(self, f_drive: float, input_field: complex) -> float:
    #     return abs(self.measurement_response_separation(f_drive, input_field))

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
    def from_measured_dressed(
        cls,
        f_qubit_dressed: float,
        f_resonator_ground: float,
        anharmonicity_dressed: float,
        chi_over_2pi: float,
        kappa_external_over_2pi: float,
        kappa_internal_over_2pi: float = 0.0,
        coupling_geometry: str = "single_sided",
        ng: float = 0.0,
        n_cutoff: int = 30,
        photon_cutoff: int = 5,
        charge_drive_line: ChargeDriveLine | None = None,
    ) -> "TransmonCircuit":
        """Construct a TransmonCircuit from measured dressed quantities.

        Parameters
        ----------
        f_qubit_dressed:
            Measured dressed qubit transition frequency in Hz.
        f_resonator_ground:
            Measured readout resonator frequency conditioned on |g>, in Hz.
        anharmonicity_dressed:
            Measured dressed anharmonicity f12 - f01, in Hz.
        chi_over_2pi:
            Measured dispersive shift chi/2pi in Hz.

            Convention:
                f_resonator_excited - f_resonator_ground = 2 chi/2pi

        kappa_external_over_2pi:
            External readout linewidth in Hz.
        kappa_internal_over_2pi:
            Internal readout linewidth in Hz.
        """

        if f_qubit_dressed <= 0:
            raise ValueError("f_qubit_dressed must be positive")
        if f_resonator_ground <= 0:
            raise ValueError("f_resonator_ground must be positive")
        if anharmonicity_dressed >= 0:
            raise ValueError("anharmonicity_dressed should be negative")
        if chi_over_2pi == 0:
            raise ValueError("chi_over_2pi must be nonzero")

        # First get a perturbative initial guess.
        alpha0 = anharmonicity_dressed
        chi0 = chi_over_2pi

        f_res_dressed_center = f_resonator_ground + chi0
        dressed_detuning = f_qubit_dressed - f_res_dressed_center

        denominator = 1.0 + 2.0 * chi0 / alpha0
        if np.isclose(denominator, 0.0):
            raise ValueError("Singular initial guess from dressed parameters")

        detuning0 = (dressed_detuning - chi0) / denominator

        g0_squared = chi0 * detuning0 * (detuning0 + alpha0) / alpha0
        if g0_squared <= 0:
            raise ValueError(
                "Initial inferred g^2 is non-positive. "
                "Check the sign convention of chi_over_2pi."
            )

        g0 = np.sqrt(g0_squared)

        f_qubit_bare0 = f_qubit_dressed - g0_squared / detuning0
        f_resonator_bare0 = f_res_dressed_center + g0_squared / (detuning0 + alpha0)

        transmon0 = Transmon.from_f01_anharmonicity(
            f01=f_qubit_bare0,
            anharmonicity=alpha0,
            ng=ng,
            n_cutoff=n_cutoff,
        )

        x0 = np.log([
            transmon0.EJ_over_h,
            transmon0.EC_over_h,
            f_resonator_bare0,
            g0,
        ])

        target = np.array([
            f_qubit_dressed,
            f_resonator_ground,
            anharmonicity_dressed,
            chi_over_2pi,
        ])

        scale = np.array([
            abs(f_qubit_dressed),
            abs(f_resonator_ground),
            abs(anharmonicity_dressed),
            abs(chi_over_2pi),
        ])

        def make_resonator(f_resonator: float, g_over_2pi: float) -> ReadoutResonator:
            if coupling_geometry == "single_sided":
                return ReadoutResonator.single_sided(
                    bare_frequency=f_resonator,
                    g_over_2pi=g_over_2pi,
                    kappa_external_over_2pi=kappa_external_over_2pi,
                    kappa_internal_over_2pi=kappa_internal_over_2pi,
                )

            if coupling_geometry == "side_coupled":
                return ReadoutResonator.side_coupled(
                    bare_frequency=f_resonator,
                    g_over_2pi=g_over_2pi,
                    kappa_external_over_2pi=kappa_external_over_2pi,
                    kappa_internal_over_2pi=kappa_internal_over_2pi,
                )

            raise ValueError(
                "This constructor supports single_sided or side_coupled. "
                "For two_sided, pass an explicit ReadoutResonator separately."
            )

        def residual(log_params: np.ndarray) -> np.ndarray:
            EJ_over_h, EC_over_h, f_res_bare, g_over_2pi = np.exp(log_params)

            transmon = Transmon(
                EJ_over_h=EJ_over_h,
                EC_over_h=EC_over_h,
                ng=ng,
                n_cutoff=n_cutoff,
            )

            resonator = make_resonator(
                f_resonator=f_res_bare,
                g_over_2pi=g_over_2pi,
            )

            circuit = cls(
                transmon=transmon,
                readout_resonator=resonator,
                charge_drive_line=charge_drive_line,
            )

            model = np.array([
                circuit.f_qubit_dressed(method="exact"),
                circuit.f_resonator_for_transmon_level(0),
                circuit.anharmonicity_dressed(method="exact"),
                circuit.chi_readout_over_2pi(method="exact"),
            ])

            return (model - target) / scale

        result = least_squares(
            residual,
            x0=x0,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=300,
        )

        EJ_fit, EC_fit, f_res_fit, g_fit = np.exp(result.x)

        transmon = Transmon(
            EJ_over_h=EJ_fit,
            EC_over_h=EC_fit,
            ng=ng,
            n_cutoff=n_cutoff,
        )

        resonator = make_resonator(
            f_resonator=f_res_fit,
            g_over_2pi=g_fit,
        )

        return cls(
            transmon=transmon,
            readout_resonator=resonator,
            charge_drive_line=charge_drive_line,
        )