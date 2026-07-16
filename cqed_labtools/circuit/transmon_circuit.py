"""Composite transmon circuit model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.optimize import least_squares

from .control_line import ChargeDriveLine
from .readout_resonator import ReadoutResonator
from .transmon import Transmon
from ..constants import H_PLANCK

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

    def _lamb_shift(self, m: int, solver = "approx"):
        if solver == "approx":
            if m == 0:
                return 0
            elif m > 0:
                return m * self.g_over_2pi ** 2 / (
                    self.qr_detuning(m - 1, representation="bare", solver="approx")
                )
            else:
                raise ValueError("m cannot be negative.")

    def f_qubit(
        self, m: int = 0, representation: str = "bare", solver: str = "approx"
    ) -> float:
        """
        Dressed transition frequency between |m> and |m+1> levels of transmon
        """
        if representation == "bare":
            return self.transmon.transition_frequency(m, m + 1, solver = solver)
        elif representation == "dressed":
            if solver == "exact":
                return self.dressed_transition_frequency(initial = (m, 0), final = (m + 1, 0))
            elif solver == "approx":
                return (
                    self.f_qubit(m, representation = "bare", solver = "approx")
                    + self._lamb_shift(m + 1, solver="approx") - self._lamb_shift(m, solver="approx")
                )
            else:
                raise ValueError("Invalid solver. Choose 'exact' or 'approx'.")
        else:
            raise ValueError("Invalid representation. Choose 'bare' or 'dressed'.")

    def anharmonicity(self, representation: str = "bare", solver: str = "approx") -> float:
        if representation == "bare":
            return self.transmon.anharmonicity(solver = solver)
        elif representation == "dressed":
            if solver == "exact":
                f01 = self.dressed_transition_frequency(initial=(0, 0), final=(1, 0))
                f12 = self.dressed_transition_frequency(initial=(1, 0), final=(2, 0))
                return f12 - f01
            elif solver == "approx":
                f01 = self.f_qubit(0, representation="dressed", solver="approx")
                f12 = self.f_qubit(1, representation="dressed", solver="approx")
                return f12 - f01
            else:
                raise ValueError("Invalid solver. Choose 'exact' or 'approx'.")
        else:
            raise ValueError("Invalid representation. Choose 'bare' or 'dressed'.")

    def f_resonator(self, representation = "bare", solver = "approx"):
        """
        Bare: Readout resonator frequency in the absence of qubit (after punchout)
        Dressed: Reference readout resonator frequency before state-dependent shift +/- chi."""
        if representation == "bare":
            return self._require_readout().frequency
        elif representation == "dressed":
            if solver == "exact":
                return 0.5 * (
                    self.f_resonator_for_transmon_level(0)
                    + self.f_resonator_for_transmon_level(1)
                )
            elif solver == "approx":
                return (
                    self.f_resonator(representation="bare", solver="approx")
                    - self.g_over_2pi**2 / self.qr_detuning(1, representation="bare", solver="approx")
                )
            else:
                raise ValueError("Invalid solver. Choose 'exact' or 'approx'.")
        else:
            raise ValueError("Invalid representation. Choose 'bare' or 'dressed'.")

    @property
    def g_over_2pi(self) -> float:
        resonator = self._require_readout()
        if resonator.g_over_2pi is None:
            raise ValueError("readout_resonator.g_over_2pi is required for dispersive properties")
        return resonator.g_over_2pi

    def qr_detuning(self, m: int = 0, representation = "bare", solver = "approx") -> float:
        """Qubit-readout detuning between
        m, m+1 transition frequency of transmon and resonator frequency.
        
        m = 0: f01 - fr -> known as the detuning
        m = 1: f12 - fr -> known as the straddling detuning
        """
        return (
            self.f_qubit(m = m, representation=representation, solver = solver)
            - self.f_resonator(representation=representation, solver = solver)
        )

    def f_resonator_for_transmon_level(self, m: int, solver = "approx"):
        """Resonator frequency associated in transmon level `m`."""
        if solver == "approx": # "blais formula"
            return (
                self.f_resonator(representation="bare", solver="approx")
                + self._lamb_shift(m, solver="approx") - self._lamb_shift(m + 1, solver="approx")
            )
        elif solver == "exact":
            return self.dressed_transition_frequency(initial=(m, 0), final=(m, 1))

    def dispersive_shift(self, m: int = 0, solver: str = "approx") -> float:
        """Transmon dispersive shift chi/2pi = (f_r,|e> - f_r,|g>) / 2."""
        return (
            self.f_resonator_for_transmon_level(1, solver=solver)
            - self.f_resonator_for_transmon_level(0, solver=solver)
        ) / 2

    @property
    def critical_photon_number(self) -> float:
        return self.qr_detuning(representation="bare", solver="exact")**2 / (4.0 * self.g_over_2pi**2)

    @property
    def purcell_decay_rate_over_2pi(self) -> float:
        """Purcell decay rate Gamma_P/2pi in Hz."""
        readout = self._require_readout()
        return (
            self.g_over_2pi
            / self.qr_detuning(0, representation="bare", solver="exact")
        ) ** 2 * readout.kappa_over_2pi

    @property
    def purcell_decay_rate(self) -> float:
        """Purcell decay rate in angular frequency [rad/s]."""
        return 2.0 * np.pi * self.purcell_decay_rate_over_2pi

    @property
    def purcell_limited_t1(self) -> float:
        """Purcell-limited T1 in time units."""
        return 1.0 / self.purcell_decay_rate

    @property
    def g(self) -> float:
        return self.g_over_2pi * 2 * np.pi

    # TODO: add property for photon-shot-noise dephasing rate Gamma_phi,PSN/2pi = 8 chi^2 n_bar / kappa

    def full_hamiltonian(self) -> np.ndarray:
        """Full transmon-readout Hamiltonian in charge x photon basis.

        Returns H in energy units.
        """
        readout = self._require_readout()

        Hq = self.transmon.hamiltonian().astype(complex)
        Hr = readout.hamiltonian().astype(complex)

        n_op = self.transmon.n_operator.astype(complex)
        n_g = self.transmon.ng
        x_r = readout.quadrature_operator(angle=0.0).astype(complex)

        # We choose the prefactor so that the matrix element between |e,0> and |g,1> is g.
        n01 = self.transmon.n_matrix_element(0, 1)
        charge_resonator_coupling_prefactor  = self.g_over_2pi / np.abs(n01)

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
    def dressed_eigensystem(self) -> tuple[np.ndarray, np.ndarray]:
        """dressed eigensystem."""
        H = self.full_hamiltonian()
        return np.linalg.eigh(H)

    def _bare_product_state(self, transmon_level: int, photon_number: int) -> np.ndarray:
        """Bare product state |transmon_level, photon_number>."""
        readout = self._require_readout()

        q_state = self.transmon.eigenstates[:, transmon_level]

        r_state = np.zeros(readout.photon_cutoff + 1, dtype=complex)
        r_state[photon_number] = 1.0

        return np.kron(q_state, r_state)

    def dressed_energy(self, transmon_level: int, photon_number: int) -> float:
        """Find dressed energy adiabatically connected to |level, photon_number>."""
        evals, evecs = self.dressed_eigensystem
        target = self._bare_product_state(
            transmon_level=transmon_level, photon_number=photon_number
        )

        overlaps = np.abs(evecs.conj().T @ target) ** 2
        idx = int(np.argmax(overlaps))

        return float(evals[idx])

    def dressed_transition_frequency(self, initial: tuple[int, int], final: tuple[int, int]) -> float:
        """Dressed transition energy between two dressed product-like states."""
        fi = self.dressed_energy(*initial)
        ff = self.dressed_energy(*final)
        return ff - fi

    def _readout_with_frequency(self, frequency: float) -> ReadoutResonator:
        r = self._require_readout()
        return ReadoutResonator(
            frequency=frequency,
            kappa_internal_over_2pi=r.kappa_internal_over_2pi,
            coupling_geometry=r.coupling_geometry,
            kappa_input_over_2pi=r.kappa_input_over_2pi,
            kappa_output_over_2pi=r.kappa_output_over_2pi,
            g_over_2pi=r.g_over_2pi,
        )

    def readout_resonator_for_transmon_level(self, m: int, solver: str = "approx") -> ReadoutResonator:
        """Resonator conditioned on transmon transmon in level m."""
        return self._readout_with_frequency(self.f_resonator_for_transmon_level(m, solver = solver))

    # def ground_state_field(self, f_drive: float, input_field: complex) -> complex:
    #     return self.readout_resonator_ground.intracavity_field(f_drive, input_field)

    # def excited_state_field(self, f_drive: float, input_field: complex) -> complex:
    #     return self.readout_resonator_excited.intracavity_field(f_drive, input_field)

    # def measurement_response_separation(self, f_drive: float, input_field: complex) -> complex:
    #     return self.excited_state_field(f_drive, input_field) - self.ground_state_field(f_drive, input_field)

    # def measurement_response_separation_abs(self, f_drive: float, input_field: complex) -> float:
    #     return abs(self.measurement_response_separation(f_drive, input_field))

    def n_photon_from_input_power(
        self, m: int, f_drive: float, input_power: float, solver = "approx"
    ) -> float:
        res = self.readout_resonator_for_transmon_level(m, solver = solver)
        return res.n_photon_from_input_power(f_drive, input_power)

    def input_power_for_n_photon(
        self, m: int, f_drive: float, n_photon: float, solver = "approx"
    ) -> float:
        res = self.readout_resonator_for_transmon_level(m, solver = solver)
        return res.input_power_for_n_photon(f_drive, n_photon)

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
    def from_measured_dressed_with_punchout_shift(
        cls,
        f_qubit_dressed: float,
        f_resonator_ground: float,
        anharmonicity_dressed: float,
        punchout_shift: float,
        kappa_internal_over_2pi: float,
        kappa_input_over_2pi: float | None,
        kappa_output_over_2pi: float,
        coupling_geometry: str = "single_sided",
        ng: float = 0.0,
        n_cutoff: int = 30,
        photon_cutoff: int = 5,
        charge_drive_line: ChargeDriveLine | None = None,
        solver = "approx"
    ) -> "TransmonCircuit":
        """Construct from dressed qubit/readout data and punchout shift.

        punchout_shift = f_resonator_bare - f_resonator_ground ≈ (g/2pi)^2 / (f_q - f_r).
        """

        if f_qubit_dressed <= 0:
            raise ValueError("f_qubit_dressed must be positive")
        if f_resonator_ground <= 0:
            raise ValueError("f_resonator_ground must be positive")
        if anharmonicity_dressed >= 0:
            raise ValueError("anharmonicity_dressed should be negative")
        if punchout_shift == 0:
            raise ValueError("punchout_shift must be nonzero")

        # By definition of punchout shift.
        f_resonator_bare = f_resonator_ground + punchout_shift

        # Perturbative initial guess.
        f_qubit_bare0 = f_qubit_dressed - punchout_shift
        detuning0 = f_qubit_bare0 - f_resonator_bare
        g0_squared = punchout_shift * detuning0

        if g0_squared <= 0:
            raise ValueError(
                "Inferred g^2 is non-positive. "
                "Check the sign of punchout_shift. "
                "punchout_shift and detuning must have the same sign."
            )

        g0 = np.sqrt(g0_squared)

        # construct a quadratic equation to find bare anharmonicity from the derived parameters.
        anharmonicity0_roots = np.roots([
            1, detuning0 - anharmonicity_dressed - punchout_shift,
            - detuning0 * (anharmonicity_dressed + punchout_shift)]
        )
        anharmonicity0 = anharmonicity0_roots[np.argmin(np.abs(anharmonicity0_roots - anharmonicity_dressed))]

        if solver == "approx":
            transmon = Transmon.from_f01_anharmonicity(f01=f_qubit_bare0, anharmonicity=anharmonicity0, solver="approx")
            readout_resonator = ReadoutResonator(
                frequency=f_resonator_bare, 
                kappa_internal_over_2pi=kappa_internal_over_2pi,
                kappa_input_over_2pi=kappa_input_over_2pi,
                coupling_geometry=coupling_geometry,
                kappa_output_over_2pi=kappa_output_over_2pi, g_over_2pi=g0,
                photon_cutoff=photon_cutoff
            )
            return cls(
                transmon=transmon, readout_resonator=readout_resonator, 
                charge_drive_line=charge_drive_line
            )

        target = np.array([
            f_qubit_dressed,
            f_resonator_ground,
            f_resonator_bare,
            anharmonicity_dressed,
        ])

        scale = np.array([
            1e7,
            1e7,
            1e7,
            1e6,
        ])

        def make_resonator(
            f_resonator: float,
            g_over_2pi: float,
        ) -> ReadoutResonator:
            if coupling_geometry == "single_sided":
                return ReadoutResonator.single_sided(
                    frequency=f_resonator,
                    g_over_2pi=g_over_2pi,
                    kappa_external_over_2pi=kappa_external_over_2pi,
                    kappa_internal_over_2pi=kappa_internal_over_2pi,
                    photon_cutoff=photon_cutoff,
                )

            if coupling_geometry == "side_coupled":
                return ReadoutResonator.side_coupled(
                    frequency=f_resonator,
                    g_over_2pi=g_over_2pi,
                    kappa_external_over_2pi=kappa_external_over_2pi,
                    kappa_internal_over_2pi=kappa_internal_over_2pi,
                    photon_cutoff=photon_cutoff,
                )

            raise ValueError(
                "This constructor supports single_sided or side_coupled. "
                "For two_sided, pass an explicit ReadoutResonator separately."
            )

        def residual(log_params: np.ndarray) -> np.ndarray:
            f_qubit_bare, minus_alpha_bare, g_over_2pi = np.exp(log_params)
            alpha_bare = -minus_alpha_bare

            transmon = Transmon.from_f01_anharmonicity(
                f01=f_qubit_bare,
                anharmonicity=alpha_bare,
                ng=ng,
                n_cutoff=n_cutoff,
            )

            resonator = make_resonator(
                f_resonator=f_resonator_bare,
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
                circuit.anharmonicity(representation="dressed", solver="exact"),
            ])

            return (model - target) / scale

        x0 = np.log([
            f_qubit_bare0,
            -alpha0,
            g0,
        ])

        lower = np.log([
            f_qubit_bare0 * 0.98,
            -alpha0 * 0.5,
            g0 * 0.2,
        ])

        upper = np.log([
            f_qubit_bare0 * 1.02,
            -alpha0 * 2.0,
            g0 * 5.0,
        ])

        result = least_squares(
            residual,
            x0=x0,
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=200,
        )

        f_qubit_fit, minus_alpha_fit, g_fit = np.exp(result.x)
        alpha_fit = -minus_alpha_fit

        transmon = Transmon.from_f01_anharmonicity(
            f01=f_qubit_fit,
            anharmonicity=alpha_fit,
            ng=ng,
            n_cutoff=n_cutoff,
        )

        resonator = make_resonator(
            f_resonator=f_res_bare_fixed,
            g_over_2pi=g_fit,
        )

        return cls(
            transmon=transmon,
            readout_resonator=resonator,
            charge_drive_line=charge_drive_line,
        )

    @classmethod
    def from_measured_dressed_with_dispersive_shift(
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
        """Construct from measured dressed qubit/readout quantities."""

        if f_qubit_dressed <= 0:
            raise ValueError("f_qubit_dressed must be positive")
        if f_resonator_ground <= 0:
            raise ValueError("f_resonator_ground must be positive")
        if anharmonicity_dressed >= 0:
            raise ValueError("anharmonicity_dressed should be negative")
        if chi_over_2pi == 0:
            raise ValueError("chi_over_2pi must be nonzero")

        alpha0_over_2pi = anharmonicity_dressed
        chi0_over_2pi = chi_over_2pi

        f_res_dressed_center = f_resonator_ground + chi0_over_2pi
        dressed_detuning = f_qubit_dressed - f_res_dressed_center

        denominator = 1.0 + 2.0 * chi0_over_2pi / alpha0_over_2pi
        if np.isclose(denominator, 0.0):
            raise ValueError("Singular initial guess from dressed parameters")

        detuning0 = (dressed_detuning - chi0_over_2pi) / denominator

        g0_squared = chi0_over_2pi * detuning0 * (detuning0 + alpha0_over_2pi) / alpha0_over_2pi
        if g0_squared <= 0:
            raise ValueError(
                "Initial inferred g^2 is non-positive. "
                "Check the sign convention of chi_over_2pi."
            )

        g0 = np.sqrt(g0_squared)

        f_qubit_bare0 = f_qubit_dressed - g0_squared / detuning0
        f_resonator_bare0 = (
            f_res_dressed_center
            + g0_squared / (detuning0 + alpha0_over_2pi)
        )

        target = np.array([
            f_qubit_dressed,
            f_resonator_ground,
            anharmonicity_dressed,
            chi_over_2pi,
        ])

        scale = np.array([
            1e6,
            1e6,
            1e6,
            1e5,
        ])

        def make_resonator(
            f_resonator: float,
            g_over_2pi: float,
        ) -> ReadoutResonator:
            if coupling_geometry == "single_sided":
                return ReadoutResonator.single_sided(
                    frequency=f_resonator,
                    g_over_2pi=g_over_2pi,
                    kappa_external_over_2pi=kappa_external_over_2pi,
                    kappa_internal_over_2pi=kappa_internal_over_2pi,
                    photon_cutoff=photon_cutoff,
                )

            if coupling_geometry == "side_coupled":
                return ReadoutResonator.side_coupled(
                    frequency=f_resonator,
                    g_over_2pi=g_over_2pi,
                    kappa_external_over_2pi=kappa_external_over_2pi,
                    kappa_internal_over_2pi=kappa_internal_over_2pi,
                    photon_cutoff=photon_cutoff,
                )

            raise ValueError(
                "This constructor supports single_sided or side_coupled. "
                "For two_sided, pass an explicit ReadoutResonator separately."
            )

        def residual(log_params: np.ndarray) -> np.ndarray:
            f_qubit_bare, minus_alpha_bare, f_res_bare, g_over_2pi = np.exp(log_params)
            alpha_bare = -minus_alpha_bare

            transmon = Transmon.from_f01_anharmonicity(
                f01=f_qubit_bare,
                anharmonicity=alpha_bare,
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
                circuit.f_qubit(0, representation="dressed", solver="exact"),
                circuit.f_resonator_for_transmon_level(0, solver="exact"),
                circuit.anharmonicity(representation="dressed", solver="exact"),
                circuit.dispersive_shift(solver="exact")
            ])

            return (model - target) / scale

        x0 = np.log([
            f_qubit_bare0,
            -alpha0_over_2pi,
            f_resonator_bare0,
            g0,
        ])

        lower = np.log([
            f_qubit_bare0 * 0.98,
            -alpha0_over_2pi * 0.5,
            f_resonator_bare0 * 0.98,
            g0 * 0.2,
        ])

        upper = np.log([
            f_qubit_bare0 * 1.02,
            -alpha0_over_2pi * 2.0,
            f_resonator_bare0 * 1.02,
            g0 * 5.0,
        ])

        result = least_squares(
            residual,
            x0=x0,
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=200,
        )

        f_qubit_fit, minus_alpha_fit, f_res_fit, g_fit = np.exp(result.x)
        alpha_fit = -minus_alpha_fit

        transmon = Transmon.from_f01_anharmonicity(
            f01=f_qubit_fit,
            anharmonicity=alpha_fit,
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