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
    # TODO: purcell_filter

    def _require_readout(self) -> ReadoutResonator:
        if self.readout_resonator is None:
            raise ValueError("No readout_resonator is attached")
        return self.readout_resonator

    def _require_drive_line(self) -> ChargeDriveLine:
        if self.charge_drive_line is None:
            raise ValueError("No charge_drive_line is attached")
        return self.charge_drive_line

    def lamb_shift(self, m: int, solver = "exact"):
        """
        The Lamb shift of transmon level |m> due to coupling with resonator.
        """
        if solver == "approx":
            if m == 0:
                return 0
            elif m > 0:
                return m * self.g_over_2pi ** 2 / (
                    self.qr_detuning(m - 1, representation="bare", solver="approx")
                )
            else:
                raise ValueError("m cannot be negative.")
        elif solver == "exact":
            return self.dressed_energy(m, 0) - self.transmon.eigenenergies[m]
        else:
            raise ValueError("Invalid solver. Choose 'exact' or 'approx'.")

    def f_qubit(
        self, m: int = 0, representation: str = "dressed", solver: str = "exact"
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
                    + self.lamb_shift(m + 1, solver="approx") - self.lamb_shift(m, solver="approx")
                )
            else:
                raise ValueError("Invalid solver. Choose 'exact' or 'approx'.")
        else:
            raise ValueError("Invalid representation. Choose 'bare' or 'dressed'.")

    def anharmonicity(self, representation: str = "dressed", solver: str = "exact") -> float:
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

    def f_resonator(self, representation = "dressed", solver = "exact"):
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

    def qr_detuning(self, m: int = 0, representation = "bare", solver = "exact") -> float:
        """Qubit-readout detuning between
        m, m+1 transition frequency of transmon and resonator frequency.
        
        m = 0: f01 - fr -> known as the detuning
        m = 1: f12 - fr -> known as the straddling detuning
        """
        return (
            self.f_qubit(m = m, representation=representation, solver = solver)
            - self.f_resonator(representation=representation, solver = solver)
        )

    def f_resonator_for_transmon_level(self, m: int, solver = "exact"):
        """Resonator frequency associated in transmon level `m`."""
        if solver == "approx":
            return (
                self.f_resonator(representation="bare", solver="approx")
                + self.lamb_shift(m, solver="approx")
                - self.lamb_shift(m + 1, solver="approx")
            )
        if solver == "exact":
            return self.dressed_transition_frequency(initial=(m, 0), final=(m, 1))
        raise ValueError("Invalid solver. Choose 'exact' or 'approx'.")

    def dispersive_shift(self, m: int = 0, solver: str = "exact") -> float:
        """Transmon dispersive shift chi/2pi = (f_r,|e> - f_r,|g>) / 2."""
        return (
            self.f_resonator_for_transmon_level(m + 1, solver=solver)
            - self.f_resonator_for_transmon_level(m, solver=solver)
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

    def thermal_photon_dephasing_rate_over_2pi(self, n_th: float) -> float:
        """Return the thermal-photon-induced dephasing rate Gamma_phi/2pi.

        This uses the non-perturbative expression for qubit dephasing due to
        thermal photons in a dispersively coupled readout resonator,

            Gamma_phi = kappa_tot / 2 * Re[
                sqrt((1 + 2 i chi / kappa_tot)^2
                    + 8 i chi n_th / kappa_tot)
                - 1
            ],

        where all rates are expressed in ordinary frequency units, i.e.
        divided by 2pi.

        Parameters
        ----------
        n_th : float
            Effective thermal photon occupation of the readout resonator.

        Returns
        -------
        float
            Thermal-photon-induced pure dephasing rate Gamma_phi/2pi in Hz.

        Notes
        -----
        This expression follows the formula used in
        Phys. Rev. B 86, 100506(R) (2012). Here ``chi`` is obtained from
        ``self.dispersive_shift()`` and ``kappa_tot`` from
        ``self.readout_resonator.kappa_over_2pi``.
        """
        resonator = self._require_readout()
        kappa_tot = resonator.kappa_over_2pi
        chi = self.dispersive_shift()

        return kappa_tot / 2 * np.real(
            np.sqrt((1 + 2j * chi / kappa_tot) ** 2 + (8j * chi * n_th / kappa_tot)) - 1
        )

    def full_hamiltonian(self) -> np.ndarray:
        """Full transmon-readout Hamiltonian in charge x photon basis.

        Returns H in energy units.
        """
        readout = self._require_readout()

        Hq = self.transmon.hamiltonian().astype(complex)
        Hr = readout.hamiltonian().astype(complex)

        n_op = self.transmon.n_operator.astype(complex)
        n_g = self.transmon.ng
        x_r = 2 * readout.quadrature_operator(angle=0.0).astype(complex) # a + a†

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

        return evals[idx]

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
    def from_dressed_freqs_with_punchout_shift(
        cls,
        f_qubit_dressed: float,
        f_resonator_ground: float,
        anharmonicity_dressed: float,
        punchout_shift: float,
        ng: float = 0.0,
        charge_cutoff: int = 30,
        photon_cutoff: int = 10,
        solver: str = "exact",
    ) -> "TransmonCircuit":
        """Construct from dressed qubit/readout data and punchout shift.

        The punchout shift is defined as

            punchout_shift = f_resonator_bare - f_resonator_ground = g_over_2pi^2 / detuning.

        For solver="approx", the inversion uses perturbative dispersive formulas.

        For solver="exact", the bare resonator frequency is fixed by the punchout
        definition, and the bare qubit frequency, bare anharmonicity, and coupling
        strength are refined by exact diagonalization.
        """

        if solver not in {"approx", "exact"}:
            raise ValueError("solver must be 'approx' or 'exact'")

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
                "Check the sign convention of punchout_shift and detuning. "
                "They must have the same sign."
            )

        g0 = np.sqrt(g0_squared)

        # Estimate bare anharmonicity from the dressed anharmonicity, 
        # bare detuning, and punchout shift.
        roots = np.roots([
            1.0,
            detuning0 - anharmonicity_dressed - 2 * punchout_shift,
            -detuning0 * anharmonicity_dressed,
        ])

        real_roots = roots[np.isclose(roots.imag, 0.0)].real
        if len(real_roots) == 0:
            anharmonicity_bare0 = anharmonicity_dressed
        else:
            anharmonicity_bare0 = real_roots[
                np.argmin(np.abs(real_roots - anharmonicity_dressed))
            ]

        def make_resonator(g_over_2pi: float) -> ReadoutResonator:
            return ReadoutResonator(
                frequency=f_resonator_bare,
                g_over_2pi=g_over_2pi,
                photon_cutoff=photon_cutoff,
            )

        def make_circuit(
            f_qubit_bare: float,
            anharmonicity_bare: float,
            g_over_2pi: float,
        ) -> "TransmonCircuit":
            transmon = Transmon.from_f01_anharmonicity(
                f01=f_qubit_bare,
                anharmonicity=anharmonicity_bare,
                ng=ng,
                charge_cutoff=charge_cutoff,
                solver="approx",
            )

            resonator = make_resonator(g_over_2pi=g_over_2pi)

            return cls(
                transmon=transmon,
                readout_resonator=resonator
            )

        if solver == "approx":
            return make_circuit(
                f_qubit_bare=f_qubit_bare0,
                anharmonicity_bare=anharmonicity_bare0,
                g_over_2pi=g0,
            )

        target = np.array([
            f_qubit_dressed,
            f_resonator_ground,
            anharmonicity_dressed,
        ])

        scale = np.array([
            1e6,
            1e6,
            1e6,
        ])

        def residual(params: np.ndarray) -> np.ndarray:
            f_qubit_bare, minus_anharmonicity_bare, g_over_2pi = params
            anharmonicity_bare = -minus_anharmonicity_bare

            circuit = make_circuit(
                f_qubit_bare=f_qubit_bare,
                anharmonicity_bare=anharmonicity_bare,
                g_over_2pi=g_over_2pi,
            )

            model = np.array([
                circuit.f_qubit(0, representation="dressed", solver="exact"),
                circuit.f_resonator_for_transmon_level(0, solver="exact"),
                circuit.anharmonicity(representation="dressed", solver="exact"),
            ])

            return (model - target) / scale

        x0 = [f_qubit_bare0, -anharmonicity_bare0, g0]
        lower = [f_qubit_bare0 * 0.95, -anharmonicity_bare0 * 0.8, g0 * 0.7]
        upper = [f_qubit_bare0 * 1.05, -anharmonicity_bare0 * 1.2, g0 * 1.3]

        result = least_squares(
            residual,
            x0=x0,
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=1000,
        )

        f_qubit_fit, minus_anharmonicity_fit, g_fit = result.x
        anharmonicity_fit = -minus_anharmonicity_fit

        return make_circuit(
            f_qubit_bare=f_qubit_fit,
            anharmonicity_bare=anharmonicity_fit,
            g_over_2pi=g_fit,
        )

    @classmethod
    def from_dressed_freqs_with_dispersive_shift(
        cls,
        f_qubit_dressed: float,
        f_resonator_ground: float,
        anharmonicity_dressed: float,
        chi_over_2pi: float,
        ng: float = 0.0,
        charge_cutoff: int = 30,
        photon_cutoff: int = 10,
        solver: str = "exact",
    ) -> "TransmonCircuit":
        """Construct from dressed qubit/readout data and dispersive shift.

        The measured readout frequency is assumed to be the resonator frequency
        conditioned on the transmon ground state,

            f_resonator_ground = f_r,|g>.

        The dispersive shift is defined as

            chi_over_2pi = (f_r,|e> - f_r,|g>) / 2.

        For solver="approx", the inversion uses perturbative dispersive formulas.

        For solver="exact", the bare qubit frequency, bare resonator frequency,
        bare anharmonicity, and coupling strength are refined by exact
        diagonalization.
        """

        if solver not in {"approx", "exact"}:
            raise ValueError("solver must be 'approx' or 'exact'")

        if f_qubit_dressed <= 0:
            raise ValueError("f_qubit_dressed must be positive")
        if f_resonator_ground <= 0:
            raise ValueError("f_resonator_ground must be positive")
        if anharmonicity_dressed >= 0:
            raise ValueError("anharmonicity_dressed should be negative")
        if chi_over_2pi == 0:
            raise ValueError("chi_over_2pi must be nonzero")

        # Approximate inversion based on the same lamb-shift convention used by
        # f_qubit(..., representation="dressed", solver="approx") and
        # f_resonator_for_transmon_level(..., solver="approx").
        #
        # Definitions:
        #   Delta = f_qubit_bare - f_resonator_bare
        #   alpha = anharmonicity_bare
        #   chi = g^2 / Delta - g^2 / (Delta + alpha)
        #
        # The dressed anharmonicity is approximately
        #   alpha_dressed = alpha - 2 chi.
        alpha_bare0 = anharmonicity_dressed + 2.0 * chi_over_2pi

        if alpha_bare0 >= 0:
            raise ValueError(
                "Inferred bare anharmonicity is non-negative. "
                "Check the sign convention of chi_over_2pi."
            )

        f_resonator_center = f_resonator_ground + chi_over_2pi
        dressed_detuning_to_center = f_qubit_dressed - f_resonator_center

        denominator = 1.0 + 2.0 * chi_over_2pi / alpha_bare0
        if np.isclose(denominator, 0.0):
            raise ValueError("Singular inversion from dressed parameters")

        detuning0 = (dressed_detuning_to_center - chi_over_2pi) / denominator

        g0_squared = (
            chi_over_2pi
            * detuning0
            * (detuning0 + alpha_bare0)
            / alpha_bare0
        )

        if g0_squared <= 0:
            raise ValueError(
                "Inferred g^2 is non-positive. "
                "Check the sign convention of chi_over_2pi, detuning, and anharmonicity."
            )

        g0 = np.sqrt(g0_squared)

        # From f_r,|g> = f_resonator_bare - g^2 / Delta.
        f_resonator_bare0 = f_resonator_ground + g0_squared / detuning0

        # From f_q,dressed = f_qubit_bare + g^2 / Delta.
        f_qubit_bare0 = f_qubit_dressed - g0_squared / detuning0

        def make_resonator(
            f_resonator_bare: float,
            g_over_2pi: float,
        ) -> ReadoutResonator:
            return ReadoutResonator(
                frequency=f_resonator_bare,
                g_over_2pi=g_over_2pi,
                photon_cutoff=photon_cutoff,
            )

        def make_circuit(
            f_qubit_bare: float,
            anharmonicity_bare: float,
            f_resonator_bare: float,
            g_over_2pi: float,
        ) -> "TransmonCircuit":
            transmon = Transmon.from_f01_anharmonicity(
                f01=f_qubit_bare,
                anharmonicity=anharmonicity_bare,
                ng=ng,
                charge_cutoff=charge_cutoff,
                solver="approx",
            )

            resonator = make_resonator(
                f_resonator_bare=f_resonator_bare,
                g_over_2pi=g_over_2pi,
            )

            return cls(
                transmon=transmon,
                readout_resonator=resonator,
            )

        if solver == "approx":
            return make_circuit(
                f_qubit_bare=f_qubit_bare0,
                anharmonicity_bare=alpha_bare0,
                f_resonator_bare=f_resonator_bare0,
                g_over_2pi=g0,
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

        def residual(params: np.ndarray) -> np.ndarray:
            f_qubit_bare, minus_anharmonicity_bare, f_resonator_bare, g_over_2pi = params
            anharmonicity_bare = -minus_anharmonicity_bare

            circuit = make_circuit(
                f_qubit_bare=f_qubit_bare,
                anharmonicity_bare=anharmonicity_bare,
                f_resonator_bare=f_resonator_bare,
                g_over_2pi=g_over_2pi,
            )

            model = np.array([
                circuit.f_qubit(0, representation="dressed", solver="exact"),
                circuit.f_resonator_for_transmon_level(0, solver="exact"),
                circuit.anharmonicity(representation="dressed", solver="exact"),
                circuit.dispersive_shift(solver="exact"),
            ])

            return (model - target) / scale

        x0 = [
            f_qubit_bare0,
            -alpha_bare0,
            f_resonator_bare0,
            g0,
        ]

        lower = [
            f_qubit_bare0 * 0.95,
            -alpha_bare0 * 0.8,
            f_resonator_bare0 * 0.98,
            g0 * 0.7,
        ]

        upper = [
            f_qubit_bare0 * 1.05,
            -alpha_bare0 * 1.2,
            f_resonator_bare0 * 1.02,
            g0 * 1.3,
        ]

        result = least_squares(
            residual,
            x0=x0,
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=1000,
        )

        f_qubit_fit, minus_anharmonicity_fit, f_resonator_fit, g_fit = result.x
        anharmonicity_fit = -minus_anharmonicity_fit

        return make_circuit(
            f_qubit_bare=f_qubit_fit,
            anharmonicity_bare=anharmonicity_fit,
            f_resonator_bare=f_resonator_fit,
            g_over_2pi=g_fit,
        )