"""Bare transmon Hamiltonian in the charge basis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.optimize import least_squares

from ..constants import E_CHARGE, H_PLANCK

@dataclass(frozen=True)
class Transmon:
    """Bare transmon system, with Hamiltonian constructed in the charge basis.

    Parameters
    ----------
    EJ_over_h:
        Josephson energy EJ/h in Hz.
    EC_over_h:
        Charging energy EC/h in Hz.
    ng:
        Static offset charge in units of Cooper-pair charge 2e.
    n_cutoff:
        Charge basis runs from -n_cutoff to +n_cutoff.
    """

    EJ_over_h: float
    EC_over_h: float
    ng: float = 0.0
    n_cutoff: int = 30


    def __post_init__(self) -> None:
        if self.EJ_over_h <= 0:
            raise ValueError("EJ_over_h must be positive")
        if self.EC_over_h <= 0:
            raise ValueError("EC_over_h must be positive")
        if self.n_cutoff < 2:
            raise ValueError("n_cutoff must be at least 2")

    @property
    def EJ(self) -> float:
        """Return the Josephson energy EJ in Joules."""
        return self.EJ_over_h * H_PLANCK  # Planck's constant in J*s

    @property
    def EC(self) -> float:
        """Return the charging energy EC in Joules."""
        return self.EC_over_h * H_PLANCK  # Planck's constant in J*s

    @property
    def C_eff(self) -> float:
        """Effective total capacitance e^2 / (2 EC), in F."""
        return (E_CHARGE ** 2) / (2 * self.EC)  # C = (e)^2 / (2EC)

    @property
    def Ic(self) -> float:
        """Josephson critical current Ic = 2e EJ / hbar, in A."""
        return (2 * E_CHARGE * 2 * np.pi) * self.EJ_over_h  # Ic = (2e/hbar) * EJ = (2e * 2pi) * EJ/h

    @property
    def n_values(self) -> np.ndarray:
        """Charge-basis values n = -n_cutoff, ..., +n_cutoff."""
        return np.arange(-self.n_cutoff, self.n_cutoff + 1, dtype=float)

    @cached_property
    def n_operator(self) -> np.ndarray:
        """Charge-number operator in the charge basis."""
        return np.diag(self.n_values)


    def hamiltonian(
        self,
        EJ_over_h: float | None = None,
        EC_over_h: float | None = None,
        ng: float | None = None,
        n_cutoff: int | None = None,
    ) -> np.ndarray:
        """Construct the transmon Hamiltonian in the charge basis.

        The Hamiltonian is returned as H/h, so all matrix elements are expressed
        in frequency units (Hz).

        Parameters
        ----------
        EJ_over_h : float, optional
            Josephson energy divided by Planck's constant, in Hz.
            Defaults to the value stored in the object.
        EC_over_h : float, optional
            Charging energy divided by Planck's constant, in Hz.
            Defaults to the value stored in the object.
        ng : float, optional
            Offset charge in units of Cooper-pair charge (2e).
            Defaults to the value stored in the object.
        n_cutoff : int, optional
            Charge basis extends from -n_cutoff to +n_cutoff.
            Defaults to the value stored in the object.

        Returns
        -------
        ndarray
            Hamiltonian matrix H/h in the charge basis.
        """
        if EJ_over_h is None:
            EJ_over_h = self.EJ_over_h
        if EC_over_h is None:
            EC_over_h = self.EC_over_h
        if ng is None:
            ng = self.ng
        if n_cutoff is None:
            n_cutoff = self.n_cutoff

        n_vals = np.arange(-n_cutoff, n_cutoff + 1, dtype=float)
        dim = len(n_vals)

        H = np.zeros((dim, dim), dtype=float)
        np.fill_diagonal(H, 4.0 * EC_over_h * (n_vals - ng) ** 2)

        offdiag = -0.5 * EJ_over_h * np.ones(dim - 1)
        H += np.diag(offdiag, k=1)
        H += np.diag(offdiag, k=-1)

        return H

    def n_matrix_element(self, i: int, j: int) -> complex:
        """Return the charge-operator matrix element.

        Computes

            <i|n|j>

        where |i> and |j> are transmon eigenstates.

        Parameters
        ----------
        i, j : int
            Transmon eigenstate indices.

        Returns
        -------
        complex
            Charge matrix element.
        """

        vi = self.eigenstates[:, i]
        vj = self.eigenstates[:, j]
        return np.vdot(vi, self.n_operator @ vj)

    @cached_property
    def eigensystem(self) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues and eigenvectors in the charge basis."""
        return np.linalg.eigh(self.hamiltonian())

    @cached_property
    def eigenstates(self) -> np.ndarray:
        """Eigenvectors as columns in the charge basis."""
        return self.eigensystem[1]

    @cached_property
    def eigenenergies(self) -> np.ndarray:
        """All eigenvalues of H/h in Hz."""
        return self.eigensystem[0]

    @cached_property
    def energy_spectrum(self) -> np.ndarray:
        """Bound transmon spectrum in Hz."""
        return self.eigenenergies[self.eigenenergies < self.EJ_over_h]

    def frequency_spectrum(self, solver: str = "exact") -> np.ndarray:
        """Return the transmon energy spectrum.

        Parameters
        ----------
        solver : {"exact", "approx"}, optional
            Method used to compute the spectrum.

            - "exact": numerical diagonalization of the charge-basis Hamiltonian.
            - "approx": transmon approximation
            E_m/h ≈ m f01 - EC/h · m(m-1)/2.

        Returns
        -------
        ndarray
            Energy levels in Hz, referenced to the ground state.
        """
        if solver == "exact":
            return self.energy_spectrum

        if solver == "approx":
            f01 = np.sqrt(8 * self.EJ_over_h * self.EC_over_h) - self.EC_over_h
            alpha = -self.EC_over_h
            return np.array([
                m * f01 + 0.5 * alpha * m * (m - 1)
                for m in range(self.n_levels)
            ])

        raise ValueError("Invalid solver. Choose 'exact' or 'approx'.")

    @property
    def n_levels(self) -> int:
        """Return the number of bound energy levels."""
        return len(self.energy_spectrum)

    def transition_frequency(self, i: int, j: int, solver = "approx") -> float:
        """Return the transition frequency between two transmon levels.

        Parameters
        ----------
        i, j : int
            Initial and final transmon levels.
        solver : {"exact", "approx"}, optional
            Method used to compute the energy spectrum.

        Returns
        -------
        float
            Transition frequency f_ij in Hz.
        """
        freqs = self.frequency_spectrum(solver = solver)
        return freqs[j] - freqs[i]

    def f01(self, solver: str = "exact") -> float:
        return self.transition_frequency(0, 1, solver = solver)

    def f02(self, solver: str = "exact") -> float:
        return self.transition_frequency(0, 2, solver = solver)

    def f12(self, solver: str = "exact") -> float:
        return self.transition_frequency(1, 2, solver = solver)

    def anharmonicity(self, solver: str = "exact") -> float:
        """Return the anharmonicity alpha/2pi = f12 - f01."""
        return self.f12(solver=solver) - self.f01(solver=solver)

    @property
    def EJ_over_EC(self) -> float:
        return self.EJ_over_h / self.EC_over_h

    @classmethod
    def from_f01_anharmonicity(
        cls,
        f01: float,
        anharmonicity: float,
        ng: float = 0.0,
        n_cutoff: int = 30,
    ) -> "Transmon":
        """Construct a transmon from measured transition frequency.

        The method numerically determines EJ/h and EC/h such that the
        numerically diagonalized transmon Hamiltonian reproduces the
        specified f01 transition frequency and anharmonicity.

        Parameters
        ----------
        f01 : float
            Target qubit transition frequency in Hz.
        anharmonicity : float
            Target anharmonicity f12 - f01 in Hz.
        ng : float, optional
            Offset charge.
        n_cutoff : int, optional
            Charge basis cutoff.

        Returns
        -------
        Transmon
            Bare transmon whose numerically computed spectrum matches the
            requested transition frequency and anharmonicity.
        """
        if f01 <= 0:
            raise ValueError("f01 must be positive")
        if anharmonicity >= 0:
            raise ValueError("anharmonicity should be negative for a transmon")

        target_f01 = f01
        target_anh = anharmonicity
        EC0_over_h = -anharmonicity
        EJ0_over_h = (f01 + EC0_over_h) ** 2 / (8.0 * EC0_over_h)

        def residual(log_params: np.ndarray) -> np.ndarray:
            EJ_over_h, EC_over_h = np.exp(log_params)
            t = cls(EJ_over_h=EJ_over_h, EC_over_h=EC_over_h, ng=ng, n_cutoff=n_cutoff)
            f01_fit = t.f01(solver = "exact")
            anh_fit = t.anharmonicity(solver = "exact")
            return np.array([
                (f01_fit - target_f01) / 1e6,
                (anh_fit - target_anh) / 1e6,
            ])

        result = least_squares(
            residual,
            x0=np.log([EJ0_over_h, EC0_over_h]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        EJ_fit, EC_fit = np.exp(result.x)
        return cls(EJ_over_h=EJ_fit, EC_over_h=EC_fit, ng=ng, n_cutoff=n_cutoff)
