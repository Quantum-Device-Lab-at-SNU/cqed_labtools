"""Approximate transmon parameter extraction."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from functools import cached_property

from constants import E_CHARGE, H_PLANCK


@dataclass(frozen=True)
class Transmon:
    """Bare transmon Hamiltonian in the charge basis.

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
        """Return the Josephson energy EJ in Watts."""
        return self.EJ_over_h * H_PLANCK  # Planck's constant in J*s

    @property
    def EC(self) -> float:
        """Return the charging energy EC in Watts."""
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
        self, EJ: float | None = None, EC: float | None = None, ng: float = 0.0, n_cutoff: int = 30
    ) -> np.ndarray:
        """
        Build the charge-basis transmon Hamiltonian.

        Parameters
        ----------
        EJ: float
            Josephson energy EJ.
        EC: float
            Charging energy EC.
        ng : float
            Offset charge in units of Cooper-pair charge 2e.
        n_cutoff : int
            Charge states from -n_cutoff to +n_cutoff are included.

        Returns
        -------
        H : ndarray
            Hamiltonian matrix with shape (2*n_cutoff + 1, 2*n_cutoff + 1).
        """
        if EJ is None:
            EJ = self.EJ
        if EC is None:
            EC = self.EC
        if ng is None:
            ng = self.ng
        if n_cutoff is None:
            n_cutoff = self.n_cutoff

        n_vals = np.arange(-n_cutoff, n_cutoff + 1, dtype=float)
        dim = len(n_vals)

        H = np.zeros((dim, dim), dtype=float)

        # Charging term: 4 EC (n - ng)^2
        np.fill_diagonal(H, 4.0 * EC * (n_vals - ng) ** 2)

        # Josephson term: -EJ/2 coupling between neighboring charge states
        offdiag = -0.5 * EJ * np.ones(dim - 1)
        H += np.diag(offdiag, k=1)
        H += np.diag(offdiag, k=-1)

        return H

    @cached_property
    def eigensystem(self) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues and eigenvectors in the charge basis."""
        return np.linalg.eigh(self.hamiltonian())

    @cached_property
    def eigenenergies(self) -> np.ndarray:
        """All eigenvalues in Hz."""
        return self.eigensystem[0]

    @cached_property
    def eigenstates(self) -> np.ndarray:
        """Eigenvectors as columns in the charge basis."""
        return self.eigensystem[1]

    def n_matrix_element(self, i: int, j: int) -> complex:
        """Charge operator matrix element <i|n|j>."""
        vi = self.eigenstates[:, i]
        vj = self.eigenstates[:, j]
        return np.vdot(vi, self.n_operator @ vj)

    @cached_property
    def energy_spectrum(self) -> np.ndarray:
        """transmon bound energy spectrum in frequency, sorted in ascending order."""
        return self.eigenenergies[self.eigenenergies < self.EJ_over_h]
    
    @property
    def frequency_spectrum(self) -> np.ndarray:
        """Return the bound energy spectrum in frequency, sorted in ascending order."""
        return self.energy_spectrum / H_PLANCK  # Convert energy to frequency (Hz)

    @property
    def n_levels(self) -> int:
        """Return the number of bound energy levels."""
        return len(self.energy_spectrum)

    def transition_frequency(self, i: int, j: int) -> float:
        """Transition frequency f_ij = E_j - E_i in Hz."""
        return self.frequency_spectrum[j] - self.frequency_spectrum[i]

    @property
    def f01(self) -> float:
        return self.transition_frequency(0, 1)

    @property
    def f02(self) -> float:
        return self.transition_frequency(0, 2)

    @property
    def f12(self) -> float:
        return self.transition_frequency(1, 2)

    @property
    def anharmonicity(self) -> float:
        """Return the anharmonicity alpha/2pi = f12 - f01."""
        return self.f12 - self.f01

    @property
    def EJ_over_EC(self) -> float:
        return self.EJ_over_h / self.EC_over_h

    @property
    def f01_approx(self) -> float:
        """Approximate f01 transition frequency using the transmon limit formula."""
        return np.sqrt(8 * self.EJ_over_h * self.EC_over_h) - self.EC_over_h

    @property
    def anharmonicity_approx(self) -> float:
        """Approximate anharmonicity alpha/2pi using the transmon limit formula."""
        return -self.EC_over_h

    @classmethod
    def from_f01_anharmonicity(
        cls,
        f01: float,
        anharmonicity: float,
        ng: float = 0.0,
        n_cutoff: int = 30,
    ) -> "Transmon":
        """Numerically infer EJ/h and EC/h from f01 and anharmonicity."""
        if f01 <= 0:
            raise ValueError("f01 must be positive")
        if anharmonicity >= 0:
            raise ValueError("anharmonicity should be negative for a transmon")

        target_f01 = f01
        target_f12 = f01 + anharmonicity
        EC0_over_h = -anharmonicity
        EJ0_over_h = (f01 + EC0_over_h) ** 2 / (8.0 * EC0_over_h)

        def residual(log_params: np.ndarray) -> np.ndarray:
            EJ_over_h, EC_over_h = np.exp(log_params)
            t = cls(EJ_over_h=EJ_over_h, EC_over_h=EC_over_h, ng=ng, n_cutoff=n_cutoff)
            f01_fit = t.f01
            f12_fit = t.f12
            return np.array([
                (f01_fit - target_f01) / target_f01,
                (f12_fit - target_f12) / abs(anharmonicity),
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
