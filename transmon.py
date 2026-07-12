"""Approximate transmon parameter extraction."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from functools import cached_property

@dataclass(frozen=True)
class Transmon:
    """Bare Transmon parameters stored as frequencies EJ/h and EC/h in Hz."""

    EJ_over_h: float
    EC_over_h: float
    ng: float = 0.0
    n_cutoff: int = 30


    @property
    def EJ(self) -> float:
        """Return the Josephson energy EJ in Watts."""
        return self.EJ_over_h * 6.62607015e-34  # Planck's constant in J*s

    @property
    def EC(self) -> float:
        """Return the charging energy EC in Watts."""
        return self.EC_over_h * 6.62607015e-34  # Planck's constant in J*s

    @property
    def C_Sigma(self) -> float:
        """Return the effective capacitance C of the transmon in Farads."""
        return (1.602176634e-19) ** 2 / (2 * self.EC)  # C = (e)^2 / (2EC)

    @property
    def Ic(self) -> float:
        """Return the critical current Ic of the Josephson junction in Amperes."""
        return (2 * 1.602176634e-19 * 2 * np.pi) * self.EJ_over_h  # Ic = (2e/hbar) * EJ = (2e * 2pi) * EJ/h

    def transmon_hamiltonian(
        self, EJ_over_h: float, EC_over_h: float, ng: float = 0.0, n_cutoff: int = 30
    ) -> np.ndarray:
        """
        Build the charge-basis transmon Hamiltonian.

        Parameters
        ----------
        EJ_over_h : float
            Josephson energy EJ/h, in Hz.
        EC_over_h : float
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
        np.fill_diagonal(H, 4.0 * EC_over_h * (n_vals - ng) ** 2)

        # Josephson term: -EJ/2 coupling between neighboring charge states
        offdiag = -0.5 * EJ_over_h * np.ones(dim - 1)
        H += np.diag(offdiag, k=1)
        H += np.diag(offdiag, k=-1)

        return H

    @cached_property
    def spectrum(self) -> np.ndarray:
        """transmon bound energy spectrum in frequency, sorted in ascending order."""
        H = self.transmon_hamiltonian(self.EJ_over_h, self.EC_over_h, ng=self.ng, n_cutoff=self.n_cutoff)
        energies = np.linalg.eigvalsh(H)
        # Bound states only
        return energies[energies < self.EJ_over_h]

    @property
    def n_levels(self) -> int:
        """Return the number of bound energy levels."""
        return len(self.spectrum)

    @property
    def f01(self) -> float:
        """Return the f01 transition frequency."""
        spectrum = self.spectrum
        return spectrum[1] - spectrum[0]


    @property
    def f02(self) -> float:
        """Return the f02 transition frequency."""
        spectrum = self.spectrum
        return spectrum[2] - spectrum[0]

    @property
    def f12(self) -> float:
        """Return the f12 transition frequency."""
        spectrum = self.spectrum
        return spectrum[2] - spectrum[1]

    @property
    def anharmonicity(self) -> float:
        """Return the anharmonicity alpha/2pi = f12 - f01."""
        spectrum = self.spectrum
        return spectrum[2] - 2 * spectrum[1] + spectrum[0]

    @classmethod
    def from_f01_anharmonicity(cls, f01: float, anharmonicity: float, n_cutoff: int = 30) -> "Transmon":
        """ 
        Create a Transmon instance from the f01 transition frequency and anharmonicity.

        Parameters
        ----------
        f01 : float
            The f01 transition frequency in Hz.
        anharmonicity : float
            The anharmonicity alpha/2pi in Hz.
        n_cutoff : int, optional
            The number of charge states to include in the calculation, by default 30.

        Returns
        -------
        Transmon
            A Transmon instance with the specified f01 and anharmonicity.
        """
        if f01 <= 0:
            raise ValueError("f01 must be positive")
        if anharmonicity >= 0:
            raise ValueError("anharmonicity should be negative")
        target_f01 = f01
        target_f12 = f01 + anharmonicity

        EC0_h = -anharmonicity
        EJ0_h = (f01 + EC0_h) ** 2 / (8 * EC0_h)

        def residual(log_params):
            EJ_h, EC_h = np.exp(log_params)
            t = cls(EJ_over_h=EJ_h, EC_over_h=EC_h, n_cutoff=n_cutoff)

            H = t.transmon_hamiltonian(EJ_h, EC_h, ng=t.ng, n_cutoff=n_cutoff)
            e = np.linalg.eigvalsh(H)[:3]

            f01 = e[1] - e[0]
            f12 = e[2] - e[1]

            return np.array([
                (f01 - target_f01) / target_f01,
                (f12 - target_f12) / abs(anharmonicity),
            ])

        result = least_squares(
            residual,
            x0=np.log([EJ0_h, EC0_h]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )

        EJ_h_fit, EC_h_fit = np.exp(result.x)
        return cls(EJ_over_h=EJ_h_fit, EC_over_h=EC_h_fit)

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

