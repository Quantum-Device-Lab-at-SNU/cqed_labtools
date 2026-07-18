"""Readout resonator models with input-output coupling geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ..constants import H_PLANCK

CouplingGeometry = Literal["single_sided", "two_sided", "side_coupled"]


@dataclass(frozen=True)
class ReadoutResonator:
    """Readout resonator model.

    The basic object represents the coherent resonator Hamiltonian. Optional
    input-output parameters may be provided when linewidths, photon calibration,
    or scattering properties are needed.

    Parameters
    ----------
    frequency : float
        Bare resonator frequency in Hz.
    g_over_2pi : float, optional
        Qubit-resonator coupling strength g/2pi in Hz.
    photon_cutoff : int, optional
        Photon cutoff. The Fock basis contains states |0>, ..., |photon_cutoff>.
    kappa_internal_over_2pi : float, optional
        Internal loss rate kappa_int/2pi in Hz.
    coupling_geometry : {"single_sided", "two_sided", "side_coupled"}, optional
        Input-output coupling geometry. Required only for port-dependent
        calculations.
    kappa_input_over_2pi : float, optional
        Input coupling rate kappa_in/2pi in Hz.
    kappa_output_over_2pi : float, optional
        Output coupling rate kappa_out/2pi in Hz.
    """

    frequency: float
    g_over_2pi: float | None = None
    photon_cutoff: int = 10

    kappa_internal_over_2pi: float | None = None
    coupling_geometry: CouplingGeometry | None = None
    kappa_input_over_2pi: float | None = None
    kappa_output_over_2pi: float | None = None

    def __post_init__(self) -> None:
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")

        if self.g_over_2pi is not None and self.g_over_2pi <= 0:
            raise ValueError("g_over_2pi must be positive when provided")

        if self.photon_cutoff < 1:
            raise ValueError("photon_cutoff must be at least 1")

        if (
            self.kappa_internal_over_2pi is not None
            and self.kappa_internal_over_2pi < 0
        ):
            raise ValueError("kappa_internal_over_2pi must be non-negative")

        if (
            self.kappa_input_over_2pi is not None
            and self.kappa_input_over_2pi <= 0
        ):
            raise ValueError("kappa_input_over_2pi must be positive when provided")

        if (
            self.kappa_output_over_2pi is not None
            and self.kappa_output_over_2pi <= 0
        ):
            raise ValueError("kappa_output_over_2pi must be positive when provided")

        if self.coupling_geometry is not None:
            if self.coupling_geometry not in {
                "single_sided",
                "two_sided",
                "side_coupled",
            }:
                raise ValueError(
                    "coupling_geometry must be 'single_sided', "
                    "'two_sided', or 'side_coupled'"
                )

    def _require_coupling_geometry(self) -> CouplingGeometry:
        if self.coupling_geometry is None:
            raise ValueError("coupling_geometry is required for this calculation")
        return self.coupling_geometry


    def _require_kappa_internal_over_2pi(self) -> float:
        if self.kappa_internal_over_2pi is None:
            return 0.0
        return self.kappa_internal_over_2pi


    def _require_kappa_input_over_2pi(self) -> float:
        if self.kappa_input_over_2pi is None:
            raise ValueError("kappa_input_over_2pi is required for this calculation")
        return self.kappa_input_over_2pi


    def _require_kappa_output_over_2pi(self) -> float:
        if self.kappa_output_over_2pi is None:
            raise ValueError("kappa_output_over_2pi is required for this calculation")
        return self.kappa_output_over_2pi

    # coherent methods
    @property
    def g(self) -> float:
        """Qubit-resonator coupling"""
        if self.g_over_2pi is None:
            raise ValueError("g_over_2pi is not set")
        return 2.0 * np.pi * self.g_over_2pi

    def annihilation_operator(self) -> np.ndarray:
        """Photon annihilation operator in the resonator Fock basis.

        photon_cutoff means photon states |0>, ..., |photon_cutoff>.
        """
        dim = self.photon_cutoff + 1
        a = np.zeros((dim, dim), dtype=complex)

        for n in range(1, dim):
            a[n - 1, n] = np.sqrt(n)

        return a

    def creation_operator(self) -> np.ndarray:
        """Photon creation operator."""
        return self.annihilation_operator().conj().T

    def number_operator(self) -> np.ndarray:
        """Photon number operator a†a."""
        a = self.annihilation_operator()
        return a.conj().T @ a

    def quadrature_operator(self, angle: float = 0.0) -> np.ndarray:
        """Resonator field quadrature 0.5 * (a e^{iθ} + a† e^{-iθ})."""
        a = self.annihilation_operator()
        return 0.5 * (a * np.exp(1j * angle) + a.conj().T * np.exp(-1j * angle))

    def hamiltonian(self) -> np.ndarray:
        """Bare resonator Hamiltonian H/h = f_r a†a in Hz."""
        return self.frequency * self.number_operator()

    # methods to add external port information
    def with_single_sided_port(
        self,
        kappa_external_over_2pi: float,
        kappa_internal_over_2pi: float = 0.0,
    ) -> "ReadoutResonator":
        """Return a copy configured with a single-sided input-output port."""
        return replace(
            self,
            coupling_geometry="single_sided",
            kappa_internal_over_2pi=kappa_internal_over_2pi,
            kappa_input_over_2pi=kappa_external_over_2pi,
            kappa_output_over_2pi=None,
        )

    def with_side_coupled_port(
        self,
        kappa_external_over_2pi: float,
        kappa_internal_over_2pi: float = 0.0,
    ) -> "ReadoutResonator":
        """Return a copy configured with a side-coupled feedline port."""
        return replace(
            self,
            coupling_geometry="side_coupled",
            kappa_internal_over_2pi=kappa_internal_over_2pi,
            kappa_input_over_2pi=kappa_external_over_2pi / 2.0,
            kappa_output_over_2pi=kappa_external_over_2pi / 2.0,
        )

    def with_two_sided_ports(
        self,
        kappa_input_over_2pi: float,
        kappa_output_over_2pi: float,
        kappa_internal_over_2pi: float = 0.0,
    ) -> "ReadoutResonator":
        """Return a copy configured with separate input and output ports."""
        return replace(
            self,
            coupling_geometry="two_sided",
            kappa_internal_over_2pi=kappa_internal_over_2pi,
            kappa_input_over_2pi=kappa_input_over_2pi,
            kappa_output_over_2pi=kappa_output_over_2pi,
        )

    @property
    def kappa_external_over_2pi(self) -> float:
        if self.coupling_geometry == "single_sided":
            return float(self.kappa_input_over_2pi)
        return float(self.kappa_input_over_2pi + self.kappa_output_over_2pi)

    @property
    def kappa_over_2pi(self) -> float:
        """Total resonator linewidth."""
        return self.kappa_internal_over_2pi + self.kappa_external_over_2pi

    @property
    def kappa_input(self) -> float:
        return 2.0 * np.pi * float(self.kappa_input_over_2pi)

    @property
    def kappa_output(self) -> float | None:
        if self.kappa_output_over_2pi is None:
            return None
        return 2.0 * np.pi * self.kappa_output_over_2pi

    @property
    def kappa_internal(self) -> float:
        return 2.0 * np.pi * self.kappa_internal_over_2pi

    @property
    def kappa_external(self) -> float:
        return 2.0 * np.pi * self.kappa_external_over_2pi

    @property
    def kappa(self) -> float:
        return 2.0 * np.pi * self.kappa_over_2pi

    @property
    def quality_factor(self) -> float:
        """Loaded quality factor Q = ω_resonator / kappa = f_resonator / (kappa / 2pi)."""
        return self.frequency / self.kappa_over_2pi

    @property
    def internal_quality_factor(self) -> float:
        """Internal quality factor Q_int = ω_resonator / kappa_internal = f_resonator / (kappa_internal / 2pi)."""
        if self.kappa_internal_over_2pi == 0:
            return np.inf
        return self.frequency / self.kappa_internal_over_2pi

    @property
    def external_quality_factor(self) -> float:
        """External quality factor Q_ext = ω_resonator / kappa_ext = f_resonator / (kappa_ext / 2pi)."""
        return self.frequency / self.kappa_external_over_2pi

    @property
    def ringdown_time(self) -> float:
        """Energy ringdown time 1/kappa in seconds."""
        return 1.0 / self.kappa

    @property
    def is_reflection_geometry(self) -> bool:
        return self.coupling_geometry == "single_sided"

    @property
    def is_transmission_geometry(self) -> bool:
        return self.coupling_geometry in {"side_coupled", "two_sided"}

    @classmethod
    def single_sided(
        cls,
        frequency: float,
        kappa_external_over_2pi: float,
        kappa_internal_over_2pi: float = 0.0,
        g_over_2pi: float | None = None,
        photon_cutoff: int = 10,
    ) -> "ReadoutResonator":
        return cls(
            frequency=frequency,
            kappa_internal_over_2pi=kappa_internal_over_2pi,
            coupling_geometry="single_sided",
            kappa_input_over_2pi=kappa_external_over_2pi,
            g_over_2pi=g_over_2pi,
            photon_cutoff=photon_cutoff
        )

    @classmethod
    def two_sided(
        cls,
        frequency: float,
        kappa_input_over_2pi: float,
        kappa_output_over_2pi: float,
        kappa_internal_over_2pi: float = 0.0,
        g_over_2pi: float | None = None,
        photon_cutoff: int = 10,
    ) -> "ReadoutResonator":
        return cls(
            frequency=frequency,
            kappa_internal_over_2pi=kappa_internal_over_2pi,
            coupling_geometry="two_sided",
            kappa_input_over_2pi=kappa_input_over_2pi,
            kappa_output_over_2pi=kappa_output_over_2pi,
            g_over_2pi=g_over_2pi,
            photon_cutoff=photon_cutoff
        )

    @classmethod
    def side_coupled(
        cls,
        frequency: float,
        kappa_external_over_2pi: float,
        kappa_internal_over_2pi: float = 0.0,
        g_over_2pi: float | None = None,
        photon_cutoff: int = 10,
    ) -> "ReadoutResonator":
        return cls(
            frequency=frequency,
            kappa_internal_over_2pi=kappa_internal_over_2pi,
            coupling_geometry="side_coupled",
            kappa_input_over_2pi=kappa_external_over_2pi / 2.0,
            kappa_output_over_2pi=kappa_external_over_2pi / 2.0,
            g_over_2pi=g_over_2pi,
            photon_cutoff=photon_cutoff
        )

    def detuning(self, f_drive: float) -> float:
        """Resonator-drive detuning f_r - f_d."""
        return self.frequency - f_drive

    def delta(self, f_drive: float) -> float:
        """Resonator-drive detuning in angular frequency (rad/s)."""
        return 2 * np.pi * self.detuning(f_drive)

    def intracavity_field(self, f_drive: float, input_field: complex) -> complex:
        """Steady-state intracavity field amplitude.

        input_field has units sqrt(photons/s).

        Equation:
            da/dt = -(i Delta + kappa/2) a + sqrt(kappa_in) a_in

        with Delta = f_r - f_d in Hz convention:

            a = sqrt(kappa_in) a_in / (kappa/2 + i Delta)
        """
        delta = self.delta(f_drive)
        return np.sqrt(self.kappa_input) * input_field / (self.kappa / 2.0 + 1j * delta)

    def n_photon_from_input_flux(self, f_drive: float, photon_flux: float) -> float:
        """Intracavity photon number from input photon flux.

        photon_flux is photons/s.

        Equation:
            n = |a|^2 = kappa_in * photon_flux / (Delta^2 + (kappa/2)^2)
        """
        if photon_flux < 0:
            raise ValueError("photon_flux must be non-negative")

        delta = self.delta(f_drive)

        return self.kappa_input * photon_flux / (delta**2 + (self.kappa / 2.0) ** 2)

    def input_flux_for_n_photon(self, f_drive: float, n_photon: float) -> float:
        """Required input photon flux for target intracavity photon number.
        
        n_photon is dimensionless.
        
        Equation:
            photon_flux = n * (Delta^2 + (kappa/2)^2) / kappa_in
        """
        if n_photon < 0:
            raise ValueError("n_photon must be non-negative")

        delta = self.delta(f_drive)

        return n_photon * (delta**2 + (self.kappa / 2.0) ** 2) / self.kappa_input

    def n_photon_from_input_power(self, f_drive: float, input_power: float) -> float:
        """Intracavity photon number from input power at the chip.
        
        input_power is in Watts.
        
        Equation:
            photon_flux = input_power / (h * f_drive)
        """
        if input_power < 0:
            raise ValueError("input_power must be non-negative")
        photon_flux = input_power / (H_PLANCK * f_drive)
        return self.n_photon_from_input_flux(f_drive=f_drive, photon_flux=photon_flux)

    def input_power_for_n_photon(self, f_drive: float, n_photon: float) -> float:
        """Input power at the chip required for target intracavity photon number.
        
        n_photon is dimensionless.
        
        Equation:
            input_power = photon_flux * h * f_drive
        """
        photon_flux = self.input_flux_for_n_photon(f_drive=f_drive, n_photon=n_photon)
        return photon_flux * H_PLANCK * f_drive    


    # def s11(self, f_drive: float) -> complex:
    #     """Reflection coefficient.

    #     Most appropriate for single-sided and side-coupled geometries.

    #     Formula:
    #         S11 = 1 - kappa_in / (kappa/2 + i Delta)
    #     """
    #     delta = self.delta(f_drive)

    #     return 1.0 - self.kappa_input / (
    #         self.kappa / 2.0 + 1j * delta
    #     )

    # def s21(self, f_drive: float) -> complex:
    #     """Forward transmission coefficient.

    #     For two-sided resonator:
    #         S21 = -sqrt(kappa_in kappa_out) / (kappa/2 + i Delta)

    #     For side-coupled notch geometry:
    #         S21 = 1 - kappa_ext / (kappa/2 + i Delta)
    #     """
    #     delta = self.delta(f_drive)
    #     denominator = self.kappa / 2.0 + 1j * delta

    #     if self.coupling_geometry == "two_sided":
    #         return -np.sqrt(self.kappa_input * self.kappa_output) / denominator

    #     if self.coupling_geometry == "side_coupled":
    #         return 1.0 - self.kappa_external / denominator

    #     raise ValueError("s21 is not defined for single_sided geometry")