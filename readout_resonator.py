from dataclasses import dataclass
from typing import Literal
import numpy as np


CouplingGeometry = Literal["single_sided", "two_sided", "side_coupled"]


@dataclass(frozen=True)
class ReadoutResonator:
    """Readout resonator with explicit input-output coupling geometry.

    All frequency-like quantities are in Hz.

    Attributes
    ----------
    f_resonator : float
        Resonator frequency in Hz.
    kappa_internal_over_2pi : float
        Internal coupling rate to the resonator in Hz.
    coupling_geometry : CouplingGeometry
        Geometry of the input-output coupling.
    kappa_input_over_2pi : float | None
        Input coupling rate to the resonator in Hz. Required for single_sided and two_sided geometries.
    kappa_output_over_2pi : float | None
        Output coupling rate from the resonator in Hz. Required for two_sided and side_coupled geometries.
    """

    f_resonator: float
    kappa_internal_over_2pi: float
    coupling_geometry: CouplingGeometry

    kappa_input_over_2pi: float | None = None
    kappa_output_over_2pi: float | None = None

    def __post_init__(self) -> None:
        if self.f_resonator <= 0:
            raise ValueError("resonator frequency must be positive")
        if self.kappa_internal_over_2pi < 0:
            raise ValueError("kappa_internal must be non-negative")

        if self.coupling_geometry == "single_sided":
            if self.kappa_input_over_2pi is None:
                raise ValueError("single_sided resonator requires kappa_input")
            if self.kappa_output_over_2pi is not None:
                raise ValueError("single_sided resonator should not use kappa_output")

        elif self.coupling_geometry == "two_sided":
            if self.kappa_input_over_2pi is None or self.kappa_output_over_2pi is None:
                raise ValueError("two_sided resonator requires kappa_input and kappa_output")

        elif self.coupling_geometry == "side_coupled":
            if self.kappa_input_over_2pi is None or self.kappa_output_over_2pi is None:
                raise ValueError("side_coupled resonator requires kappa_input and kappa_output")

        else:
            raise ValueError(
                "coupling_geometry must be 'single_sided', 'two_sided', or 'side_coupled'"
            )

        if self.kappa_input_over_2pi is not None and self.kappa_input_over_2pi <= 0:
            raise ValueError("kappa_input must be positive")
        if self.kappa_output_over_2pi is not None and self.kappa_output_over_2pi <= 0:
            raise ValueError("kappa_output must be positive")

    @property
    def kappa_external_over_2pi(self) -> float:
        """Total coupling rate to external measurement ports."""
        if self.coupling_geometry == "single_sided":
            return self.kappa_input_over_2pi

        if self.coupling_geometry in {"two_sided", "side_coupled"}:
            return self.kappa_input_over_2pi + self.kappa_output_over_2pi

    @property
    def kappa_input(self) -> float:
        """Input coupling rate to the resonator in angular frequency (rad/s)."""
        return 2 * np.pi * self.kappa_input_over_2pi

    @property
    def kappa_output(self) -> float:
        """Output coupling rate from the resonator in angular frequency (rad/s)."""
        return 2 * np.pi * self.kappa_output_over_2pi

    @property
    def kappa_internal(self) -> float:
        """Internal coupling rate to the resonator in angular frequency (rad/s)."""
        return 2 * np.pi * self.kappa_internal_over_2pi

    @property
    def kappa_external(self) -> float:
        """Total coupling rate to external measurement ports in angular frequency (rad/s)."""
        return 2 * np.pi * self.kappa_external_over_2pi

    @property
    def kappa(self) -> float:
        """Total resonator linewidth in angular frequency (rad/s)."""
        return 2 * np.pi * self.kappa_over_2pi

    @property
    def kappa_over_2pi(self) -> float:
        """Total resonator linewidth."""
        return self.kappa_internal_over_2pi + self.kappa_external_over_2pi

    @property
    def quality_factor(self) -> float:
        """Loaded quality factor Q = ω_resonator / kappa = f_resonator / (kappa / 2pi)."""
        return self.f_resonator / self.kappa_over_2pi

    @property
    def internal_quality_factor(self) -> float:
        """Internal quality factor Q_int = ω_resonator / kappa_internal = f_resonator / (kappa_internal / 2pi)."""
        if self.kappa_internal_over_2pi == 0:
            return np.inf
        return self.f_resonator / self.kappa_internal_over_2pi

    @property
    def external_quality_factor(self) -> float:
        """External quality factor Q_ext = ω_resonator / kappa_ext = f_resonator / (kappa_ext / 2pi)."""
        return self.f_resonator / self.kappa_external_over_2pi

    @property
    def ringdown_time(self) -> float:
        """Energy ringdown time 1 / kappa."""
        return 1.0 / (2 * np.pi * self.kappa_over_2pi)

    @property
    def is_reflection_geometry(self) -> bool:
        return self.coupling_geometry in {"single_sided"}

    @property
    def is_transmission_geometry(self) -> bool:
        return self.coupling_geometry in {"side_coupled", "two_sided"}

    @classmethod
    def single_sided(
        cls,
        f_resonator: float,
        kappa_external_over_2pi: float,
        kappa_internal_over_2pi: float = 0.0,
    ) -> "ReadoutResonator":
        return cls(
            f_resonator=f_resonator,
            kappa_internal_over_2pi=kappa_internal_over_2pi,
            coupling_geometry="single_sided",
            kappa_input_over_2pi=kappa_external_over_2pi,
        )

    @classmethod
    def two_sided(
        cls,
        f_resonator: float,
        kappa_input_over_2pi: float,
        kappa_output_over_2pi: float,
        kappa_internal_over_2pi: float = 0.0,
    ) -> "ReadoutResonator":
        return cls(
            f_resonator=f_resonator,
            kappa_internal_over_2pi=kappa_internal_over_2pi,
            coupling_geometry="two_sided",
            kappa_input_over_2pi=kappa_input_over_2pi,
            kappa_output_over_2pi=kappa_output_over_2pi,
        )

    @classmethod
    def side_coupled(
        cls,
        f_resonator: float,
        kappa_external_over_2pi: float,
        kappa_internal_over_2pi: float = 0.0,
    ) -> "ReadoutResonator":
        return cls(
            f_resonator=f_resonator,
            kappa_internal_over_2pi=kappa_internal_over_2pi,
            coupling_geometry="side_coupled",
            kappa_input_over_2pi=kappa_external_over_2pi / 2,
            kappa_output_over_2pi=kappa_external_over_2pi / 2
        )

    def detuning(self, f_drive: float) -> float:
        """Resonator-drive detuning f_r - f_d."""
        return self.f_resonator - f_drive

    def delta(self, f_drive: float) -> float:
        """Resonator-drive detuning in angular frequency (rad/s)."""
        return 2 * np.pi * self.detuning(f_drive)

    def intracavity_field(
        self,
        f_drive: float,
        input_field: complex,
    ) -> complex:
        """Steady-state intracavity field amplitude.

        input_field has units sqrt(photons/s).

        Equation:
            da/dt = -(i Delta + kappa/2) a + sqrt(kappa_in) a_in

        with Delta = f_r - f_d in Hz convention:

            a = sqrt(kappa_in) a_in / (kappa/2 + i Delta)
        """
        delta = self.delta(f_drive)
        return np.sqrt(self.kappa_input) * input_field / (
            self.kappa / 2.0 + 1j * delta
        )

    def n_photon_from_input_flux(
        self,
        f_drive: float,
        photon_flux: float,
    ) -> float:
        """Intracavity photon number from input photon flux.

        photon_flux is photons/s.

        Equation:
            n = |a|^2 = kappa_in * photon_flux / (Delta^2 + (kappa/2)^2)
        """
        if photon_flux < 0:
            raise ValueError("photon_flux must be non-negative")

        delta = self.delta(f_drive)

        return (
            self.kappa_input
            * photon_flux
            / (delta**2 + (self.kappa / 2.0) ** 2)
        )

    def input_flux_for_n_photon(
        self,
        f_drive: float,
        n_photon: float,
    ) -> float:
        """Required input photon flux for target intracavity photon number.
        
        n_photon is dimensionless.
        
        Equation:
            photon_flux = n * (Delta^2 + (kappa/2)^2) / kappa_in
        """
        if n_photon < 0:
            raise ValueError("n_photon must be non-negative")

        delta = self.delta(f_drive)

        return (
            n_photon
            * (delta**2 + (self.kappa / 2.0) ** 2)
            / self.kappa_input
        )

    def n_photon_from_input_power(
        self,
        f_drive: float,
        input_power: float,
    ) -> float:
        """Intracavity photon number from input power at the chip.
        
        input_power is in Watts.
        
        Equation:
            photon_flux = input_power / (h * f_drive)
        """
        if input_power < 0:
            raise ValueError("input_power must be non-negative")
        h = 6.62607015e-34
        photon_flux = input_power / (h * f_drive)

        return self.n_photon_from_input_flux(
            f_drive=f_drive,
            photon_flux=photon_flux,
        )

    def input_power_for_n_photon(
        self,
        f_drive: float,
        n_photon: float,
    ) -> float:
        """Input power at the chip required for target intracavity photon number.
        
        n_photon is dimensionless.
        
        Equation:
            input_power = photon_flux * h * f_drive
        """
        h = 6.62607015e-34

        photon_flux = self.input_flux_for_n_photon(
            f_drive=f_drive,
            n_photon=n_photon,
        )

        return photon_flux * h * f_drive    

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