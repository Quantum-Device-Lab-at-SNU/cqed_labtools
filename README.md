# cqed-labtools

Lightweight Python tools for superconducting-circuit experiment calibration and modeling.

`cqed-labtools` is intended for circuit-QED experimental workflows where one wants to move between measured quantities, approximate dispersive formulas, and numerical Hamiltonian diagonalization.

## Current components

- `Transmon`  
  Charge-basis transmon Hamiltonian, transition frequencies, anharmonicity, charge matrix elements, and extraction of `EJ/h` and `EC/h` from `f01` and anharmonicity.

- `ReadoutResonator`  
  Readout resonator model with bare resonator frequency, optional qubit-resonator coupling `g/2pi`, Fock-space operators, resonator Hamiltonian, and optional input-output port information for linewidths and photon-number calibration.

- `ChargeDriveLine`  
  Capacitive charge-drive line with attenuation and Rabi-frequency estimates.

- `TransmonCircuit`  
  Composite transmon + readout + optional drive-line model with charge-basis x Fock-basis diagonalization, bare and dressed frequencies, dispersive shift, Purcell estimate, photon-number estimates, and construction from measured dressed parameters.

## Installation

Editable install from the repository root:

```bash
pip install -e .
```

Standard install:

```bash
pip install .
```

Developer test dependencies:

```bash
pip install -e .[dev]
pytest
```

## Basic example

```python
from cqed_labtools import Transmon, ReadoutResonator, ChargeDriveLine, TransmonCircuit

GHz = 1e9
MHz = 1e6

transmon = Transmon.from_f01_anharmonicity(
    f01=5.0 * GHz,
    anharmonicity=-250 * MHz,
    n_cutoff=20,
    solver="exact",
)

readout = ReadoutResonator(
    frequency=6.5 * GHz,
    g_over_2pi=100 * MHz,
    photon_cutoff=4,
)

circuit = TransmonCircuit(
    transmon=transmon,
    readout_resonator=readout,
)

print(circuit.f_qubit(0, representation="bare", solver="exact") / GHz)
print(circuit.f_qubit(0, representation="dressed", solver="exact") / GHz)
print(circuit.f_resonator_for_transmon_level(0, solver="exact") / GHz)
print(circuit.dispersive_shift(solver="exact") / MHz)
```

## Adding input-output port information

A `ReadoutResonator` can be initialized with only coherent Hamiltonian information. Linewidths and port geometry can be added only when needed for photon-number calibration, quality factors, or input-output calculations.

For example:

```python
readout = ReadoutResonator(
    frequency=6.5 * GHz,
    g_over_2pi=100 * MHz,
    photon_cutoff=4,
)

readout = readout.with_single_sided_port(
    kappa_external_over_2pi=1.0 * MHz,
    kappa_internal_over_2pi=0.05 * MHz,
)
```

or for a side-coupled feedline geometry:

```python
readout = readout.with_side_coupled_port(
    kappa_external_over_2pi=1.0 * MHz,
    kappa_internal_over_2pi=0.05 * MHz,
)
```

Then photon-number calibration can be used:

```python
nbar = readout.n_photon_from_input_power(
    f_drive=readout.frequency,
    input_power=1e-15,
)

print(nbar)
```

## Charge-drive example

```python
drive = ChargeDriveLine(
    coupling_capacitance=0.1e-15,
    attenuation_db=60,
)

circuit = TransmonCircuit(
    transmon=transmon,
    readout_resonator=readout,
    charge_drive_line=drive,
)

f_rabi = circuit.rabi_frequency_from_charge_voltage(
    voltage=0.1,
    voltage_is_at_chip=False,
)

print(f_rabi / MHz)
```

## Constructing from measured dressed quantities

The package provides constructors for estimating a bare circuit model from measured dressed quantities.

### From dressed frequencies and punchout shift

```python
circuit = TransmonCircuit.from_dressed_freqs_with_punchout_shift(
    f_qubit_dressed=5.0 * GHz,
    f_resonator_ground=6.5 * GHz,
    anharmonicity_dressed=-250 * MHz,
    punchout_shift=2.0 * MHz,
    kappa_input_over_2pi=1.0 * MHz,
    kappa_internal_over_2pi=0.05 * MHz,
    coupling_geometry="single_sided",
    n_cutoff=20,
    photon_cutoff=4,
    solver="exact",
)
```

Here `punchout_shift` is interpreted as

```text
punchout_shift = f_resonator_bare - f_resonator_ground.
```

For `solver="approx"`, the constructor uses perturbative dispersive formulas.  
For `solver="exact"`, the initial estimate is refined using exact diagonalization of the composite transmon-readout Hamiltonian.

### From dressed frequencies and dispersive shift

```python
circuit = TransmonCircuit.from_dressed_freqs_with_dispersive_shift(
    f_qubit_dressed=5.0 * GHz,
    f_resonator_ground=6.5 * GHz,
    anharmonicity_dressed=-250 * MHz,
    chi_over_2pi=-1.0 * MHz,
    kappa_input_over_2pi=1.0 * MHz,
    kappa_internal_over_2pi=0.05 * MHz,
    coupling_geometry="single_sided",
    n_cutoff=20,
    photon_cutoff=4,
    solver="exact",
)
```

The dispersive shift is defined as

```text
chi/2pi = (f_resonator_e - f_resonator_g) / 2.
```

## Conventions

Most frequency-like quantities are ordinary frequencies in Hz, not angular frequencies.

- Quantities ending in `_over_2pi` are in Hz.
- Quantities such as `kappa`, `g`, and angular detunings are in rad/s.
- Hamiltonians returned by `Transmon.hamiltonian()`, `ReadoutResonator.hamiltonian()`, and `TransmonCircuit.full_hamiltonian()` are expressed as `H/h`, so matrix elements are in Hz.
- `representation="bare"` refers to uncoupled component frequencies.
- `representation="dressed"` refers to frequencies of the coupled transmon-readout system.
- `solver="approx"` uses perturbative dispersive formulas.
- `solver="exact"` uses numerical diagonalization where available.

## Development notes

The component classes are designed to keep coherent Hamiltonian parameters separate from input-output parameters. In particular, `g_over_2pi` denotes the coherent qubit-resonator coupling, while `kappa_*` parameters describe resonator linewidths and ports.
