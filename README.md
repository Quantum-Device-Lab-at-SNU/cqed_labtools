# cqed-labtools

Lightweight Python tools for superconducting-circuit experiment calibration and modeling.

Current components:

- `Transmon`: charge-basis transmon Hamiltonian, transition frequencies, anharmonicity, charge matrix elements, and numerical extraction of `EJ/h`, `EC/h` from `f01` and anharmonicity.
- `ReadoutResonator`: readout resonator with single-sided, two-sided, or side-coupled input-output geometry, photon-number calibration, Fock-space operators, and scattering parameters.
- `ChargeDriveLine`: capacitive charge-drive line with attenuation and Rabi-frequency estimates.
- `TransmonCircuit`: composed transmon + readout + optional drive line with full charge-basis x Fock-basis diagonalization, dressed frequencies, dispersive shift, Purcell estimate, and fitting from measured dressed parameters.

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

## Example

```python
from cqed_labtools import Transmon, ReadoutResonator, ChargeDriveLine, TransmonCircuit

GHz = 1e9
MHz = 1e6

transmon = Transmon.from_f01_anharmonicity(
    f01=5.0 * GHz,
    anharmonicity=-250 * MHz,
    n_cutoff=20,
)

readout = ReadoutResonator.single_sided(
    bare_frequency=6.5 * GHz,
    kappa_external_over_2pi=1.0 * MHz,
    kappa_internal_over_2pi=0.05 * MHz,
    g_over_2pi=100 * MHz,
    photon_cutoff=4,
)

drive = ChargeDriveLine(coupling_capacitance=0.1e-15, attenuation_db=60)

circuit = TransmonCircuit(transmon, readout, drive)

print(circuit.f_qubit_dressed() / GHz)
print(circuit.f_resonator_ground / GHz)
print(circuit.chi_readout_over_2pi() / MHz)
print(circuit.purcell_limited_t1)
print(circuit.rabi_frequency_from_charge_voltage(0.1, voltage_is_at_chip=False) / MHz)
```

## Notes

Most frequency-like quantities are ordinary frequencies in Hz, not angular frequencies. Rate properties ending in `_over_2pi` are in Hz. Properties without `_over_2pi`, such as `kappa`, are angular rates in rad/s.
