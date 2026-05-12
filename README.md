# axibem

Axisymmetric collocation Boundary Element Method (BEM) solver for
electrostatic capacitance problems on bodies of revolution.

## Scope

`axibem` discretises an axisymmetric conducting surface into ring panels of
constant surface charge density and assembles the boundary integral
equation for the electrostatic potential. The rotationally-integrated
free-space Green's function reduces to a complete elliptic integral of the
first kind, which is evaluated via `scipy.special.ellipk`.

Reference problems:

- isolated thin charged disk of radius `R` (Kelvin: `C = 8 ε₀ R`)
- coaxial truncated cylinders, axisymmetric needle, oblate spheroid

## Usage

```python
from axibem.disk import disk_capacitance

C = disk_capacitance(radius=1.0, n_panels=200)
# compare with the Kelvin closed form 8 * eps0 * R
```

## Install

```
pip install -e .
```

## Tests

```
pytest -q
```

## License

MIT
