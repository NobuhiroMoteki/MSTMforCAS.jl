# MSTMforCAS.jl

Julia implementation of the Multi-Sphere T-Matrix (MSTM) method, specialized for Complex Amplitude Sensing (CAS) applications.

## Purpose

Generate surrogate model training data by computing light scattering properties of fractal aggregates of homogeneous spheres, with parameter sweeps over monomer count, fractal dimension, and complex refractive index.

Reference implementation: [MSTM-v4.0_for_CAS](https://github.com/NobuhiroMoteki/MSTM-v4.0_for_CAS) (Fortran).

## Physical Scope

- Infinite homogeneous medium, plane wave incidence along z-axis
- Aggregates of homogeneous spheres (each with its own radius)
- Fixed orientation (random orientation averaging is a future extension)
- Output per (aggregate, refractive index) pair:
  - Forward/backward scattering amplitude matrix S₁–S₄ (BH83 convention, dimensionless) and S₁₁, S₂₂, S₁₂, S₂₁ (MI02 convention, dimension of length)
  - Polarization-averaged efficiency factors Q_ext, Q_abs, Q_sca (unpolarized incidence)

## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick Start

```julia
using MSTMforCAS

# Read aggregate geometry (coordinates in μm)
agg = read_aggregate_file("aggregate.ptsa")

# Compute scattering
wavelength = 0.6328  # [μm]
n_medium   = 1.0
k = 2π * n_medium / wavelength
m_rel = (1.6 + 0.0123im) / n_medium

result = compute_scattering(agg, m_rel, k)

# Results
result.S_forward   # (S₁, S₂, S₃, S₄) BH83 at θ=0°
result.S_backward  # (S₁, S₂, S₃, S₄) BH83 at θ=180°
result.Q_ext       # extinction efficiency
result.Q_abs       # absorption efficiency
result.Q_sca       # scattering efficiency
```

## Parameter Sweep

```julia
config = SweepConfig(
    aggregate_files    = readdir("aggregates/", join=true),
    refractive_indices = [1.5+0.0im, 1.6+0.0123im, 1.7+0.1im],
    wavelength         = 0.6328,       # [μm]
    medium_refindex    = 1.0,
    monomer_radius     = 1.0,          # 1.0 if files already in physical units
)
df = run_parameter_sweep(config; output_file="results.csv")  # or "results.h5"
```

Run with multi-threading: `julia -t auto --project=. scripts/run_sweep_small.jl`

Output DataFrame includes BH83 amplitudes (S1–S4), MI02 amplitudes (S11, S22, S12, S21), volume-equivalent radius R_ve, efficiency factors, and solver diagnostics.

## Project Structure

```text
MSTMforCAS.jl/
├── Project.toml
├── src/
│   ├── MSTMforCAS.jl          # Top-level module
│   ├── MieCoefficients.jl     # Lorenz-Mie coefficients aₙ, bₙ
│   ├── TranslationCoefs.jl    # VSWF translation addition theorem
│   ├── TMatrixSolver.jl       # CBICG iterative solver
│   ├── ScatteringAmplitude.jl  # Amplitudes + cross sections
│   ├── AggregateIO.jl         # Read .ptsa/.pos geometry files
│   └── ParameterSweep.jl      # Parallel parameter sweeps
├── test/
├── scripts/                   # Example computation scripts
└── ref/                       # Reference Fortran code (gitignored)
```

## Validation

Validated against Fortran MSTM-v4.0_for_CAS reference output:

| Test case | Spheres | Q_ext %err | Q_abs %err | Q_sca %err | S1 fwd %err |
|-----------|---------|------------|------------|------------|-------------|
| testcase1 | 2       | 0.001%     | 0.050%     | 0.015%     | 0.014%      |
| testcase2 | 1000    | 0.003%     | 0.017%     | 0.001%     | 0.021%      |

Run validation: `julia --project=. scripts/validate_testcases.jl`

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## References

- Bohren, C.F. & Huffman, D.R. (1983). *Absorption and Scattering of Light by Small Particles*. Wiley.
- Mishchenko, M.I., Travis, L.D. & Lacis, A.A. (2002). *Scattering, Absorption, and Emission of Light by Small Particles*. Cambridge University Press.
- Mackowski, D.W. (1996). JOSA A, 13(11), 2266-2278.
- Mackowski, D.W. & Mishchenko, M.I. (2011). JQSRT, 112(13), 2182-2192.
- Moteki, N. & Adachi, K. (2024). Optics Express, 32(21), 36500-36522.

## License

MIT License
