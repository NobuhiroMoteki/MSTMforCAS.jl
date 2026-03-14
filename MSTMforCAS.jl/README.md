# MSTMforCAS.jl

Julia implementation of the Multi-Sphere T-Matrix (MSTM) method, specialized for Complex Amplitude Sensing (CAS) applications.

## Purpose

Generate surrogate model training data by computing light scattering properties of fractal aggregates of homogeneous spheres, with parameter sweeps over monomer count, fractal dimension, and complex refractive index.

## Physical Scope

- Infinite homogeneous medium (plane wave incidence)
- Aggregates of identical homogeneous spheres
- Fixed orientation
- Output: forward/backward complex scattering amplitudes (S₁, S₂, S₃, S₄ in BH83 convention) and polarization-averaged cross sections (C_ext, C_abs, C_sca)

## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick Start

```julia
using MSTMforCAS

# Read aggregate geometry
agg = read_aggregate_file("aggregate.dat"; scale_factor=50e-9)  # 50nm monomer radius

# Compute scattering for a single refractive index
k = 2π / 532e-9  # 532 nm wavelength in vacuum, air medium
m_rel = 1.58 + 0.0im  # relative refractive index
result = compute_scattering(agg, m_rel, ComplexF64(k))

# Access results
println("S₁(0°) = ", result.S_forward[1])
println("C_ext  = ", result.C_ext)
```

## Parameter Sweep

```julia
config = SweepConfig(
    aggregate_files = readdir("aggregates/", join=true),
    refractive_indices = [1.5+0im, 1.5+0.01im, 1.5+0.1im, 1.75+0im],
    wavelength = 532e-9,
    medium_refindex = 1.0,
    monomer_radius = 50e-9,
)
results = run_parameter_sweep(config; output_file="sweep_results.h5")
```

Run with multi-threading: `julia -t auto --project=. scripts/sweep.jl`

## Testing

```julia
using Pkg
Pkg.test()
```

## References

- Bohren & Huffman (1983), *Absorption and Scattering of Light by Small Particles*
- Mackowski (1996), JOSA A 13:2266
- Mackowski & Mishchenko (2011), JQSRT 112:2182
- Moteki & Adachi (2024), Optics Express 32:36500

## License

MIT License
