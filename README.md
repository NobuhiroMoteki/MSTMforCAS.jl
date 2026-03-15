# MSTMforCAS.jl

Julia implementation of the Multi-Sphere T-Matrix (MSTM) method, specialized for Complex Amplitude Sensing (CAS) applications.

## Purpose

Generate surrogate model training data by computing light scattering properties of fractal aggregates of homogeneous spheres, with parameter sweeps over monomer count, fractal dimension, and complex refractive index.

Reference implementation: [MSTM-v4.0_for_CAS](https://github.com/NobuhiroMoteki/MSTM-v4.0_for_CAS) (Fortran/MPI).

## Physical Scope

- Infinite homogeneous medium, plane wave incidence along z-axis (fixed orientation)
- Aggregates of homogeneous spheres (each monomer with its own radius, common complex refractive index)
- Output per (aggregate, refractive index) pair:
  - Forward/backward scattering amplitude matrix S₁–S₄ (BH83 convention, dimensionless) and S₁₁, S₂₂, S₁₂, S₂₁ (MI02 convention, dimension of length)
  - Polarization-averaged efficiency factors Q_ext, Q_abs, Q_sca (unpolarized incidence)

### Convention definitions

**BH83** = Bohren & Huffman (1983), Eq. 3.12. The amplitude matrix relates scattered to incident field:

```
[E_∥_sca]   exp(ikr)  [S₂  S₃] [E_∥_inc]
[E_⊥_sca] = -------  [S₄  S₁] [E_⊥_inc]
               -ikr
```

**MI02** = Mishchenko, Travis & Lacis (2002). Conversion from BH83:

```
S₁₁(MI02) = S₂(BH83) / (-ik)    S₁₂(MI02) = S₃(BH83) / (ik)
S₂₁(MI02) = S₄(BH83) / (ik)     S₂₂(MI02) = S₁(BH83) / (-ik)
where k = k_medium = 2π × n_medium / λ₀
```

**Efficiency factors**: Q = C / (π R_ve²), where C is the cross section and R_ve = (Σ rᵢ³)^{1/3} is the volume-equivalent sphere radius.

## Installation

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Dependencies

| Package | Purpose |
|---------|---------|
| SpecialFunctions.jl | Spherical Bessel functions |
| FFTW.jl | FFT-accelerated translation |
| HDF5.jl | Output storage for large parameter sweeps |
| CSV.jl + DataFrames.jl | Alternative tabular output format |
| LinearAlgebra (stdlib) | Matrix operations |
| Printf (stdlib) | Formatted output |

## Usage

### Input parameters and units

All length inputs (positions, radii, wavelength) must be given in **the same physical unit** (e.g., all in μm, or all in m). The package does NOT assume a particular unit; consistency is all that matters.

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `wavelength` | `Float64` | Vacuum wavelength [length] | `0.6328` (μm) |
| `medium_refindex` | `Float64` | Real refractive index of surrounding medium | `1.0` (air) |
| `refractive_indices` | `Vector{ComplexF64}` | **Absolute** complex RI of sphere material (n + iκ, positive κ = absorption, BH83 sign convention). NOT relative to medium. | `[1.6+0.0123im]` |
| `aggregate_files` | `Vector{String}` | Paths to geometry files (.ptsa or .pos) | `["agg1.ptsa", "agg2.ptsa"]` |
| `monomer_radius` | `Float64` | Scale factor for file coordinates. Set `1.0` if file coordinates are already in physical units. Set to the monomer radius [length] if files use dimensionless coordinates normalized by monomer radius. | `1.0` |

**Aggregate geometry files** (.ptsa, .pos): plain ASCII, one sphere per line, 4 whitespace-separated columns:

```
x   y   z   r
```

No header. `x, y, z` = sphere center position, `r` = sphere radius (both in physical units or dimensionless — see `monomer_radius`).

### Single aggregate computation

```bash
julia --project=. scripts/run_single.jl
```

Or in Julia:

```julia
using MSTMforCAS

# Read aggregate geometry (coordinates in μm)
agg = read_aggregate_file("aggregate.ptsa")

# Physical parameters
wavelength = 0.6328   # [μm]
n_medium   = 1.0
m_sphere   = 1.6 + 0.0123im   # absolute complex RI

# Derived
k_medium = 2π * n_medium / wavelength   # [μm⁻¹]
m_rel    = m_sphere / n_medium           # relative RI

# Compute (direct translation)
result = compute_scattering(agg, m_rel, k_medium)

# Compute (FFT-accelerated translation — faster for N > ~100 spheres)
result = compute_scattering(agg, m_rel, k_medium; use_fft=true)

# Access results
result.S_forward    # (S₁, S₂, S₃, S₄) BH83, dimensionless, θ=0°
result.S_backward   # (S₁, S₂, S₃, S₄) BH83, dimensionless, θ=180°
result.Q_ext        # extinction efficiency (dimensionless)
result.Q_abs        # absorption efficiency (dimensionless)
result.Q_sca        # scattering efficiency (dimensionless)
result.converged    # solver convergence flag
result.n_iterations # number of CBICG iterations
```

### Parameter sweep

```bash
# Single thread
julia --project=. scripts/run_sweep_small.jl

# Multi-threaded (4 threads)
julia -t 4 --project=. scripts/run_sweep_small.jl
```

Or in Julia:

```julia
using MSTMforCAS

config = SweepConfig(
    aggregate_files    = readdir("aggregates/", join=true),
    refractive_indices = [1.5+0.0im, 1.6+0.0123im, 1.7+0.1im],
    wavelength         = 0.6328,      # [μm]
    medium_refindex    = 1.0,
    monomer_radius     = 1.0,         # 1.0 if files already in physical units
    max_iterations     = 200,
    convergence_epsilon = 1e-6,
)

# Output to CSV
df = run_parameter_sweep(config; output_file="results.csv")

# Output to HDF5 (more compact)
df = run_parameter_sweep(config; output_file="results.h5")
```

### Output data format

The `DataFrame` (and corresponding CSV/HDF5 file) contains one row per (aggregate_file, refractive_index) pair, with the following 43 columns:

#### Geometry and material

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `aggregate_file` | String | — | Basename of aggregate geometry file |
| `n_monomers` | Int | — | Number of monomers in aggregate |
| `R_ve` | Float64 | same as file coords | Volume-equivalent sphere radius (Σ rᵢ³)^{1/3} |
| `m_real` | Float64 | — | Real part of **absolute** refractive index |
| `m_imag` | Float64 | — | Imaginary part of **absolute** refractive index |

#### Forward scattering amplitudes (θ = 0°)

| Columns | Convention | Unit | Description |
|---------|------------|------|-------------|
| `S1_fwd_re`, `S1_fwd_im` | BH83 | dimensionless | S₁(0°) = perpendicular amplitude |
| `S2_fwd_re`, `S2_fwd_im` | BH83 | dimensionless | S₂(0°) = parallel amplitude |
| `S3_fwd_re`, `S3_fwd_im` | BH83 | dimensionless | S₃(0°) = cross-pol amplitude |
| `S4_fwd_re`, `S4_fwd_im` | BH83 | dimensionless | S₄(0°) = cross-pol amplitude |
| `S11_fwd_re`, `S11_fwd_im` | MI02 | [length] | S₁₁(0°) = S₂/(−ik) |
| `S22_fwd_re`, `S22_fwd_im` | MI02 | [length] | S₂₂(0°) = S₁/(−ik) |
| `S12_fwd_re`, `S12_fwd_im` | MI02 | [length] | S₁₂(0°) = S₃/(ik) |
| `S21_fwd_re`, `S21_fwd_im` | MI02 | [length] | S₂₁(0°) = S₄/(ik) |

#### Backward scattering amplitudes (θ = 180°)

Same structure as forward with `_bwd` suffix (16 columns total).

#### Efficiency factors and diagnostics

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `Q_ext` | Float64 | dimensionless | Extinction efficiency (unpolarized incidence) |
| `Q_abs` | Float64 | dimensionless | Absorption efficiency |
| `Q_sca` | Float64 | dimensionless | Scattering efficiency (= Q_ext − Q_abs) |
| `converged` | Bool | — | Whether CBICG solver converged |
| `n_iterations` | Int | — | Number of CBICG iterations used |

**Unit note for MI02 amplitudes**: The MI02 amplitude unit [length] is the same as the wavelength and coordinate unit. For example, if `wavelength` is in μm, then S₁₁ etc. are in μm.

**Physical cross sections** are not stored in the output, but can be computed as: C = Q × π × R_ve² (same length unit squared).

### Reading output files

**CSV:**

```julia
using CSV, DataFrames
df = CSV.read("results.csv", DataFrame)
```

or with the provided script:

```bash
julia --project=. scripts/read_results_csv.jl results.csv
```

**HDF5:**

```julia
using HDF5
h5open("results.h5", "r") do fid
    Q_ext = read(fid["Q_ext"])             # Vector{Float64}
    S1_fwd = read(fid["S1_fwd_re"]) .+ im .* read(fid["S1_fwd_im"])  # Vector{ComplexF64}
    # ... etc.
end
```

or with the provided script:

```bash
julia --project=. scripts/read_results_h5.jl results.h5
```

HDF5 files also store sweep metadata as root-level attributes: `wavelength`, `medium_refindex`, `monomer_radius`, `n_jobs`.

## Project Structure

```text
MSTMforCAS.jl/
├── Project.toml
├── src/
│   ├── MSTMforCAS.jl          # Top-level module
│   ├── MieCoefficients.jl     # Lorenz-Mie coefficients aₙ, bₙ
│   ├── TranslationCoefs.jl    # VSWF translation addition theorem
│   ├── FFTTranslation.jl      # FFT-accelerated translation (O(N + M log M))
│   ├── TMatrixSolver.jl       # CBICG iterative solver
│   ├── ScatteringAmplitude.jl  # Amplitudes + cross sections
│   ├── AggregateIO.jl         # Read .ptsa/.pos geometry files
│   └── ParameterSweep.jl      # Parallel parameter sweeps
├── test/
├── scripts/
│   ├── run_single.jl          # Single aggregate computation example
│   ├── run_sweep_small.jl     # Parameter sweep example
│   ├── validate_testcases.jl  # Validation against Fortran reference
│   ├── read_results_csv.jl    # Read and summarize CSV output
│   └── read_results_h5.jl     # Read and summarize HDF5 output
└── ref/                       # Reference Fortran code and test data (gitignored)
```

## Validation

Validated against Fortran MSTM-v4.0_for_CAS reference output. All primary quantities agree to < 0.1%.

| Test case | Spheres | Translation | Q_ext %err | Q_abs %err | Q_sca %err | S₁ fwd %err | S₂ fwd %err | Iterations |
|-----------|---------|------------|------------|------------|------------|-------------|-------------|------------|
| testcase1 | 2       | direct     | 0.001%     | 0.050%     | 0.015%     | 0.014%      | 0.001%      | 3          |
| testcase2 | 1000    | direct     | 0.003%     | 0.017%     | 0.001%     | 0.021%      | 0.016%      | 18         |
| testcase3 | 1000    | FFT        | 0.001%     | 1.6%       | 0.056%     | 0.024%      | 0.007%      | 18         |

Testcase3 uses the same geometry and refractive index as testcase2 but with FFT-accelerated translation. The larger Q_abs error (1.6%) is expected because Q_abs = Q_ext − Q_sca is a small residual (Q_abs ≈ 0.37 vs Q_ext ≈ 11.2), amplifying the FFT approximation error. The Fortran FFT reference shows the same effect.

Run validation:

```bash
julia --project=. scripts/validate_testcases.jl
```

## Performance

### Direct vs FFT translation

The FFT-accelerated translation replaces O(N²) pairwise sphere-to-sphere translation with O(N + M log M) grid-based convolution (M = number of grid cells). This provides a large speedup for aggregates with many monomers.

| Mode | Algorithm | Complexity | Best for |
|------|-----------|------------|----------|
| `use_fft=false` (default) | Direct pairwise | O(N² × n_order²) per iteration | N < ~100 |
| `use_fft=true` | FFT convolution + near-field correction | O(N + M log M) per iteration | N > ~100 |

### Benchmark: testcase2/3 (1000 spheres, x=1.0, m=1.6+0.0123i)

All timings on a single core (no MPI, no multi-threading). Julia version: 1.11, Fortran: MSTM-v4.0_for_CAS with `mpiexec -n 1`.

| | Julia direct | Julia FFT | Fortran direct (1 proc) | Fortran FFT (1 proc) |
|---|---|---|---|---|
| Solver time | 413 s | 10.5 s | 96.4 s | 3.3 s |
| Speedup vs Julia direct | 1× | **40×** | — | — |
| Iterations | 18 | 18 | 18 | 18 |
| Peak memory | ~2 GB | ~0.3 GB | ~0.5 GB | ~0.1 GB |

**Julia vs Fortran speed**: The Julia direct solver is ~4× slower than single-process Fortran for this test case, primarily because the Fortran code uses optimized in-place matrix operations and pre-allocated work arrays at a lower level. The Julia FFT solver is ~3× slower than the Fortran FFT solver. These ratios are expected to improve with further Julia-side optimization (e.g., LoopVectorization, precomputed rotation matrices).

**Memory**: Julia's higher memory usage is due to garbage collector overhead and less aggressive in-place reuse compared to Fortran. For practical CAS parameter sweeps (N ≤ 1000), memory is not a bottleneck.

**Multi-threading**: The parameter sweep (`run_parameter_sweep`) parallelizes over the (aggregate × refractive_index) grid via `Threads.@threads`. Start Julia with `julia -t auto` or `julia -t N` for speedup. Thread safety: the translation coefficient cache is pre-warmed single-threaded before the parallel section.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## References

- Bohren, C.F. & Huffman, D.R. (1983). *Absorption and Scattering of Light by Small Particles*. Wiley.
- Mishchenko, M.I., Travis, L.D. & Lacis, A.A. (2002). *Scattering, Absorption, and Emission of Light by Small Particles*. Cambridge University Press.
- Mackowski, D.W. (1996). JOSA A, 13(11), 2266-2278.
- Mackowski, D.W. & Mishchenko, M.I. (2011). JQSRT, 112(13), 2182-2192.
- Mackowski, D.W. (2014). Direct simulation of scattering and absorption by particle deposits. Proc. ICHMT.
- Moteki, N. & Adachi, K. (2024). Optics Express, 32(21), 36500-36522.

## License

MIT License
