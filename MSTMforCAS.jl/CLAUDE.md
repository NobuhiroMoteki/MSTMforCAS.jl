# CLAUDE.md — MSTMforCAS.jl

## Project Overview

Julia implementation of the Multi-Sphere T-Matrix (MSTM) method, specialized for
Complex Amplitude Sensing (CAS) applications. This is NOT a general-purpose MSTM code;
it targets a specific, well-defined computational task.

**Purpose:** Generate surrogate model training data by computing light scattering
properties of fractal aggregates of homogeneous spheres over large parameter sweeps
(monomer count, fractal dimension, complex refractive index).

**Reference implementation:** Mackowski's MSTM v4.0 Fortran code
(https://github.com/dmckwski/MSTM), modified version at
https://github.com/NobuhiroMoteki/MSTM-v4.0_for_CAS

## Physical Scope (FIXED — do not expand without explicit instruction)

- Infinite homogeneous medium (no layered structures, no plane boundaries)
- Homogeneous sphere monomers (no core-shell, no optically active materials)
- Plane wave incidence only (no Gaussian beam)
- Fixed orientation (random orientation averaging is a future extension)
- Output: forward/backward scattering amplitudes + polarization-averaged cross sections

## Output Specification

For each (aggregate geometry, complex refractive index) pair, compute:

1. Forward scattering amplitude matrix at θ=0°: S₁, S₂, S₃, S₄ (BH83 convention) — 4 complex numbers
2. Backward scattering amplitude matrix at θ=180°: S₁, S₂, S₃, S₄ (BH83 convention) — 4 complex numbers
3. Polarization-averaged cross sections: C_ext, C_abs, C_sca — 3 real numbers

Convention reference:
- BH83 = Bohren & Huffman 1983, "Absorption and Scattering of Light by Small Particles", Eq. 3.12
- MI02 = Mishchenko, Travis & Lacis 2002, "Scattering, Absorption, and Emission of Light by Small Particles"
- Conversion: S₁₁(MI02) = S₂(BH83)/(-ik), S₂₂(MI02) = S₁(BH83)/(-ik),
              S₁₂(MI02) = S₃(BH83)/(ik),  S₂₁(MI02) = S₄(BH83)/(ik)
  where k = vacuum wavenumber × medium refractive index

## Module Architecture

```
src/
├── MSTMforCAS.jl          # Top-level module, re-exports public API
├── MieCoefficients.jl     # Step 1: Mie scattering coefficients aₙ, bₙ
├── TranslationCoefs.jl    # Step 2: VSWF translation addition theorem coefficients
├── TMatrixSolver.jl       # Step 3: Multi-sphere interaction equation solver
├── ScatteringAmplitude.jl # Step 4: Forward/backward amplitudes + cross sections
├── AggregateIO.jl         # Read aggregate_generator_PTSA output files
└── ParameterSweep.jl      # Step 5: Orchestrate parameter sweeps, parallel execution
```

## Computational Flow

```
Input: aggregate file (positions, radii) + m (complex refractive index) + k (wavenumber)
  │
  ▼
Step 1: MieCoefficients
  For each monomer sphere, compute Mie coefficients aₙ, bₙ (n = 1, ..., N_order)
  using Bohren-Huffman algorithm (downward recurrence for ψₙ/ξₙ ratios)
  │
  ▼
Step 2: TranslationCoefs
  For each sphere pair (i,j), compute VSWF translation coefficients
  A^{ij}_{mnp,mlq} and B^{ij}_{mnp,mlq}
  Algorithm: Mackowski 1996 (JOSA A 13:2266) recursive scheme
  │
  ▼
Step 3: TMatrixSolver
  Solve the multi-sphere interaction equation:
    aᵢ = Tᵢ · (pᵢ + Σ_{j≠i} H_{ij} · aⱼ)
  where Tᵢ = Mie T-matrix of sphere i, pᵢ = incident field coefficients,
  H_{ij} = translation operator from j to i
  Method: Order-of-scattering iteration (primary) or BiCGSTAB (fallback)
  │
  ▼
Step 4: ScatteringAmplitude
  From solved expansion coefficients {aᵢ}, compute:
  - S₁, S₂, S₃, S₄ at θ=0° and θ=180° (BH83 Eq. 3.12)
    (exploit special values of associated Legendre functions at cos θ = ±1)
  - C_ext via optical theorem: C_ext = (4π/k²) Re[S(0°)] averaged over 2 polarizations
  - C_abs from individual sphere absorption (internal field coefficients)
  - C_sca = C_ext - C_abs
  │
  ▼
Step 5: ParameterSweep
  Loop over: aggregate files × complex refractive indices
  Parallelism: Julia multi-threading (Threads.@threads) over parameter grid
  Output: HDF5 or CSV with all results
```

## Validation Strategy

### Unit tests (per module)

- **MieCoefficients:** Compare with MieScat_Py (https://github.com/NobuhiroMoteki/MieScat_Py)
  for known size parameters and refractive indices. Also compare with
  Bohren-Huffman Table 4.1/4.3 values.

- **TranslationCoefs:** 2-sphere axially symmetric case, compare with Fortran MSTM output.
  Also verify symmetry relations: A^{ij} ↔ A^{ji} reciprocity.

- **TMatrixSolver:** Single sphere case must reduce to Mie theory exactly.
  2-sphere case compared with Fortran MSTM.

- **ScatteringAmplitude:**
  - Single homogeneous sphere: S₁(0°) = S₂(0°), S₃ = S₄ = 0 (spherical symmetry)
  - Optical theorem consistency: C_ext computed from S(0°) vs computed from coefficients
  - Energy conservation: C_ext = C_sca + C_abs (always)
  - Non-absorbing sphere (Im(m) = 0): C_abs → 0

### End-to-end tests

- Compare full output (8 complex + 3 real) against Fortran MSTM-v4.0_for_CAS
  for several aggregate geometries from aggregate_generator_PTSA

## Key Numerical Considerations

- Floating point: use Float64 (ComplexF64) throughout
- Mie coefficient computation: use logarithmic derivative ratio method (Wiscombe 1980)
  to avoid overflow for large size parameters
- Translation coefficients: Mackowski's recursive scheme is numerically stable;
  do NOT use direct factorial expressions
- Convergence criterion for T-matrix iteration: relative change in scattering
  coefficients < 1e-6 (configurable)
- Validation tolerance: relative error < 1e-10 against Fortran reference
  (exact bit-match is NOT expected due to compiler/LLVM differences)

## Dependencies

- SpecialFunctions.jl  — spherical Bessel functions, gamma functions
- HDF5.jl             — output storage for large parameter sweeps
- CSV.jl + DataFrames.jl — alternative output format
- Test (stdlib)        — unit testing
- LinearAlgebra (stdlib) — matrix operations
- Printf (stdlib)      — formatted output

## Coding Conventions

- All angles in radians internally; degree conversion only at I/O boundary
- Wavenumber k = 2π/λ × n_medium (in medium, not vacuum)
- Size parameter x = k × radius (in medium)
- Complex refractive index m = n + iκ (positive imaginary part = absorption)
  NOTE: Some references use m = n - iκ. We follow BH83 convention (positive κ = absorption).
- Array indexing: 1-based (native Julia)
- Multipole order n starts from 1 (not 0)
- Module-level docstrings in Julia format for all public functions
- Type annotations on all function signatures

## Input Format

Aggregate geometry files from aggregate_generator_PTSA:
- Text file, one line per monomer
- Each line: x y z r (position and radius, whitespace-separated)
- Coordinates normalized by monomer radius (dimensionless) or in physical units
  (the code should handle both via a scale_factor parameter)

## Development Workflow

1. Implement and test modules bottom-up: MieCoefficients → TranslationCoefs →
   TMatrixSolver → ScatteringAmplitude → ParameterSweep
2. Each module must pass its unit tests before proceeding to the next
3. Run `julia --project=. -e 'using Pkg; Pkg.test()'` to execute full test suite
4. Format check: no strict formatter enforced, but keep consistent style

## File Locations

- Source: `src/`
- Tests: `test/`
- Design notes and reference PDFs: `.claude/` (gitignored)
- Example aggregate files: `test/fixtures/`
- Sweep scripts: `scripts/`

## References

- Bohren, C.F. & Huffman, D.R. (1983). Absorption and Scattering of Light by Small Particles. Wiley.
- Mackowski, D.W. (1996). Calculation of total cross sections of multiple-sphere clusters. JOSA A, 13(11), 2266-2278.
- Mackowski, D.W. & Mishchenko, M.I. (2011). A multiple sphere T-matrix Fortran code for use on parallel computer clusters. JQSRT, 112(13), 2182-2192.
- Mishchenko, M.I., Travis, L.D. & Lacis, A.A. (2002). Scattering, Absorption, and Emission of Light by Small Particles. Cambridge University Press.
- Wiscombe, W.J. (1980). Improved Mie scattering algorithms. Applied Optics, 19(9), 1505-1509.
- Xu, Y.-l. (1996). Calculation of the addition coefficients in electromagnetic multisphere-scattering theory. J. Comput. Phys., 127(2), 285-298.
- Moteki, N. & Adachi, K. (2024). CAS-v2 protocol. Optics Express, 32(21), 36500-36522.
