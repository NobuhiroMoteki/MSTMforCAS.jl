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
Source code and documents are in `ref/MSTM-v4.0_for_CAS/` (gitignored).

## Physical Scope (FIXED — do not expand without explicit instruction)

- Infinite homogeneous medium (no layered structures, no plane boundaries)
- Homogeneous sphere monomers, each with its own radius and refractive index
  (aggregates of homogeneous spheres — no core-shell, no optically active materials)
- Plane wave incidence only (no Gaussian beam)
- Fixed orientation (random orientation averaging is a future extension)
- Output: forward/backward scattering amplitudes + polarization-averaged cross section efficiencies

## Output Specification

For each (aggregate geometry, complex refractive index) pair, compute:

1. Forward scattering amplitude matrix at θ=0°: S₁, S₂, S₃, S₄ (BH83 convention) — 4 complex numbers
2. Backward scattering amplitude matrix at θ=180°: S₁, S₂, S₃, S₄ (BH83 convention) — 4 complex numbers
3. Polarization-averaged scattering efficiencies (unpolarized incidence): Q_ext, Q_abs, Q_sca — 3 real numbers
   (dimensionless: Q = cross_section / π / a_eff², where a_eff = volume-equivalent sphere radius)

Convention reference:
- BH83 = Bohren & Huffman 1983, "Absorption and Scattering of Light by Small Particles", Eq. 3.12
  The amplitude matrix relates scattered to incident field as:
    [E_∥_sca]   exp(ikr) [S₂  S₃] [E_∥_inc]
    [E_⊥_sca] = ------- × [S₄  S₁] × [E_⊥_inc]
                  -ikr
- MI02 = Mishchenko, Travis & Lacis 2002
- Conversion: S₁₁(MI02) = S₂(BH83)/(-ik), S₂₂(MI02) = S₁(BH83)/(-ik),
              S₁₂(MI02) = S₃(BH83)/(ik),  S₂₁(MI02) = S₄(BH83)/(ik)
  where k = vacuum wavenumber × medium refractive index = k_medium

**NOTE on S₃, S₄:** These are zero only for special symmetry cases (single sphere,
symmetric 2-sphere dimer). For general aggregates, S₃ and S₄ are non-zero.

## Validation Test Cases

Three test cases are defined in `ref/MSTM-v4.0_for_CAS/code/`:

### testcase1 (2-sphere dimer, serial)

- `mstm-v4.0_testcase1.inp`, run with: `mpiexec -n 1 ./mstm-ampmat_fb-mpi mstm-v4.0_testcase1.inp`
- 2 spheres: sphere1 at (0,0,0) r=0.2, sphere2 at (0.4,0,0) r=0.2 (in input units)
- `length_scale_factor = 7.534` → size parameters x₁=x₂ = 7.534×0.2 = 1.507
- Refractive index per sphere: m = 1.58+0.05i (given inline in .inp)
- Medium: n_medium = 1.0
- Incident: β=α=0° (plane wave along z-axis)
- Expected output (`testcase1_output.dat`):
  - Forward: S₁=(1.147-2.425i), S₂=(1.504-2.258i), S₃≈0, S₄≈0
  - Backward: S₁=(0.0819-0.1902i), S₂=(-0.4150-0.1322i), S₃≈0, S₄≈0
  - Q_ext(unpol)=1.4708, Q_abs(unpol)=0.3185, Q_sca(unpol)=1.1523
  - T-matrix order N=5, converged in 3 iterations

### testcase2 (1000-sphere aggregate, MPI, direct translation)

- `mstm-v4.0_testcase2.inp`, run with: `mpiexec -n 4 ./mstm-ampmat_fb-mpi mstm-v4.0_testcase2.inp`
- 1000 spheres from `random_in_cylinder_1000.pos` (4-column file: x y z r)
- `length_scale_factor = 1.0`, `ref_index_scale_factor = (1.6, 0.0123)` (m = 1.6+0.0123i for all spheres)
- All radii = 1.0 → size parameter x = 1.0 for each sphere
- Medium: n_medium = 1.0
- `fft_translation_option = .false.` (direct translation)
- Expected output (`testcase2_output.dat`):
  - Forward: S₁=(261.5-181.5i), S₂=(261.9-181.8i), S₃=(-0.6731-1.653i), S₄=(0.3981-1.356i)
  - Backward: S₁=(2.566-7.266i), S₂=(-1.419+10.34i), S₃=(0.02001+0.08706i), S₄=(-0.02373-0.08486i)
  - Q_ext(unpol)=11.190, Q_abs(unpol)=0.3652, Q_sca(unpol)=10.825
  - T-matrix order N=31, converged in 18 iterations

### testcase3 (same as testcase2 but with FFT translation)

- `mstm-v4.0_testcase3.inp`, run with: `mpiexec -n 4 ./mstm-ampmat_fb-mpi mstm-v4.0_testcase3.inp`
- Same geometry as testcase2, but `fft_translation_option = .true.`
- Slightly different results due to FFT approximation (~0.1% difference from testcase2)
- Expected output (`testcase3_output.dat`):
  - Forward: S₁=(261.5-181.3i), S₂=(261.9-181.6i), S₃=(-0.6812-1.650i), S₄=(0.4180-1.379i)
  - Q_ext(unpol)=11.188, converged in 18 iterations (3.26s vs 96.4s for testcase2)

**Validation criterion:** Each Julia output must agree with the corresponding Fortran reference
output to within 0.01% relative error in all quantities.

## Module Architecture

```
src/
├── MSTMforCAS.jl          # Top-level module, re-exports public API
├── MieCoefficients.jl     # Step 1: Mie scattering coefficients aₙ, bₙ
├── TranslationCoefs.jl    # Step 2: VSWF translation addition theorem coefficients
├── TMatrixSolver.jl       # Step 3: Multi-sphere interaction equation solver
├── ScatteringAmplitude.jl # Step 4: Forward/backward amplitudes + cross sections
├── AggregateIO.jl         # Read aggregate geometry files (.ptsa, .pos)
└── ParameterSweep.jl      # Step 5: Orchestrate parameter sweeps, parallel execution
```

## Computational Flow

```
Input: aggregate file (x,y,z,r per sphere) + m_sphere (complex RI) + length_scale_factor
  │
  ▼
Preprocessing
  Scale all positions and radii: r_scaled = length_scale_factor × r_file
  (r_scaled is the dimensionless size parameter x = k × r, where k = length_scale_factor)
  │
  ▼
Step 1: MieCoefficients  [module mie in Fortran; validated against MieScat_Py]
  For each sphere i, compute Lorenz-Mie coefficients a_n^{(i)}, b_n^{(i)} (n=1,...,N_i)
  using the BH83 (Bohren & Huffman 1983, Eq. 4.88) algorithm:

    Input: x_i = k_medium × r_i  (size parameter in medium)
           m_r = m_sphere / m_medium  (complex relative refractive index)

    1. Downward recurrence for logarithmic derivative D_n(m_r * x):
         n_start = floor(max(N_max, |m_r*x|)) + 15
         D_{n-1}(y) = n/y - 1 / (D_n(y) + n/y),   n = n_start, n_start-1, ..., 1
         (numerically stable downward recurrence; y = m_r * x)

    2. Forward recurrence for ψ_n(x) via ratio R_n = ψ_n/ψ_{n-1}:
         Downward: R_n = 1 / ((2n+1)/x - R_{n+1}),  R_{N_start} ≈ 0
         ψ_0 = sin(x), then ψ_n = R_n * ψ_{n-1}

    3. Forward recurrence for χ_n(x) (second kind):
         χ_0 = -cos(x), χ_1 = χ_0/x - sin(x)
         χ_n = ((2n-1)/x) * χ_{n-1} - χ_{n-2}

    4. ξ_n(x) = ψ_n(x) + i*χ_n(x)  (Riccati-Hankel)

    5. Mie coefficients (BH83 Eq. 4.88):
         a_n = (D_n(m_r*x)/m_r + n/x) * ψ_n(x) - ψ_{n-1}(x)
               ─────────────────────────────────────────────────
               (D_n(m_r*x)/m_r + n/x) * ξ_n(x) - ξ_{n-1}(x)

         b_n = (m_r * D_n(m_r*x) + n/x) * ψ_n(x) - ψ_{n-1}(x)
               ──────────────────────────────────────────────────
               (m_r * D_n(m_r*x) + n/x) * ξ_n(x) - ξ_{n-1}(x)

  Mie order determination (mirrors Fortran mieoa convergence):
    N_start = nint(x + 4*x^(1/3)) + 5   [upper bound]
    Converge: reduce N until Q_ext converges to within mie_eps = 1e-6
    Actual N for testcases: N=5 (x=1.507, m=1.58+0.05i), N=4 (x=1.0, m=1.6+0.0123i)

  Output: a_n, b_n for n=1,...,N  (used as diagonal T-matrix entries in solver)
  Also compute: Q_ext_single = (2/x²) Σ(2n+1) Re(a_n+b_n)
                Q_sca_single = (2/x²) Σ(2n+1) (|a_n|²+|b_n|²)
  │
  ▼
Step 2: TranslationCoefs  [module translation in Fortran]
  For each sphere pair (i,j) with i≠j, compute VSWF translation matrix H_{ij}
  using rotation-translation-rotation decomposition:
    1. Rotate coordinate frame so j→i vector aligns with z-axis (rotcoef)
    2. Compute axial translation coefficients via recurrence (axialtrancoefrecurrence)
    3. Rotate back (rotcoef + ephicoef)
  Reference: Mackowski 1996 (JOSA A 13:2266), Xu 1996 (J. Comput. Phys. 127:285)
  │
  ▼
Step 3: TMatrixSolver  [module solver in Fortran]
  Solve the multi-sphere interaction linear system:
    a_i = T_i · (p_i^{inc} + Σ_{j≠i} H_{ij} · a_j)
  where T_i = Mie T-matrix of sphere i, p_i^{inc} = incident plane wave coefficients,
  H_{ij} = VSWF translation matrix from j to i

  Algorithm: CBICG (Conjugate Biconjugate Gradient) iterative solver
  Convergence: relative residual < solution_eps = 1e-6 (default)
  Max iterations: 10000 (default)
  Additional convergence check: relative change in Q_ext < convergence_eps = 1e-6

  Expansion coefficients stored as: amn0(m1, n, p, q) where:
    - m1 = m for m≥0, m1 = n+1 for m<0 (with n1=-m in the latter case)
    - n = multipole order
    - p = 1 (TE) or 2 (TM) for scattered field polarization
    - q = 1 or 2 for incident field polarization
  │
  ▼
Step 4: ScatteringAmplitude  [subroutine scatteringmatrix in Fortran]
  From solved expansion coefficients amn0, compute amplitude vector sa(4):

    sa = 0; qsca = 0
    for n=1:N, m=-n:n, p=1:2:
      (m1,n1) = m≥0 ? (m,n) : (n+1,-m)
      qsca += |amn0(m1,n1,p,1)|² + |amn0(m1,n1,p,2)|²
      a = amn0(m1,n1,p,1)          [phi=0 case]
      b = -amn0(m1,n1,p,2)
      sa(1) += (-i)^n       * τ(m1,n1,3-p) * b * exp(i*m*phi)
      sa(2) += i*(-i)^n     * τ(m1,n1,p)   * a * exp(i*m*phi)
      sa(3) += i*(-i)^n     * τ(m1,n1,p)   * b * exp(i*m*phi)
      sa(4) += (-i)^n       * τ(m1,n1,3-p) * a * exp(i*m*phi)
    qsca *= 2
    sa *= 4 / sqrt(1/π)        [normalize_s11=false: qsca replaced by 1/π]
    sa *= 4 * sqrt(π)          [equivalently]

  where τ(m,n,p) = angular functions from taufunc (based on Wigner d-matrix at cos θ)

  Final BH83 amplitudes written to output:
    S_i (BH83) = sa(i) / (-sqrt(4π))  for i=1,2,3,4

  Therefore: S_i (BH83) = [4*sqrt(π)*raw_sum_i] / (-sqrt(4π)) = -2 * raw_sum_i
  where raw_sum_i = Σ_{n,m,p} [(-i)^n or i*(-i)^n] * τ * a/b * exp(imφ)

  Cross sections: Q_ext, Q_abs, Q_sca (dimensionless efficiencies)
  - Q_ext = -2/x² * Σ_n (2n+1) * Re[a_n(1,1) + b_n(1,1)]  (optical theorem, single sphere)
  - Q_abs = Q_ext - Q_sca
  - Q_sca = 2/x² * Σ_n (2n+1) * (|a_n|² + |b_n|²)
  - For multi-sphere: summed contributions per sphere, normalized by appropriate area
  │
  ▼
Step 5: ParameterSweep
  Loop over: aggregate files × complex refractive indices
  Parallelism: Julia multi-threading (Threads.@threads) over parameter grid
  Output: HDF5 or CSV with all results
```

## Input File Formats

### Aggregate geometry files (.ptsa and .pos)

Both formats are identical: plain ASCII text, one line per sphere, 4 columns:

```
x  y  z  r
```

- x, y, z: sphere center position (physical units, e.g., μm)
- r: sphere radius (same physical units)
- Whitespace-separated, scientific notation (e.g., `2.7366866e-02  -2.4160867e-02  1.1478261e-02  2.2460919e-02`)
- NO header line, NO refractive index per sphere (RI is given globally)

### Key input parameters (from MSTM .inp files)

- `length_scale_factor` = k_medium = 2π × n_medium / λ₀ (wavenumber in medium, units: 1/length).
  Multiplying file coordinates/radii by this factor gives dimensionless size parameters x = k × r.
- `ref_index_scale_factor = (n_real, n_imag)` = complex refractive index of spheres m_sphere
  (absolute, not relative to medium; all spheres share one value in parameter sweep mode).
- Medium refractive index n_medium is embedded in `length_scale_factor` (not separately specified).
  For testcases 1-3: n_medium = 1.0 (air), so length_scale_factor = k_0 = k_medium.
- `incident_beta_deg = 0`, `incident_alpha_deg = 0`: plane wave along z-axis.
- `fft_translation_option`: .false. = direct (exact), .true. = FFT-accelerated (approximate).

### PTSA filename convention (from aggregate_generator_PTSA)

`agg_num={i}_k={k}_Df={Df}_meanRp={a}um_rstdRp={σ}_Np={N}_Rve={Rve}um_Rg={Rg}um_epsagg={ε}.ptsa`

The `k` in the filename is the packing parameter (fractal prefactor), NOT the wavenumber.

## Validation Strategy

### Priority order: testcase2 first, then testcase1, then testcase3

- testcase1 is simplest (2 spheres) → validates Mie + translation + solver core
- testcase2 is the primary CAS use case (1000 spheres, direct translation)
- testcase3 validates FFT-accelerated translation (optional for initial development)

### Unit tests (per module)

- **MieCoefficients:**

  Primary reference: [MieScat_Py](https://github.com/NobuhiroMoteki/MieScat_Py)
  (validated to rtol=1e-12; provides ground-truth a_n, b_n, Q values, S1/S2 amplitudes)

  **Algorithm equivalence:** MieScat_Py's `_compute_bessel_and_dd` + `_compute_efficiencies_and_amplitudes`
  uses the identical BH83 algorithm described in Step 1 above. Results must agree to rtol ~ 1e-10.

  **Comparison protocol with MieScat_Py** (using `miescat(wl_0, m_m, d_p, m_p_real, m_p_imag)`):
  - Both use `x = π×m_m×d_p/wl_0` and `m_r = (m_p_real + i×m_p_imag) / m_m`
  - Compare Q_sca, Q_ext, Q_abs directly (dimensionless)
  - Compare S1(θ), S2(θ) computed from a_n, b_n using BH83 Eq. 4.74:
      `S1(θ) = Σ_n (2n+1)/(n(n+1)) × (a_n×π_n(cosθ) + b_n×τ_n(cosθ))`
      `S2(θ) = Σ_n (2n+1)/(n(n+1)) × (a_n×τ_n(cosθ) + b_n×π_n(cosθ))`
  - At θ=0°: π_n(1) = τ_n(1) = n(n+1)/2, so S1(0°)=S2(0°)=Σ (2n+1)/2 × (a_n+b_n) ✓

  **⚠️ Convention warning — never confuse:**
  - S1, S2, S3, S4 (BH83): dimensionless, output by MSTM and MieScat_Py BH83 formulas
  - S11, S12, S21, S22 (MI02): dimension of length (m), output as MieScat_Py S11/S22 columns
  - Conversion (from MSTM-v4.0_for_CAS README):
      S11(MI02) = S2(BH83)/(-ik),  S12(MI02) = S3(BH83)/(ik)
      S21(MI02) = S4(BH83)/(ik),   S22(MI02) = S1(BH83)/(-ik)
      where k = k_medium (wavenumber in medium)

  **Single-sphere cross-check with MSTM (confirmed by README):**
  The README states: "For homogeneous spherical particles, I confirmed that S1(0°), S2(0°),
  S1(180°), S2(180°) [BH83] computed by MSTM agreed with MieScat_Py."
  Therefore for N=1 sphere, Julia MSTM output S1,S2 (BH83) must match:
    BH83 S1(θ) computed directly from MieScat_Py a_n, b_n via Eq. 4.74
  (MieScat_Py also outputs MI02 S11,S22 columns; convert back via S1=S22×(-ik) if needed)

  **Specific test values for single-sphere unit tests:**
  From README example (x=1.507, m_r=1.58+0.0i, non-absorbing):
  - Q_ext=1.0525, Q_abs≈0, forward S1=S2=(0.5974-1.401i), backward S1=-S2=(0.2559-0.0672i)
  From testcase1 sphere parameters (x=1.507, m_r=1.58+0.05i, absorbing):
  - Q_ext_single≈1.2517, Q_abs_single≈0.2855 (from testcase1 per-sphere output)
  From testcase2 sphere parameters (x=1.0, m_r=1.6+0.0123i):
  - Q_ext_single≈0.3399, Q_abs_single≈0.0356 (from testcase2 mean Mie efficiencies)
  From BH83 Table 4.1 (x=1.0, m=1.5+1.0i):
  - Q_sca≈0.5274, Q_ext≈2.336

  **⚠️ MieScat_Py scope limitation:**
  MieScat_Py computes single-sphere Mie theory only. It CANNOT validate:
  - testcase1 (2-sphere cluster) full scattering result — multi-sphere interactions change S1,S2
  - testcase2 (1000-sphere cluster)
  Multi-sphere test cases must be compared against Fortran reference output directly.

  **nstop difference note:**
  - MieScat_Py: nstop = floor(x + 4*x^(1/3) + 2)   [BH83 standard]
  - Julia/MSTM:  nstop = nint(x + 4*x^(1/3)) + 5 with convergence check [more conservative]
  - For MieCoefficients unit tests, use same nstop as MieScat_Py; results must agree to rtol~1e-10

- **TranslationCoefs:**
  - 2-sphere axially symmetric case, compare with Fortran MSTM testcase1 (N=2)
  - Symmetry relation: H_{ij} related to H_{ji} by conjugation/transposition

- **TMatrixSolver:**
  - Single sphere case must reduce to Mie theory exactly (no coupling)
  - 2-sphere case (testcase1): Q_ext(unpol)=1.4708, Q_abs=0.3185

- **ScatteringAmplitude:**
  - **Single-sphere test (N=1, validated by MieScat_Py):**
    - x=1.507, m_r=1.58+0.0i (non-absorbing): forward S1=S2=(0.5974-1.401i) [BH83]
    - S3=S4=0 (spherical symmetry), backward S1=-S2 (known property)
    - Q_abs→0 for Im(m_r)=0
  - **Multi-sphere tests (validated by Fortran reference only):**
    - testcase1 (2 spheres): forward S1=(1.147-2.425i), S2=(1.504-2.258i)
    - testcase2 (1000 spheres): forward S1=(261.5-181.5i), S2=(261.9-181.8i)
  - Energy conservation: Q_ext = Q_sca + Q_abs (always check)

### End-to-end validation
Match Fortran reference outputs to < 0.01% relative error:
- testcase1: 2 spheres, x=1.507, m=1.58+0.05i
- testcase2: 1000 spheres, x=1.0, m=1.6+0.0123i (direct translation)
- testcase3: same as testcase2 but FFT translation (optional)

## Key Numerical Considerations

- Floating point: use Float64 (ComplexF64) throughout
- Mie coefficient computation:
  - Ricatti-Bessel ψ_n(z) via cricbessel (complex downward recurrence for large n/x ratio)
  - Ricatti-Hankel ξ_n(z) via crichankel
  - Use logarithmic derivative ratios: psipn(n) = ψ_{n-1}/ψ_n - n/x to avoid overflow
  - Mie order: N = nint(x + 4*x^(1/3)) + 5 for convergence-based determination
- Translation coefficients: rotation-axial-rotation decomposition with Mackowski recurrence;
  do NOT use direct factorial/Gaunt coefficient expressions
- CBICG solver: default solution_eps = 1e-6, max_iterations = 10000
- For large N_sphere (testcase2: N=1000), direct translation is O(N²×N_order²) — acceptable
  for testcase sizes; FFT option needed for N>10000
- Validation tolerance: 0.01% relative error (1e-4) against Fortran reference
  (exact bit-match NOT expected; CBICG is non-deterministic in parallel)

## Dependencies

- SpecialFunctions.jl  — spherical Bessel functions
- HDF5.jl             — output storage for large parameter sweeps
- CSV.jl + DataFrames.jl — alternative output format
- Test (stdlib)        — unit testing
- LinearAlgebra (stdlib) — matrix operations
- Printf (stdlib)      — formatted output

## Coding Conventions

- All angles in radians internally; degree conversion only at I/O boundary
- `length_scale_factor` = k_medium = wavenumber in medium
- Size parameter x = k_medium × r (dimensionless)
- Complex refractive index m = n + iκ (positive κ = absorption, BH83 convention)
  NOTE: Some references use m = n - iκ. We ALWAYS use positive imaginary part = absorption.
- Array indexing: 1-based (native Julia)
- Multipole order n starts from 1 (not 0)
- Index convention for m<0: store at position (n+1, -m) instead of (m, n)
  (mirrors Fortran's m1=n+1, n1=-m for m≤-1)
- Module-level docstrings in Julia format for all public functions
- Type annotations on all function signatures

## Design Decisions (resolved)

1. **Output cross sections**: Output Q_ext, Q_abs, Q_sca (dimensionless efficiency factors),
   consistent with Fortran reference output. Physical cross section C = Q × π × a_eff²
   can be computed by the caller if needed.

2. **Medium refractive index**: `n_medium` is passed explicitly as a parameter in the
   Julia API. Internally, k_medium = 2π × n_medium / λ₀ is derived from n_medium and λ₀.

3. **FFT translation**: NOT a development target until testcase1 and testcase2 pass.
   Implement only direct (exact) translation first.

## Development Workflow

1. Implement and test modules bottom-up: MieCoefficients → TranslationCoefs →
   TMatrixSolver → ScatteringAmplitude → AggregateIO → ParameterSweep
2. Each module must pass its unit tests before proceeding to the next
3. Run `julia --project=. -e 'using Pkg; Pkg.test()'` to execute full test suite
4. Validate against testcase1, then testcase2 (testcase3 optional)

## File Locations

- Source: `src/`
- Tests: `test/`
- Reference Fortran code, PDFs, test cases: `ref/MSTM-v4.0_for_CAS/` (gitignored)
- Example aggregate files: `test/fixtures/`
- Sweep scripts: `scripts/`

## References

- Bohren, C.F. & Huffman, D.R. (1983). Absorption and Scattering of Light by Small Particles. Wiley.
- Mackowski, D.W. (1991). Analysis of radiative scattering for multiple sphere configurations. Proc. R. Soc. Lond. A, 433, 599-614.
- Mackowski, D.W. (1994). Calculation of total cross sections of multiple-sphere clusters. JOSA A, 11(11), 2851-2861.
- Mackowski, D.W. (1996). Calculation of total cross sections of multiple-sphere clusters. JOSA A, 13(11), 2266-2278.
- Mackowski, D.W. & Mishchenko, M.I. (2011). A multiple sphere T-matrix Fortran code for use on parallel computer clusters. JQSRT, 112(13), 2182-2192.
- Mackowski, D.W. (2014). Direct simulation of scattering and absorption by particle deposits. Proc. ICHMT.
- Mishchenko, M.I., Travis, L.D. & Lacis, A.A. (2002). Scattering, Absorption, and Emission of Light by Small Particles. Cambridge University Press.
- Wiscombe, W.J. (1980). Improved Mie scattering algorithms. Applied Optics, 19(9), 1505-1509.
- Xu, Y.-l. (1996). Calculation of the addition coefficients in electromagnetic multisphere-scattering theory. J. Comput. Phys., 127(2), 285-298.
- Moteki, N. & Adachi, K. (2024). CAS-v2 protocol. Optics Express, 32(21), 36500-36522.
