# MSTMforCAS.jl

Julia implementation of the Multi-Sphere T-Matrix (MSTM) method, specialized for Complex Amplitude Sensing (CAS) applications.

## Purpose

Compute numerically exact light scattering solutions for particles consisting of multiple homogeneous spheres (possibly of different sizes) at high speed. This package re-implements, from scratch in Julia, the subset of [Mackowski's MSTM v4.0](https://github.com/dmckwski/MSTM) (Fortran/MPI) functionality required for Complex Amplitude Sensing (CAS). In practice, the code is well suited for parameter sweeps over wavelength, medium refractive index, particle refractive index, and particle morphology (e.g., fractal dimension of aggregates).

Reference implementation: [MSTM-v4.0_for_CAS](https://github.com/NobuhiroMoteki/MSTM-v4.0_for_CAS) (Fortran/MPI).

For a detailed mathematical formulation of the MSTM theory (VSWF expansion, Mie coefficients, translation addition theorem, multi-sphere interaction equation, CBICG solver, and output quantities), see [docs/technical_note_mstm_theory.pdf](docs/technical_note_mstm_theory.pdf).

## Physical Scope

- Infinite homogeneous medium, plane wave incidence along z-axis (fixed orientation)
- Aggregates of homogeneous spheres (each monomer with its own radius, common complex refractive index)
- Output per (aggregate, medium condition, refractive index) combination:
  - Forward/backward scattering amplitude matrix S₁–S₄ (BH83 convention, dimensionless) and S₁₁, S₂₂, S₁₂, S₂₁ (MI02 convention, dimension of length)
  - Polarization-averaged efficiency factors $Q_\mathrm{ext}$, $Q_\mathrm{abs}$, $Q_\mathrm{sca}$ (unpolarized incidence)

**Note on particle orientation:** MSTMforCAS does not internally support rotation of aggregate orientation. To compute scattering at different orientations, apply the desired rotation to the aggregate geometry file (.ptsa/.pos) before passing it to the code.

### Convention definitions

**BH83** = Bohren & Huffman (1983), Eq. 3.12. The amplitude matrix relates scattered to incident field:

```math
\begin{pmatrix} E_{\parallel}^{\mathrm{sca}} \\ E_{\perp}^{\mathrm{sca}} \end{pmatrix} = \frac{e^{ikr}}{-ikr} \begin{pmatrix} S_2 & S_3 \\ S_4 & S_1 \end{pmatrix} \begin{pmatrix} E_{\parallel}^{\mathrm{inc}} \\ E_{\perp}^{\mathrm{inc}} \end{pmatrix}
```

where $S_1, S_2, S_3, S_4$ are dimensionless complex scattering amplitudes.

**MI02** = Mishchenko, Travis & Lacis (2002). Conversion from BH83:

```math
S_{11}^{\mathrm{MI02}} = \frac{S_2^{\mathrm{BH83}}}{-ik}, \quad S_{12}^{\mathrm{MI02}} = \frac{S_3^{\mathrm{BH83}}}{ik}, \quad S_{21}^{\mathrm{MI02}} = \frac{S_4^{\mathrm{BH83}}}{ik}, \quad S_{22}^{\mathrm{MI02}} = \frac{S_1^{\mathrm{BH83}}}{-ik}
```

where $k = k_{\mathrm{medium}} = 2\pi n_{\mathrm{medium}} / \lambda_0$ is the wavenumber in the medium. The MI02 amplitudes have dimension of length.

**Efficiency factors**:

```math
Q = \frac{C}{\pi R_{\mathrm{ve}}^2}, \qquad R_{\mathrm{ve}} = \left(\sum_i r_i^3\right)^{1/3}
```

where $C$ is the cross section and $R_{\rm ve}$ is the volume-equivalent sphere radius.

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
| HDF5.jl | Incremental output storage for large parameter sweeps |
| CSV.jl + DataFrames.jl | Reading aggregate catalog (input side) |
| LinearAlgebra (stdlib) | Matrix operations |
| Printf (stdlib) | Formatted output |

## Usage

### Input parameters and units

All length inputs (positions, radii, wavelength) must be given in **the same physical unit** (e.g., all in μm). The package does NOT assume a particular unit; consistency is all that matters.

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `medium_conditions` | `Vector{Tuple{Float64,Float64}}` | List of (wavelength, medium_refindex) pairs. Wavelength in the same unit as coordinates. | `[(0.6328, 1.0), (0.6328, 1.33)]` |
| `m_real_range` | `Tuple{Float64,Float64,Int}` | (min, max, n_grid) for real part of **absolute** complex RI (BH83 sign convention). Single value: `(1.6, 1.6, 1)`. | `(1.5, 1.7, 3)` |
| `m_imag_range` | `Tuple{Float64,Float64,Int}` | (min, max, n_grid) for imaginary part (positive κ = absorption). | `(0.0, 0.1, 2)` |
| `use_fft` | `Bool` | FFT-accelerated translation (default `false`) | `true` |
| `truncation_order` | `Union{Int,Nothing}` | Override VSWF truncation order (default `nothing` = auto) | `8` |

The sweep covers the full Cartesian product: **aggregates × medium_conditions × m_real × m_imag**.

**VSWF truncation order**: By default, the truncation order N for each sphere is determined automatically from the size parameter x using the Wiscombe criterion: $N = \mathrm{round}(x + 4x^{1/3}) + 5$, refined by a convergence check on the Mie efficiencies. Typical values: N=4 for x=1.0, N=5 for x=1.5. To override, pass `truncation_order=N` (applied uniformly to all spheres; if smaller than the auto-determined value, the auto value is used as a floor). The actual truncation order used is reported in the output as `truncation_order`.

### Aggregate geometry input

Two input formats are supported:

**1. HDF5 + CSV catalog** (recommended, from [aggregate_generator_PTSA](https://github.com/NobuhiroMoteki/aggregate_generator_PTSA)):

```julia
aggregates = read_aggregate_catalog("aggregates.h5", "catalog.csv")
# Optional filters:
aggregates = read_aggregate_catalog("aggregates.h5", "catalog.csv";
    Df_range=(1.5, 2.5), Np_range=(100, 500), agg_num_range=(0, 2))
```

The CSV catalog columns: `mean_rp, rel_std_rp, k, Df, Np, agg_num, Rve, Rg, eps_agg, h5_file, h5_key`. The HDF5 file stores monomer coordinates (`xp`, 3×Np) and radii (`rp`, Np) at each `h5_key` path. All metadata is automatically populated into `AggregateGeometry` and propagated to the output DataFrame.

**2. Text files** (.ptsa, .pos): plain ASCII, one sphere per line, 4 whitespace-separated columns:

```
x   y   z   r
```

No header. Coordinates and radii in physical units (μm).

```julia
agg = read_aggregate_file("aggregate.ptsa")
```

### Single aggregate computation

```bash
# Direct translation (default)
julia --project=. scripts/run_single.jl

# FFT-accelerated translation (faster for N > ~100 spheres)
julia --project=. scripts/run_single.jl --fft

# Override automatic VSWF truncation order (use N=8 for all spheres)
julia --project=. scripts/run_single.jl --truncation-order 8
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

# Compute with user-specified truncation order
result = compute_scattering(agg, m_rel, k_medium; truncation_order=8)

# Access results
result.S_forward        # (S₁, S₂, S₃, S₄) BH83, dimensionless, θ=0°
result.S_backward       # (S₁, S₂, S₃, S₄) BH83, dimensionless, θ=180°
result.Q_ext            # extinction efficiency (dimensionless)
result.Q_abs            # absorption efficiency (dimensionless)
result.Q_sca            # scattering efficiency (dimensionless)
result.converged        # solver convergence flag
result.n_iterations     # number of CBICG iterations
result.truncation_order # maximum VSWF truncation order used
```

### Parameter sweep

**From HDF5 + CSV catalog** (recommended):

```bash
# Single thread
julia --project=. scripts/run_sweep_h5.jl

# Multi-threaded (4 threads) with FFT
julia -t 4 --project=. scripts/run_sweep_h5.jl --fft
```

Or in Julia:

```julia
using MSTMforCAS

# Load aggregates from HDF5 + CSV catalog
aggregates = read_aggregate_catalog("data/aggregates/aggregates.h5",
                                     "data/aggregates/catalog.csv")

config = SweepConfig(
    medium_conditions = [(0.6328, 1.0), (0.6328, 1.33)],  # (wavelength [μm], n_medium)
    m_real_range = (1.5, 1.7, 3),    # (min, max, n_grid) for Re(m)
    m_imag_range = (0.0, 0.1, 2),    # (min, max, n_grid) for Im(m)
    use_fft          = true,          # FFT-accelerated translation
    truncation_order = nothing,       # nothing = automatic, or Int to override
)

# Run sweep — results written incrementally to HDF5 (safe to interrupt and resume)
run_parameter_sweep(aggregates, config; output_h5="results.h5")
```

`run_parameter_sweep` returns `nothing`; all results are written directly to the HDF5 file. If the file already exists, completed jobs are detected automatically and skipped (resume support).

**From .ptsa text files** (alternative):

```bash
julia -t 4 --project=. scripts/run_sweep_small.jl [--fft] [--truncation-order N]
```

```julia
aggregates = [read_aggregate_file(f) for f in readdir("aggregates/", join=true)]
run_parameter_sweep(aggregates, config; output_h5="results.h5")
```

### Output data format

Results are stored in HDF5 format. Each column is a 1-D dataset of length equal to the number of completed jobs. There are 51 columns per row (one row = one aggregate × medium_condition × refractive_index combination):

#### Aggregate metadata and material

| Column | Type | Unit | Description | Category |
|--------|------|------|-------------|----------|
| `source` | String | — | Source identifier (h5_key or file path) | — |
| `mean_rp` | Float64 | [μm] | Mean monomer radius | constant |
| `rel_std_rp` | Float64 | — | Relative std dev of monomer radius | constant |
| `k_f` | Float64 | — | Fractal prefactor | constant |
| `Df` | Float64 | — | Fractal dimension | sweep |
| `n_monomers` | Int | — | Number of monomers | sweep |
| `agg_num` | Int | — | Aggregate index | sweep |
| `R_ve` | Float64 | [μm] | Volume-equivalent sphere radius | derived |
| `R_g` | Float64 | [μm] | Radius of gyration | derived |
| `eps_agg` | Float64 | — | Aggregate porosity | derived |
| `wavelength` | Float64 | [μm] | Vacuum wavelength | sweep |
| `medium_refindex` | Float64 | — | Medium refractive index | sweep |
| `m_real` | Float64 | — | Real part of **absolute** refractive index | sweep |
| `m_imag` | Float64 | — | Imaginary part of **absolute** refractive index | sweep |

Note: When reading from .ptsa text files, metadata fields (`mean_rp`, `rel_std_rp`, `k_f`, `Df`, `agg_num`) are automatically parsed from the filename if it follows the naming convention `meanRp={val}um_rstdRp={val}_k={val}_Df={val}_Np={val}_agg_num={val}.ptsa`. Fields `R_g` and `eps_agg` are only available via the HDF5+CSV catalog and will be `NaN` when reading text files.

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
| `converged` | Int8 | — | Whether CBICG solver converged (1 = yes, 0 = no) |
| `n_iterations` | Int | — | Number of CBICG iterations used |
| `truncation_order` | Int | — | Maximum VSWF truncation order used |

**Unit note for MI02 amplitudes**: The MI02 amplitude unit [length] is the same as the wavelength and coordinate unit. For example, if `wavelength` is in μm, then S₁₁ etc. are in μm.

**Physical cross sections** are not stored in the output, but can be computed as: $C = Q \times \pi \times R_\mathrm{ve}^2$ (same length unit squared).

### Reading output files

**HDF5:**

```julia
using HDF5

h5open("data/results/results_fullsweep_agg20260316_00.h5", "r") do fid
    # Each column is a 1-D array (length = number of jobs)
    wavelength     = read(fid["wavelength"])
    medium_refindex = read(fid["medium_refindex"])
    agg_num    = read(fid["agg_num"])
    n_monomers = read(fid["n_monomers"])
    Df         = read(fid["Df"])
    m_real     = read(fid["m_real"])
    m_imag     = read(fid["m_imag"])
    R_ve       = read(fid["R_ve"])
    Q_ext      = read(fid["Q_ext"])
    Q_abs      = read(fid["Q_abs"])
    Q_sca      = read(fid["Q_sca"])

    # MI02 forward scattering amplitudes (complex, dimension of length [μm])
    S11_fwd = read(fid["S11_fwd_re"]) .+ im .* read(fid["S11_fwd_im"])
    S22_fwd = read(fid["S22_fwd_re"]) .+ im .* read(fid["S22_fwd_im"])
    S12_fwd = read(fid["S12_fwd_re"]) .+ im .* read(fid["S12_fwd_im"])
    S21_fwd = read(fid["S21_fwd_re"]) .+ im .* read(fid["S21_fwd_im"])

    # MI02 backward scattering amplitudes
    S11_bwd = read(fid["S11_bwd_re"]) .+ im .* read(fid["S11_bwd_im"])
    S22_bwd = read(fid["S22_bwd_re"]) .+ im .* read(fid["S22_bwd_im"])
    S12_bwd = read(fid["S12_bwd_re"]) .+ im .* read(fid["S12_bwd_im"])
    S21_bwd = read(fid["S21_bwd_re"]) .+ im .* read(fid["S21_bwd_im"])

    # Filter by condition: wavelength=0.6328, n_medium=1.0, agg_num=0, Np=100, Df=2.35, m=1.6+0.0123i
    idx = findall(i -> wavelength[i] == 0.6328 && medium_refindex[i] == 1.0 &&
                       agg_num[i] == 0 && n_monomers[i] == 100 &&
                       Df[i] == 2.35 && m_real[i] == 1.6 && m_imag[i] == 0.0123,
                  eachindex(agg_num))

    for i in idx
        println("R_ve=$(R_ve[i]), Q_ext=$(Q_ext[i]), Q_abs=$(Q_abs[i]), Q_sca=$(Q_sca[i])")
        println("  MI02 fwd: S11=$(S11_fwd[i]), S22=$(S22_fwd[i]), S12=$(S12_fwd[i]), S21=$(S21_fwd[i])")
        println("  MI02 bwd: S11=$(S11_bwd[i]), S22=$(S22_bwd[i]), S12=$(S12_bwd[i]), S21=$(S21_bwd[i])")
    end
end
```

or with the provided script:

```bash
julia --project=. scripts/read_results_h5.jl data/results/results_fullsweep_agg20260316_00.h5
```

HDF5 files also store sweep config as root-level attributes: `medium_conditions_wavelength`, `medium_conditions_refindex`, `m_real_range`, `m_imag_range`, `n_jobs`.

**CAS-specific parameter lookup** (interactive use):

```bash
julia --project=. scripts/read_results_h5_for_cas.jl <results.h5> [i_medium] [i_agg] [i_Np] [i_Df] [i_m_real] [i_m_imag]
```

This script reads an HDF5 results file and outputs CAS-v2 observable quantities for a specific parameter combination specified by 1-based array indices:

- `Ss_fwd = S11_fwd + S12_fwd * i` — s-polarization forward complex scattering amplitude [μm]
- `Sp_fwd = S22_fwd - S21_fwd * i` — p-polarization forward complex scattering amplitude [μm]
- `S_bak = (-S11_bwd + S22_bwd) / √2` — backward scattering amplitude [μm]
- `R_ve`, `Q_ext`, `Q_abs`, `Q_sca`

where S11, S12, S21, S22 are the MI02-convention amplitude matrix elements. When run without index arguments (or with all indices set to 1), it first lists the available parameter values and their counts:

```text
Available parameter values:
  medium   (2): [(0.6328, 1.0), (0.6328, 1.33)]  # (wavelength, n_medium)
  agg_num  (3): [0, 1, 2]
  Np       (4): [100, 200, 300, 400]
  Df       (2): [2.35, 2.95]
  m_real   (2): [1.5, 1.6]
  m_imag   (2): [0.0, 0.0123]
```

The numbers in parentheses are the upper limits for each index. In this example, valid ranges are `i_medium`=1–2, `i_agg`=1–3, `i_Np`=1–4, `i_Df`=1–2, `i_m_real`=1–2, `i_m_imag`=1–2.

Example:

```bash
# Select medium=(0.6328,1.33) (2nd), agg_num=1 (2nd), Np=200 (2nd), Df=2.35 (1st), m=1.6+0.0123i
julia --project=. scripts/read_results_h5_for_cas.jl data/results/results_fullsweep_agg20260316_00.h5 2 2 2 1 2 2
```

**Using results from Python**: The `.h5` file is self-contained and can be read directly with `h5py` without any Julia dependency. See [docs/technical_note_use_results_from_python.md](docs/technical_note_use_results_from_python.md) for the HDF5 structure reference, CAS-v2 observable formulas, and a ready-to-use `read_cas_results()` Python function.

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
│   ├── AggregateIO.jl         # Read .ptsa/.pos and HDF5+CSV catalog
│   └── ParameterSweep.jl      # Parallel parameter sweeps
├── data/
│   ├── aggregates/            # HDF5 + CSV from aggregate_generator_PTSA
│   └── results/               # MSTM computation results (CSV/HDF5)
├── scripts/
│   ├── run_single.jl          # Single aggregate computation
│   ├── run_sweep_h5.jl        # Parameter sweep from HDF5 catalog
│   ├── run_sweep_small.jl     # Parameter sweep from .ptsa files
│   ├── validate_testcases.jl  # Validation against Fortran reference
│   ├── read_results_csv.jl    # Read and summarize CSV output
│   ├── read_results_h5.jl     # Read and summarize HDF5 output
│   └── read_results_h5_for_cas.jl  # CAS-specific parameter lookup from HDF5
├── docs/
│   ├── technical_note_mstm_theory.md              # Mathematical formulation of MSTM theory
│   └── technical_note_use_results_from_python.md  # Python reader guide + code
└── test/
```

## Validation

Validated against Fortran MSTM-v4.0_for_CAS reference output. All primary quantities agree to < 0.1%.

| Test case | Spheres | Translation | $Q_\mathrm{ext}$ %err | $Q_\mathrm{abs}$ %err | $Q_\mathrm{sca}$ %err | S₁ fwd %err | S₂ fwd %err | Iterations |
|-----------|---------|------------|------------|------------|------------|-------------|-------------|------------|
| testcase1 | 2       | direct     | 0.001%     | 0.050%     | 0.015%     | 0.014%      | 0.001%      | 3          |
| testcase2 | 1000    | direct     | 0.003%     | 0.017%     | 0.001%     | 0.021%      | 0.016%      | 18         |
| testcase3 | 1000    | FFT        | 0.001%     | 1.6%       | 0.056%     | 0.024%      | 0.007%      | 18         |

Testcase3 uses the same geometry and refractive index as testcase2 but with FFT-accelerated translation. The larger $Q_\mathrm{abs}$ error (1.6%) is expected because $Q_\mathrm{abs} = Q_\mathrm{ext} - Q_\mathrm{sca}$ is a small residual ($Q_\mathrm{abs} \approx 0.37$ vs $Q_\mathrm{ext} \approx 11.2$), amplifying the FFT approximation error. The Fortran FFT reference shows the same effect.

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
| Solver time | 301 s | 7.2 s | 96.4 s | 3.3 s |
| Speedup vs Julia direct | 1× | **42×** | — | — |
| Iterations | 18 | 18 | 18 | 18 |
| Peak memory | ~2 GB | ~0.5 GB | ~0.5 GB | ~0.1 GB |

**Julia vs Fortran speed**: The Julia direct solver is ~3× slower than single-process Fortran for this test case. The Julia FFT solver is ~2× slower than the Fortran FFT solver. Key optimizations applied: stride-1 translation coefficient layout, batched FFTW plans (6× on FFT convolution), precomputed near-field translation matrices, and precomputed T-matrix lookup tables.

**Memory**: Julia FFT mode uses ~0.5 GB (including ~190 MB for cached near-field translation matrices). For practical CAS parameter sweeps (N ≤ 1000), memory is not a bottleneck.

**Multi-threading**: Parallelization is at the **parameter sweep level**, not within a single aggregate computation. `run_parameter_sweep` distributes the (aggregate × medium_condition × refractive_index) job grid across threads via `Threads.@threads`. Start Julia with `julia -t auto` or `julia -t N`.

This task-level parallelism is far more efficient than parallelizing within a single solver call because: (1) each job is fully independent — zero inter-thread communication and synchronization, giving near-linear scaling; (2) intra-solver parallelism (e.g., threading the O(N²) translation loop) would require synchronization at every BiCG iteration, and Amdahl's law limits the practical speedup to 2–5×. For typical CAS parameter sweeps (thousands of jobs across aggregates, medium conditions, and refractive indices), task-level parallelism with C cores gives ~C× speedup as long as the number of jobs exceeds C.

Thread safety: the translation coefficient cache is pre-warmed single-threaded before the parallel section.

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
