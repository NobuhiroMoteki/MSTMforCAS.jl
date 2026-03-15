"""
    ParameterSweep

Orchestrate parameter sweeps over aggregate geometries and refractive indices.
Supports multi-threaded parallel execution via Julia's Threads.@threads.

# Typical usage

```julia
config = SweepConfig(
    aggregate_files    = readdir("aggregates/", join=true),
    refractive_indices = [1.5+0.0im, 1.5+0.01im, 1.75+0.05im],
    wavelength         = 0.6328e-6,   # [m] He-Ne laser
    medium_refindex    = 1.0,
    monomer_radius     = 50e-9,       # [m], only needed when files use dimensionless coords
)
df = run_parameter_sweep(config; output_file="results.csv")
```

# Threading notes
The sweep parallelises over the (aggregate × refractive-index) grid with
`Threads.@threads`. Start Julia with `julia -t auto` (or `-t N`) for speedup.
Thread safety: `TranslationCoefs` uses a global cache that is initialised by a
single-threaded pre-warm step before `@threads` is entered, so all threads only
read from the cache during the sweep.
"""

export run_parameter_sweep, SweepConfig

using DataFrames, CSV, HDF5

"""
    SweepConfig

Configuration for a parameter sweep computation.

# Fields
- `aggregate_files::Vector{String}`: Paths to aggregate geometry files (.ptsa or .pos)
- `refractive_indices::Vector{ComplexF64}`: Absolute complex refractive indices of the spheres
  (NOT relative to medium; conversion to m_rel = m_abs / n_medium is done internally)
- `wavelength::Float64`: Vacuum wavelength in the same physical units as file coordinates
  (e.g., if files are in μm, give wavelength in μm; if files are in m, give wavelength in m)
- `medium_refindex::Float64`: Real refractive index of the surrounding medium (default 1.0)
- `monomer_radius::Float64`: Scale factor to convert file coordinates to physical units.
  Set to 1.0 (default) when files already use physical units consistent with `wavelength`.
  Set to the monomer radius in meters when files store dimensionless coords normalised by r_mono.
- `max_iterations::Int`: Maximum CBICG iterations (default 200)
- `convergence_epsilon::Float64`: Solver convergence threshold (default 1e-6)
"""
Base.@kwdef struct SweepConfig
    aggregate_files::Vector{String}
    refractive_indices::Vector{ComplexF64}
    wavelength::Float64
    medium_refindex::Float64 = 1.0
    monomer_radius::Float64  = 1.0
    max_iterations::Int      = 200
    convergence_epsilon::Float64 = 1e-6
end

# ─────────────────────────────────────────────────────────────
# AggregateGeometry overload of compute_scattering
# ─────────────────────────────────────────────────────────────

"""
    compute_scattering(agg::AggregateGeometry, m_rel::ComplexF64, k::Float64;
                       tol, max_iter) -> ScatteringResult

Compute scattering for an `AggregateGeometry` object.

# Arguments
- `agg`: aggregate geometry (positions and radii in physical units consistent with `k`)
- `m_rel`: complex refractive index relative to medium, m_sphere / n_medium
- `k`: wavenumber in the medium, k = 2π × n_medium / λ₀ (same units⁻¹ as agg coordinates)

The function scales `agg.positions` and `agg.radii` by `k` to obtain dimensionless size
parameters before calling the core solver.
"""
function compute_scattering(
    agg    ::AggregateGeometry,
    m_rel  ::ComplexF64,
    k      ::Float64;
    tol    ::Float64 = 1e-6,
    max_iter::Int    = 200
)::ScatteringResult
    positions_x = agg.positions .* k   # dimensionless (size parameters)
    radii_x     = agg.radii     .* k
    return compute_scattering(positions_x, radii_x, m_rel; tol=tol, max_iter=max_iter)
end

# ─────────────────────────────────────────────────────────────
# Main sweep function
# ─────────────────────────────────────────────────────────────

"""
    run_parameter_sweep(config::SweepConfig; output_file=nothing) -> DataFrame

Execute a parameter sweep: for every (aggregate_file, refractive_index) pair,
compute the full scattering result.

# Returns
`DataFrame` with one row per (file, m) pair and columns:
- `aggregate_file`: basename of the source file
- `n_monomers`: number of monomers
- `R_ve`: volume-equivalent radius (Σ rᵢ³)^{1/3}, same units as file coordinates
- `m_real`, `m_imag`: components of the absolute refractive index
- `S1_fwd_re/im` … `S4_fwd_re/im`: forward amplitude S₁–S₄ (BH83 convention, dimensionless, θ=0°)
- `S11_fwd_re/im` … `S21_fwd_re/im`: forward amplitude S₁₁–S₂₁ (MI02 convention, dim. of length, θ=0°)
- `S1_bwd_re/im` … `S4_bwd_re/im`: backward amplitude S₁–S₄ (BH83, θ=180°)
- `S11_bwd_re/im` … `S21_bwd_re/im`: backward amplitude S₁₁–S₂₁ (MI02, θ=180°)
- `Q_ext`, `Q_abs`, `Q_sca`: polarization-averaged scattering efficiencies (unpolarized incidence)
- `converged`: solver convergence flag
- `n_iterations`: number of CBICG iterations used

MI02 conversion: S₁₁=S₂/(-ik), S₂₂=S₁/(-ik), S₁₂=S₃/(ik), S₂₁=S₄/(ik) where k=k_medium

# Output file
When `output_file` is provided, the DataFrame is written to disk:
- `*.csv` → CSV (human-readable, portable)
- `*.h5` or `*.hdf5` → HDF5 (compact, fast for large sweeps)
"""
function run_parameter_sweep(
    config::SweepConfig;
    output_file::Union{String,Nothing} = nothing
)::DataFrame

    n_files = length(config.aggregate_files)
    n_mrel  = length(config.refractive_indices)
    n_jobs  = n_files * n_mrel

    if n_jobs == 0
        return DataFrame()
    end

    # k_medium: wavenumber in the medium (same units as file coordinates after monomer_radius scaling)
    k = 2π * config.medium_refindex / config.wavelength

    # ── Read all aggregate files (sequential: avoids I/O races in @threads) ──
    aggregates = Vector{AggregateGeometry}(undef, n_files)
    for fi in 1:n_files
        aggregates[fi] = read_aggregate_file(
            config.aggregate_files[fi]; scale_factor=config.monomer_radius
        )
    end

    # ── Pre-warm global caches (single-threaded, must run before @threads) ──
    # TranslationCoefs uses a module-level Ref cache (_TCC, _NC) that is not
    # thread-safe during initialisation. Running one full computation here
    # ensures the cache is populated before the parallel section.
    # Since ntrani = nois (Mie order), the cache size is bounded by the
    # maximum Mie order across all spheres, which is small and stable.
    let agg0  = aggregates[1],
        m_rel0 = config.refractive_indices[1] / config.medium_refindex
        compute_scattering(agg0, m_rel0, k;
            tol=config.convergence_epsilon, max_iter=config.max_iterations)
    end

    # ── Flat job grid: (file_index, m_rel_index) ─────────────────────────────
    grid = [(fi, mi) for fi in 1:n_files for mi in 1:n_mrel]

    # ── Pre-allocate result rows (one slot per job, indexed by position) ─────
    rows = Vector{NamedTuple}(undef, n_jobs)

    # ── Parallel sweep ────────────────────────────────────────────────────────
    ik = im * k   # i × k_medium (for BH83 → MI02 conversion)

    Threads.@threads for idx in 1:n_jobs
        fi, mi  = grid[idx]
        agg     = aggregates[fi]
        m_abs   = config.refractive_indices[mi]
        m_rel   = m_abs / config.medium_refindex

        r = compute_scattering(agg, m_rel, k;
            tol=config.convergence_epsilon, max_iter=config.max_iterations)

        # Volume-equivalent radius: R_ve = (Σ r_i³)^{1/3}
        R_ve = cbrt(sum(ri^3 for ri in agg.radii))

        # MI02 amplitudes: S11=S2/(-ik), S22=S1/(-ik), S12=S3/(ik), S21=S4/(ik)
        S11_fwd = r.S_forward[2] / (-ik);  S22_fwd = r.S_forward[1] / (-ik)
        S12_fwd = r.S_forward[3] / (ik);   S21_fwd = r.S_forward[4] / (ik)
        S11_bwd = r.S_backward[2] / (-ik); S22_bwd = r.S_backward[1] / (-ik)
        S12_bwd = r.S_backward[3] / (ik);  S21_bwd = r.S_backward[4] / (ik)

        rows[idx] = (
            aggregate_file = basename(agg.filename),
            n_monomers     = agg.n_monomers,
            R_ve           = R_ve,
            m_real         = real(m_abs),
            m_imag         = imag(m_abs),
            # Forward scattering amplitudes — BH83 (dimensionless)
            S1_fwd_re = real(r.S_forward[1]),  S1_fwd_im = imag(r.S_forward[1]),
            S2_fwd_re = real(r.S_forward[2]),  S2_fwd_im = imag(r.S_forward[2]),
            S3_fwd_re = real(r.S_forward[3]),  S3_fwd_im = imag(r.S_forward[3]),
            S4_fwd_re = real(r.S_forward[4]),  S4_fwd_im = imag(r.S_forward[4]),
            # Forward scattering amplitudes — MI02 (dimension of length)
            S11_fwd_re = real(S11_fwd), S11_fwd_im = imag(S11_fwd),
            S22_fwd_re = real(S22_fwd), S22_fwd_im = imag(S22_fwd),
            S12_fwd_re = real(S12_fwd), S12_fwd_im = imag(S12_fwd),
            S21_fwd_re = real(S21_fwd), S21_fwd_im = imag(S21_fwd),
            # Backward scattering amplitudes — BH83 (dimensionless)
            S1_bwd_re = real(r.S_backward[1]), S1_bwd_im = imag(r.S_backward[1]),
            S2_bwd_re = real(r.S_backward[2]), S2_bwd_im = imag(r.S_backward[2]),
            S3_bwd_re = real(r.S_backward[3]), S3_bwd_im = imag(r.S_backward[3]),
            S4_bwd_re = real(r.S_backward[4]), S4_bwd_im = imag(r.S_backward[4]),
            # Backward scattering amplitudes — MI02 (dimension of length)
            S11_bwd_re = real(S11_bwd), S11_bwd_im = imag(S11_bwd),
            S22_bwd_re = real(S22_bwd), S22_bwd_im = imag(S22_bwd),
            S12_bwd_re = real(S12_bwd), S12_bwd_im = imag(S12_bwd),
            S21_bwd_re = real(S21_bwd), S21_bwd_im = imag(S21_bwd),
            # Cross-section efficiencies (unpolarized incidence)
            Q_ext = r.Q_ext,
            Q_abs = r.Q_abs,
            Q_sca = r.Q_sca,
            # Solver diagnostics
            converged    = r.converged,
            n_iterations = r.n_iterations,
        )
    end

    df = DataFrame(rows)

    # ── Optional output ───────────────────────────────────────────────────────
    if output_file !== nothing
        if endswith(output_file, ".csv")
            CSV.write(output_file, df)
        elseif endswith(output_file, ".h5") || endswith(output_file, ".hdf5")
            _write_sweep_hdf5(df, output_file, config)
        else
            @warn "Unknown extension for output_file=\"$output_file\"; skipping write."
        end
    end

    return df
end

# ─────────────────────────────────────────────────────────────
# HDF5 writer
# ─────────────────────────────────────────────────────────────

function _write_sweep_hdf5(df::DataFrame, path::String, config::SweepConfig)
    HDF5.h5open(path, "w") do fid
        # Store config metadata as attributes on root
        HDF5.attrs(fid)["wavelength"]      = config.wavelength
        HDF5.attrs(fid)["medium_refindex"] = config.medium_refindex
        HDF5.attrs(fid)["monomer_radius"]  = config.monomer_radius
        HDF5.attrs(fid)["n_jobs"]          = nrow(df)

        # Write each numeric column as a 1-D dataset
        for col in names(df)
            v = df[!, col]
            if eltype(v) <: AbstractString
                fid[col] = collect(String, v)
            else
                fid[col] = collect(v)
            end
        end
    end
end
