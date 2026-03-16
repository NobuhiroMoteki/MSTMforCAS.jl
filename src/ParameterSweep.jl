"""
    ParameterSweep

Orchestrate parameter sweeps over aggregate geometries, medium conditions, and
refractive indices. Supports multi-threaded parallel execution via Julia's
Threads.@threads. Supports automatic resume: if the output CSV already exists,
completed jobs are skipped and only remaining jobs are computed.

# Typical usage

```julia
aggregates = read_aggregate_catalog("aggregates.h5", "catalog.csv")
config = SweepConfig(
    medium_conditions = [(0.6328, 1.0), (0.6328, 1.33)],  # (wavelength, n_medium)
    m_real_range = (1.5, 1.7, 3),   # min, max, n_grid
    m_imag_range = (0.0, 0.1, 2),   # min, max, n_grid
)
df = run_parameter_sweep(aggregates, config; output_file="results.csv")
```

# Threading notes
The sweep parallelises over the (aggregate × medium_condition × m_real × m_imag)
grid with `Threads.@threads`. Start Julia with `julia -t auto` (or `-t N`) for
speedup. Thread safety: `TranslationCoefs` uses a global cache that is initialised
by a single-threaded pre-warm step before `@threads` is entered, so all threads
only read from the cache during the sweep.
"""

export run_parameter_sweep, write_results_hdf5, SweepConfig

# CSV, DataFrames, HDF5 already imported by AggregateIO.jl (included earlier)

"""
    SweepConfig

Configuration for a parameter sweep computation.

# Fields
- `medium_conditions::Vector{Tuple{Float64,Float64}}`: List of (wavelength, medium_refindex)
  pairs. Wavelength in the same physical units as aggregate coordinates. Each pair defines
  a medium condition to sweep over.
- `m_real_range::Tuple{Float64,Float64,Int}`: (min, max, n_grid) for real part of absolute
  complex refractive index. Generates `n_grid` equally spaced values from min to max.
  For a single value, use e.g. `(1.5, 1.5, 1)`.
- `m_imag_range::Tuple{Float64,Float64,Int}`: (min, max, n_grid) for imaginary part.
  The sweep covers the full Cartesian product: medium_conditions × m_real × m_imag.
- `max_iterations::Int`: Maximum CBICG iterations (default 200)
- `convergence_epsilon::Float64`: Solver convergence threshold (default 1e-6)
- `use_fft::Bool`: Use FFT-accelerated translation (default false)
- `truncation_order::Union{Int,Nothing}`: Override VSWF truncation order (default nothing = auto)
"""
Base.@kwdef struct SweepConfig
    medium_conditions::Vector{Tuple{Float64,Float64}}
    m_real_range::Tuple{Float64,Float64,Int}
    m_imag_range::Tuple{Float64,Float64,Int}
    max_iterations::Int      = 200
    convergence_epsilon::Float64 = 1e-6
    use_fft::Bool            = false
    truncation_order::Union{Int,Nothing} = nothing
end

"""
    _make_ri_grid(config::SweepConfig) -> Vector{ComplexF64}

Generate the Cartesian product grid of complex refractive indices from the
(min, max, n_grid) ranges for real and imaginary parts.
"""
function _make_ri_grid(config::SweepConfig)::Vector{ComplexF64}
    mr_min, mr_max, mr_n = config.m_real_range
    mi_min, mi_max, mi_n = config.m_imag_range
    m_real_vals = mr_n == 1 ? [mr_min] : collect(range(mr_min, mr_max, length=mr_n))
    m_imag_vals = mi_n == 1 ? [mi_min] : collect(range(mi_min, mi_max, length=mi_n))
    grid = ComplexF64[]
    for mr in m_real_vals, mi in m_imag_vals
        push!(grid, mr + mi * im)
    end
    return grid
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
    max_iter::Int    = 200,
    use_fft::Bool    = false,
    truncation_order::Union{Int,Nothing} = nothing
)::ScatteringResult
    positions_x = agg.positions .* k   # dimensionless (size parameters)
    radii_x     = agg.radii     .* k
    return compute_scattering(positions_x, radii_x, m_rel; tol=tol, max_iter=max_iter,
                              use_fft=use_fft, truncation_order=truncation_order)
end

# ─────────────────────────────────────────────────────────────
# Main sweep function
# ─────────────────────────────────────────────────────────────

"""
    run_parameter_sweep(aggregates::Vector{AggregateGeometry}, config::SweepConfig;
                        output_file=nothing) -> DataFrame

Execute a parameter sweep over the Cartesian product:
  aggregates × medium_conditions × m_real_values × m_imag_values.

# Resume support
If `output_file` points to an existing CSV file, previously completed jobs are
loaded and skipped. Only remaining jobs are computed. The merged result (old + new)
is written back to the same file.

# Arguments
- `aggregates`: Vector of `AggregateGeometry` (e.g., from `read_aggregate_catalog`)
- `config`: `SweepConfig` with medium conditions, RI ranges, etc.
- `output_file`: Optional path for CSV or HDF5 output

# Returns
`DataFrame` with one row per (aggregate, medium_condition, m) combination,
including aggregate metadata, scattering amplitudes, and efficiency factors.
"""
function run_parameter_sweep(
    aggregates::Vector{AggregateGeometry},
    config::SweepConfig;
    output_file::Union{String,Nothing} = nothing
)::DataFrame

    ri_grid = _make_ri_grid(config)
    n_agg   = length(aggregates)
    n_mc    = length(config.medium_conditions)
    n_mrel  = length(ri_grid)
    n_jobs_total = n_agg * n_mc * n_mrel

    if n_jobs_total == 0
        return DataFrame()
    end

    # ── Check for existing results (resume support) ──────────────────────────
    df_existing = DataFrame()
    completed_keys = Set{Tuple{String, Float64, Float64, Float64, Float64}}()

    if output_file !== nothing && endswith(output_file, ".csv") && isfile(output_file)
        df_tmp = CSV.read(output_file, DataFrame)
        required_cols = (:source, :wavelength, :medium_refindex, :m_real, :m_imag)
        if all(c -> hasproperty(df_tmp, c), required_cols)
            df_existing = df_tmp
            for row in eachrow(df_existing)
                push!(completed_keys, (row.source, row.wavelength, row.medium_refindex,
                                       row.m_real, row.m_imag))
            end
            @info "Resume: found $(length(completed_keys)) completed jobs in $(basename(output_file))"
        else
            @warn "Existing $(basename(output_file)) has incompatible format (missing columns); ignoring and recomputing all jobs."
        end
    end

    # ── Build job grid, excluding completed jobs ─────────────────────────────
    # Each job: (aggregate_index, medium_condition_index, ri_index)
    grid = Tuple{Int,Int,Int}[]
    for ai in 1:n_agg, mci in 1:n_mc, mi in 1:n_mrel
        wl, n_med = config.medium_conditions[mci]
        m_abs = ri_grid[mi]
        key = (aggregates[ai].source, wl, n_med, real(m_abs), imag(m_abs))
        if key ∉ completed_keys
            push!(grid, (ai, mci, mi))
        end
    end

    n_jobs = length(grid)
    if n_jobs == 0
        @info "All $n_jobs_total jobs already completed. Nothing to compute."
        return df_existing
    end

    n_skipped = n_jobs_total - n_jobs
    if n_skipped > 0
        @info "Resuming: $n_skipped jobs already done, $n_jobs remaining"
    end

    # ── Pre-warm global caches (single-threaded, must run before @threads) ──
    let (ai0, mci0, mi0) = grid[1],
        agg0  = aggregates[ai0],
        (wl0, n_med0) = config.medium_conditions[mci0],
        k0    = 2π * n_med0 / wl0,
        m_rel0 = ri_grid[mi0] / n_med0
        compute_scattering(agg0, m_rel0, k0;
            tol=config.convergence_epsilon, max_iter=config.max_iterations,
            use_fft=config.use_fft, truncation_order=config.truncation_order)
    end

    # ── Pre-allocate result rows ─────────────────────────────────────────────
    rows = Vector{NamedTuple}(undef, n_jobs)

    # ── Parallel sweep ────────────────────────────────────────────────────────
    Threads.@threads for idx in 1:n_jobs
        ai, mci, mi = grid[idx]
        agg     = aggregates[ai]
        wl, n_med = config.medium_conditions[mci]
        k       = 2π * n_med / wl
        m_abs   = ri_grid[mi]
        m_rel   = m_abs / n_med

        r = compute_scattering(agg, m_rel, k;
            tol=config.convergence_epsilon, max_iter=config.max_iterations,
            use_fft=config.use_fft, truncation_order=config.truncation_order)

        # MI02 amplitudes: S11=S2/(-ik), S22=S1/(-ik), S12=S3/(ik), S21=S4/(ik)
        ik = im * k
        S11_fwd = r.S_forward[2] / (-ik);  S22_fwd = r.S_forward[1] / (-ik)
        S12_fwd = r.S_forward[3] / (ik);   S21_fwd = r.S_forward[4] / (ik)
        S11_bwd = r.S_backward[2] / (-ik); S22_bwd = r.S_backward[1] / (-ik)
        S12_bwd = r.S_backward[3] / (ik);  S21_bwd = r.S_backward[4] / (ik)

        rows[idx] = (
            # Aggregate metadata — source
            source     = agg.source,
            # Aggregate metadata — constant parameters
            mean_rp    = agg.mean_rp,
            rel_std_rp = agg.rel_std_rp,
            k_f        = agg.k_f,
            # Aggregate metadata — sweep parameters
            Df         = agg.Df,
            n_monomers = agg.n_monomers,
            agg_num    = agg.agg_num,
            # Aggregate metadata — derived quantities
            R_ve       = agg.R_ve,
            R_g        = agg.R_g,
            eps_agg    = agg.eps_agg,
            # Medium condition
            wavelength     = wl,
            medium_refindex = n_med,
            # Material
            m_real     = real(m_abs),
            m_imag     = imag(m_abs),
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
            converged        = r.converged,
            n_iterations     = r.n_iterations,
            truncation_order = r.truncation_order,
        )
    end

    df_new = DataFrame(rows)

    # ── Merge with existing results ──────────────────────────────────────────
    df = nrow(df_existing) > 0 ? vcat(df_existing, df_new) : df_new

    # ── Write output ─────────────────────────────────────────────────────────
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

"""
    write_results_hdf5(df::DataFrame, path::String, config::SweepConfig)

Write sweep results DataFrame to HDF5 format.
Each column is stored as a 1-D dataset; sweep config metadata is stored as root attributes.
"""
function write_results_hdf5(df::DataFrame, path::String, config::SweepConfig)
    _write_sweep_hdf5(df, path, config)
end

function _write_sweep_hdf5(df::DataFrame, path::String, config::SweepConfig)
    HDF5.h5open(path, "w") do fid
        # Store config metadata as attributes on root
        wl_vals  = [mc[1] for mc in config.medium_conditions]
        nmed_vals = [mc[2] for mc in config.medium_conditions]
        HDF5.attrs(fid)["medium_conditions_wavelength"]  = wl_vals
        HDF5.attrs(fid)["medium_conditions_refindex"]    = nmed_vals
        HDF5.attrs(fid)["m_real_range"]    = Float64[config.m_real_range...]
        HDF5.attrs(fid)["m_imag_range"]    = Float64[config.m_imag_range...]
        HDF5.attrs(fid)["n_jobs"]          = nrow(df)

        # Write each column as a 1-D dataset
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
