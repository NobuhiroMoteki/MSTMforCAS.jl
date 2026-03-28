"""
    ParameterSweep

Orchestrate parameter sweeps over aggregate geometries, medium conditions, and
refractive indices. Supports multi-threaded parallel execution via Julia's
Threads.@threads. Results are written incrementally to HDF5 for safety and
memory efficiency. Supports automatic resume: if the output HDF5 already exists,
completed jobs are skipped and only remaining jobs are computed.

# Typical usage

```julia
aggregates = read_aggregate_catalog("aggregates.h5", "catalog.csv")
config = SweepConfig(
    medium_conditions = [(0.6328, 1.0), (0.6328, 1.33)],  # (wavelength, n_medium)
    m_real_range = (1.5, 1.7, 3),   # min, max, n_grid
    m_imag_range = (0.0, 0.1, 2),   # min, max, n_grid
)
run_parameter_sweep(aggregates, config; output_h5="results.h5")
```

# Threading notes
The sweep parallelises over the (aggregate × medium_condition × m_real × m_imag)
grid with `Threads.@threads`. Start Julia with `julia -t auto` (or `-t N`) for
speedup. Thread safety: `TranslationCoefs` uses a global cache that is initialised
by a single-threaded pre-warm step before `@threads` is entered, so all threads
only read from the cache during the sweep. HDF5 writes are serialised via a lock.
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
    solver::Symbol           = :cbicg   # :cbicg or :gmres
end

"""
    _make_ri_grid(config::SweepConfig) -> Vector{ComplexF64}

Generate the Cartesian product grid of complex refractive indices from the
(min, max, n_grid) ranges for real and imaginary parts.

Uses snake (boustrophedon) ordering: even-indexed m_real rows sweep m_imag
forward, odd-indexed rows sweep backward. This keeps consecutive grid points
close in the complex plane, improving continuation (warm-start) quality at
row boundaries (|Δm| ≈ Δm_real instead of the full m_imag range).
"""
function _make_ri_grid(config::SweepConfig)::Vector{ComplexF64}
    mr_min, mr_max, mr_n = config.m_real_range
    mi_min, mi_max, mi_n = config.m_imag_range
    m_real_vals = mr_n == 1 ? [mr_min] : collect(range(mr_min, mr_max, length=mr_n))
    m_imag_vals = mi_n == 1 ? [mi_min] : collect(range(mi_min, mi_max, length=mi_n))
    grid = ComplexF64[]
    sizehint!(grid, mr_n * mi_n)
    for (ri, mr) in enumerate(m_real_vals)
        if iseven(ri)
            for mi in reverse(m_imag_vals)
                push!(grid, mr + mi * im)
            end
        else
            for mi in m_imag_vals
                push!(grid, mr + mi * im)
            end
        end
    end
    return grid
end

function _fmt_hms(s::Float64)
    isfinite(s) || return "--:--:--"
    h, r  = divrem(round(Int, s), 3600)
    m, sec = divrem(r, 60)
    @sprintf("%02d:%02d:%02d", h, m, sec)
end

# ─────────────────────────────────────────────────────────────
# HDF5 incremental I/O helpers
# ─────────────────────────────────────────────────────────────

# (column_name, storage_type) — order must match row NamedTuple field order
const _SWEEP_COLS = Tuple{String,DataType}[
    ("source",           String),
    ("mean_rp",          Float64),
    ("rel_std_rp",       Float64),
    ("k_f",              Float64),
    ("Df",               Float64),
    ("n_monomers",       Int64),
    ("agg_num",          Int64),
    ("R_ve",             Float64),
    ("R_g",              Float64),
    ("eps_agg",          Float64),
    ("wavelength",       Float64),
    ("medium_refindex",  Float64),
    ("m_real",           Float64),
    ("m_imag",           Float64),
    ("S1_fwd_re",        Float64), ("S1_fwd_im",  Float64),
    ("S2_fwd_re",        Float64), ("S2_fwd_im",  Float64),
    ("S3_fwd_re",        Float64), ("S3_fwd_im",  Float64),
    ("S4_fwd_re",        Float64), ("S4_fwd_im",  Float64),
    ("S11_fwd_re",       Float64), ("S11_fwd_im", Float64),
    ("S22_fwd_re",       Float64), ("S22_fwd_im", Float64),
    ("S12_fwd_re",       Float64), ("S12_fwd_im", Float64),
    ("S21_fwd_re",       Float64), ("S21_fwd_im", Float64),
    ("S1_bwd_re",        Float64), ("S1_bwd_im",  Float64),
    ("S2_bwd_re",        Float64), ("S2_bwd_im",  Float64),
    ("S3_bwd_re",        Float64), ("S3_bwd_im",  Float64),
    ("S4_bwd_re",        Float64), ("S4_bwd_im",  Float64),
    ("S11_bwd_re",       Float64), ("S11_bwd_im", Float64),
    ("S22_bwd_re",       Float64), ("S22_bwd_im", Float64),
    ("S12_bwd_re",       Float64), ("S12_bwd_im", Float64),
    ("S21_bwd_re",       Float64), ("S21_bwd_im", Float64),
    ("Q_ext",            Float64),
    ("Q_abs",            Float64),
    ("Q_sca",            Float64),
    ("converged",        Int8),    # Bool stored as Int8 (0/1)
    ("n_iterations",     Int64),
    ("truncation_order", Int64),
]

const _H5_CHUNK = 10_000   # rows per HDF5 chunk (affects IO and compression granularity)

"""Create a new HDF5 results file with extendable datasets and config metadata."""
function _h5_init!(path::String, config::SweepConfig)
    HDF5.h5open(path, "w") do fid
        wl_vals   = [mc[1] for mc in config.medium_conditions]
        nmed_vals = [mc[2] for mc in config.medium_conditions]
        HDF5.attrs(fid)["medium_conditions_wavelength"] = wl_vals
        HDF5.attrs(fid)["medium_conditions_refindex"]   = nmed_vals
        HDF5.attrs(fid)["m_real_range"] = Float64[config.m_real_range...]
        HDF5.attrs(fid)["m_imag_range"] = Float64[config.m_imag_range...]

        for (col, T) in _SWEEP_COLS
            HDF5.create_dataset(fid, col, HDF5.datatype(T),
                HDF5.dataspace((0,), (-1,)); chunk=(_H5_CHUNK,))
        end
    end
end

"""Append a batch of result rows to an existing HDF5 results file."""
function _h5_append!(path::String, buf::Vector)
    isempty(buf) && return
    n_new = length(buf)
    HDF5.h5open(path, "r+") do fid
        for (col, T) in _SWEEP_COLS
            ds    = fid[col]
            n_old = size(ds, 1)
            HDF5.set_extent_dims(ds, (n_old + n_new,))
            sym  = Symbol(col)
            vals = if T === Int8
                Int8[getproperty(row, sym) ? Int8(1) : Int8(0) for row in buf]
            elseif T === Int64
                Int64[getproperty(row, sym) for row in buf]
            elseif T === String
                String[getproperty(row, sym) for row in buf]
            else  # Float64
                Float64[getproperty(row, sym) for row in buf]
            end
            ds[(n_old+1):(n_old+n_new)] = vals
        end
    end
end

# ─────────────────────────────────────────────────────────────
# AggregateGeometry overload of compute_scattering
# ─────────────────────────────────────────────────────────────

"""
    compute_scattering(agg::AggregateGeometry, m_rel::ComplexF64, k::Float64;
                       tol, max_iter, initial_amn) -> (ScatteringResult, Matrix{ComplexF64})

Compute scattering for an `AggregateGeometry` object.

# Arguments
- `agg`: aggregate geometry (positions and radii in physical units consistent with `k`)
- `m_rel`: complex refractive index relative to medium, m_sphere / n_medium
- `k`: wavenumber in the medium, k = 2π × n_medium / λ₀ (same units⁻¹ as agg coordinates)
- `initial_amn`: if provided, use as initial guess for the iterative solver
  (continuation method for refractive index sweeps)

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
    truncation_order::Union{Int,Nothing} = nothing,
    precomputed_fft::Union{FFTGridData, Nothing} = nothing,
    solver::Symbol = :cbicg,
    initial_amn::Union{Nothing, Matrix{ComplexF64}} = nothing
)::Tuple{ScatteringResult, Matrix{ComplexF64}}
    positions_x = agg.positions .* k   # dimensionless (size parameters)
    radii_x     = agg.radii     .* k
    return compute_scattering(positions_x, radii_x, m_rel; tol=tol, max_iter=max_iter,
                              use_fft=use_fft, truncation_order=truncation_order,
                              precomputed_fft=precomputed_fft, solver=solver,
                              initial_amn=initial_amn)
end

# ─────────────────────────────────────────────────────────────
# Main sweep function
# ─────────────────────────────────────────────────────────────

"""
    run_parameter_sweep(aggregates::Vector{AggregateGeometry}, config::SweepConfig;
                        output_h5=nothing)

Execute a parameter sweep over the Cartesian product:
  aggregates × medium_conditions × m_real_values × m_imag_values.

Results are written incrementally to `output_h5` (HDF5) as computation proceeds,
so partial results are preserved if the run is interrupted.

# Resume support
If `output_h5` points to an existing file, previously completed jobs are
detected by reading the key columns and skipped automatically.

# Arguments
- `aggregates`: Vector of `AggregateGeometry` (e.g., from `read_aggregate_catalog`)
- `config`: `SweepConfig` with medium conditions, RI ranges, etc.
- `output_h5`: Path for HDF5 output (created or appended to)
"""
function run_parameter_sweep(
    aggregates::Vector{AggregateGeometry},
    config::SweepConfig;
    output_h5::Union{String,Nothing} = nothing
)::Nothing

    ri_grid = _make_ri_grid(config)
    n_agg   = length(aggregates)
    n_mc    = length(config.medium_conditions)
    n_mrel  = length(ri_grid)
    n_jobs_total = n_agg * n_mc * n_mrel

    if n_jobs_total == 0
        return nothing
    end

    # ── Check for existing results (resume support) ──────────────────────────
    completed_keys = Set{Tuple{String, Float64, Float64, Float64, Float64}}()

    if output_h5 !== nothing && isfile(output_h5)
        HDF5.h5open(output_h5, "r") do fid
            if haskey(fid, "source") && size(fid["source"], 1) > 0
                sources  = read(fid["source"])
                wls      = read(fid["wavelength"])
                n_meds   = read(fid["medium_refindex"])
                m_reals  = read(fid["m_real"])
                m_imags  = read(fid["m_imag"])
                for i in eachindex(sources)
                    push!(completed_keys,
                        (sources[i], wls[i], n_meds[i], m_reals[i], m_imags[i]))
                end
                @info "Resume: found $(length(completed_keys)) completed jobs in $(basename(output_h5))"
            end
        end
    end

    # ── Build job grid, excluding completed jobs ─────────────────────────────
    # Group jobs by (ai, mci) so that all m_rel values for the same
    # (aggregate, medium_condition) are processed together, sharing FFT setup.
    # groups[(ai,mci)] = [mi1, mi2, ...]
    groups = Dict{Tuple{Int,Int}, Vector{Int}}()
    n_skipped_count = 0
    for ai in 1:n_agg, mci in 1:n_mc, mi in 1:n_mrel
        wl, n_med = config.medium_conditions[mci]
        m_abs = ri_grid[mi]
        key = (aggregates[ai].source, wl, n_med, real(m_abs), imag(m_abs))
        if key ∉ completed_keys
            gk = (ai, mci)
            if !haskey(groups, gk)
                groups[gk] = Int[]
            end
            push!(groups[gk], mi)
        else
            n_skipped_count += 1
        end
    end
    group_list = collect(pairs(groups))  # Vector of (ai,mci) => [mi...]
    n_jobs = sum(length(mis) for (_, mis) in group_list; init=0)

    if n_jobs == 0
        @info "All $n_jobs_total jobs already completed. Nothing to compute."
        return nothing
    end
    if n_skipped_count > 0
        @info "Resuming: $n_skipped_count jobs already done, $n_jobs remaining"
    end

    # ── Initialize HDF5 file if needed ──────────────────────────────────────
    if output_h5 !== nothing && !isfile(output_h5)
        _h5_init!(output_h5, config)
    end

    # ── Pre-warm global caches (single-threaded, must run before @threads) ──
    let (gk0, mis0) = group_list[1],
        (ai0, mci0) = gk0,
        agg0  = aggregates[ai0],
        (wl0, n_med0) = config.medium_conditions[mci0],
        k0    = 2π * n_med0 / wl0,
        m_rel0 = ri_grid[mis0[1]] / n_med0
        compute_scattering(agg0, m_rel0, k0;
            tol=config.convergence_epsilon, max_iter=config.max_iterations,
            use_fft=config.use_fft, truncation_order=config.truncation_order,
            solver=config.solver)  # returns (ScatteringResult, amn); result discarded
    end

    # ── Weighted ETA: estimate total work using cost proxy per job ───────────
    # Direct mode: O(Np^2) per iteration → weight ∝ Np^2
    # FFT mode:    O(Np + M log M) per iteration → weight ∝ Np (approx linear)
    job_weights = Float64[]
    for (gk, mis) in group_list
        np = Float64(aggregates[gk[1]].n_monomers)
        w  = config.use_fft ? np : np^2
        for _ in mis
            push!(job_weights, w)
        end
    end
    total_weight    = sum(job_weights)
    weight_done     = Threads.Atomic{Float64}(0.0)

    # ── Progress state ────────────────────────────────────────────────────────
    n_done      = Threads.Atomic{Int}(0)
    io_lock     = ReentrantLock()
    save_buf    = NamedTuple[]
    t_sw        = time()
    t_last_flush = Threads.Atomic{Float64}(t_sw)
    snapshot_row = Ref{Union{NamedTuple, Nothing}}(nothing)  # last result for display
    flush_interval = 60.0   # seconds — flush to HDF5 at least this often

    # ── Sort groups heavy-first for better dynamic load balancing ────────────
    sort!(group_list, by = ((gk, mis),) -> -aggregates[gk[1]].n_monomers)

    # ── Parallel sweep (grouped by geometry+medium for FFT reuse) ────────────
    Threads.@threads :dynamic for gidx in eachindex(group_list)
        (ai, mci), mi_list = group_list[gidx]
        agg     = aggregates[ai]
        wl, n_med = config.medium_conditions[mci]
        k       = 2π * n_med / wl
        np      = agg.n_monomers
        job_w   = Float64(np)^2

        # Precompute FFT grid once per (aggregate, medium_condition) group
        local fft_cache::Union{FFTGridData, Nothing} = nothing
        if config.use_fft && agg.n_monomers >= 2 && length(mi_list) > 1
            positions_x = agg.positions .* k
            radii_x     = agg.radii     .* k
            local max_noi = 0
            for mi in mi_list
                m_rel_tmp = ri_grid[mi] / n_med
                nois_tmp  = [mie_nmax(radii_x[s], m_rel_tmp) for s in 1:length(radii_x)]
                noi_tmp   = maximum(nois_tmp)
                if config.truncation_order !== nothing
                    noi_tmp = max(noi_tmp, config.truncation_order)
                end
                max_noi = max(max_noi, noi_tmp)
            end
            nois_for_grid = fill(max_noi, length(radii_x))
            fft_cache = init_fft_grid(positions_x, radii_x, nois_for_grid)
        end

        # Continuation method: carry previous solution as initial guess
        local prev_amn::Union{Nothing, Matrix{ComplexF64}} = nothing

        for mi in mi_list
            m_abs   = ri_grid[mi]
            m_rel   = m_abs / n_med

            r, amn_out = compute_scattering(agg, m_rel, k;
                tol=config.convergence_epsilon, max_iter=config.max_iterations,
                use_fft=config.use_fft, truncation_order=config.truncation_order,
                precomputed_fft=fft_cache, solver=config.solver,
                initial_amn=prev_amn)
            prev_amn = amn_out

            # MI02 amplitudes: S11=S2/(-ik), S22=S1/(-ik), S12=S3/(ik), S21=S4/(ik)
            ik = im * k
            S11_fwd = r.S_forward[2] / (-ik);  S22_fwd = r.S_forward[1] / (-ik)
            S12_fwd = r.S_forward[3] / (ik);   S21_fwd = r.S_forward[4] / (ik)
            S11_bwd = r.S_backward[2] / (-ik); S22_bwd = r.S_backward[1] / (-ik)
            S12_bwd = r.S_backward[3] / (ik);  S21_bwd = r.S_backward[4] / (ik)

            local row = (
                source     = agg.source,
                mean_rp    = agg.mean_rp,
                rel_std_rp = agg.rel_std_rp,
                k_f        = agg.k_f,
                Df         = agg.Df,
                n_monomers = agg.n_monomers,
                agg_num    = agg.agg_num,
                R_ve       = agg.R_ve,
                R_g        = agg.R_g,
                eps_agg    = agg.eps_agg,
                wavelength     = wl,
                medium_refindex = n_med,
                m_real     = real(m_abs),
                m_imag     = imag(m_abs),
                S1_fwd_re = real(r.S_forward[1]),  S1_fwd_im = imag(r.S_forward[1]),
                S2_fwd_re = real(r.S_forward[2]),  S2_fwd_im = imag(r.S_forward[2]),
                S3_fwd_re = real(r.S_forward[3]),  S3_fwd_im = imag(r.S_forward[3]),
                S4_fwd_re = real(r.S_forward[4]),  S4_fwd_im = imag(r.S_forward[4]),
                S11_fwd_re = real(S11_fwd), S11_fwd_im = imag(S11_fwd),
                S22_fwd_re = real(S22_fwd), S22_fwd_im = imag(S22_fwd),
                S12_fwd_re = real(S12_fwd), S12_fwd_im = imag(S12_fwd),
                S21_fwd_re = real(S21_fwd), S21_fwd_im = imag(S21_fwd),
                S1_bwd_re = real(r.S_backward[1]), S1_bwd_im = imag(r.S_backward[1]),
                S2_bwd_re = real(r.S_backward[2]), S2_bwd_im = imag(r.S_backward[2]),
                S3_bwd_re = real(r.S_backward[3]), S3_bwd_im = imag(r.S_backward[3]),
                S4_bwd_re = real(r.S_backward[4]), S4_bwd_im = imag(r.S_backward[4]),
                S11_bwd_re = real(S11_bwd), S11_bwd_im = imag(S11_bwd),
                S22_bwd_re = real(S22_bwd), S22_bwd_im = imag(S22_bwd),
                S12_bwd_re = real(S12_bwd), S12_bwd_im = imag(S12_bwd),
                S21_bwd_re = real(S21_bwd), S21_bwd_im = imag(S21_bwd),
                Q_ext = r.Q_ext,
                Q_abs = r.Q_abs,
                Q_sca = r.Q_sca,
                converged        = r.converged,
                n_iterations     = r.n_iterations,
                truncation_order = r.truncation_order,
            )

            done = Threads.atomic_add!(n_done, 1) + 1
            Threads.atomic_add!(weight_done, job_w)

            lock(io_lock) do
                push!(save_buf, row)
                snapshot_row[] = row   # keep latest for display
                now = time()
                time_since_flush = now - t_last_flush[]
                should_flush = time_since_flush >= flush_interval || done == n_jobs

                if should_flush
                    elapsed        = now - t_sw
                    rate           = done / elapsed
                    remaining_jobs = n_jobs_total - (done + n_skipped_count)
                    remaining_days = rate > 0 ? remaining_jobs / rate / 86400.0 : Inf

                    @printf("\n  [elapsed %s | remaining %.2f days]  %d / %d  (%.2f%%)  %.2f jobs/s",
                        _fmt_hms(elapsed), remaining_days,
                        done + n_skipped_count, n_jobs_total,
                        100.0 * (done + n_skipped_count) / n_jobs_total,
                        rate)

                    # Snapshot: show last computed result
                    sr = snapshot_row[]
                    if sr !== nothing
                        Ss_fwd = ComplexF64(sr.S11_fwd_re, sr.S11_fwd_im) +
                                 im * ComplexF64(sr.S12_fwd_re, sr.S12_fwd_im)
                        Sp_fwd = ComplexF64(sr.S22_fwd_re, sr.S22_fwd_im) -
                                 im * ComplexF64(sr.S21_fwd_re, sr.S21_fwd_im)
                        src_short = basename(sr.source)
                        if length(src_short) > 40; src_short = src_short[1:37] * "..."; end
                        @printf("\n    last: Np=%d m=%.3f+%.4fi Df=%.2f agg#%d  Ss=%.3e%+.3ei Sp=%.3e%+.3ei Q=%.4f",
                            sr.n_monomers, sr.m_real, sr.m_imag, sr.Df, sr.agg_num,
                            real(Ss_fwd), imag(Ss_fwd), real(Sp_fwd), imag(Sp_fwd), sr.Q_ext)
                    end

                    if output_h5 !== nothing && !isempty(save_buf)
                        _h5_append!(output_h5, save_buf)
                    end
                    empty!(save_buf)
                    t_last_flush[] = now
                end
            end
        end  # for mi
    end  # @threads for gidx
    println()  # end progress line

    return nothing
end

# ─────────────────────────────────────────────────────────────
# Utility: read HDF5 results into a DataFrame
# ─────────────────────────────────────────────────────────────

"""
    write_results_hdf5(df::DataFrame, path::String, config::SweepConfig)

Write sweep results DataFrame to HDF5 format (one-shot, for post-processing use).
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
