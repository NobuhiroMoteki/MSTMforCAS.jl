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

# GPU dispatch hook — set by the CUDA extension's __init__
const _gpu_batch_solve_ref = Ref{Union{Function, Nothing}}(nothing)

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
- `truncation_eps::Union{Float64,Nothing}`: Target relative accuracy of the forward amplitudes
  S₁(0°), S₂(0°) with respect to the VSWF truncation order (default nothing = single-sphere
  Mie criterion only, the behaviour of earlier versions). When set, the truncation order is
  chosen for every refractive index by `contact_truncation_order`, a preliminary two-sphere
  calculation that resolves the coupling between touching monomers. Suggested values:
  `1e-2` for routine use, `1e-3` for precise calculations. Each sphere uses the larger of this
  order, its own Mie order and `truncation_order`. On the GPU path all refractive indices of an
  (aggregate, medium condition) group share one order, the largest over the refractive index
  grid.
- `fft_far_order::Union{Int,Nothing}`: Multipole order that the FFT grid carries (default
  nothing = all orders of every sphere, the behaviour of earlier versions). An integer `F`
  limits the grid to the order `max(F, Mie order)`, where the Mie order is the largest
  single-sphere Mie order of the aggregate over the refractive index grid. `F = 0` selects
  that Mie order. Orders above it then couple near pairs only, by direct translation (see
  `init_fft_grid`). This keeps the memory and the time of the FFT part independent of the
  orders that `truncation_eps` or `truncation_order` add.
- `fft_near_factor::Float64`: Near-pair criterion for `fft_far_order` (default 1.5). Spheres
  are a near pair if their center distance is at most this factor times the sum of their radii.
- `shared_basis::Bool`: Solve all refractive indices of an (aggregate, medium condition)
  group with one shared Krylov basis (`sweep_refractive_index`) instead of one CBICG solve per
  refractive index (default false = the behaviour of earlier versions). All spheres and all
  refractive indices of a group then use one truncation order: the largest of the Mie orders,
  of `truncation_order` and of the orders that `truncation_eps` selects over the refractive
  index grid. Every row meets `convergence_epsilon`: refractive indices that the basis does
  not resolve are solved by CBICG, started from the basis solution. The column `n_iterations`
  holds the basis size, plus the CBICG iterations of such rows. A warning is issued if the
  sweep lies outside the validated range (`shared_basis_range_issues`). Not available
  together with `use_gpu` or with `solver = :gmres`.
- `phase_reference::Symbol`: Point to which the phase of the amplitudes refers (see
  `compute_scattering`). `:particle` (default): the unweighted mean of the sphere centres, so
  the amplitudes do not depend on the coordinate origin of the aggregate files. `:lab`:
  incident wave at the plane z = 0, the only convention before version 1.3.0. The output file
  records it in the attribute `phase_reference`. Files without the attribute are taken as
  `:lab`, and rows of the two conventions cannot share a file.
- `use_gpu::Bool`: Use GPU-accelerated batched solver via CUDA.jl extension (default false).
  Requires `using CUDA` before calling `run_parameter_sweep`. Implies `use_fft=true`.
- `gpu_float32::Bool`: Use Float32 precision for GPU BiCG iterations (default false). It has
  an effect only together with `use_gpu = true`. Mie coefficients and post-processing remain
  Float64. Convergence tolerance is automatically relaxed to max(tol, 1e-5) for Float32 mode.
"""
Base.@kwdef struct SweepConfig
    medium_conditions::Vector{Tuple{Float64,Float64}}
    m_real_range::Tuple{Float64,Float64,Int}
    m_imag_range::Tuple{Float64,Float64,Int}
    max_iterations::Int      = 200
    convergence_epsilon::Float64 = 1e-6
    use_fft::Bool            = false
    truncation_order::Union{Int,Nothing} = nothing
    truncation_eps::Union{Float64,Nothing} = nothing
    fft_far_order::Union{Int,Nothing} = nothing
    fft_near_factor::Float64 = 1.5
    solver::Symbol           = :cbicg   # :cbicg or :gmres
    shared_basis::Bool       = false    # one Krylov basis per (aggregate, medium) group
    phase_reference::Symbol  = :particle  # :particle (default since 1.3.0) or :lab
    use_gpu::Bool            = false
    gpu_float32::Bool        = false
    # Hybrid GPU+CPU threshold: groups with n_monomers <= gpu_np_threshold are
    # processed by CPU threads while the GPU handles larger groups concurrently.
    # 0 = all groups use GPU (original behaviour). Default 500 works well for 16+ cores.
    gpu_np_threshold::Int    = 500
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
        # NaN = truncation order from the single-sphere Mie criterion only
        HDF5.attrs(fid)["truncation_eps"] =
            config.truncation_eps === nothing ? NaN : config.truncation_eps
        # -1 = the FFT grid carries all orders of every sphere
        HDF5.attrs(fid)["fft_far_order"] =
            config.fft_far_order === nothing ? -1 : config.fft_far_order
        HDF5.attrs(fid)["fft_near_factor"] = config.fft_near_factor
        # 1 = rows computed with the shared Krylov basis (one truncation order per group)
        HDF5.attrs(fid)["shared_basis"] = config.shared_basis ? 1 : 0
        # "particle" or "lab": point to which the phase of the amplitudes refers
        HDF5.attrs(fid)["phase_reference"] = String(config.phase_reference)

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
- `phase_reference`: `:particle` (default) or `:lab`, as in the method for positions

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
    initial_amn::Union{Nothing, Matrix{ComplexF64}} = nothing,
    fft_far_order::Union{Int,Nothing} = nothing,
    fft_near_factor::Float64 = 1.5,
    phase_reference::Symbol = :particle
)::Tuple{ScatteringResult, Matrix{ComplexF64}}
    positions_x = agg.positions .* k   # dimensionless (size parameters)
    radii_x     = agg.radii     .* k
    return compute_scattering(positions_x, radii_x, m_rel; tol=tol, max_iter=max_iter,
                              use_fft=use_fft, truncation_order=truncation_order,
                              precomputed_fft=precomputed_fft, solver=solver,
                              initial_amn=initial_amn, fft_far_order=fft_far_order,
                              fft_near_factor=fft_near_factor, phase_reference=phase_reference)
end

# ─────────────────────────────────────────────────────────────
# Truncation order from the two-sphere preliminary calculation
# ─────────────────────────────────────────────────────────────

"""
    _x_bucket(agg::AggregateGeometry, k::Float64) -> Float64

Mean monomer size parameter of an aggregate, rounded to two significant digits.
The contact truncation order depends only weakly on the size parameter, so
aggregates with nearly equal mean monomer size share one preliminary calculation.
"""
function _x_bucket(agg::AggregateGeometry, k::Float64)::Float64
    return round(k * sum(agg.radii) / length(agg.radii); sigdigits = 2)
end

"""
    _build_contact_order_table(keys, eps) -> Dict{Tuple{Float64,ComplexF64},Int}

Run the two-sphere preliminary calculation (`contact_truncation_order`) for every
(mean size parameter, relative refractive index) pair in `keys`, in parallel.
"""
function _build_contact_order_table(
    keys::Vector{Tuple{Float64,ComplexF64}},
    eps::Float64
)::Dict{Tuple{Float64,ComplexF64},Int}
    isempty(keys) && return Dict{Tuple{Float64,ComplexF64},Int}()
    # Largest |m| first: it needs the highest order. Running it alone grows the lazily
    # built translation-coefficient caches before the threaded loop starts.
    sorted = sort(keys; by = key -> -abs(key[2]))
    orders = Vector{Int}(undef, length(sorted))
    orders[1] = contact_truncation_order(sorted[1][1], sorted[1][2]; eps = eps)
    Threads.@threads :dynamic for i in 2:length(sorted)
        x, m_rel = sorted[i]
        orders[i] = contact_truncation_order(x, m_rel; eps = eps)
    end
    return Dict(zip(sorted, orders))
end

"""
    _read_truncation_eps(path::String) -> Union{Float64,Nothing}

`truncation_eps` recorded in an existing results file. Files written before this
attribute existed used the single-sphere Mie criterion, which is reported as `nothing`.
"""
function _read_truncation_eps(path::String)::Union{Float64,Nothing}
    return HDF5.h5open(path, "r") do fid
        a = HDF5.attrs(fid)
        if haskey(a, "truncation_eps")
            v = Float64(a["truncation_eps"])
            isnan(v) ? nothing : v
        else
            nothing
        end
    end
end

"""
    _read_fft_far_settings(path::String) -> Tuple{Union{Int,Nothing},Float64}

`(fft_far_order, fft_near_factor)` recorded in an existing results file. Files written
before these attributes existed let the FFT grid carry all orders. This is reported as
`(nothing, NaN)`.
"""
function _read_fft_far_settings(path::String)::Tuple{Union{Int,Nothing},Float64}
    return HDF5.h5open(path, "r") do fid
        a = HDF5.attrs(fid)
        haskey(a, "fft_far_order") || return (nothing, NaN)
        v = Int(a["fft_far_order"])
        v < 0 ? (nothing, NaN) : (v, Float64(a["fft_near_factor"]))
    end
end

"""
    _read_shared_basis(path::String) -> Bool

Whether an existing results file was written with `shared_basis = true`. Files written
before this attribute existed used one CBICG solve per refractive index (`false`).
"""
function _read_shared_basis(path::String)::Bool
    return HDF5.h5open(path, "r") do fid
        a = HDF5.attrs(fid)
        haskey(a, "shared_basis") && Int(a["shared_basis"]) != 0
    end
end

"""
    _read_phase_reference(path::String) -> Symbol

Phase reference of the amplitudes in an existing results file. Files written before this
attribute existed (versions before 1.3.0) used `:lab`.
"""
function _read_phase_reference(path::String)::Symbol
    return HDF5.h5open(path, "r") do fid
        a = HDF5.attrs(fid)
        haskey(a, "phase_reference") ? Symbol(String(a["phase_reference"])) : :lab
    end
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

    # ── GPU availability check ────────────────────────────────────────────────
    gpu_active = config.use_gpu && _gpu_batch_solve_ref[] !== nothing
    config.gpu_float32 && !config.use_gpu && @warn(
        "gpu_float32 = true has no effect without use_gpu = true. The sweep runs on the CPU in double precision.")
    if config.use_gpu && _gpu_batch_solve_ref[] === nothing
        @warn "use_gpu=true but CUDA extension not loaded. Run `using CUDA` first. Falling back to CPU."
    end
    # FFT mode: explicit config, GPU mode, or auto (Np > 20 for any aggregate)
    auto_fft = any(a.n_monomers > 20 for a in aggregates)
    effective_use_fft = config.use_fft || gpu_active || auto_fft

    if n_jobs_total == 0
        return nothing
    end

    _check_phase_reference(config.phase_reference)
    if config.shared_basis
        config.use_gpu && error("shared_basis = true is not available together with use_gpu = true")
        config.solver == :cbicg || error("shared_basis = true needs solver = :cbicg")
    end

    # ── Check for existing results (resume support) ──────────────────────────
    completed_keys = Set{Tuple{String, Float64, Float64, Float64, Float64}}()

    if output_h5 !== nothing && isfile(output_h5)
        # Rows computed with different truncation settings must not share a file.
        eps_file = _read_truncation_eps(output_h5)
        isequal(eps_file, config.truncation_eps) || error(
            "$(basename(output_h5)) was written with truncation_eps = $(repr(eps_file)), " *
            "but this sweep uses truncation_eps = $(repr(config.truncation_eps)). " *
            "Use a different output file.")
        far_file = _read_fft_far_settings(output_h5)
        far_now  = config.fft_far_order === nothing ? (nothing, NaN) :
                   (config.fft_far_order, config.fft_near_factor)
        isequal(far_file, far_now) || error(
            "$(basename(output_h5)) was written with (fft_far_order, fft_near_factor) = " *
            "$(repr(far_file)), but this sweep uses $(repr(far_now)). " *
            "Use a different output file.")
        # The shared basis uses one truncation order per group, so its rows differ from
        # those of the solve per refractive index at the level of the truncation error.
        sb_file = _read_shared_basis(output_h5)
        sb_file == config.shared_basis || error(
            "$(basename(output_h5)) was written with shared_basis = $(sb_file), " *
            "but this sweep uses shared_basis = $(config.shared_basis). " *
            "Use a different output file.")
        # The amplitudes of the two conventions differ by exp(i k z_r0).
        pr_file = _read_phase_reference(output_h5)
        pr_file == config.phase_reference || error(
            "$(basename(output_h5)) was written with phase_reference = :$(pr_file), " *
            "but this sweep uses phase_reference = :$(config.phase_reference). " *
            "Use a different output file, or set phase_reference = :$(pr_file) to continue it.")
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

    # ── Truncation order per refractive index (two-sphere preliminary calculation) ──
    # order_table[(x_bucket, m_rel)] = contact truncation order; empty if truncation_eps is unset.
    order_table = Dict{Tuple{Float64,ComplexF64},Int}()
    if config.truncation_eps !== nothing
        key_set = Set{Tuple{Float64,ComplexF64}}()
        for ((ai, mci), mis) in group_list
            wl, n_med = config.medium_conditions[mci]
            xb = _x_bucket(aggregates[ai], 2π * n_med / wl)
            # The shared basis and the GPU path take one order per group from the whole
            # refractive index grid, so that a resumed sweep uses the same order as an
            # uninterrupted one.
            for mi in ((config.shared_basis || gpu_active) ? (1:n_mrel) : mis)
                push!(key_set, (xb, ri_grid[mi] / n_med))
            end
        end
        t_tab = time()
        order_table = _build_contact_order_table(collect(key_set), config.truncation_eps)
        @info @sprintf("Truncation orders for truncation_eps = %.1e: %d preliminary two-sphere calculations in %.0f s, orders %d to %d",
                       config.truncation_eps, length(order_table), time() - t_tab,
                       minimum(values(order_table)), maximum(values(order_table)))
    end
    # Truncation order passed to the solver for group (ai, mci) and refractive index mi.
    # Each sphere still uses at least its own single-sphere Mie order.
    _order_for = function (ai::Int, mci::Int, mi::Int)
        config.truncation_eps === nothing && return config.truncation_order
        wl, n_med = config.medium_conditions[mci]
        o = order_table[(_x_bucket(aggregates[ai], 2π * n_med / wl), ri_grid[mi] / n_med)]
        return config.truncation_order === nothing ? o : max(o, config.truncation_order)
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
            use_fft=effective_use_fft, truncation_order=_order_for(ai0, mci0, mis0[1]),
            solver=config.solver, fft_far_order=config.fft_far_order,
            fft_near_factor=config.fft_near_factor,
            phase_reference=config.phase_reference)  # returns (ScatteringResult, amn); result discarded
    end

    # ── Shared basis: range check and caches for the largest order ──────────
    n_sb_fallback = Threads.Atomic{Int}(0)
    if config.shared_basis
        x_max_all = 0.0
        m_rel_all = ComplexF64[]
        for (wl, n_med) in config.medium_conditions
            k_mc = 2π * n_med / wl
            x_max_all = max(x_max_all, k_mc * maximum(maximum(a.radii) for a in aggregates))
            append!(m_rel_all, ri_grid ./ n_med)
        end
        issues = shared_basis_range_issues(m_rel_all, x_max_all)
        isempty(issues) || @warn(
            "shared_basis = true: the sweep lies outside the range in which the shared-basis " *
            "sweep was validated: " * join(issues, "; ") * ". Every row still meets " *
            "convergence_epsilon through the CBICG fallback, but the gain in time can be lost.")
        # Largest order that any group can use, so that no thread grows the global caches.
        ord_bound = maximum(mie_nmax(x_max_all, m) for m in m_rel_all)
        config.truncation_order === nothing || (ord_bound = max(ord_bound, config.truncation_order))
        isempty(order_table) || (ord_bound = max(ord_bound, maximum(values(order_table))))
        nc_warm = _ensure_numconstants(ord_bound + 2)
        _ensure_tran_coef(ord_bound, nc_warm.fnr, nc_warm.bcof)
    end

    # ── Weighted ETA: estimate total work using cost proxy per job ───────────
    # Direct mode: O(Np^2) per iteration → weight ∝ Np^2
    # FFT mode:    O(Np + M log M) per iteration → weight ∝ Np (approx linear)
    job_weights = Float64[]
    for (gk, mis) in group_list
        np = Float64(aggregates[gk[1]].n_monomers)
        w  = effective_use_fft ? np : np^2
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

    # ── Sweep (grouped by geometry+medium for FFT reuse) ─────────────────────
    # GPU mode: sequential outer loop for large-Np groups (GPU provides parallelism
    #   within each group); CPU threads process small-Np groups concurrently.
    # CPU mode: threaded outer loop with dynamic scheduling.
    # force_cpu=true: use CPU FFT path even when gpu_active (for hybrid small-Np groups)
    _sweep_body! = function (gidx; force_cpu::Bool=false)
        (ai, mci), mi_list = group_list[gidx]
        agg     = aggregates[ai]
        wl, n_med = config.medium_conditions[mci]
        k       = 2π * n_med / wl
        np      = agg.n_monomers
        job_w   = effective_use_fft ? Float64(np) : Float64(np)^2

        # Precompute FFT grid once per (aggregate, medium_condition) group
        # With fft_far_order the grid is also built for a single refractive index, so that
        # a resumed sweep uses the same operator as an uninterrupted one.
        # Shared basis and GPU path: one order for the group, from the whole refractive index
        # grid and not from the jobs that remain
        use_sb_here = config.shared_basis && agg.n_monomers >= 2
        use_gpu_here = gpu_active && !force_cpu
        local sb_order::Int = 0
        local sb_mie::Int = 0
        local fft_cache::Union{FFTGridData, Nothing} = nothing
        if use_sb_here || (use_gpu_here && agg.n_monomers >= 2)
            positions_sb = agg.positions .* k
            radii_sb     = agg.radii     .* k
            for m_abs in ri_grid, s in 1:length(radii_sb)
                sb_mie = max(sb_mie, mie_nmax(radii_sb[s], m_abs / n_med))
            end
            sb_order = sb_mie
            for mi in 1:n_mrel
                o = _order_for(ai, mci, mi)
                o === nothing || (sb_order = max(sb_order, o))
            end
            if effective_use_fft
                nois_sb = fill(sb_order, length(radii_sb))
                far_sb  = config.fft_far_order === nothing ? sb_order :
                          min(sb_order, max(config.fft_far_order, sb_mie))
                fft_cache = init_fft_grid(positions_sb, radii_sb, nois_sb;
                                          far_nois=fill(far_sb, length(radii_sb)),
                                          near_factor=config.fft_near_factor)
            end
        elseif effective_use_fft && agg.n_monomers >= 2 &&
           (length(mi_list) > 1 || config.fft_far_order !== nothing)
            positions_x = agg.positions .* k
            radii_x     = agg.radii     .* k
            local max_noi = 0
            for mi in mi_list
                m_rel_tmp = ri_grid[mi] / n_med
                nois_tmp  = [mie_nmax(radii_x[s], m_rel_tmp) for s in 1:length(radii_x)]
                noi_tmp   = maximum(nois_tmp)
                order_tmp = _order_for(ai, mci, mi)
                if order_tmp !== nothing
                    noi_tmp = max(noi_tmp, order_tmp)
                end
                max_noi = max(max_noi, noi_tmp)
            end
            nois_for_grid = fill(max_noi, length(radii_x))
            if config.fft_far_order === nothing
                fft_cache = init_fft_grid(positions_x, radii_x, nois_for_grid)
            else
                # Far order from the whole refractive index grid, not from the jobs that
                # remain, for the same reason.
                local max_mie = 0
                for m_abs in ri_grid, s in 1:length(radii_x)
                    max_mie = max(max_mie, mie_nmax(radii_x[s], m_abs / n_med))
                end
                far_order = min(max_noi, max(config.fft_far_order, max_mie))
                fft_cache = init_fft_grid(positions_x, radii_x, nois_for_grid;
                                          far_nois=fill(far_order, length(radii_x)),
                                          near_factor=config.fft_near_factor)
            end
        end

        # ── Record one result (shared by GPU and CPU paths) ─────────────
        function _record_one!(r, m_abs::ComplexF64; force_flush::Bool=false)
            ik = im * k
            S11_fwd = r.S_forward[2] / (-ik);  S22_fwd = r.S_forward[1] / (-ik)
            S12_fwd = r.S_forward[3] / (ik);   S21_fwd = r.S_forward[4] / (ik)
            S11_bwd = r.S_backward[2] / (-ik); S22_bwd = r.S_backward[1] / (-ik)
            S12_bwd = r.S_backward[3] / (ik);  S21_bwd = r.S_backward[4] / (ik)

            row = (
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
                snapshot_row[] = row
                now = time()
                time_since_flush = now - t_last_flush[]
                should_flush = time_since_flush >= flush_interval || done == n_jobs || force_flush

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
        end  # _record_one!

        # ── GPU batch path or CPU path ────────────────────────────────────
        if use_gpu_here
            ri_batch = [ri_grid[mi] for mi in mi_list]
            # Callback: record each result as soon as its batch completes
            # (enables incremental HDF5 flush and crash recovery)
            n_ri_in_group = length(ri_batch)
            function _on_gpu_result(ri_idx, r, _amn)
                # Force flush on last RI of each group to ensure progress display
                _record_one!(r, ri_batch[ri_idx]; force_flush=(ri_idx == n_ri_in_group))
            end
            # The batched solver uses one truncation order for the whole group: the order
            # of the whole refractive index grid (sb_order), which the FFT grid above has too.
            local order_gpu::Union{Int,Nothing}
            if agg.n_monomers >= 2
                order_gpu = sb_order
            else
                orders_batch = [_order_for(ai, mci, mi) for mi in 1:n_mrel]
                order_gpu = any(isnothing, orders_batch) ? nothing : maximum(orders_batch)
            end
            _gpu_batch_solve_ref[](
                agg, k, ri_batch, n_med, fft_cache,
                (tol=config.convergence_epsilon, max_iter=config.max_iterations,
                 truncation_order=order_gpu,
                 float32=config.gpu_float32,
                 phase_reference=config.phase_reference,
                 on_result=_on_gpu_result))
        elseif use_sb_here
            # ── CPU path with one Krylov basis for all refractive indices ──
            m_abs_list = [ri_grid[mi] for mi in mi_list]
            res_sb, info_sb = sweep_refractive_index(agg.positions .* k, agg.radii .* k,
                m_abs_list ./ n_med;
                tol=config.convergence_epsilon, max_iter=config.max_iterations,
                use_fft=effective_use_fft, truncation_order=sb_order,
                precomputed_fft=fft_cache, warn_range=false, max_mie_order=sb_mie,
                phase_reference=config.phase_reference)
            Threads.atomic_add!(n_sb_fallback, info_sb.n_fallback)
            for (jj, m_abs) in enumerate(m_abs_list)
                _record_one!(res_sb[jj], m_abs; force_flush=(jj == length(m_abs_list)))
            end
        else
            # ── CPU sequential path (continuation method) ─────────────────
            local prev_amn::Union{Nothing, Matrix{ComplexF64}} = nothing
            for mi in mi_list
                m_abs   = ri_grid[mi]
                m_rel   = m_abs / n_med
                r, amn_out = compute_scattering(agg, m_rel, k;
                    tol=config.convergence_epsilon, max_iter=config.max_iterations,
                    use_fft=effective_use_fft, truncation_order=_order_for(ai, mci, mi),
                    precomputed_fft=fft_cache, solver=config.solver,
                    initial_amn=prev_amn, fft_far_order=config.fft_far_order,
                    fft_near_factor=config.fft_near_factor,
                    phase_reference=config.phase_reference)
                prev_amn = amn_out
                _record_one!(r, m_abs)
            end
        end
    end  # _sweep_body!

    # ── Dispatch ─────────────────────────────────────────────────────────────
    # GPU hybrid mode (gpu_np_threshold > 0):
    #   - GPU task: large-Np groups processed sequentially with batched BiCG;
    #               CPU prep/post parallelised via Threads.@threads inside the
    #               GPU batch solver (Approach A).
    #   - CPU threads: small-Np groups processed in parallel concurrently with
    #                  the GPU task (Approach B).
    # CPU-only mode: threaded outer loop with dynamic scheduling.
    if gpu_active
        np_thresh = config.gpu_np_threshold
        gpu_idxs = [i for i in eachindex(group_list)
                    if aggregates[group_list[i][1][1]].n_monomers > np_thresh]
        cpu_idxs = [i for i in eachindex(group_list)
                    if aggregates[group_list[i][1][1]].n_monomers <= np_thresh]

        if isempty(cpu_idxs)
            # All groups go to GPU (threshold=0 or all aggregates are large)
            for gidx in gpu_idxs
                _sweep_body!(gidx)
            end
        else
            @info "Hybrid GPU+CPU mode" gpu_groups=length(gpu_idxs) cpu_groups=length(cpu_idxs) np_threshold=np_thresh
            # GPU task runs on its own thread; CPU threads handle the rest simultaneously
            gpu_task = Threads.@spawn begin
                for gidx in gpu_idxs
                    _sweep_body!(gidx)
                end
            end
            Threads.@threads :dynamic for gidx in cpu_idxs
                _sweep_body!(gidx; force_cpu=true)
            end
            wait(gpu_task)
        end
    else
        Threads.@threads :dynamic for gidx in eachindex(group_list)
            _sweep_body!(gidx)
        end
    end
    # Rows that were recorded after the last flush. Several threads can finish at almost the
    # same time, and the thread that sees `done == n_jobs` can reach the lock before another
    # thread has stored its row. Without this flush such a row would be lost.
    lock(io_lock) do
        if output_h5 !== nothing && !isempty(save_buf)
            _h5_append!(output_h5, save_buf)
        end
        empty!(save_buf)
    end
    println()  # end progress line
    if config.shared_basis
        @info @sprintf("Shared basis: %d of %d rows were solved by the CBICG fallback",
                       n_sb_fallback[], n_jobs)
    end

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
        HDF5.attrs(fid)["phase_reference"] = String(config.phase_reference)

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
