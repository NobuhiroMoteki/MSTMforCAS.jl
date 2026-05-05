"""
Extension sweep for extending the existing LUT
(results_fullsweep_agg20260323_00.h5) to cover the L-shaped differential region:

  Region A: m_real ∈ {2.45, 2.50, 2.55, 2.60}            × m_imag ∈ [0.15, 1.60] step 0.05 (30 pts) → 120 RI pts
  Region B: m_real ∈ [1.55, 2.40] step 0.05 (18 pts)     × m_imag ∈ {1.45, 1.50, 1.55, 1.60}       →  72 RI pts

Both regions are written to a separate output HDF5 file so the existing LUT
is not touched. The two regions are disjoint in (m_real, m_imag), so they can
share the same output file via run_parameter_sweep's resume logic.

Run with (mirror flags of run_sweep_h5.jl):
  julia -t auto --project=. scripts/run_sweep_h5_extension.jl --gpu --float32
"""

using MSTMforCAS, Printf

gpu_float32 = "--float32" in ARGS
use_gpu = "--gpu" in ARGS || gpu_float32
if use_gpu
    @info "GPU mode requested — loading CUDA.jl..."
    using CUDA
end

use_fft = "--fft" in ARGS || use_gpu
truncation_order = let to = nothing
    for (i, arg) in enumerate(ARGS)
        if arg == "--truncation-order" && i < length(ARGS)
            to = parse(Int, ARGS[i+1])
        end
    end
    to
end

# ─── Aggregate data files ────────────────────────────────────────────────────
agg_dir  = joinpath(@__DIR__, "..", "data", "aggregates")
h5_file  = joinpath(agg_dir, "aggregates_20260323_00.h5")
csv_file = joinpath(agg_dir, "catalog_20260323_00.csv")

agg_id = let m = match(r"(\d{8}_\d{2})", basename(h5_file))
    m !== nothing ? m[1] : "unknown"
end

println("Reading aggregate catalog:")
println("  H5:  $h5_file")
println("  CSV: $csv_file")
println("  Aggregate ID: $agg_id")

aggregates = read_aggregate_catalog(h5_file, csv_file)
println("  Loaded $(length(aggregates)) aggregates")
println()

# ─── Output path (separate file — does NOT touch the existing LUT) ───────────
results_dir = joinpath(@__DIR__, "..", "data", "results")
mkpath(results_dir)
output_h5 = joinpath(results_dir, "results_fullsweep_agg$(agg_id)_extension.h5")

println("Output: $output_h5")
if isfile(output_h5)
    println("  (existing file found — will resume from where left off)")
end
println()

# ─── Common sweep settings ───────────────────────────────────────────────────
medium_conditions = [(0.453, 1.0), (0.638, 1.0), (0.834, 1.0)]

function make_config(m_real_range, m_imag_range)
    SweepConfig(
        medium_conditions   = medium_conditions,
        m_real_range        = m_real_range,
        m_imag_range        = m_imag_range,
        max_iterations      = 200,
        convergence_epsilon = 1e-6,
        use_fft             = use_fft,
        truncation_order    = truncation_order,
        use_gpu             = use_gpu,
        gpu_float32         = gpu_float32,
        gpu_np_threshold    = 500,
    )
end

# ─── Region A: new m_real × all m_imag (120 RI pts) ─────────────────────────
# m_real ∈ {2.45, 2.50, 2.55, 2.60}
# m_imag ∈ 0.15 .. 1.60 step 0.05 (30 pts)
config_A = make_config((2.45, 2.60, 4), (0.15, 1.60, 30))

# ─── Region B: old m_real × new m_imag (72 RI pts) ──────────────────────────
# m_real ∈ 1.55 .. 2.40 step 0.05 (18 pts) — SAME range call as original LUT
# m_imag ∈ {1.45, 1.50, 1.55, 1.60}
config_B = make_config((1.55, 2.40, 18), (1.45, 1.60, 4))

n_mc = length(medium_conditions)
n_agg = length(aggregates)
n_ri_A = 4 * 30
n_ri_B = 18 * 4
n_jobs_A = n_agg * n_mc * n_ri_A
n_jobs_B = n_agg * n_mc * n_ri_B
n_jobs_total = n_jobs_A + n_jobs_B

println("Medium conditions ($n_mc): $medium_conditions")
println("Region A: m_real ∈ [2.45, 2.60] (n=4)  × m_imag ∈ [0.15, 1.60] (n=30) → $n_ri_A RI pts, $n_jobs_A jobs")
println("Region B: m_real ∈ [1.55, 2.40] (n=18) × m_imag ∈ [1.45, 1.60] (n=4)  → $n_ri_B RI pts, $n_jobs_B jobs")
println("Total jobs: $n_jobs_total")
println("Threads: $(Threads.nthreads()), use_fft: $use_fft, use_gpu: $use_gpu, gpu_float32: $gpu_float32, truncation_order: $(truncation_order === nothing ? "auto" : truncation_order)")
if !use_gpu && Threads.nthreads() == 1
    @warn "Running CPU mode with 1 thread. Use `julia -t auto` for multi-threaded parallel execution."
end
println()

# ─── Run Region A ───────────────────────────────────────────────────────────
println("═══ Region A (new m_real × all m_imag) ═══")
t0_A = time()
run_parameter_sweep(aggregates, config_A; output_h5=output_h5)
dt_A = time() - t0_A
println("Region A done in $(round(dt_A, digits=1)) s")
println()

# ─── Run Region B ───────────────────────────────────────────────────────────
println("═══ Region B (old m_real × new m_imag) ═══")
t0_B = time()
run_parameter_sweep(aggregates, config_B; output_h5=output_h5)
dt_B = time() - t0_B
println("Region B done in $(round(dt_B, digits=1)) s")
println()

println("All extension sweeps done in $(round(dt_A + dt_B, digits=1)) s")
println("Results saved to: $output_h5")
