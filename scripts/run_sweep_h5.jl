"""
Parameter sweep using HDF5 + CSV catalog from aggregate_generator_PTSA.
Results are written incrementally to HDF5 — safe to interrupt and resume.

Run with:
  julia --project=. scripts/run_sweep_h5.jl [--fft] [--truncation-order N]
  julia -t auto --project=. scripts/run_sweep_h5.jl [--fft]

Options:
  --fft                : use FFT-accelerated translation (faster for N > ~100 spheres)
  --truncation-order N : override automatic VSWF truncation order
"""

using MSTMforCAS, Printf

use_fft = "--fft" in ARGS
truncation_order = let to = nothing
    for (i, arg) in enumerate(ARGS)
        if arg == "--truncation-order" && i < length(ARGS)
            to = parse(Int, ARGS[i+1])
        end
    end
    to
end

# ─── Aggregate data files ────────────────────────────────────────────────────
# HDF5 + CSV catalog produced by aggregate_generator_PTSA
agg_dir  = joinpath(@__DIR__, "..", "data", "aggregates")
h5_file  = joinpath(agg_dir, "aggregates_20260323_00.h5")
csv_file = joinpath(agg_dir, "catalog_20260323_00.csv")

# Extract YYYYMMDD_xx identifier from aggregate filename
agg_id = let m = match(r"(\d{8}_\d{2})", basename(h5_file))
    m !== nothing ? m[1] : "unknown"
end

println("Reading aggregate catalog:")
println("  H5:  $h5_file")
println("  CSV: $csv_file")
println("  Aggregate ID: $agg_id")

# agg_num range for this machine (0-3 here; 4-5 on other machine)
agg_num_lo, agg_num_hi = 0, 3
aggregates = read_aggregate_catalog(h5_file, csv_file; agg_num_range=(agg_num_lo, agg_num_hi))

println("  Loaded $(length(aggregates)) aggregates")
println()

# ─── Print summary of aggregate parameters ───────────────────────────────────
Df_vals  = sort(unique(a.Df for a in aggregates))
Np_vals  = sort(unique(a.n_monomers for a in aggregates))
agg_nums = sort(unique(a.agg_num for a in aggregates))
println("  Df values:  $Df_vals")
println("  Np values:  $Np_vals")
println("  agg_num:    $agg_nums")
println()

# ─── Configure sweep ─────────────────────────────────────────────────────────
config = SweepConfig(
    medium_conditions = [(0.453, 1.0), (0.638, 1.0), (0.834, 1.0)],  # (wavelength [μm], n_medium)
    m_real_range = (1.55, 2.4, 18),    # (min, max, n_grid)
    m_imag_range = (0.15, 1.4, 26),    # (min, max, n_grid)
    max_iterations     = 200,
    convergence_epsilon = 1e-6,
    use_fft            = use_fft,
    truncation_order   = truncation_order,
)

n_mc = length(config.medium_conditions)
mr_min, mr_max, mr_n = config.m_real_range
mi_min, mi_max, mi_n = config.m_imag_range
n_ri  = mr_n * mi_n
n_jobs = length(aggregates) * n_mc * n_ri
println("Medium conditions ($n_mc): $(config.medium_conditions)")
println("RI grid: m_real ∈ [$mr_min, $mr_max] (n=$mr_n) × m_imag ∈ [$mi_min, $mi_max] (n=$mi_n) → $n_ri RI values")
println("Total jobs: $n_jobs ($(length(aggregates)) aggregates × $n_mc media × $n_ri RIs)")
println("Threads: $(Threads.nthreads()), use_fft: $use_fft, truncation_order: $(truncation_order === nothing ? "auto" : truncation_order)")
println()

# ─── Output path ─────────────────────────────────────────────────────────────
results_dir = joinpath(@__DIR__, "..", "data", "results")
mkpath(results_dir)
output_h5 = joinpath(results_dir, "results_fullsweep_agg$(agg_id)_num$(agg_num_lo)-$(agg_num_hi).h5")

println("Output: $output_h5")
if isfile(output_h5)
    println("  (existing file found — will resume from where left off)")
end
println()

# ─── Run sweep ───────────────────────────────────────────────────────────────
println("Running sweep...")
t0 = time()
run_parameter_sweep(aggregates, config; output_h5=output_h5)
dt = time() - t0

println("Done in $(round(dt, digits=1)) s")
println("Results saved to: $output_h5")
