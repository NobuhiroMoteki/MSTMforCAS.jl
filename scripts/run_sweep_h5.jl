"""
Parameter sweep using HDF5 + CSV catalog from aggregate_generator_PTSA.

Run with:
  julia --project=. scripts/run_sweep_h5.jl [--fft] [--truncation-order N]
  julia -t 4 --project=. scripts/run_sweep_h5.jl [--fft]

Options:
  --fft                : use FFT-accelerated translation (faster for N > ~100 spheres)
  --truncation-order N : override automatic VSWF truncation order
"""

using MSTMforCAS, Printf, DataFrames

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
h5_file  = joinpath(agg_dir, "aggregates_20260316_00.h5")
csv_file = joinpath(agg_dir, "catalog_20260316_00.csv")

# Extract YYYYMMDD_xx identifier from aggregate filename
agg_id = let m = match(r"(\d{8}_\d{2})", basename(h5_file))
    m !== nothing ? m[1] : "unknown"
end

println("Reading aggregate catalog:")
println("  H5:  $h5_file")
println("  CSV: $csv_file")
println("  Aggregate ID: $agg_id")

# Read all aggregates (or filter with Df_range, Np_range, agg_num_range)
aggregates = read_aggregate_catalog(h5_file, csv_file)

println("  Loaded $(length(aggregates)) aggregates")
println()

# ─── Print summary of aggregate parameters ───────────────────────────────────
Df_vals = sort(unique(a.Df for a in aggregates))
Np_vals = sort(unique(a.n_monomers for a in aggregates))
agg_nums = sort(unique(a.agg_num for a in aggregates))
println("  Df values:  $Df_vals")
println("  Np values:  $Np_vals")
println("  agg_num:    $agg_nums")
println()

# ─── Configure sweep ─────────────────────────────────────────────────────────
config = SweepConfig(
    medium_conditions = [(0.6328, 1.0), (0.6328, 1.33)],  # (wavelength [μm], n_medium)
    m_real_range = (1.5, 1.6, 2),    # (min, max, n_grid)
    m_imag_range = (0.0, 0.0123, 2), # (min, max, n_grid)
    max_iterations     = 200,
    convergence_epsilon = 1e-6,
    use_fft            = use_fft,
    truncation_order   = truncation_order,
)

n_mc = length(config.medium_conditions)
mr_min, mr_max, mr_n = config.m_real_range
mi_min, mi_max, mi_n = config.m_imag_range
n_ri = mr_n * mi_n
n_jobs = length(aggregates) * n_mc * n_ri
println("Medium conditions ($n_mc): $(config.medium_conditions)")
println("RI grid: m_real ∈ [$mr_min, $mr_max] (n=$mr_n) × m_imag ∈ [$mi_min, $mi_max] (n=$mi_n) → $n_ri RI values")
println("Total jobs: $n_jobs ($(length(aggregates)) aggregates × $n_mc media × $n_ri RIs)")
println("Threads: $(Threads.nthreads()), use_fft: $use_fft, truncation_order: $(truncation_order === nothing ? "auto" : truncation_order)")
println()

# ─── Output paths ────────────────────────────────────────────────────────────
results_dir = joinpath(@__DIR__, "..", "data", "results")
mkpath(results_dir)
output_csv = joinpath(results_dir, "results_fullsweep_agg$(agg_id).csv")
output_h5  = joinpath(results_dir, "results_fullsweep_agg$(agg_id).h5")

# ─── Run sweep ───────────────────────────────────────────────────────────────
println("Running sweep...")
t0 = time()
df = run_parameter_sweep(aggregates, config; output_file=output_csv)
dt = time() - t0

# Also write HDF5
write_results_hdf5(df, output_h5, config)

println("Done in $(round(dt, digits=1)) s")
println()
println("Results saved to:")
println("  $output_csv")
println("  $output_h5")
println("$(nrow(df)) rows × $(ncol(df)) columns")
println()

# ─── Summary ─────────────────────────────────────────────────────────────────
println("Summary (first 10 rows):")
for row in eachrow(first(df, 10))
    @printf("  wl=%.4f n_med=%.4f  Df=%.2f Np=%3d agg=%d  m=(%.3f+%.4fi)  Q_ext=%.4e  Q_sca=%.4e  conv=%s  iter=%d  norder=%d\n",
        row.wavelength, row.medium_refindex,
        row.Df, row.n_monomers, row.agg_num,
        row.m_real, row.m_imag,
        row.Q_ext, row.Q_sca,
        row.converged ? "Y" : "N", row.n_iterations, row.truncation_order)
end
