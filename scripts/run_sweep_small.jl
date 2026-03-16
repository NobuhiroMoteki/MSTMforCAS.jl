"""
Small parameter sweep example using .ptsa files.
Run with: julia --project=. scripts/run_sweep_small.jl [--fft] [--truncation-order N]
For parallel: julia -t 4 --project=. scripts/run_sweep_small.jl [--fft]

  --fft                : use FFT-accelerated translation (faster for N > ~100 spheres)
  --truncation-order N : override automatic VSWF truncation order (use N for all spheres)
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

# ─── Collect small aggregate files (Np ≤ 100) ────────────────────────────
ptsa_dir = joinpath(@__DIR__, "..", "data", "aggregates",
                    "ptsa_files")

# Filter for small files across different Df values
small_files = String[]
for f in readdir(ptsa_dir, join=true)
    bn = basename(f)
    m = match(r"Np=(\d+)", bn)
    m === nothing && continue
    np = parse(Int, m[1])
    np <= 100 && push!(small_files, f)
end
sort!(small_files)

println("Found $(length(small_files)) aggregate files with Np ≤ 20")
for f in small_files
    println("  ", basename(f))
end
println()

# ─── Read aggregates ─────────────────────────────────────────────────────
aggregates = [read_aggregate_file(f) for f in small_files]

# ─── Configure sweep ─────────────────────────────────────────────────────
config = SweepConfig(
    medium_conditions = [(0.6328, 1.0)],   # (wavelength [μm], n_medium)
    m_real_range = (1.5, 1.7, 3),          # (min, max, n_grid)
    m_imag_range = (0.0, 0.1, 3),          # (min, max, n_grid)
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
println("Total jobs: $n_jobs ($(length(aggregates)) files × $n_mc media × $n_ri RIs)")
println("Threads: $(Threads.nthreads()), use_fft: $use_fft, truncation_order: $(truncation_order === nothing ? "auto" : truncation_order)")
println()

# ─── Output paths ─────────────────────────────────────────────────────────
results_dir = joinpath(@__DIR__, "..", "data", "results")
mkpath(results_dir)
output_csv = joinpath(results_dir, "results_ptsa_small.csv")

# ─── Run sweep ────────────────────────────────────────────────────────────
println("Running sweep...")
t0 = time()
df = run_parameter_sweep(aggregates, config; output_file=output_csv)
dt = time() - t0

println("Done in $(round(dt, digits=1)) s")
println()
println("Results saved to: $output_csv")
println("$(nrow(df)) rows × $(ncol(df)) columns")
println()

# ─── Summary ──────────────────────────────────────────────────────────────
println("Summary (first 10 rows, Q values are for unpolarized incidence):")
for row in eachrow(first(df, 10))
    @printf("  %s  m=(%.4e+%.4ei)  Np=%d  Q_ext=%.4e  Q_sca=%.4e  conv=%s  iter=%d  norder=%d\n",
        row.source[1:min(40,end)], row.m_real, row.m_imag,
        row.n_monomers, row.Q_ext, row.Q_sca,
        row.converged ? "Y" : "N", row.n_iterations, row.truncation_order)
end
