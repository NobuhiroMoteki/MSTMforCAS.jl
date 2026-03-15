"""
Small parameter sweep example.
Run with: julia --project=. scripts/run_sweep_small.jl
For parallel: julia -t 4 --project=. scripts/run_sweep_small.jl
"""

using MSTMforCAS, Printf, DataFrames

# ─── Collect small aggregate files (Np ≤ 100) ────────────────────────────
ptsa_dir = joinpath(@__DIR__, "..", "ref", "MSTM-v4.0_for_CAS",
                    "code", "aggregate_ptsa_files")

# Filter for small files across different Df values
small_files = String[]
for f in readdir(ptsa_dir, join=true)
    bn = basename(f)
    # Extract Np from filename
    m = match(r"Np=(\d+)", bn)
    m === nothing && continue
    np = parse(Int, m[1])
    np <= 20 && push!(small_files, f)
end
sort!(small_files)

println("Found $(length(small_files)) aggregate files with Np ≤ 20")
for f in small_files
    println("  ", basename(f))
end
println()

# ─── Configure sweep ─────────────────────────────────────────────────────
config = SweepConfig(
    aggregate_files    = small_files,
    refractive_indices = [1.5+0.0im, 1.6+0.0123im, 1.7+0.1im],
    wavelength         = 0.6328,       # [μm] He-Ne laser
    medium_refindex    = 1.0,
    monomer_radius     = 1.0,          # file coords already in μm
    max_iterations     = 200,
    convergence_epsilon = 1e-6,
)

n_jobs = length(small_files) * length(config.refractive_indices)
println("Total jobs: $n_jobs ($(length(small_files)) files × $(length(config.refractive_indices)) RIs)")
println("Threads: $(Threads.nthreads())")
println()

# ─── Run sweep ────────────────────────────────────────────────────────────
println("Running sweep...")
t0 = time()
df = run_parameter_sweep(config; output_file="results_small_sweep.h5")
dt = time() - t0

println("Done in $(round(dt, digits=1)) s")
println()
println("Results saved to results_small_sweep.csv")
println("$(nrow(df)) rows × $(ncol(df)) columns")
println()

# ─── Summary ──────────────────────────────────────────────────────────────
println("Summary (first 10 rows, Q values are for unpolarized incidence):")
for row in eachrow(first(df, 10))
    @printf("  %s  m=(%.4e+%.4ei)  Np=%d  Q_ext=%.4e  Q_sca=%.4e  conv=%s  iter=%d\n",
        row.aggregate_file[1:min(40,end)], row.m_real, row.m_imag,
        row.n_monomers, row.Q_ext, row.Q_sca,
        row.converged ? "Y" : "N", row.n_iterations)
end
