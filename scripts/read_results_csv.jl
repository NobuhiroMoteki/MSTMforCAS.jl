"""
Read sweep results from CSV file.
Run with: julia --project=. scripts/read_results_csv.jl results_small_sweep.csv
"""

using CSV, DataFrames, Printf

_mean(x) = sum(x) / length(x)
_std(x)  = (m = _mean(x); sqrt(sum((xi - m)^2 for xi in x) / max(1, length(x) - 1)))

if length(ARGS) < 1
    println("Usage: julia --project=. scripts/read_results_csv.jl <results.csv>")
    exit(1)
end

csvfile = ARGS[1]
println("Reading: $csvfile")
df = CSV.read(csvfile, DataFrame)
println("$(nrow(df)) rows × $(ncol(df)) columns")
println()

# ─── Column names ───────────────────────────────────────────────────────
println("Columns: ", join(names(df), ", "))
println()

# ─── Summary table ──────────────────────────────────────────────────────
println("Summary (all rows, Q values for unpolarized incidence):")
println("  aggregate_file                           n_mono  R_ve         m_real       m_imag       Q_ext        Q_abs        Q_sca        conv  iter")
println("  " * "─"^140)
for row in eachrow(df)
    @printf("  %-40s  %4d  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e  %4s  %4d\n",
        row.aggregate_file[1:min(40,end)],
        row.n_monomers, row.R_ve, row.m_real, row.m_imag,
        row.Q_ext, row.Q_abs, row.Q_sca,
        row.converged ? "Y" : "N", row.n_iterations)
end
println()

# ─── Forward scattering amplitudes (first 5 rows) ──────────────────────
println("Forward scattering amplitudes (first 5 rows):")
for row in eachrow(first(df, 5))
    @printf("  %s  m=(%.4e+%.4ei)  R_ve=%.4e\n",
        row.aggregate_file[1:min(40,end)], row.m_real, row.m_imag, row.R_ve)
    println("    BH83 (dimensionless):")
    @printf("      S1 = %+13.6e %+13.6ei\n", row.S1_fwd_re, row.S1_fwd_im)
    @printf("      S2 = %+13.6e %+13.6ei\n", row.S2_fwd_re, row.S2_fwd_im)
    @printf("      S3 = %+13.6e %+13.6ei\n", row.S3_fwd_re, row.S3_fwd_im)
    @printf("      S4 = %+13.6e %+13.6ei\n", row.S4_fwd_re, row.S4_fwd_im)
    println("    MI02 (dimension of length):")
    @printf("      S11 = %+13.6e %+13.6ei\n", row.S11_fwd_re, row.S11_fwd_im)
    @printf("      S22 = %+13.6e %+13.6ei\n", row.S22_fwd_re, row.S22_fwd_im)
    @printf("      S12 = %+13.6e %+13.6ei\n", row.S12_fwd_re, row.S12_fwd_im)
    @printf("      S21 = %+13.6e %+13.6ei\n", row.S21_fwd_re, row.S21_fwd_im)
end
println()

# ─── Statistics by refractive index ─────────────────────────────────────
m_values = unique(zip(df.m_real, df.m_imag))
println("Unique refractive indices: $(length(collect(m_values)))")
for (mr, mi) in m_values
    sub = filter(row -> row.m_real == mr && row.m_imag == mi, df)
    @printf("  m = %.4e + %.4ei : %d rows, Q_ext = %.4e ± %.4e\n",
        mr, mi, nrow(sub), _mean(sub.Q_ext), _std(sub.Q_ext))
end
println()
