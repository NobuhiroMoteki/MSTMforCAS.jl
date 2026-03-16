"""
Read sweep results from HDF5 file.
Run with: julia --project=. scripts/read_results_h5.jl results.h5
"""

using HDF5, Printf

if length(ARGS) < 1
    println("Usage: julia --project=. scripts/read_results_h5.jl <results.h5>")
    exit(1)
end

h5file = ARGS[1]
println("Reading: $h5file")

h5open(h5file, "r") do fid
    # ─── Metadata ───────────────────────────────────────────────────────
    println("Metadata:")
    for key in keys(attrs(fid))
        val = attrs(fid)[key]
        @printf("  %-20s = %s\n", key, string(val))
    end
    println()

    # ─── List datasets ──────────────────────────────────────────────────
    dataset_names = keys(fid)
    println("Datasets ($(length(dataset_names))): ", join(dataset_names, ", "))

    n_rows = length(read(fid["Q_ext"]))
    println("Number of rows: $n_rows")
    println()

    # ─── Read core arrays ───────────────────────────────────────────────
    source         = read(fid["source"])
    mean_rp        = read(fid["mean_rp"])
    Df             = read(fid["Df"])
    n_monomers     = read(fid["n_monomers"])
    agg_num        = read(fid["agg_num"])
    R_ve           = read(fid["R_ve"])
    wavelength     = read(fid["wavelength"])
    medium_ri      = read(fid["medium_refindex"])
    m_real         = read(fid["m_real"])
    m_imag         = read(fid["m_imag"])
    Q_ext          = read(fid["Q_ext"])
    Q_abs          = read(fid["Q_abs"])
    Q_sca          = read(fid["Q_sca"])
    converged      = read(fid["converged"])
    n_iterations   = read(fid["n_iterations"])
    trunc_order    = read(fid["truncation_order"])

    # ─── BH83 forward scattering amplitudes ─────────────────────────────
    S1_fwd = read(fid["S1_fwd_re"]) .+ im .* read(fid["S1_fwd_im"])
    S2_fwd = read(fid["S2_fwd_re"]) .+ im .* read(fid["S2_fwd_im"])
    S3_fwd = read(fid["S3_fwd_re"]) .+ im .* read(fid["S3_fwd_im"])
    S4_fwd = read(fid["S4_fwd_re"]) .+ im .* read(fid["S4_fwd_im"])

    # ─── MI02 forward scattering amplitudes ─────────────────────────────
    S11_fwd = read(fid["S11_fwd_re"]) .+ im .* read(fid["S11_fwd_im"])
    S22_fwd = read(fid["S22_fwd_re"]) .+ im .* read(fid["S22_fwd_im"])
    S12_fwd = read(fid["S12_fwd_re"]) .+ im .* read(fid["S12_fwd_im"])
    S21_fwd = read(fid["S21_fwd_re"]) .+ im .* read(fid["S21_fwd_im"])

    # ─── Summary table ──────────────────────────────────────────────────
    println("Summary (all rows, Q values for unpolarized incidence):")
    println("  source                                   wl       n_med  mean_rp   Df    Np  agg  R_ve         m_real       m_imag       Q_ext        Q_abs        Q_sca        conv  iter  norder")
    println("  " * "─"^200)
    for i in 1:n_rows
        @printf("  %-40s  %.4f  %.4f  %.4f  %4.2f  %4d  %3d  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e  %4s  %4d  %4d\n",
            source[i][1:min(40,end)],
            wavelength[i], medium_ri[i],
            mean_rp[i], Df[i], n_monomers[i], agg_num[i],
            R_ve[i], m_real[i], m_imag[i],
            Q_ext[i], Q_abs[i], Q_sca[i],
            converged[i] ? "Y" : "N", n_iterations[i], trunc_order[i])
    end
    println()

    # ─── Forward scattering amplitudes (first 5 rows) ───────────────────
    println("Forward scattering amplitudes (first 5 rows):")
    for i in 1:min(5, n_rows)
        @printf("  %s  m=(%.4e+%.4ei)  Np=%d  Df=%.2f  R_ve=%.4e\n",
            source[i][1:min(40,end)], m_real[i], m_imag[i],
            n_monomers[i], Df[i], R_ve[i])
        println("    BH83 (dimensionless):")
        @printf("      S1 = %+13.6e %+13.6ei\n", real(S1_fwd[i]), imag(S1_fwd[i]))
        @printf("      S2 = %+13.6e %+13.6ei\n", real(S2_fwd[i]), imag(S2_fwd[i]))
        @printf("      S3 = %+13.6e %+13.6ei\n", real(S3_fwd[i]), imag(S3_fwd[i]))
        @printf("      S4 = %+13.6e %+13.6ei\n", real(S4_fwd[i]), imag(S4_fwd[i]))
        println("    MI02 (dimension of length):")
        @printf("      S11 = %+13.6e %+13.6ei\n", real(S11_fwd[i]), imag(S11_fwd[i]))
        @printf("      S22 = %+13.6e %+13.6ei\n", real(S22_fwd[i]), imag(S22_fwd[i]))
        @printf("      S12 = %+13.6e %+13.6ei\n", real(S12_fwd[i]), imag(S12_fwd[i]))
        @printf("      S21 = %+13.6e %+13.6ei\n", real(S21_fwd[i]), imag(S21_fwd[i]))
    end
    println()

    # ─── Energy conservation check ──────────────────────────────────────
    conservation = Q_ext .- Q_sca .- Q_abs
    max_err = maximum(abs.(conservation))
    @printf("Energy conservation: max|Q_ext - Q_sca - Q_abs| = %.2e\n", max_err)
end
