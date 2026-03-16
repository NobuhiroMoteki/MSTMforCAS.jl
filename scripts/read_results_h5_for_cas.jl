"""
Read CAS-relevant quantities from HDF5 sweep results by specifying array indices
for the sweep parameters (medium, agg_num, Np, Df, m_real, m_imag).

Run with:
  julia --project=. scripts/read_results_h5_for_cas.jl <results.h5> [i_medium] [i_agg] [i_Np] [i_Df] [i_m_real] [i_m_imag]

Arguments:
  results.h5  : path to HDF5 results file
  i_medium    : 1-based index into unique (wavelength, medium_refindex) pairs (default: 1)
  i_agg       : 1-based index into unique agg_num values (default: 1)
  i_Np        : 1-based index into unique Np values (default: 1)
  i_Df        : 1-based index into unique Df values (default: 1)
  i_m_real    : 1-based index into unique m_real values (default: 1)
  i_m_imag    : 1-based index into unique m_imag values (default: 1)

Example:
  julia --project=. scripts/read_results_h5_for_cas.jl data/results/results_fullsweep_agg20260316_00.h5 1 1 2 1 1 1
"""

using HDF5, Printf

if length(ARGS) < 1
    println("Usage: julia --project=. scripts/read_results_h5_for_cas.jl <results.h5> [i_medium] [i_agg] [i_Np] [i_Df] [i_m_real] [i_m_imag]")
    exit(1)
end

h5file   = ARGS[1]
i_medium = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
i_agg    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
i_Np     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
i_Df     = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
i_m_real = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1
i_m_imag = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 1

println("Reading: $h5file")
println()

h5open(h5file, "r") do fid
    # ─── Read parameter arrays ───────────────────────────────────────────
    wavelength      = read(fid["wavelength"])
    medium_refindex = read(fid["medium_refindex"])
    agg_num    = read(fid["agg_num"])
    n_monomers = read(fid["n_monomers"])
    Df         = read(fid["Df"])
    m_real     = read(fid["m_real"])
    m_imag     = read(fid["m_imag"])

    # ─── Build unique sorted values for each parameter ───────────────────
    # Medium conditions: unique (wavelength, medium_refindex) pairs
    medium_pairs = sort(unique(collect(zip(wavelength, medium_refindex))))
    u_agg    = sort(unique(agg_num))
    u_Np     = sort(unique(n_monomers))
    u_Df     = sort(unique(Df))
    u_m_real = sort(unique(m_real))
    u_m_imag = sort(unique(m_imag))

    println("Available parameter values:")
    println("  medium   ($(length(medium_pairs))): $medium_pairs  # (wavelength, n_medium)")
    println("  agg_num  ($(length(u_agg))): $u_agg")
    println("  Np       ($(length(u_Np))): $u_Np")
    println("  Df       ($(length(u_Df))): $u_Df")
    println("  m_real   ($(length(u_m_real))): $u_m_real")
    println("  m_imag   ($(length(u_m_imag))): $u_m_imag")
    println()

    # ─── Resolve indices to values ───────────────────────────────────────
    val_wl, val_nmed = medium_pairs[i_medium]
    val_agg    = u_agg[i_agg]
    val_Np     = u_Np[i_Np]
    val_Df     = u_Df[i_Df]
    val_m_real = u_m_real[i_m_real]
    val_m_imag = u_m_imag[i_m_imag]

    @printf("Selected: medium=(%.4f μm, n=%.4f), agg_num=%d, Np=%d, Df=%.2f, m=%.4f+%.4fi\n",
            val_wl, val_nmed, val_agg, val_Np, val_Df, val_m_real, val_m_imag)
    println()

    # ─── Find matching row ───────────────────────────────────────────────
    idx = findall(i -> wavelength[i] == val_wl &&
                       medium_refindex[i] == val_nmed &&
                       agg_num[i] == val_agg &&
                       n_monomers[i] == val_Np &&
                       Df[i] == val_Df &&
                       m_real[i] == val_m_real &&
                       m_imag[i] == val_m_imag,
                  eachindex(agg_num))

    if isempty(idx)
        println("ERROR: No matching row found for the specified parameter combination.")
        return
    end

    # ─── Read result arrays ──────────────────────────────────────────────
    R_ve  = read(fid["R_ve"])
    Q_ext = read(fid["Q_ext"])
    Q_abs = read(fid["Q_abs"])
    Q_sca = read(fid["Q_sca"])

    S11_fwd = read(fid["S11_fwd_re"]) .+ im .* read(fid["S11_fwd_im"])
    S12_fwd = read(fid["S12_fwd_re"]) .+ im .* read(fid["S12_fwd_im"])
    S21_fwd = read(fid["S21_fwd_re"]) .+ im .* read(fid["S21_fwd_im"])
    S22_fwd = read(fid["S22_fwd_re"]) .+ im .* read(fid["S22_fwd_im"])

    S11_bwd = read(fid["S11_bwd_re"]) .+ im .* read(fid["S11_bwd_im"])
    S21_bwd = read(fid["S21_bwd_re"]) .+ im .* read(fid["S21_bwd_im"])
    S22_bwd = read(fid["S22_bwd_re"]) .+ im .* read(fid["S22_bwd_im"])

    # ─── CAS-v2 observables ────────────────────────────────────────────
    # Ss_fwd: s-polarization forward complex scattering amplitude
    # Sp_fwd: p-polarization forward complex scattering amplitude
    # S_bak:  backward scattering amplitude (depolarization-sensitive)
    Ss_fwd = S11_fwd .+ S12_fwd .* im
    Sp_fwd = S22_fwd .- S21_fwd .* im
    S_bak  = (-S11_bwd .+ S22_bwd) ./ sqrt(2)

    # ─── Output ──────────────────────────────────────────────────────────
    for i in idx
        println("─── Result ───")
        @printf("  wavelength       = %.6f μm\n", val_wl)
        @printf("  medium_refindex  = %.4f\n", val_nmed)
        @printf("  agg_num          = %d\n", val_agg)
        @printf("  Np               = %d\n", val_Np)
        @printf("  Df               = %.2f\n", val_Df)
        @printf("  m                = %.4f + %.4fi\n", val_m_real, val_m_imag)
        @printf("  R_ve             = %.6e μm\n", R_ve[i])
        println()
        @printf("  Q_ext            = %.6e\n", Q_ext[i])
        @printf("  Q_abs            = %.6e\n", Q_abs[i])
        @printf("  Q_sca            = %.6e\n", Q_sca[i])
        println()
        println("  CAS-v2 complex scattering amplitudes [μm]:")
        @printf("    Ss_fwd = S11+S12*i = %+13.6e %+13.6ei\n", real(Ss_fwd[i]), imag(Ss_fwd[i]))
        @printf("    Sp_fwd = S22-S21*i = %+13.6e %+13.6ei\n", real(Sp_fwd[i]), imag(Sp_fwd[i]))
        @printf("    S_bak  = (-S11+S22)/√2 = %+13.6e %+13.6ei\n", real(S_bak[i]), imag(S_bak[i]))
    end
end
