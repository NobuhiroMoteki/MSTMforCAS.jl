"""
Single aggregate scattering computation example.
Run with: julia --project=. scripts/run_single.jl [--fft] [--truncation-order N]

  --fft                : use FFT-accelerated translation (faster for N > ~100 spheres)
  --truncation-order N : override automatic VSWF truncation order (use N for all spheres)

Usage:
  compute_scattering(positions_x, radii_x, m_rel)
    - positions_x: [3 × N] matrix of dimensionless positions (= k_medium × r_physical)
    - radii_x:     [N] vector of size parameters (= k_medium × r_physical)
    - m_rel:       complex relative refractive index (m_sphere / n_medium)

  compute_scattering(agg, m_rel, k)
    - agg: AggregateGeometry from read_aggregate_file()
    - m_rel: complex relative RI
    - k: wavenumber in medium [same units⁻¹ as file coordinates]
"""

using MSTMforCAS, Printf

use_fft = "--fft" in ARGS
truncation_order = nothing
for (i, arg) in enumerate(ARGS)
    if arg == "--truncation-order" && i < length(ARGS)
        truncation_order = parse(Int, ARGS[i+1])
    end
end

# ─── Physical parameters ──────────────────────────────────────────────────
wavelength = 0.6328   # [μm] He-Ne laser
n_medium   = 1.0      # air
m_sphere   = 1.6 + 0.0123im   # absolute complex RI of monomers

# Derived quantities
k_medium = 2π * n_medium / wavelength   # wavenumber in medium [μm⁻¹]
m_rel    = m_sphere / n_medium           # relative RI

# ─── Read aggregate file (ptsa: coordinates in μm) ────────────────────────
# Small files for quick testing:
#   Np=20  → ~1 s
#   Np=50  → ~5 s
#   Np=100 → ~30 s
ptsa_dir = joinpath(@__DIR__, "..", "data", "aggregates",
                    "ptsa_files")

# Pick a small one (Np=100)
files = filter(f -> contains(f, "Np=00100"), readdir(ptsa_dir, join=true))
aggfile = first(files)

println("Aggregate file: $(basename(aggfile))")
agg = read_aggregate_file(aggfile)   # coordinates already in μm
println("N monomers: $(agg.n_monomers)")
@printf("Monomer radii range: %.4e – %.4e μm\n", minimum(agg.radii), maximum(agg.radii))
@printf("k_medium = %.4e μm⁻¹\n", k_medium)
@printf("Size parameter range: %.4e – %.4e\n", minimum(agg.radii)*k_medium, maximum(agg.radii)*k_medium)

# Volume-equivalent radius: R_ve = (Σ r_i³)^{1/3}
R_ve = cbrt(sum(r^3 for r in agg.radii))   # [μm]
@printf("Volume-equivalent radius: R_ve = %.4e μm\n", R_ve)
println()

# ─── Compute scattering ──────────────────────────────────────────────────
println("Computing (use_fft=$use_fft, truncation_order=$(truncation_order === nothing ? "auto" : truncation_order))...")
t0 = time()
result = compute_scattering(agg, m_rel, k_medium; use_fft=use_fft, truncation_order=truncation_order)
dt = time() - t0
println("Elapsed: $(round(dt, digits=2)) s")
println("Converged: $(result.converged), iterations: $(result.n_iterations), truncation_order: $(result.truncation_order)")
println()

# ─── Print results ────────────────────────────────────────────────────────
# BH83 convention (dimensionless):
#   [E_∥_sca]   exp(ikr)  [S₂  S₃] [E_∥_inc]
#   [E_⊥_sca] = -------  [S₄  S₁] [E_⊥_inc]
#                 -ikr
#
# MI02 convention (dimension of length [μm]):
#   S₁₁(MI02) = S₂(BH83)/(-ik),  S₂₂(MI02) = S₁(BH83)/(-ik)
#   S₁₂(MI02) = S₃(BH83)/(ik),   S₂₁(MI02) = S₄(BH83)/(ik)
#   where k = k_medium [μm⁻¹]
ik = im * k_medium

println("Forward scattering amplitudes (θ=0°):")
println("  BH83 (dimensionless):")
for (i, name) in enumerate(["S1", "S2", "S3", "S4"])
    s = result.S_forward[i]
    @printf("    %s = %+13.6e %+13.6ei\n", name, real(s), imag(s))
end
println("  MI02 [μm]:  S11=S2/(-ik), S22=S1/(-ik), S12=S3/(ik), S21=S4/(ik)")
S11_fwd = result.S_forward[2] / (-ik)
S22_fwd = result.S_forward[1] / (-ik)
S12_fwd = result.S_forward[3] / (ik)
S21_fwd = result.S_forward[4] / (ik)
for (name, s) in zip(["S11","S22","S12","S21"], [S11_fwd, S22_fwd, S12_fwd, S21_fwd])
    @printf("    %s = %+13.6e %+13.6ei\n", name, real(s), imag(s))
end

println()
println("Backward scattering amplitudes (θ=180°):")
println("  BH83 (dimensionless):")
for (i, name) in enumerate(["S1", "S2", "S3", "S4"])
    s = result.S_backward[i]
    @printf("    %s = %+13.6e %+13.6ei\n", name, real(s), imag(s))
end
println("  MI02 [μm]:  S11=S2/(-ik), S22=S1/(-ik), S12=S3/(ik), S21=S4/(ik)")
S11_bwd = result.S_backward[2] / (-ik)
S22_bwd = result.S_backward[1] / (-ik)
S12_bwd = result.S_backward[3] / (ik)
S21_bwd = result.S_backward[4] / (ik)
for (name, s) in zip(["S11","S22","S12","S21"], [S11_bwd, S22_bwd, S12_bwd, S21_bwd])
    @printf("    %s = %+13.6e %+13.6ei\n", name, real(s), imag(s))
end
println()
println("Efficiency factors (unpolarized incidence, normalized by π R_ve²):")
@printf("  Q_ext = %.6e\n", result.Q_ext)
@printf("  Q_abs = %.6e\n", result.Q_abs)
@printf("  Q_sca = %.6e\n", result.Q_sca)
@printf("  Energy conservation: Q_ext - Q_sca - Q_abs = %.2e\n",
    result.Q_ext - result.Q_sca - result.Q_abs)
println()

# Cross sections: C = Q × π × R_ve²  [μm²]
geo = π * R_ve^2
println("Cross sections (unpolarized incidence): C = Q × π R_ve²")
@printf("  π R_ve² = %.6e μm²\n", geo)
@printf("  C_ext = %.6e μm²\n", result.Q_ext * geo)
@printf("  C_abs = %.6e μm²\n", result.Q_abs * geo)
@printf("  C_sca = %.6e μm²\n", result.Q_sca * geo)
