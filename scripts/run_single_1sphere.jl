"""
Single sphere (Lorenz-Mie) scattering computation.
Run with: julia --project=. scripts/run_single_1sphere.jl

This is the N=1 special case of compute_scattering.
For a single sphere: S1(fwd)=S2(fwd), S3=S4=0, Q values match Lorenz-Mie theory.
"""

using MSTMforCAS, Printf

# ─── Physical parameters ──────────────────────────────────────────────────
wavelength = 0.6328   # [μm] He-Ne laser
n_medium   = 1.33154      # air
m_sphere   = 1.457 + 0.0im   # absolute complex RI
r_sphere   = 0.73/2    # [μm] sphere radius

# Derived quantities
k_medium = 2π * n_medium / wavelength   # wavenumber in medium [μm⁻¹]
m_rel    = m_sphere / n_medium           # relative RI
x        = k_medium * r_sphere           # size parameter

println("Single sphere Lorenz-Mie scattering")
println("────────────────────────────────────")
@printf("Wavelength:   %.4e μm\n", wavelength)
@printf("n_medium:     %.4e\n", n_medium)
@printf("m_sphere:     %.4e + %.4ei\n", real(m_sphere), imag(m_sphere))
@printf("m_rel:        %.4e + %.4ei\n", real(m_rel), imag(m_rel))
@printf("Radius:       %.4e μm\n", r_sphere)
@printf("k_medium:     %.4e μm⁻¹\n", k_medium)
@printf("Size param x: %.4e\n", x)
println()

# ─── Compute scattering (single sphere at origin) ────────────────────────
positions = zeros(3, 1)             # [3 × 1], sphere at origin
radii_x   = [x]                    # size parameter
println("Computing...")
t0 = time()
result = compute_scattering(positions, radii_x, m_rel)
dt = time() - t0
println("Elapsed: $(round(dt, digits=2)) s")
println("Converged: $(result.converged), iterations: $(result.n_iterations)")
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
println("Symmetry checks (single sphere):")
@printf("  |S1_fwd - S2_fwd| = %.2e  (should be 0)\n",
    abs(result.S_forward[1] - result.S_forward[2]))
@printf("  |S3_fwd|          = %.2e  (should be 0)\n", abs(result.S_forward[3]))
@printf("  |S4_fwd|          = %.2e  (should be 0)\n", abs(result.S_forward[4]))
@printf("  |S1_bwd + S2_bwd| = %.2e  (should be 0)\n",
    abs(result.S_backward[1] + result.S_backward[2]))
@printf("  |S3_bwd|          = %.2e  (should be 0)\n", abs(result.S_backward[3]))
@printf("  |S4_bwd|          = %.2e  (should be 0)\n", abs(result.S_backward[4]))

println()
R_ve = r_sphere   # for single sphere, R_ve = r_sphere
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
