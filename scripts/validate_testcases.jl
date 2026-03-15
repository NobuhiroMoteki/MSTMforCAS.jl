"""
Validate Julia MSTM against Fortran reference outputs for testcase1 and testcase2.
Run with: julia --project=. scripts/validate_testcases.jl
"""

using MSTMforCAS, Printf, DelimitedFiles

# ─── Testcase 1: 2-sphere dimer ────────────────────────────────────────────
println("=" ^ 60)
println("TESTCASE 1: 2-sphere dimer")
println("=" ^ 60)

# sphere1 at (0,0,0) r=0.2, sphere2 at (0.4,0,0) r=0.2
# length_scale_factor = 7.534  → positions/radii already scaled
positions_1 = [0.0  0.4*7.534;   # x
               0.0  0.0;          # y
               0.0  0.0]          # z
radii_1 = [0.2*7.534, 0.2*7.534]
m_rel_1  = ComplexF64(1.58, 0.05)

t0 = time()
res1 = compute_scattering(positions_1, radii_1, m_rel_1)
dt1 = time() - t0

println("Elapsed: $(round(dt1, digits=2)) s, converged=$(res1.converged), iters=$(res1.n_iterations)")
println()
println("  Quantity    Julia            Fortran          %err")
println("  ────────────────────────────────────────────────────")

function pct(julia, fortran)
    abs_err = abs(julia - fortran)
    mag = max(abs(fortran), 1e-15)
    return abs_err / mag * 100
end

# Fortran reference
ref_S1f = ComplexF64(1.147, -2.425)
ref_S2f = ComplexF64(1.504, -2.258)
ref_S1b = ComplexF64(0.0819, -0.1902)
ref_S2b = ComplexF64(-0.4150, -0.1322)
ref_Qext = 1.4708
ref_Qabs = 0.3185
ref_Qsca = 1.1523

@printf("  S1_fwd  = %10.4f%+10.4fi  %10.4f%+10.4fi  %.3f%%\n",
    real(res1.S_forward[1]), imag(res1.S_forward[1]),
    real(ref_S1f), imag(ref_S1f), pct(res1.S_forward[1], ref_S1f))
@printf("  S2_fwd  = %10.4f%+10.4fi  %10.4f%+10.4fi  %.3f%%\n",
    real(res1.S_forward[2]), imag(res1.S_forward[2]),
    real(ref_S2f), imag(ref_S2f), pct(res1.S_forward[2], ref_S2f))
@printf("  S1_bwd  = %10.4f%+10.4fi  %10.4f%+10.4fi  %.3f%%\n",
    real(res1.S_backward[1]), imag(res1.S_backward[1]),
    real(ref_S1b), imag(ref_S1b), pct(res1.S_backward[1], ref_S1b))
@printf("  S2_bwd  = %10.4f%+10.4fi  %10.4f%+10.4fi  %.3f%%\n",
    real(res1.S_backward[2]), imag(res1.S_backward[2]),
    real(ref_S2b), imag(ref_S2b), pct(res1.S_backward[2], ref_S2b))
@printf("  Q_ext   = %10.6f           %10.6f           %.3f%%\n",
    res1.Q_ext, ref_Qext, pct(res1.Q_ext, ref_Qext))
@printf("  Q_abs   = %10.6f           %10.6f           %.3f%%\n",
    res1.Q_abs, ref_Qabs, pct(res1.Q_abs, ref_Qabs))
@printf("  Q_sca   = %10.6f           %10.6f           %.3f%%\n",
    res1.Q_sca, ref_Qsca, pct(res1.Q_sca, ref_Qsca))

# ─── Testcase 2: 1000-sphere aggregate ─────────────────────────────────────
println()
println("=" ^ 60)
println("TESTCASE 2: 1000-sphere aggregate")
println("=" ^ 60)

# Read geometry: length_scale_factor = 1.0, all r=1.0
pos_file = joinpath(@__DIR__, "..", "ref", "MSTM-v4.0_for_CAS", "code", "random_in_cylinder_1000.pos")
data = readdlm(pos_file)   # 1000 × 4 matrix: x y z r
N2 = size(data, 1)
positions_2 = Matrix{Float64}(data[:, 1:3]')   # 3 × N
radii_2 = Vector{Float64}(data[:, 4])           # N
m_rel_2  = ComplexF64(1.6, 0.0123)

println("N spheres = $N2, x per sphere = $(radii_2[1])")

t0 = time()
res2 = compute_scattering(positions_2, radii_2, m_rel_2)
dt2 = time() - t0

println("Elapsed: $(round(dt2, digits=1)) s, converged=$(res2.converged), iters=$(res2.n_iterations)")
println()
println("  Quantity    Julia             Fortran           %err")
println("  ─────────────────────────────────────────────────────")

ref2_S1f = ComplexF64(261.5, -181.5)
ref2_S2f = ComplexF64(261.9, -181.8)
ref2_S3f = ComplexF64(-0.6731, -1.653)
ref2_S4f = ComplexF64(0.3981, -1.356)
ref2_S1b = ComplexF64(2.566, -7.266)
ref2_S2b = ComplexF64(-1.419, 10.34)
ref2_Qext = 11.190
ref2_Qabs = 0.3652
ref2_Qsca = 10.825

@printf("  S1_fwd  = %10.2f%+10.2fi  %10.2f%+10.2fi  %.3f%%\n",
    real(res2.S_forward[1]), imag(res2.S_forward[1]),
    real(ref2_S1f), imag(ref2_S1f), pct(res2.S_forward[1], ref2_S1f))
@printf("  S2_fwd  = %10.2f%+10.2fi  %10.2f%+10.2fi  %.3f%%\n",
    real(res2.S_forward[2]), imag(res2.S_forward[2]),
    real(ref2_S2f), imag(ref2_S2f), pct(res2.S_forward[2], ref2_S2f))
@printf("  S3_fwd  = %10.4f%+10.4fi  %10.4f%+10.4fi  %.3f%%\n",
    real(res2.S_forward[3]), imag(res2.S_forward[3]),
    real(ref2_S3f), imag(ref2_S3f), pct(res2.S_forward[3], ref2_S3f))
@printf("  S4_fwd  = %10.4f%+10.4fi  %10.4f%+10.4fi  %.3f%%\n",
    real(res2.S_forward[4]), imag(res2.S_forward[4]),
    real(ref2_S4f), imag(ref2_S4f), pct(res2.S_forward[4], ref2_S4f))
@printf("  S1_bwd  = %10.4f%+10.4fi  %10.4f%+10.4fi  %.3f%%\n",
    real(res2.S_backward[1]), imag(res2.S_backward[1]),
    real(ref2_S1b), imag(ref2_S1b), pct(res2.S_backward[1], ref2_S1b))
@printf("  S2_bwd  = %10.4f%+10.4fi  %10.4f%+10.4fi  %.3f%%\n",
    real(res2.S_backward[2]), imag(res2.S_backward[2]),
    real(ref2_S2b), imag(ref2_S2b), pct(res2.S_backward[2], ref2_S2b))
@printf("  Q_ext   = %10.4f            %10.4f            %.3f%%\n",
    res2.Q_ext, ref2_Qext, pct(res2.Q_ext, ref2_Qext))
@printf("  Q_abs   = %10.4f            %10.4f            %.3f%%\n",
    res2.Q_abs, ref2_Qabs, pct(res2.Q_abs, ref2_Qabs))
@printf("  Q_sca   = %10.4f            %10.4f            %.3f%%\n",
    res2.Q_sca, ref2_Qsca, pct(res2.Q_sca, ref2_Qsca))
