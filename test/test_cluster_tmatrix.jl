# Validation for the cluster T-matrix (design phase P2).
# Requires the full package (solver internals); run on a machine where
# MSTMforCAS precompiles in reasonable time:
#   julia --project=. test/test_cluster_tmatrix.jl
#
# Checks:
#  (1) Ω=identity: cluster-T forward BH83 amplitudes reproduce the direct
#      fixed-z compute_scattering result.
#  (2) rotate-vs-resolve: cluster-T + Wigner-D at several Ω matches rotating the
#      geometry by the same Ω and re-solving directly.

using Test
using LinearAlgebra
using MSTMforCAS
using MSTMforCAS: build_cluster_tmatrix, cluster_forward_amplitudes

# active intrinsic ZYZ rotation matrix (particle frame), matches wigner convention
Rz(a) = [cos(a) -sin(a) 0.0; sin(a) cos(a) 0.0; 0.0 0.0 1.0]
Ry(b) = [cos(b) 0.0 sin(b); 0.0 1.0 0.0; -sin(b) 0.0 cos(b)]
Rzyz(a, b, g) = Rz(a) * Ry(b) * Rz(g)

# a small non-symmetric aggregate (few small spheres) → small nodrt, light solves
positions = [ 0.0  0.35 -0.20;
              0.0  0.10  0.40;
             -0.55 0.30  0.50 ]          # (3, 3)
radii = [0.45, 0.35, 0.40]
m_rel = ComplexF64(1.5, 0.02)

@testset "ClusterTMatrix" begin
    # reference: direct fixed-z solve
    res0, _ = compute_scattering(positions, radii, m_rel; tol=1e-9)
    ref_fwd = res0.S_forward

    T_cl, nodrt, r0, conv = build_cluster_tmatrix(positions, radii, m_rel; tol=1e-9)
    @test conv

    @testset "Ω = identity reproduces direct z-solve" begin
        S = cluster_forward_amplitudes(T_cl, nodrt, r0, 0.0, 0.0, 0.0)
        for i in 1:4
            # agreement to solver precision (~1e-6 abs on O(0.06) amplitudes);
            # cluster-T assembles from 2·nodrt(nodrt+2) mode solves vs the direct 2-RHS solve
            @test S[i] ≈ ref_fwd[i] atol=1e-5 rtol=1e-5
        end
    end

    @testset "rotate-vs-resolve" begin
        for (α, β, γ) in [(0.0, 0.6, 0.0), (0.7, 1.1, 0.0), (1.3, 0.8, 2.0)]
            # cluster-T + Wigner-D
            Scl = cluster_forward_amplitudes(T_cl, nodrt, r0, α, β, γ)
            # direct: rotate the geometry about its centroid (so the centroid — and
            # thus the plane-wave phase reference — is unchanged), re-solve at fixed z
            Rp = Rzyz(α, β, γ)
            pos_rot = Rp * (positions .- r0) .+ r0
            res, _ = compute_scattering(pos_rot, radii, m_rel; tol=1e-9)
            Sdir = res.S_forward
            for i in 1:4
                @test Scl[i] ≈ Sdir[i] atol=1e-5 rtol=1e-5
            end
        end
    end
end
