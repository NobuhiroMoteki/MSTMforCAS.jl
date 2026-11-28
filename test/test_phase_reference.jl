# Phase reference of the amplitudes (version 1.3.0).
#
# :particle (default) refers the incident and the scattered wave to the unweighted mean of the
# sphere centres, so translating the input leaves every output unchanged. :lab phases the
# incident wave at z = 0, the convention of MSTM-v4.0_for_CAS and of all versions before 1.3.0. The two
# differ by the factor exp(i z_r0), z_r0 = k_medium × z of the mean centre.

using LinearAlgebra: norm

@testset "Phase reference" begin
    # Off-centre aggregate of six unequal spheres, dimensionless (k_medium × length)
    pos = [0.0  1.0  0.3  -0.6  0.5  1.4;
           0.0  0.2  1.1   0.4 -0.7  0.9;
           0.0  0.1 -0.4   0.8  0.6  0.2] .+ [0.4, -0.3, 0.9]   # shape: (3, 6)
    rad = [0.5, 0.45, 0.4, 0.42, 0.38, 0.47]                     # shape: (6,)
    m = ComplexF64(1.95, 0.79)
    z0 = sum(pos[3, :]) / size(pos, 2)
    x2 = sum(rad .^ 3)^(2 / 3)                 # square of the volume-equivalent size parameter
    vecS(r) = [r.S_forward..., r.S_backward...]
    vecQ(r) = [r.Q_ext, r.Q_abs, r.Q_sca]

    @testset "direct translation: outputs do not depend on the origin" begin
        r1, _ = compute_scattering(pos, rad, m; tol = 1e-12)
        for d in ([0.0, 0.0, 0.7], [0.9, -0.4, 0.0], [-1.3, 2.1, -3.7])
            r2, _ = compute_scattering(pos .+ d, rad, m; tol = 1e-12)
            @test maximum(abs.(vecS(r2) .- vecS(r1))) <= 1e-9 * maximum(abs.(vecS(r1)))
            @test vecQ(r2) ≈ vecQ(r1) rtol = 1e-10
        end
    end

    @testset ":lab equals :particle times exp(i z_r0)" begin
        ph = cis(z0)
        for use_fft in (false, true)
            rp, ap = compute_scattering(pos, rad, m; use_fft = use_fft)
            rl, al = compute_scattering(pos, rad, m; use_fft = use_fft, phase_reference = :lab)
            @test vecS(rl) ≈ ph .* vecS(rp) rtol = 1e-13
            @test vecQ(rl) == vecQ(rp)
            @test al == ap                     # the solver coefficients do not depend on it
        end
        ms = ComplexF64[1.8 + 0.4im, 2.0 + 0.6im, 2.2 + 0.8im, 1.9 + 0.5im]
        for kw in ((;), (; max_basis = 2))     # basis rows, and rows solved by the fallback
            rp, ip = sweep_refractive_index(pos, rad, ms; warn_range = false, kw...)
            rl, il = sweep_refractive_index(pos, rad, ms; warn_range = false, phase_reference = :lab, kw...)
            @test ip.n_fallback == il.n_fallback
            for j in eachindex(ms)
                @test vecS(rl[j]) ≈ ph .* vecS(rp[j]) rtol = 1e-13
                @test vecQ(rl[j]) == vecQ(rp[j])
            end
        end
        @test sweep_refractive_index(pos, rad, ms; warn_range = false, max_basis = 2)[2].n_fallback > 0
    end

    @testset "optical theorem holds with :particle" begin
        # order 8: at the Mie order (3) the expansion about the mean centre leaves ~2e-6
        for pr in (:particle, :lab)
            r, _ = compute_scattering(pos, rad, m; tol = 1e-12, truncation_order = 8,
                                      phase_reference = pr)
            s0 = (r.S_forward[1] + r.S_forward[2]) / 2
            if pr === :particle
                @test 4 / x2 * real(s0) ≈ r.Q_ext rtol = 1e-9
            else
                @test !isapprox(4 / x2 * real(s0), r.Q_ext; rtol = 1e-2)   # z_r0 = 1.12 here
            end
        end
    end

    @testset "single sphere off the origin gives the Mie amplitudes" begin
        r0, _ = compute_scattering(zeros(3, 1), [1.2], m)
        r1, _ = compute_scattering(reshape([0.3, -0.8, 2.5], 3, 1), [1.2], m)
        @test vecS(r1) ≈ vecS(r0) rtol = 1e-13
        rl, _ = compute_scattering(reshape([0.3, -0.8, 2.5], 3, 1), [1.2], m; phase_reference = :lab)
        @test vecS(rl) ≈ cis(2.5) .* vecS(r0) rtol = 1e-13
    end

    @testset "cluster T-matrix matches any centre of rotation" begin
        T, nodrt, r0, conv = build_cluster_tmatrix(pos, rad, m; tol = 1e-10)
        @test conv
        rotz(a) = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1]
        roty(b) = [cos(b) 0 sin(b); 0 1 0; -sin(b) 0 cos(b)]
        for (α, β, γ) in [(0.0, 0.0, 0.0), (0.7, 1.1, 0.0), (1.3, 0.8, 2.0)]
            S = collect(cluster_forward_amplitudes(T, nodrt, r0, α, β, γ))
            R = rotz(α) * roty(β) * rotz(γ)
            for c in (zeros(3), r0, [2.0, -1.0, 3.0])        # rotate about different points
                rd, _ = compute_scattering(R * (pos .- c) .+ c, rad, m; tol = 1e-10)
                @test norm(S .- collect(rd.S_forward)) <= 1e-5 * norm(S)
            end
            # :lab keeps the behaviour before 1.3.0: phase of the unrotated mean centre
            Sl = collect(cluster_forward_amplitudes(T, nodrt, r0, α, β, γ; phase_reference = :lab))
            @test Sl ≈ cis(r0[3]) .* S rtol = 1e-12
        end
    end

    @testset "unknown values are refused" begin
        @test_throws ArgumentError compute_scattering(pos, rad, m; phase_reference = :origin)
        @test_throws ArgumentError sweep_refractive_index(pos, rad, [m, m + 0.1, m + 0.2];
                                                          phase_reference = :origin)
        T, nodrt, r0, _ = build_cluster_tmatrix(pos[:, 1:2], rad[1:2], m)
        @test_throws ArgumentError cluster_forward_amplitudes(T, nodrt, r0; phase_reference = :origin)
    end

    @testset "sweep files record the convention and refuse to mix" begin
        HDF5 = MSTMforCAS.HDF5
        r = 0.015                                         # monomer radius [um]
        agg = AggregateGeometry([0.0 2r 2r; 0.0 0.0 2r; 0.03 0.03 0.03], fill(r, 3), 3, "trimer_z")
        wl = 0.834; k = 2pi / wl
        base = (medium_conditions = [(wl, 1.0)],
                m_real_range = (1.6, 2.2, 2), m_imag_range = (0.2, 1.0, 2),
                convergence_epsilon = 1e-20)
        s1(path) = HDF5.h5open(f -> complex.(read(f["S1_fwd_re"]), read(f["S1_fwd_im"])), path, "r")
        mktempdir() do dir
            pp = joinpath(dir, "particle.h5"); pl = joinpath(dir, "lab.h5")
            run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base...); output_h5 = pp)
            run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base..., phase_reference = :lab); output_h5 = pl)
            @test MSTMforCAS._read_phase_reference(pp) === :particle
            @test MSTMforCAS._read_phase_reference(pl) === :lab
            zr0 = k * sum(agg.positions[3, :]) / 3
            @test s1(pl) ≈ cis(zr0) .* s1(pp) rtol = 1e-12
            # resuming with the other convention is refused
            @test_throws ErrorException run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base...);
                                                            output_h5 = pl)
            # a file without the attribute (versions before 1.3.0) counts as :lab
            HDF5.h5open(f -> HDF5.delete_attribute(f, "phase_reference"), pl, "r+")
            @test MSTMforCAS._read_phase_reference(pl) === :lab
            @test_throws ErrorException run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base...);
                                                            output_h5 = pl)
            @test_throws ArgumentError run_parameter_sweep([agg],
                MSTMforCAS.SweepConfig(; base..., phase_reference = :origin); output_h5 = joinpath(dir, "x.h5"))
        end
    end
end
