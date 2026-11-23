# Refractive-index sweep with a shared Krylov basis (sweep_refractive_index) against
# compute_scattering, and the wiring of SweepConfig.shared_basis.

@testset "Shared-basis refractive-index sweep" begin
    # Spheres of unequal radii on a cubic lattice. Neighbours with the largest radius touch.
    function lattice(nx::Int, ny::Int, nz::Int, x0::Float64)
        local xyz = Float64[]
        local rr = Float64[]
        local c = 0
        for ix in 0:nx-1, iy in 0:ny-1, iz in 0:nz-1
            append!(xyz, 2x0 .* (ix, iy, iz)); c += 1
            push!(rr, x0 * (0.8 + 0.2 * mod(7c, 10) / 9))
        end
        return reshape(xyz, 3, :), rr                    # shape: (3, N), (N,)
    end
    relerr(a, b) = abs(a - b) / abs(b)
    # Largest relative differences to compute_scattering at a tight tolerance and the same order
    function errors(xyz, rr, mlist, res, order; kw...)
        local e = (fwd = 0.0, bwd = 0.0, qext = 0.0, qabs = 0.0)
        for (j, m) in enumerate(mlist)
            local ref
            ref, _ = compute_scattering(xyz, rr, m; tol = 1e-16, max_iter = 3000,
                                        truncation_order = order, kw...)
            @test ref.converged
            e = (fwd = max(e.fwd, relerr(res[j].S_forward[1], ref.S_forward[1]),
                           relerr(res[j].S_forward[2], ref.S_forward[2])),
                 bwd = max(e.bwd, relerr(res[j].S_backward[1], ref.S_backward[1]),
                           relerr(res[j].S_backward[2], ref.S_backward[2])),
                 qext = max(e.qext, relerr(res[j].Q_ext, ref.Q_ext)),
                 qabs = max(e.qabs, relerr(res[j].Q_abs, ref.Q_abs)))
        end
        return e
    end
    ms = vec([ComplexF64(a, b) for a in (1.6, 1.8, 2.0, 2.2), b in (0.2, 0.5, 0.8, 1.1)])
    pos, rad = lattice(3, 3, 2, 0.2)                     # 18 spheres, x = 0.16 to 0.20

    @testset "direct translation, default tolerance" begin
        res, info = sweep_refractive_index(pos, rad, ms)
        @test info.method == :shared_basis
        @test info.truncation_order == maximum(MSTMforCAS.mie_nmax(x, m) for x in rad, m in ms)
        @test all(r -> r.converged, res)
        @test info.n_fallback == 0
        @test maximum(info.relres)^2 < 1e-6
        @test all(r -> r.truncation_order == info.truncation_order, res)
        @test all(r -> r.n_iterations == maximum(info.basis_size), res)
        e = errors(pos, rad, ms, res, info.truncation_order)
        @test e.fwd < 1e-4
        @test e.bwd < 1e-4
        @test e.qext < 1e-4
        @test e.qabs < 1e-4
        # energy balance and the order of the input
        @test all(r -> isapprox(r.Q_ext, r.Q_abs + r.Q_sca; rtol = 1e-12), res)
        res_rev, _ = sweep_refractive_index(pos, rad, reverse(ms))
        @test all(isapprox(res_rev[end+1-j].Q_ext, res[j].Q_ext; rtol = 1e-8) for j in eachindex(ms))
    end

    @testset "truncation order as a lower bound" begin
        res, info = sweep_refractive_index(pos, rad, ms; truncation_order = 5)
        @test info.truncation_order == 5
        e = errors(pos, rad, ms, res, 5)
        @test e.fwd < 1e-4
        @test e.qabs < 1e-4
    end

    @testset "tight tolerance uses the CBICG fallback" begin
        res, info = sweep_refractive_index(pos, rad, ms; tol = 1e-13)
        @test info.n_fallback > 0
        @test all(r -> r.converged, res)
        e = errors(pos, rad, ms, res, info.truncation_order)
        @test e.fwd < 1e-5
        @test e.bwd < 1e-5
        @test e.qext < 1e-5
        @test e.qabs < 1e-5
        # without the fallback the same rows are flagged and a warning is issued
        res_nf, info_nf = @test_logs (:warn, r"do not meet tol") sweep_refractive_index(
            pos, rad, ms; tol = 1e-13, fallback = false)
        @test info_nf.n_fallback == 0
        @test count(r -> !r.converged, res_nf) == info.n_fallback
        @test [r.converged for r in res_nf] == [maximum(info_nf.relres[j, :])^2 < 1e-13 for j in eachindex(ms)]
    end

    @testset "FFT translation" begin
        pos2, rad2 = lattice(4, 4, 3, 0.2)               # 48 spheres
        res, info = sweep_refractive_index(pos2, rad2, ms; use_fft = true)
        @test all(r -> r.converged, res)
        e = errors(pos2, rad2, ms, res, info.truncation_order; use_fft = true)
        @test e.fwd < 1e-4
        @test e.bwd < 1e-4
        # far order of the grid below the sphere order
        res, info = sweep_refractive_index(pos2, rad2, ms; use_fft = true, truncation_order = 5,
                                           fft_far_order = 3)
        @test info.truncation_order == 5
        e = errors(pos2, rad2, ms, res, 5; use_fft = true, fft_far_order = 3)
        @test e.fwd < 1e-4
        @test e.qext < 1e-4
    end

    @testset "outside the validated range" begin
        @test isempty(shared_basis_range_issues(ms, maximum(rad)))
        @test length(shared_basis_range_issues([ComplexF64(0.2, 3.0), ComplexF64(1.5, 0.1)], 0.1)) == 2
        @test length(shared_basis_range_issues(ms, 1.0)) == 1
        # large monomers: the basis does not resolve all refractive indices, the fallback does
        pos3, rad3 = lattice(3, 3, 2, 1.0)
        ms3 = vec([ComplexF64(a, b) for a in (1.6, 2.0, 2.5), b in (0.1, 0.5)])
        res, info = @test_logs (:warn, r"outside the range") sweep_refractive_index(
            pos3, rad3, ms3; tol = 1e-12, max_iter = 2000)
        @test info.n_fallback > 0
        @test all(r -> r.converged, res)
        e = errors(pos3, rad3, ms3, res, info.truncation_order)
        @test e.fwd < 1e-5
        @test e.bwd < 1e-5
        @test e.qabs < 1e-5
        # no warning when it is switched off
        @test_logs sweep_refractive_index(pos3, rad3, ms3; warn_range = false)
    end

    @testset "cases without a shared basis" begin
        res, info = sweep_refractive_index(pos[:, 1:1], rad[1:1], ms)
        @test info.method == :independent
        for (j, m) in enumerate(ms)
            ref, _ = compute_scattering(pos[:, 1:1], rad[1:1], m; truncation_order = info.truncation_order)
            @test res[j].Q_ext ≈ ref.Q_ext rtol = 1e-12
        end
        res, info = sweep_refractive_index(pos, rad, ms[1:2]; tol = 1e-16)
        @test info.method == :independent
        e = errors(pos, rad, ms[1:2], res, info.truncation_order)
        @test e.fwd < 1e-7
    end

    @testset "arguments" begin
        @test_throws ArgumentError sweep_refractive_index(pos, rad, ComplexF64[])
        @test_throws ArgumentError sweep_refractive_index(pos[1:2, :], rad, ms)
        @test_throws ArgumentError sweep_refractive_index(pos, rad, ms; tol = 0.0)
        @test_throws ArgumentError sweep_refractive_index(pos, rad, [ComplexF64(1.0, 0.0), ms...])
    end

    @testset "SweepConfig.shared_basis" begin
        HDF5 = MSTMforCAS.HDF5
        r0 = 0.015                                       # monomer radius [um]
        pos_um, rad_x = lattice(3, 2, 2, 1.0)
        agg = AggregateGeometry(pos_um .* r0, rad_x .* r0, length(rad_x), "lattice12")
        wl = 0.453; k = 2pi / wl
        base = (medium_conditions = [(wl, 1.0)],
                m_real_range = (1.6, 2.2, 3), m_imag_range = (0.2, 1.0, 3))
        read_cols(path) = HDF5.h5open(path, "r") do f
            (m = complex.(read(f["m_real"]), read(f["m_imag"])), order = read(f["truncation_order"]),
             qext = read(f["Q_ext"]), qabs = read(f["Q_abs"]), conv = read(f["converged"]),
             s1 = complex.(read(f["S1_fwd_re"]), read(f["S1_fwd_im"])),
             s1b = complex.(read(f["S1_bwd_re"]), read(f["S1_bwd_im"])))
        end
        mktempdir() do dir
            path = joinpath(dir, "sb.h5")
            run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base..., shared_basis = true); output_h5 = path)
            c = read_cols(path)
            @test length(c.m) == 9
            @test all(c.conv .== 1)
            @test MSTMforCAS._read_shared_basis(path)
            order = maximum(MSTMforCAS.mie_nmax(k * x, m) for x in agg.radii, m in c.m)
            @test all(c.order .== order)                 # one order for the whole group
            for i in eachindex(c.m)
                ref, _ = compute_scattering(agg, c.m[i], k; tol = 1e-16, truncation_order = order)
                @test c.qext[i] ≈ ref.Q_ext rtol = 1e-4
                @test c.qabs[i] ≈ ref.Q_abs rtol = 1e-4
                @test c.s1[i] ≈ ref.S_forward[1] rtol = 1e-4
                @test c.s1b[i] ≈ ref.S_backward[1] rtol = 1e-4
            end

            # with truncation_eps the group uses the largest contact order of the grid
            path_e = joinpath(dir, "sb_teps.h5")
            run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base..., shared_basis = true,
                                truncation_eps = 1e-2); output_h5 = path_e)
            ce = read_cols(path_e)
            xb = round(k * sum(agg.radii) / length(agg.radii); sigdigits = 2)
            order_e = max(order, maximum(contact_truncation_order(xb, m; eps = 1e-2) for m in ce.m))
            @test all(ce.order .== order_e)
            @test order_e > order
            for i in eachindex(ce.m)
                ref, _ = compute_scattering(agg, ce.m[i], k; tol = 1e-16, truncation_order = order_e)
                @test ce.s1[i] ≈ ref.S_forward[1] rtol = 1e-4
            end

            # the default is the solve per refractive index, and the two do not share a file
            path_d = joinpath(dir, "default.h5")
            run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base...); output_h5 = path_d)
            @test !MSTMforCAS._read_shared_basis(path_d)
            wider = (medium_conditions = base.medium_conditions, m_real_range = (1.6, 2.2, 4),
                     m_imag_range = base.m_imag_range)
            @test_throws ErrorException run_parameter_sweep([agg],
                MSTMforCAS.SweepConfig(; wider..., shared_basis = true); output_h5 = path_d)
            @test_throws ErrorException run_parameter_sweep([agg],
                MSTMforCAS.SweepConfig(; wider...); output_h5 = path)

            # a resumed sweep keeps the order of the whole grid. The basis of the resumed
            # rows differs, so the values agree to the solver tolerance only.
            path_r = joinpath(dir, "resume.h5")
            cfg = MSTMforCAS.SweepConfig(; base..., shared_basis = true, truncation_order = 4)
            run_parameter_sweep([agg], cfg; output_h5 = path_r)
            cr = read_cols(path_r)
            keep = 1:5
            HDF5.h5open(path_r, "r+") do f
                for name in keys(f)
                    HDF5.set_extent_dims(f[name], (length(keep),))
                end
            end
            run_parameter_sweep([agg], cfg; output_h5 = path_r)
            cr2 = read_cols(path_r)
            @test length(cr2.m) == 9
            @test all(cr2.order .== 4)
            for i in eachindex(cr2.m)
                j = findfirst(==(cr2.m[i]), cr.m)
                @test cr2.qext[i] ≈ cr.qext[j] rtol = 1e-4
            end

            @test_throws ErrorException run_parameter_sweep([agg],
                MSTMforCAS.SweepConfig(; base..., shared_basis = true, use_gpu = true))
            @test_throws ErrorException run_parameter_sweep([agg],
                MSTMforCAS.SweepConfig(; base..., shared_basis = true, solver = :gmres))
        end
    end
end
