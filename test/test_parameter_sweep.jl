# Parameter sweep: truncation order chosen by truncation_eps (two-sphere preliminary calculation).

@testset "ParameterSweep truncation_eps" begin
    HDF5 = MSTMforCAS.HDF5
    r = 0.015                                     # monomer radius [um]
    # three touching spheres in a bent chain; coordinates in um
    positions = [0.0 2r 2r;
                 0.0 0.0 2r;
                 0.0 0.0 0.0]                     # shape: (3, 3)
    agg = AggregateGeometry(positions, fill(r, 3), 3, "bent_trimer")
    wl = 0.834; k = 2pi / wl
    base = (medium_conditions = [(wl, 1.0)],
            m_real_range = (1.6, 2.2, 2), m_imag_range = (0.2, 1.0, 2),
            convergence_epsilon = 1e-20)   # tight, so warm and cold starts agree
    read_cols(path) = HDF5.h5open(path, "r") do f
        (m = complex.(read(f["m_real"]), read(f["m_imag"])), order = read(f["truncation_order"]),
         qext = read(f["Q_ext"]), conv = read(f["converged"]),
         s1 = complex.(read(f["S1_fwd_re"]), read(f["S1_fwd_im"])))
    end

    mktempdir() do dir
        @testset "default keeps the single-sphere Mie order" begin
            path = joinpath(dir, "mie.h5")
            run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base...); output_h5 = path)
            c = read_cols(path)
            @test length(c.m) == 4
            for i in eachindex(c.m)
                @test c.order[i] == MSTMforCAS.mie_nmax(k * r, c.m[i])
                ref, _ = compute_scattering(agg, c.m[i], k; tol = 1e-20)
                @test c.qext[i] ≈ ref.Q_ext rtol = 1e-8
            end
            @test MSTMforCAS._read_truncation_eps(path) === nothing
        end

        @testset "truncation_eps selects the contact order per refractive index" begin
            for eps in (1e-2, 1e-3)
                path = joinpath(dir, "teps_$(eps).h5")
                run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base..., truncation_eps = eps); output_h5 = path)
                c = read_cols(path)
                @test all(c.conv .== 1)
                @test MSTMforCAS._read_truncation_eps(path) == eps
                for i in eachindex(c.m)
                    n = contact_truncation_order(round(k * r; sigdigits = 2), c.m[i]; eps = eps)
                    @test c.order[i] == n
                    ref, _ = compute_scattering(agg, c.m[i], k; tol = 1e-20, truncation_order = n)
                    @test c.qext[i] ≈ ref.Q_ext rtol = 1e-8
                    @test c.s1[i] ≈ ref.S_forward[1] rtol = 1e-8
                end
                @test maximum(c.order) > minimum(c.order)        # order follows the refractive index
            end
        end

        @testset "explicit truncation_order acts as a lower bound" begin
            path = joinpath(dir, "floor.h5")
            run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base..., truncation_eps = 1e-2, truncation_order = 6);
                                output_h5 = path)
            c = read_cols(path)
            for i in eachindex(c.m)
                n = contact_truncation_order(round(k * r; sigdigits = 2), c.m[i]; eps = 1e-2)
                @test c.order[i] == max(6, n)
            end
        end

        @testset "FFT path sizes its grid for the selected orders" begin
            path = joinpath(dir, "fft.h5")
            run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base..., truncation_eps = 1e-2, use_fft = true);
                                output_h5 = path)
            c = read_cols(path); d = read_cols(joinpath(dir, "teps_0.01.h5"))
            @test all(c.conv .== 1)
            pd = sortperm(d.m; by = z -> (real(z), imag(z))); pc = sortperm(c.m; by = z -> (real(z), imag(z)))
            @test c.order[pc] == d.order[pd]
            @test isapprox(c.qext[pc], d.qext[pd]; rtol = 2e-2)   # FFT translation is approximate
        end

        @testset "resume refuses a file written with another setting" begin
            @test_throws ErrorException run_parameter_sweep(
                [agg], MSTMforCAS.SweepConfig(; base..., truncation_eps = 1e-3); output_h5 = joinpath(dir, "mie.h5"))
            @test_throws ErrorException run_parameter_sweep(
                [agg], MSTMforCAS.SweepConfig(; base...); output_h5 = joinpath(dir, "teps_0.01.h5"))
            # same setting: nothing left to do, no error
            @test run_parameter_sweep([agg], MSTMforCAS.SweepConfig(; base..., truncation_eps = 1e-2);
                                      output_h5 = joinpath(dir, "teps_0.01.h5")) === nothing
        end
    end
end
