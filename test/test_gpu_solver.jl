"""
GPU batch solver tests.

These tests require a functional CUDA GPU. They are skipped automatically
if CUDA.jl is not available or no GPU is detected.

Run with:
  julia --project=. test/test_gpu_solver.jl

CUDA.jl is a weak dependency of the package. It must be loadable in the session, for example
from the default environment (`julia -e 'using Pkg; Pkg.add("CUDA")'`).
The extension is tested with CUDA.jl 5 and 6. `Pkg.test()` does not
run this file.
"""

using Test
using MSTMforCAS
using LinearAlgebra
using HDF5

# ─── Check GPU availability ──────────────────────────────────────────────────

gpu_available = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !gpu_available
    @info "CUDA not available — skipping GPU tests"
    @testset "GPU solver (skipped)" begin
        @test_skip true
    end
else
    @info "CUDA available: $(CUDA.name(CUDA.device()))"

    @testset "GPU solver" begin

        @testset "GPU dispatch hook registered" begin
            @test MSTMforCAS._gpu_batch_solve_ref[] !== nothing
        end

        @testset "SweepConfig use_gpu field" begin
            config = SweepConfig(
                medium_conditions = [(0.5, 1.0)],
                m_real_range = (1.5, 1.5, 1),
                m_imag_range = (0.1, 0.1, 1),
                use_gpu = true,
            )
            @test config.use_gpu == true
        end

        @testset "Single RI: GPU vs CPU (testcase1 geometry)" begin
            # 2-sphere dimer from testcase1
            positions = Float64[0.0 0.4; 0.0 0.0; 0.0 0.0]  # (3, 2)
            radii = Float64[0.2, 0.2]
            k = 7.534          # length_scale_factor
            m_rel = ComplexF64(1.58, 0.05)

            # CPU reference
            agg = AggregateGeometry(positions, radii, 2, "test_gpu")
            r_cpu, amn_cpu = compute_scattering(agg, m_rel, k; use_fft=true)

            # GPU batch (single RI)
            ri_values = [m_rel * 1.0]  # absolute RI (n_med=1.0)
            fft_cache_positions = positions .* k
            fft_cache_radii = radii .* k
            noi = MSTMforCAS.mie_nmax(fft_cache_radii[1], m_rel)
            fft_cache = MSTMforCAS.init_fft_grid(fft_cache_positions, fft_cache_radii, fill(noi, 2))

            gpu_results = MSTMforCAS._gpu_batch_solve_ref[](
                agg, k, ri_values, 1.0, fft_cache,
                (tol=1e-6, max_iter=200, truncation_order=nothing))

            r_gpu, amn_gpu = gpu_results[1]

            # Compare Q values
            @test isapprox(r_gpu.Q_ext, r_cpu.Q_ext, rtol=1e-3)
            @test isapprox(r_gpu.Q_abs, r_cpu.Q_abs, rtol=1e-2)
            @test isapprox(r_gpu.Q_sca, r_cpu.Q_sca, rtol=1e-3)

            # Compare S amplitudes
            for i in 1:4
                if abs(r_cpu.S_forward[i]) > 1e-10
                    @test isapprox(r_gpu.S_forward[i], r_cpu.S_forward[i], rtol=1e-2)
                end
                if abs(r_cpu.S_backward[i]) > 1e-10
                    @test isapprox(r_gpu.S_backward[i], r_cpu.S_backward[i], rtol=1e-2)
                end
            end

            @test r_gpu.converged
        end

        @testset "Batch consistency: multiple RIs" begin
            # Simple 2-sphere geometry
            positions = Float64[0.0 0.3; 0.0 0.0; 0.0 0.0]
            radii = Float64[0.15, 0.15]
            k = 8.0
            agg = AggregateGeometry(positions, radii, 2, "test_batch")

            ri_values = [ComplexF64(1.5, 0.1), ComplexF64(1.6, 0.2), ComplexF64(1.7, 0.3)]
            n_med = 1.0

            # CPU reference for each RI
            cpu_results = map(ri_values) do m_abs
                m_rel = m_abs / n_med
                compute_scattering(agg, m_rel, k; use_fft=true)
            end

            # GPU batch
            noi_max = maximum(MSTMforCAS.mie_nmax(radii[1]*k, m/n_med) for m in ri_values)
            fft_cache = MSTMforCAS.init_fft_grid(positions .* k, radii .* k, fill(noi_max, 2))
            gpu_results = MSTMforCAS._gpu_batch_solve_ref[](
                agg, k, ri_values, n_med, fft_cache,
                (tol=1e-6, max_iter=200, truncation_order=nothing))

            for (idx, m_abs) in enumerate(ri_values)
                r_cpu, _ = cpu_results[idx]
                r_gpu, _ = gpu_results[idx]
                @test isapprox(r_gpu.Q_ext, r_cpu.Q_ext, rtol=1e-3)
                @test isapprox(r_gpu.Q_sca, r_cpu.Q_sca, rtol=1e-3)
                @test r_gpu.converged
            end
        end

        @testset "Unequal radii: every sphere has its own T-matrix" begin
            # Bent chain of four touching spheres with different radii; coordinates in um.
            # Sphere 1 is the smallest, so a T-matrix shared from sphere 1 would be far off.
            radii = Float64[0.011, 0.015, 0.018, 0.0135]
            positions = zeros(Float64, 3, 4)                       # shape: (3, 4)
            positions[:, 2] = positions[:, 1] .+ (radii[1] + radii[2]) .* [1.0, 0.0, 0.0]
            positions[:, 3] = positions[:, 2] .+ (radii[2] + radii[3]) .* [0.0, 1.0, 0.0]
            positions[:, 4] = positions[:, 3] .+ (radii[3] + radii[4]) .* [0.0, 0.6, 0.8]
            k = 2pi / 0.834                                        # wavenumber [1/um]
            agg = AggregateGeometry(positions, radii, 4, "test_unequal_radii")
            ri_values = [ComplexF64(1.6, 0.2), ComplexF64(2.2, 1.0)]
            order = 4                                              # same order on both paths
            fft_cache = MSTMforCAS.init_fft_grid(positions .* k, radii .* k, fill(order, 4))

            cpu = [compute_scattering(agg, m, k; use_fft=true, truncation_order=order, tol=1e-12)[1]
                   for m in ri_values]
            for (float32, rtol_s, tol) in ((false, 1e-6, 1e-12), (true, 1e-3, 1e-5))
                gpu = MSTMforCAS._gpu_batch_solve_ref[](
                    agg, k, ri_values, 1.0, fft_cache,
                    (tol=tol, max_iter=200, truncation_order=order, float32=float32))
                for idx in eachindex(ri_values)
                    r_gpu, _ = gpu[idx]
                    r_cpu = cpu[idx]
                    @test r_gpu.converged
                    for q in 1:2
                        @test isapprox(r_gpu.S_forward[q], r_cpu.S_forward[q]; rtol=rtol_s)
                        @test isapprox(r_gpu.S_backward[q], r_cpu.S_backward[q]; rtol=rtol_s)
                    end
                    @test isapprox(r_gpu.Q_ext, r_cpu.Q_ext; rtol=rtol_s)
                    @test isapprox(r_gpu.Q_abs, r_cpu.Q_abs; rtol=rtol_s)
                end
            end
        end

        @testset "Aggregate of 100 monomers with unequal radii" begin
            # Fractal aggregate with a relative standard deviation of the radii of 0.10
            # (fixture of test_fortran_unequal_radii.jl). GPU and CPU use the same FFT grid.
            k = 13.221101                                          # wavenumber [1/um]
            agg = read_aggregate_file(joinpath(@__DIR__, "fixtures", "aggregate_unequal_radii_Np100.ptsa"))
            pos = agg.positions .* k; rad = agg.radii .* k         # shape: (3, N), (N,)
            N = length(rad)
            ri_values = [ComplexF64(1.95, 0.79), ComplexF64(1.6, 0.0123), ComplexF64(2.4, 1.2)]
            @test maximum(rad) / minimum(rad) > 1.3

            # (sphere order, far order of the grid); `nothing` = the grid carries all orders
            for (order, far) in ((3, nothing), (6, 3))
                fft_cache = far === nothing ? MSTMforCAS.init_fft_grid(pos, rad, fill(order, N)) :
                            MSTMforCAS.init_fft_grid(pos, rad, fill(order, N); far_nois = fill(far, N))
                cpu = [compute_scattering(agg, m, k; use_fft = true, truncation_order = order,
                                          precomputed_fft = fft_cache, tol = 1e-16, max_iter = 500)[1]
                       for m in ri_values]
                # Float64 on the GPU reproduces the CPU result. Float32 is limited by its tolerance.
                for (float32, rtol, tol) in ((false, 1e-7, 1e-16), (true, 1e-2, 1e-5))
                    gpu = MSTMforCAS._gpu_batch_solve_ref[](
                        agg, k, ri_values, 1.0, fft_cache,
                        (tol = tol, max_iter = 500, truncation_order = order, float32 = float32))
                    for idx in eachindex(ri_values)
                        r_gpu, _ = gpu[idx]
                        r_cpu = cpu[idx]
                        @test r_gpu.converged
                        @test r_gpu.truncation_order == order
                        for q in 1:2
                            @test isapprox(r_gpu.S_forward[q], r_cpu.S_forward[q]; rtol = rtol)
                            @test isapprox(r_gpu.S_backward[q], r_cpu.S_backward[q]; rtol = rtol)
                        end
                        @test isapprox(r_gpu.Q_ext, r_cpu.Q_ext; rtol = rtol)
                        @test isapprox(r_gpu.Q_abs, r_cpu.Q_abs; rtol = 10 * rtol)
                        @test isapprox(r_gpu.Q_sca, r_cpu.Q_sca; rtol = rtol)
                    end
                end
            end
        end

        @testset "Parameter sweep on the GPU" begin
            read_cols(path) = HDF5.h5open(path, "r") do f
                n = read(f["n_monomers"])
                m = complex.(read(f["m_real"]), read(f["m_imag"]))
                p = sortperm(collect(zip(n, real.(m), imag.(m))))
                (n = n[p], m = m[p], qext = read(f["Q_ext"])[p], qabs = read(f["Q_abs"])[p],
                 s1f = complex.(read(f["S1_fwd_re"]), read(f["S1_fwd_im"]))[p],
                 s1b = complex.(read(f["S1_bwd_re"]), read(f["S1_bwd_im"]))[p],
                 order = read(f["truncation_order"])[p], conv = read(f["converged"])[p])
            end
            agg20  = read_aggregate_file(joinpath(@__DIR__, "fixtures", "aggregate_unequal_radii_Np20.ptsa"))
            agg100 = read_aggregate_file(joinpath(@__DIR__, "fixtures", "aggregate_unequal_radii_Np100.ptsa"))
            base = (medium_conditions = [(0.6328, 1.0)], m_real_range = (1.6, 2.4, 3),
                    m_imag_range = (0.1, 1.2, 3), use_fft = true, truncation_order = 4,
                    convergence_epsilon = 1e-10)

            mktempdir() do dir
                run_parameter_sweep([agg20, agg100], SweepConfig(; base...); output_h5 = joinpath(dir, "cpu.h5"))
                c = read_cols(joinpath(dir, "cpu.h5"))
                @test length(c.m) == 18

                # hybrid dispatch (20 monomers on the CPU, 100 on the GPU), all on the GPU, Float32
                for (name, extra, rtol) in (("hybrid.h5", (use_gpu = true, gpu_np_threshold = 50), 2e-4),
                                            ("gpu.h5", (use_gpu = true, gpu_np_threshold = 0), 2e-4),
                                            ("f32.h5", (use_gpu = true, gpu_float32 = true, gpu_np_threshold = 0), 2e-2))
                    run_parameter_sweep([agg20, agg100], SweepConfig(; base..., extra...); output_h5 = joinpath(dir, name))
                    g = read_cols(joinpath(dir, name))
                    @test g.n == c.n && g.m == c.m
                    @test all(g.conv .== 1)
                    @test g.order == c.order
                    @test isapprox(g.s1f, c.s1f; rtol = rtol)
                    @test isapprox(g.s1b, c.s1b; rtol = rtol)
                    @test isapprox(g.qext, c.qext; rtol = rtol)
                    @test all(isapprox.(g.qabs, c.qabs; rtol = 5 * rtol))
                end

                # One truncation order per group, taken from the whole refractive index grid:
                # a resumed sweep uses the order of the uninterrupted one. The row of the
                # refractive index with the lowest contact order is removed and computed again.
                path = joinpath(dir, "resume.h5")
                cfg = SweepConfig(; medium_conditions = base.medium_conditions, m_real_range = base.m_real_range,
                                  m_imag_range = base.m_imag_range, use_fft = true, truncation_eps = 1e-2,
                                  use_gpu = true, gpu_np_threshold = 0)
                run_parameter_sweep([agg20], cfg; output_h5 = path)
                full = read_cols(path)
                @test length(full.m) == 9
                @test all(full.order .== full.order[1])           # one order for the group
                m_easy = complex(1.6, 0.1)
                @test MSTMforCAS.contact_truncation_order(maximum(agg20.radii) * 2pi / 0.6328, m_easy; eps = 1e-2) <
                      full.order[1]                               # the easy index alone needs less
                HDF5.h5open(path, "r+") do f
                    m_file = complex.(read(f["m_real"]), read(f["m_imag"]))
                    j = findfirst(≈(m_easy), m_file)
                    perm = vcat(setdiff(1:length(m_file), j), j)   # the easy row last
                    for name in keys(f)
                        v = read(f[name])
                        f[name][:] = v[perm]
                        HDF5.set_extent_dims(f[name], (length(m_file) - 1,))
                    end
                end
                run_parameter_sweep([agg20], cfg; output_h5 = path)
                res = read_cols(path)
                @test length(res.m) == 9
                @test res.order == full.order
                @test isapprox(res.qext, full.qext; rtol = 1e-6)
                @test isapprox(res.s1f, full.s1f; rtol = 1e-6)
            end
        end

    end  # @testset "GPU solver"
end
