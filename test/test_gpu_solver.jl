"""
GPU batch solver tests.

These tests require a functional CUDA GPU. They are skipped automatically
if CUDA.jl is not available or no GPU is detected.

Run with:
  julia --project=. test/test_gpu_solver.jl
"""

using Test
using MSTMforCAS
using LinearAlgebra

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

    end  # @testset "GPU solver"
end
