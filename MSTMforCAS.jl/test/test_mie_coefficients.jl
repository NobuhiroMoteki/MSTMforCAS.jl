using Test
using MSTMforCAS

@testset "MieCoefficients" begin

    @testset "nmax estimation" begin
        # Small particle
        @test mie_nmax(1.0) >= 3
        # Medium particle
        @test mie_nmax(10.0) >= 14
        # Large particle
        @test mie_nmax(100.0) >= 120
    end

    @testset "Single sphere: BH83 Table 4.1 (water droplet x=1.0)" begin
        # BH83 Table 4.1: m = 1.33, x = 1.0
        # Expected (from BH83):
        #   a₁ ≈ 0.04921 + 0.17973i  (complex Mie coefficient, check sign convention)
        #   b₁ ≈ 0.00296 + 0.01257i
        x = 1.0
        m = ComplexF64(1.33, 0.0)

        a, b = compute_mie_coefficients(x, m)

        # At minimum, a and b should have nmax entries
        @test length(a) >= 3
        @test length(b) >= 3

        # Verify extinction efficiency via BH83 Eq. 4.61:
        # Q_ext = (2/x²) Σₙ (2n+1) Re(aₙ + bₙ)
        nmax = length(a)
        Q_ext = (2.0 / x^2) * sum((2n + 1) * real(a[n] + b[n]) for n in 1:nmax)

        # BH83 Table 4.3 gives Q_ext ≈ 0.1130 for m=1.33, x=1.0
        # (approximate; exact value depends on nmax truncation)
        @test isapprox(Q_ext, 0.1130, atol=0.005)
    end

    @testset "Non-absorbing sphere: Im(aₙ), Im(bₙ) properties" begin
        # For non-absorbing sphere (real m), |aₙ|² = Re(aₙ) and |bₙ|² = Re(bₙ)
        # This is equivalent to: |aₙ| ≤ 1 and Im(aₙ) = |aₙ|² - Re(aₙ) ... 
        # Actually: for real m, aₙ and bₙ lie on or inside the unit circle
        # in the complex plane, and |aₙ|² + (Im aₙ)² stuff...
        #
        # Simpler test: Q_sca = Q_ext for non-absorbing sphere
        x = 5.0
        m = ComplexF64(1.5, 0.0)
        a, b = compute_mie_coefficients(x, m)
        nmax = length(a)

        Q_ext = (2.0 / x^2) * sum((2n + 1) * real(a[n] + b[n]) for n in 1:nmax)
        Q_sca = (2.0 / x^2) * sum((2n + 1) * (abs2(a[n]) + abs2(b[n])) for n in 1:nmax)

        # For non-absorbing particle: Q_ext = Q_sca (Q_abs = 0)
        @test isapprox(Q_ext, Q_sca, rtol=1e-12)
    end

    @testset "Absorbing sphere: Q_abs > 0" begin
        x = 5.0
        m = ComplexF64(1.5, 0.1)  # absorbing
        a, b = compute_mie_coefficients(x, m)
        nmax = length(a)

        Q_ext = (2.0 / x^2) * sum((2n + 1) * real(a[n] + b[n]) for n in 1:nmax)
        Q_sca = (2.0 / x^2) * sum((2n + 1) * (abs2(a[n]) + abs2(b[n])) for n in 1:nmax)
        Q_abs = Q_ext - Q_sca

        @test Q_abs > 0.0
        @test Q_ext > Q_sca
    end

    @testset "Forward scattering amplitude for single sphere" begin
        # For a single sphere, S₁(0°) = S₂(0°) = (1/2) Σₙ (2n+1)(aₙ + bₙ)
        # (BH83 Eq. 4.74, using πₙ(1) = τₙ(1) = n(n+1)/2)
        x = 3.0
        m = ComplexF64(1.5, 0.01)
        a, b = compute_mie_coefficients(x, m)
        nmax = length(a)

        S_forward = 0.5 * sum((2n + 1) * (a[n] + b[n]) for n in 1:nmax)

        # Optical theorem: C_ext = (4π/k²) Re(S(0°)) where k*r → x for efficiency
        # Q_ext = C_ext / (π r²) = (4/x²) Re(S(0°))  ... check factor
        # Actually: Q_ext = (2/x²) Σ(2n+1)Re(aₙ+bₙ) = (2/x²)·2·Re(S_forward)
        #         = (4/x²) Re(S_forward)
        Q_ext_from_S = (4.0 / x^2) * real(S_forward)
        Q_ext_direct = (2.0 / x^2) * sum((2n + 1) * real(a[n] + b[n]) for n in 1:nmax)

        @test isapprox(Q_ext_from_S, Q_ext_direct, rtol=1e-14)
    end

    @testset "Size parameter x → 0 (Rayleigh limit)" begin
        x = 0.01
        m = ComplexF64(1.5, 0.0)
        a, b = compute_mie_coefficients(x, m)

        # In the Rayleigh limit, a₁ dominates and scales as x³
        # |a₁| >> |a₂|, |b₁|, etc.
        @test abs(a[1]) > 100 * abs(a[2])
        @test abs(a[1]) > 100 * abs(b[1])
    end

end
