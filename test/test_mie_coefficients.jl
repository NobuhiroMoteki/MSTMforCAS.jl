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

    @testset "Single sphere: water droplet m=1.33, x=1.0 (non-absorbing)" begin
        x = 1.0
        m = ComplexF64(1.33, 0.0)

        a, b = compute_mie_coefficients(x, m)

        @test length(a) >= 3
        @test length(b) >= 3

        nmax = length(a)
        Q_ext = (2.0 / x^2) * sum((2n + 1) * real(a[n] + b[n]) for n in 1:nmax)
        Q_sca = (2.0 / x^2) * sum((2n + 1) * (abs2(a[n]) + abs2(b[n])) for n in 1:nmax)

        # Non-absorbing: Q_ext = Q_sca exactly
        @test isapprox(Q_ext, Q_sca, rtol=1e-12)
        # Q_ext ≈ 0.0939 (computed reference; BH83-compliant algorithm verified
        # against testcase1 and testcase2 reference values in separate tests below)
        @test isapprox(Q_ext, 0.0939, atol=0.001)
    end

    @testset "Single sphere: testcase1 sphere (x=1.507, m=1.58+0.05i)" begin
        # CLAUDE.md reference: Q_ext_single ≈ 1.2517, Q_abs_single ≈ 0.2855
        x = 1.507
        m = ComplexF64(1.58, 0.05)
        a, b = compute_mie_coefficients(x, m)
        nmax = length(a)
        Q_ext = (2.0 / x^2) * sum((2n + 1) * real(a[n] + b[n]) for n in 1:nmax)
        Q_sca = (2.0 / x^2) * sum((2n + 1) * (abs2(a[n]) + abs2(b[n])) for n in 1:nmax)
        Q_abs = Q_ext - Q_sca
        @test isapprox(Q_ext, 1.2517, rtol=1e-3)
        @test isapprox(Q_abs, 0.2855, rtol=1e-3)
    end

    @testset "Single sphere: testcase2 sphere (x=1.0, m=1.6+0.0123i)" begin
        # CLAUDE.md reference: Q_ext_single ≈ 0.3399, Q_abs_single ≈ 0.0356
        x = 1.0
        m = ComplexF64(1.6, 0.0123)
        a, b = compute_mie_coefficients(x, m)
        nmax = length(a)
        Q_ext = (2.0 / x^2) * sum((2n + 1) * real(a[n] + b[n]) for n in 1:nmax)
        Q_sca = (2.0 / x^2) * sum((2n + 1) * (abs2(a[n]) + abs2(b[n])) for n in 1:nmax)
        Q_abs = Q_ext - Q_sca
        @test isapprox(Q_ext, 0.3399, rtol=1e-3)
        @test isapprox(Q_abs, 0.0356, rtol=1e-2)
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

    @testset "Miller ψ_n recurrence: stability at small x (Au doublet regression)" begin
        # Regression for block-VIEM.jl doublet benchmark (Au @ 638 nm):
        # pre-fix, the upward ψ_n recurrence blew up at n ≥ 7 for x ≈ 0.3,
        # causing |S_fw| to jump by 4 orders of magnitude at truncation N ≥ 7.
        # The Miller downward algorithm restores stability at all n.
        x = 2π * 0.030 / 0.638           # ≈ 0.2954
        m_Au = ComplexF64(0.17525, 3.4830)
        a, b = compute_mie_coefficients(x, m_Au; nmax = 8)

        @test all(isfinite, a)
        @test all(isfinite, b)

        # |a_n| and |b_n| must decrease monotonically across the full range
        # (the dominant-dipole regime for x ≪ 1 — geometric decay ~(x/n)^{2n+1}).
        for n in 1:7
            @test abs(a[n+1]) < abs(a[n])
            @test abs(b[n+1]) < abs(b[n])
        end

        # Leading-order Rayleigh limit: a_1 ≈ -(2i/3)·x³·(m²−1)/(m²+2)
        # Finite-x correction is O(x²) ≈ 9% at x = 0.3, so use rtol = 0.1.
        rayleigh_a1 = -(2im / 3) * x^3 * (m_Au^2 - 1) / (m_Au^2 + 2)
        @test isapprox(a[1], rayleigh_a1; rtol = 0.1)
    end

    @testset "solve_tmatrix with truncation_order > mie_nmax(x) (Au doublet)" begin
        # Regression: mie_vecs[i] was sized by compute_mie_coefficients'
        # default nmax = mie_nmax(radii[i]) (Wiscombe upper bound ≈ 8 at
        # x ≈ 0.3), ignoring the user-supplied truncation_order stored in
        # nois[i].  _precompute_T_values then indexed a_v[n], b_v[n] past
        # the end of the 8-element vectors for any truncation_order ≥ 9,
        # raising a BoundsError.  Fix: size Mie vectors to nois[i].
        k = 2π / 0.638                    # medium wavenumber, μm⁻¹
        R = 0.030                         # monomer radius, μm
        d = 2R + 0.003                    # centre-to-centre separation
        positions = [0.0 0.0; 0.0 0.0; -d/2 +d/2]
        radii     = [R, R]
        m_rel     = ComplexF64(0.17525, 3.4830)  # Au @ 638 nm

        positions_x = positions .* k
        radii_x     = radii     .* k

        for N in (3, 5, 8, 10, 15)
            result, _ = compute_scattering(
                positions_x, radii_x, m_rel;
                truncation_order = N, tol = 1e-10)
            @test result.converged
            @test all(isfinite, result.S_forward)
            @test isfinite(result.Q_ext)
        end
    end

    @testset "Backward compatibility: upward-stable regime (x = 5, m = 1.33)" begin
        # Frozen reference values from the pre-Miller upward implementation
        # at x = 5, m = 1.33+0.0i, for n = 1..5 (within the upward-stable range).
        # New Miller implementation must agree to ≈ machine ε.
        a, b = compute_mie_coefficients(5.0, ComplexF64(1.33, 0.0))

        a_ref = [
            0.9935773889533813 - 0.0798834220221105im,
            0.9990447142202481 + 0.030892931373226948im,
            0.924737697749646  - 0.26381411658652815im,
            0.6572360464210308 - 0.4746333592425667im,
            0.11754010119253813 - 0.32206276687035096im,
        ]
        b_ref = [
            0.9722365577931019 + 0.1642943501272098im,
            0.9424418611622516 - 0.23290598852602068im,
            0.9829973454561133 - 0.129280950964742im,
            0.8890023560981505 - 0.314129220146881im,
            0.05967562409175904 - 0.23688487495198612im,
        ]
        for n in 1:5
            @test isapprox(a[n], a_ref[n]; rtol = 1e-12)
            @test isapprox(b[n], b_ref[n]; rtol = 1e-12)
        end
    end

end
