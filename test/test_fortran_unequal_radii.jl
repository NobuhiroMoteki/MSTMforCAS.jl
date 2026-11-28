# Regression test against the Fortran code MSTM-v4.0_for_CAS for aggregates whose monomer
# radii differ. The three test cases of the Fortran distribution have equal spheres only.
#
# Reference values: Fortran runs of 2026-11-24 with direct translation
# (fft_translation_option = .false.), solution_epsilon = 1.d-12 and
# length_scale_factor = 13.221101 (wavelength 0.6328 μm in water, n_medium = 1.33154).
# The refractive index is given relative to the medium. The Fortran output has four
# significant digits for the amplitudes and five for the efficiencies, which sets the
# tolerances below. The package differed from these values by at most 1.9e-4 in S1, S2
# and by at most 3.1e-5 in the efficiencies when the test was written.
#
# The Fortran code phases the incident wave at the lab plane z = 0. The fixtures have their
# mass centroid at the origin, not the unweighted mean of the centres, so the amplitudes are
# compared with phase_reference = :lab (the default :particle differs by exp(-i z_r0)).
#
# Fixtures: fractal aggregates with D_f = 2.35, mean monomer radius 0.020 μm and a relative
# standard deviation of the radii of 0.10 (file columns x y z r in μm).

@testset "Fortran reference, unequal radii" begin
    k_medium = 13.221101                 # wavenumber in the medium [1/μm]

    # (file, m_rel, forward S1..S4, backward S1..S4, (Q_ext, Q_abs, Q_sca)), BH83 amplitudes
    cases = [
        ("aggregate_unequal_radii_Np20.ptsa", complex(1.95, 0.79),
         (1.2500e-01 - 1.9390e-01im, 1.2080e-01 - 1.9160e-01im, 6.7420e-03 - 1.8790e-03im, 6.9630e-03 - 3.0110e-03im),
         (5.8170e-02 - 5.6880e-02im, -5.7760e-02 + 5.8460e-02im, -3.3990e-03 - 1.0560e-03im, 3.3990e-03 + 1.0550e-03im),
         (1.0719e+00, 9.2057e-01, 1.5136e-01)),
        ("aggregate_unequal_radii_Np20.ptsa", complex(1.6, 0.0123),
         (3.8930e-03 - 1.2370e-01im, 3.7750e-03 - 1.2230e-01im, 3.4570e-04 - 2.0750e-03im, 1.9600e-04 - 2.0930e-03im),
         (7.9820e-03 - 4.4840e-02im, -7.8810e-03 + 4.5000e-02im, -5.2000e-04 + 8.9140e-04im, 5.2000e-04 - 8.9140e-04im),
         (6.3484e-02, 1.8745e-02, 4.4738e-02)),
        ("aggregate_unequal_radii_Np100.ptsa", complex(1.95, 0.79),
         (8.1030e-01 - 9.4450e-01im, 7.5950e-01 - 9.4730e-01im, 4.3120e-02 - 7.8730e-03im, 4.0370e-02 - 8.4210e-03im),
         (-4.2460e-02 - 2.3060e-01im, 8.6880e-02 + 1.4260e-01im, 9.7920e-03 + 1.7400e-02im, -9.7920e-03 - 1.7400e-02im),
         (2.1143e+00, 1.5363e+00, 5.7799e-01)),
        ("aggregate_unequal_radii_Np100.ptsa", complex(1.6, 0.0123),
         (7.7350e-02 - 7.0600e-01im, 6.9100e-02 - 6.9060e-01im, 1.9530e-03 - 1.3430e-02im, 2.0010e-03 - 1.2940e-02im),
         (-9.7170e-02 - 8.5820e-02im, 7.6890e-02 + 6.1220e-02im, 6.7600e-03 - 1.4090e-03im, -6.7600e-03 + 1.4090e-03im),
         (2.3327e-01, 3.4373e-02, 1.9890e-01)),
    ]

    rtol_amp = 5e-4      # S1, S2: four printed digits
    atol_x   = 1e-4      # S3, S4: absolute, in units of |S1|
    rtol_q   = 1e-4      # efficiencies: five printed digits

    for (file, m_rel, S_fwd, S_bwd, Q) in cases
        @testset "$(file), m = $(m_rel)" begin
            agg = read_aggregate_file(joinpath(@__DIR__, "fixtures", file); scale_factor = k_medium)
            pos, rad = agg.positions, agg.radii          # shape: (3, N), (N,)
            @test maximum(rad) / minimum(rad) > 1.3       # the radii differ

            r, _ = compute_scattering(pos, rad, m_rel; use_fft = false, tol = 1e-12, max_iter = 500,
                                    phase_reference = :lab)
            @test r.converged
            @test r.truncation_order == 3                 # largest Mie order of the Fortran run

            for i in 1:2
                @test isapprox(r.S_forward[i],  S_fwd[i]; rtol = rtol_amp)
                @test isapprox(r.S_backward[i], S_bwd[i]; rtol = rtol_amp)
            end
            for i in 3:4
                @test abs(r.S_forward[i]  - S_fwd[i]) <= atol_x * abs(S_fwd[1])
                @test abs(r.S_backward[i] - S_bwd[i]) <= atol_x * abs(S_bwd[1])
            end
            @test isapprox(r.Q_ext, Q[1]; rtol = rtol_q)
            @test isapprox(r.Q_abs, Q[2]; rtol = rtol_q)
            @test isapprox(r.Q_sca, Q[3]; rtol = rtol_q)
            @test isapprox(r.Q_ext, r.Q_abs + r.Q_sca; rtol = 1e-12)
        end
    end

    # The default tolerance keeps the amplitudes and Q_ext at the level of the printed precision.
    # Q_abs = Q_ext − Q_sca is then less accurate (see the theory note, validation section).
    @testset "default tolerance" begin
        file, m_rel, S_fwd, S_bwd, Q = cases[3]
        agg = read_aggregate_file(joinpath(@__DIR__, "fixtures", file); scale_factor = k_medium)
        r, _ = compute_scattering(agg.positions, agg.radii, m_rel; use_fft = false,
                                  phase_reference = :lab)
        @test r.converged
        for i in 1:2
            @test isapprox(r.S_forward[i],  S_fwd[i]; rtol = 2e-3)
            @test isapprox(r.S_backward[i], S_bwd[i]; rtol = 2e-3)
        end
        @test isapprox(r.Q_ext, Q[1]; rtol = 2e-3)
        @test isapprox(r.Q_abs, Q[2]; rtol = 2e-3)
    end
end
