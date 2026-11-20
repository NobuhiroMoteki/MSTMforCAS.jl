# Truncation order for near-touching monomers (two-sphere preliminary calculation).

@testset "TruncationOrder" begin
    x     = 0.113                       # size parameter (15 nm radius at 834 nm)
    m_lo  = complex(1.55, 0.15)
    m_mid = complex(1.95, 0.75)
    m_hi  = complex(2.40, 1.40)

    @testset "basic properties" begin
        r2 = contact_order_scan(x, m_mid; eps = 1e-2)
        r3 = contact_order_scan(x, m_mid; eps = 1e-3)
        @test r2.reference_ok && r3.reference_ok
        @test r2.orders[1] == MSTMforCAS.mie_nmax(x, m_mid)
        @test r2.order >= r2.orders[1]
        @test r3.order >= r2.order
        @test r2.tail <= 0.25 * 1e-2
        @test r3.tail <= 0.25 * 1e-3
        @test length(r3.errors) == length(r3.orders)
        @test contact_truncation_order(x, m_mid; eps = 1e-2) == r2.order
        @test_throws ArgumentError contact_order_scan(x, m_mid; eps = 0.0)
        @test_throws ArgumentError contact_order_scan(x, m_mid; gap = -0.1)
    end

    @testset "required order grows with refractive index" begin
        @test contact_truncation_order(x, m_lo;  eps = 1e-2) == 2
        @test contact_truncation_order(x, m_mid; eps = 1e-2) == 4
        @test contact_truncation_order(x, m_hi;  eps = 1e-2) == 8
        @test contact_truncation_order(x, m_lo;  eps = 1e-3) == 5
        @test contact_truncation_order(x, m_mid; eps = 1e-3) == 9
    end

    @testset "nearly independent of size parameter for small monomers" begin
        n_small = contact_truncation_order(0.05,  m_mid; eps = 1e-3)
        n_large = contact_truncation_order(0.208, m_mid; eps = 1e-3)
        @test abs(n_small - n_large) <= 1
    end

    @testset "separated spheres need a lower order" begin
        @test contact_truncation_order(x, m_hi; eps = 1e-2, gap = 0.1) <
              contact_truncation_order(x, m_hi; eps = 1e-2, gap = 0.0)
    end

    @testset "error estimate is conservative against a high-order reference" begin
        eps = 1e-3
        r = contact_order_scan(x, m_mid; eps = eps)
        s_ref = MSTMforCAS._dimer_forward(x, m_mid, 0.0, 24)      # truncation error ~1e-5
        true_err(n) = begin
            s = MSTMforCAS._dimer_forward(x, m_mid, 0.0, n)
            max(abs(s[1] - s_ref[1]) / abs(s_ref[1]), abs(s[2] - s_ref[2]) / abs(s_ref[2]))
        end
        for (k, n) in enumerate(r.orders)
            @test true_err(n) <= r.errors[k]
        end
        @test true_err(r.order) < eps
        @test true_err(r.orders[1]) > 10 * eps        # single-sphere Mie order is far from converged
    end
end
