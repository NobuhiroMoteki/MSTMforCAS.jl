# FFT translation with far orders below the sphere orders (init_fft_grid: far_nois, near_factor).

using LinearAlgebra

"""
Deterministic cluster of `n` touching spheres with radii near `r` (dimensionless size
parameters). Every sphere touches the sphere it was attached to. No sphere overlaps another.
"""
function _touching_cluster(n::Int, r::Float64)
    state = Ref(UInt64(12345))
    function nextu()::Float64                       # linear congruential generator in [0, 1)
        state[] = state[] * 0x5851f42d4c957f2d + 0x14057b7ef767814f
        return Float64(state[] >> 11) / 2.0^53
    end
    radii = [r * (0.9 + 0.2 * nextu()) for _ in 1:n]
    pos = zeros(Float64, 3, n)                      # shape: (3, n)
    count = 1
    while count < n
        parent = 1 + floor(Int, nextu() * count)
        u = 2 * nextu() - 1
        phi = 2pi * nextu()
        s = sqrt(1 - u^2)
        c = pos[:, parent] .+ (radii[parent] + radii[count+1]) .* [s * cos(phi), s * sin(phi), u]
        free = all(norm(c .- pos[:, j]) >= (radii[j] + radii[count+1]) * (1 - 1e-12) for j in 1:count)
        free || continue
        count += 1
        pos[:, count] = c
    end
    return pos, radii
end

@testset "FFT translation: far orders" begin
    init_fft_grid = MSTMforCAS.init_fft_grid
    N = 60
    pos, rad = _touching_cluster(N, 0.113)
    n_full, n_far = 5, 2
    hnb, hnb_f = n_full * (n_full + 2), n_far * (n_far + 2)
    nois = fill(n_full, N)
    far  = fill(n_far, N)
    near_factor = 1.5

    # layout of a coefficient vector: sphere i, p-block p, index mn
    idx(i, p, mn, h) = (i - 1) * 2h + (p - 1) * h + mn
    # The helpers declare their variables `local`: a closure would otherwise assign to a
    # variable of the same name in the enclosing test set.
    function apply_fft(f, x, orders)
        local h = [o * (o + 2) for o in orders]
        local offs = [sum(2 .* h[1:i-1]; init = 0) for i in 1:N]
        local (nx, ny, nz) = f.cell_dim
        local nb = f.nblk_node
        local out = zeros(ComplexF64, length(x))
        MSTMforCAS.apply_A_fft!(out, x, pos, f, offs, h, orders,
                                zeros(ComplexF64, nx, ny, nz, nb, 2), zeros(ComplexF64, nx, ny, nz, nb, 2),
                                zeros(ComplexF64, 2nx, 2ny, 2nz, nb), zeros(ComplexF64, 2nx, 2ny, 2nz, nb))
        return out
    end
    # restriction to / embedding of the orders up to the far order
    function lower(x)
        local y = zeros(ComplexF64, N * 2hnb_f)
        for i in 1:N, p in 1:2, mn in 1:hnb_f
            y[idx(i, p, mn, hnb_f)] = x[idx(i, p, mn, hnb)]
        end
        return y
    end
    function embed(x)
        local y = zeros(ComplexF64, N * 2hnb)
        for i in 1:N, p in 1:2, mn in 1:hnb_f
            y[idx(i, p, mn, hnb)] = x[idx(i, p, mn, hnb_f)]
        end
        return y
    end
    v = [cis(0.37 * q) * (1 + 0.5 * sin(1.3 * q)) for q in 1:(N * 2hnb)]   # deterministic test vector

    f_full = init_fft_grid(pos, rad, nois)
    f_far  = init_fft_grid(pos, rad, far)
    f_dec  = init_fft_grid(pos, rad, nois; far_nois = far, near_factor = near_factor)
    is_near(i, j, nf) = norm(pos[:, i] .- pos[:, j]) <= nf * (rad[i] + rad[j])

    @testset "grid follows the far orders" begin
        @test f_dec.cell_dim == f_far.cell_dim
        @test f_dec.node_order == f_far.node_order
        @test f_dec.node_order < f_full.node_order
        @test prod(f_dec.cell_dim) > 27                 # some pairs go through the grid
        @test all(size(h, 2) == hnb_f for h in f_dec.sphere_node_H)
        @test sizeof(f_dec.cell_tran_fft_p1) < sizeof(f_full.cell_tran_fft_p1)
        @test f_dec.far_nois == far
    end

    @testset "default far orders leave the operator unchanged" begin
        f_same = init_fft_grid(pos, rad, nois; far_nois = copy(nois), near_factor = 7.0)
        @test f_same.neighbor_pairs == f_full.neighbor_pairs
        @test apply_fft(f_same, v, nois) == apply_fft(f_full, v, nois)
    end

    @testset "orders up to the far order: FFT operator of that order" begin
        w = lower(v)
        @test lower(apply_fft(f_dec, embed(w), nois)) ≈ apply_fft(f_far, w, far) rtol = 1e-12
    end

    @testset "orders above the far order: direct coupling of near pairs only" begin
        # expected = FFT operator at the far order + remaining blocks of the near pairs
        expected = embed(apply_fft(f_far, lower(v), far))
        n_near = 0
        n_near_outside = 0
        for i in 1:N, j in 1:N
            (i == j || !is_near(i, j, near_factor)) && continue
            n_near += 1
            n_near_outside += !(j in f_full.neighbor_pairs[i])
            H = compute_translation_matrix(pos[:, i] .- pos[:, j], ComplexF64(1.0), n_full, n_full;
                                           use_regular = false)       # shape: (hnb, hnb, 2)
            H[1:hnb_f, 1:hnb_f, :] .= 0
            for p in 1:2, kl in 1:hnb, mn in 1:hnb
                expected[idx(i, p, kl, hnb)] += H[kl, mn, p] * v[idx(j, p, mn, hnb)]
            end
        end
        @test n_near >= 2 * (N - 1)                     # every sphere touches its parent
        @test apply_fft(f_dec, v, nois) ≈ expected rtol = 1e-12
        @test sum(length, f_dec.neighbor_pairs) == sum(length, f_full.neighbor_pairs) + n_near_outside
    end

    @testset "all pairs near: orders above the far order are exact" begin
        f_all = init_fft_grid(pos, rad, nois; far_nois = far, near_factor = 1e3)
        v_high = v .- embed(lower(v))                   # only orders above the far order
        direct = zeros(ComplexF64, length(v))
        offs = [(i - 1) * 2hnb for i in 1:N]
        MSTMforCAS._apply_A!(direct, v_high, pos, offs, fill(hnb, N), nois)
        @test apply_fft(f_all, v_high, nois) ≈ direct rtol = 1e-10
    end

    @testset "argument checks" begin
        @test_throws ArgumentError init_fft_grid(pos, rad, nois; far_nois = fill(n_full + 1, N))
        @test_throws ArgumentError init_fft_grid(pos, rad, nois; far_nois = fill(0, N))
        @test_throws ArgumentError init_fft_grid(pos, rad, nois; far_nois = fill(n_far, N - 1))
        @test_throws ArgumentError init_fft_grid(pos, rad, nois; far_nois = far, near_factor = -1.0)
    end

    @testset "solver: fft_far_order" begin
        m = ComplexF64(1.95, 0.75)
        kw = (tol = 1e-12, max_iter = 500)
        fwd_err(a, b) = maximum(abs(a.S_forward[q] - b.S_forward[q]) / abs(b.S_forward[q]) for q in 1:2)
        direct, _ = compute_scattering(pos, rad, m; kw..., truncation_order = n_full)
        mie, _    = compute_scattering(pos, rad, m; kw...)
        @test fwd_err(mie, direct) > 1e-2                # the Mie order alone is not enough here
        # The error of the decoupled operator is the error of the FFT at the far order.
        for (far_order, tol_s) in ((0, 1.2e-2), (4, 2e-3))
            dec, _ = compute_scattering(pos, rad, m; kw..., truncation_order = n_full, use_fft = true,
                                        fft_far_order = far_order)
            n_grid = max(far_order, MSTMforCAS.mie_nmax(maximum(rad), m))
            d_low, _ = compute_scattering(pos, rad, m; kw..., truncation_order = n_grid)
            f_low, _ = compute_scattering(pos, rad, m; kw..., truncation_order = n_grid, use_fft = true)
            @test dec.converged
            @test dec.truncation_order == n_full
            @test fwd_err(dec, direct) < tol_s
            @test fwd_err(dec, direct) < 1.5 * fwd_err(f_low, d_low) + 2e-4
            @test isapprox(dec.Q_abs, direct.Q_abs; rtol = 5e-3)
        end
        # far order at or above the sphere order: same result as without the option
        full, _ = compute_scattering(pos, rad, m; kw..., truncation_order = n_full, use_fft = true)
        same, _ = compute_scattering(pos, rad, m; kw..., truncation_order = n_full, use_fft = true,
                                     fft_far_order = n_full + 2)
        @test same.S_forward == full.S_forward && same.Q_ext == full.Q_ext
        # no truncation_order: sphere orders equal the Mie orders, so far order 0 changes nothing
        a, _ = compute_scattering(pos, rad, m; use_fft = true)
        b, _ = compute_scattering(pos, rad, m; use_fft = true, fft_far_order = 0)
        @test a.S_forward == b.S_forward && a.Q_ext == b.Q_ext
    end
end
