"""
    TMatrixSolver

Solve the multi-sphere interaction equation using the Complex BiConjugate Gradient
(CBICG) iterative method in the left/right-circular (lr_tran) VSWF basis.

The interaction equation for N spheres is:
    (I - T·A) · x = p

where:
- x: scattered field expansion coefficients (global flat vector)
- T: block-diagonal single-sphere T-matrix (2×2 lr_tran Mie matrix per order n)
- A: off-diagonal translation operator (H_{ij} matrices)
- p: incident plane wave expansion coefficients (lr_tran basis, β=0, α=0)

# Coefficient representation (lr_tran basis)
For sphere i with Mie order noi, block size nblk_i = 2*noi*(noi+2):
- p=1 block (indices 1..noi*(noi+2)): "left-circular" VSWF
- p=2 block (indices noi*(noi+2)+1..nblk_i): "right-circular" VSWF
- T couples p=1 and p=2: T_n = [[-(a_n+b_n)/2, -(a_n-b_n)/2], [-(a_n-b_n)/2, -(a_n+b_n)/2]]

Multipole index: mn = n*(n+1)+m, n=1..noi, m=-n..n (1-indexed)

# Incident field (β=0, α=0 only)
Only m=±1 multipoles are non-zero in this basis for z-axis incidence.

# References
- Mackowski & Mishchenko (2011), JQSRT 112:2182
- Mackowski (1996), JOSA A 13:2266
"""

export solve_tmatrix

# ─────────────────────────────────────────────────────────────
# Helper: Mie order determination (mirrors mie_nmax from MieCoefficients.jl)
# ─────────────────────────────────────────────────────────────

"""
    _mie_order(x::Float64, m_rel::ComplexF64) -> Int

Return the converged multipole truncation order, using MSTM's mieoa criterion.
"""
function _mie_order(x::Float64, m_rel::ComplexF64)::Int
    return mie_nmax(x, m_rel)
end

# ─────────────────────────────────────────────────────────────
# Incident plane wave coefficients in lr_tran basis (β=0, α=0)
# ─────────────────────────────────────────────────────────────

"""
    _genplanewavecoef_z0(nmax::Int) -> Array{ComplexF64, 3}

Compute incident plane wave expansion coefficients in the lr_tran basis
for z-axis incidence (β=0, α=0).

Returns p0[mn, p, q] of size (nmax*(nmax+2), 2, 2) where:
- mn = n*(n+1)+m (1-indexed), n=1..nmax, m=-n..n
- p = 1 (left-circular) or 2 (right-circular)
- q = 1 or 2 (incident polarization)

Only mn with m=+1 (p=1) and m=-1 (p=2) are non-zero.
"""
function _genplanewavecoef_z0(nmax::Int)::Array{ComplexF64, 3}
    nblk = nmax * (nmax + 2)
    p0 = zeros(ComplexF64, nblk, 2, 2)

    for n in 1:nmax
        fac_n = sqrt((2n + 1) / 2.0)

        # m = +1 contributes to p=1 block
        mn_p1 = n * (n + 1) + 1    # mn for m=+1
        # m = -1 contributes to p=2 block
        mn_m1 = n * (n + 1) - 1    # mn for m=-1

        # Phases: im^(n+1) and im^(n+2)
        ip1 = im^(n + 1)
        ip2 = im^(n + 2)

        # q=1 (first incident polarization)
        p0[mn_p1, 1, 1] =  ip1 * fac_n
        p0[mn_m1, 2, 1] = -ip1 * fac_n

        # q=2 (second incident polarization)
        p0[mn_p1, 1, 2] = -ip2 * fac_n
        p0[mn_m1, 2, 2] = -ip2 * fac_n
    end

    return p0
end

# ─────────────────────────────────────────────────────────────
# T-matrix application: out = T * inp
#
# In the lr_tran basis, for a homogeneous sphere with Mie coefficients a_n (TE)
# and b_n (TM), the T-matrix is a full 2×2 matrix per multipole order n:
#
#   T_n = [ -(a_n+b_n)/2   -(a_n-b_n)/2 ]
#         [ -(a_n-b_n)/2   -(a_n+b_n)/2 ]
#
# This couples the p=1 and p=2 blocks. Derived analytically from mieoa subroutine
# (Mackowski MSTM v4.0) for the homogeneous sphere lr_tran T-matrix.
# ─────────────────────────────────────────────────────────────

"""
    _apply_T!(out, inp, mie_vecs, offsets, half_nblks, nois)

Apply the block-diagonal T-matrix to inp, storing result in out.

In the lr_tran basis, the T-matrix per sphere i and multipole order n is:
  T_n = [[-(aₙ+bₙ)/2, -(aₙ-bₙ)/2], [-(aₙ-bₙ)/2, -(aₙ+bₙ)/2]]

This couples the p=1 and p=2 VSWF blocks (Fortran: multmiecoeffmult with an1(p,q,n)).

- `mie_vecs[i]` = (a_vec, b_vec) Mie coefficients for sphere i
- `offsets[i]` = start index (0-based) of sphere i in the flat vector
- `half_nblks[i]` = noi*(noi+2) for sphere i
- `nois[i]` = Mie order for sphere i
"""
function _apply_T!(
    out::Vector{ComplexF64},
    inp::Vector{ComplexF64},
    mie_vecs::Vector{Tuple{Vector{ComplexF64}, Vector{ComplexF64}}},
    offsets::Vector{Int},
    half_nblks::Vector{Int},
    nois::Vector{Int}
)
    N = length(nois)
    for i in 1:N
        off  = offsets[i]
        hnb  = half_nblks[i]
        noi  = nois[i]
        a_v, b_v = mie_vecs[i]

        for mn in 1:hnb
            n = Int(floor(sqrt(Float64(mn))))
            t_diag = -(a_v[n] + b_v[n]) / 2
            t_off  = -(a_v[n] - b_v[n]) / 2
            inp_p1 = inp[off + mn]
            inp_p2 = inp[off + hnb + mn]
            # p=1 block: index off + mn
            out[off + mn]       = t_diag * inp_p1 + t_off  * inp_p2
            # p=2 block: index off + hnb + mn
            out[off + hnb + mn] = t_off  * inp_p1 + t_diag * inp_p2
        end
    end
end

# ─────────────────────────────────────────────────────────────
# S-parity operator: maps a_{m,n,p} → (-1)^m * a_{-m,n,p}
# Corresponds to Fortran shiftcoefficient(msign=-1, mflip=-1).
# Used in the backward BiCG pass to implement H^H = (S*H*S)^*.
# Applied in-place to both p=1 and p=2 blocks simultaneously.
# ─────────────────────────────────────────────────────────────

function _apply_S!(v::Vector{ComplexF64}, off::Int, hnb::Int, noi::Int)
    for n in 1:noi
        sign = 1
        for m in 1:n
            sign = -sign            # sign = (-1)^m
            mn_pos = n*(n+1) + m    # index for (m,  n)
            mn_neg = n*(n+1) - m    # index for (-m, n)
            # p=1 block
            t1 = v[off + mn_pos]
            v[off + mn_pos] = sign * v[off + mn_neg]
            v[off + mn_neg] = sign * t1
            # p=2 block
            t2 = v[off + hnb + mn_pos]
            v[off + hnb + mn_pos] = sign * v[off + hnb + mn_neg]
            v[off + hnb + mn_neg] = sign * t2
        end
    end
end

# ─────────────────────────────────────────────────────────────
# Translation application: out += A * inp
# For each pair (i,j), i≠j: compute H_ij on-the-fly and apply.
# ─────────────────────────────────────────────────────────────

"""
    _apply_A!(out, inp, positions, offsets, half_nblks, nois)

Apply the off-diagonal translation operator A to inp, adding result into out.

Uses `apply_translation_mvp!` to fuse translation-matrix construction with the
matrix-vector product, avoiding any per-pair heap allocation. Work buffers are
pre-allocated once per call based on the maximum multipole order.
"""
function _apply_A!(
    out::Vector{ComplexF64},
    inp::Vector{ComplexF64},
    positions::Matrix{Float64},
    offsets::Vector{Int},
    half_nblks::Vector{Int},
    nois::Vector{Int}
)
    N = length(offsets)
    nodrmax     = maximum(nois)
    wmax_global = 2 * nodrmax   # upper bound for any pair

    # Pre-allocate work buffers once (reused for every pair)
    ephim_buf = Vector{ComplexF64}(undef, 2*wmax_global + 1)
    fn_buf    = Vector{ComplexF64}(undef, wmax_global + 1)
    ymn_buf   = Matrix{Float64}(undef, 2*wmax_global + 1, wmax_global + 1)
    r_ij      = Vector{Float64}(undef, 3)

    for i in 1:N
        off_i = offsets[i]
        hnb_i = half_nblks[i]
        noi_i = nois[i]
        for j in 1:N
            i == j && continue

            off_j = offsets[j]
            hnb_j = half_nblks[j]
            noi_j = nois[j]

            r_ij[1] = positions[1, i] - positions[1, j]
            r_ij[2] = positions[2, i] - positions[2, j]
            r_ij[3] = positions[3, i] - positions[3, j]

            apply_translation_mvp!(
                out, inp, off_i, hnb_i, off_j, hnb_j,
                r_ij, noi_j, noi_i,
                ephim_buf, fn_buf, ymn_buf
            )
        end
    end
end

# ─────────────────────────────────────────────────────────────
# BiCG solver
# Solves (I - T*A) * x = p using Complex BiConjugate Gradient
# ─────────────────────────────────────────────────────────────

"""
    _solve_bicg(p, apply_L!, apply_Lstar!, norm2_p, tol, max_iter)
        -> (x, converged, iter)

Solve L·x = p using the Complex BiConjugate Gradient method.

`apply_L!(out, inp)` computes out = (I - T*A)*inp (overwrites out).
`apply_Lstar!(out, inp)` computes out = (I - A^H * T^H)*inp (overwrites out).

Uses Julia's `dot(a,b) = Σ conj(a)*b` for inner products (mirrors Fortran dot_product).
Convergence: `real(dot(r,r)) / norm2_p < tol`.
"""
function _solve_bicg(
    p::Vector{ComplexF64},
    apply_L!::Function,
    apply_Lstar!::Function,
    norm2_p::Float64,
    tol::Float64,
    max_iter::Int
)::Tuple{Vector{ComplexF64}, Bool, Int}

    neqns = length(p)
    x   = copy(p)                # initial guess: x = p
    tmp = zeros(ComplexF64, neqns)

    # r = p - L*x  (residual)
    apply_L!(tmp, x)
    r  = p .- tmp                # r = p - L*x = p - (I-TA)*x

    q  = conj.(r)                # shadow residual (BiCG choice)
    w  = copy(q)
    pv = copy(r)                 # search direction

    # Fortran CBICG uses dot_product(a,b) = Σ conj(a)*b (Hermitian).
    # csk = dot_product(conjg(cr), cr) = Σ conj(conjg(r))*r = Σ r^2 (bilinear)
    # but in Julia: dot(q, r) where q=conj(r) gives Σ conj(q)*r = Σ conj(conj(r))*r = Σ r^2 ✓
    sk = dot(q, r)

    # Check if already converged (e.g., single-sphere: r = 0 exactly)
    if real(dot(r, r)) / norm2_p < tol || abs(sk) < 1e-300
        return (x, true, 0)
    end

    converged = false
    iter = 0

    cap = zeros(ComplexF64, neqns)
    caw = zeros(ComplexF64, neqns)

    for it in 1:max_iter
        iter = it

        # cap = (I - T*A)*pv
        apply_L!(cap, pv)
        # caw = (I - A^H * T^H)*w
        apply_Lstar!(caw, w)

        denom = dot(w, cap)
        if abs(denom) < 1e-300
            # breakdown — return what we have
            break
        end
        alpha = sk / denom

        x .+= alpha .* pv
        r .-= alpha .* cap
        q .-= conj(alpha) .* caw

        sk2  = dot(q, r)
        eerr = real(dot(r, r)) / norm2_p

        if eerr < tol
            converged = true
            break
        end

        beta = sk2 / sk
        sk   = sk2
        pv   .= r .+ beta .* pv
        w    .= q .+ conj(beta) .* w
    end

    return (x, converged, iter)
end

# ─────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────

"""
    solve_tmatrix(
        positions::Matrix{Float64},
        radii::Vector{Float64},
        m_rel::ComplexF64;
        tol::Float64 = 1e-6,
        max_iter::Int = 200,
        normalize_error::Bool = true
    ) -> (amn, converged, n_iter)

Solve the multi-sphere T-matrix interaction equation using CBICG.

# Arguments
- `positions`: [3, N] sphere center coordinates, already dimensionless (scaled by k_medium)
- `radii`: [N] sphere radii, already dimensionless (size parameters x_i = k_medium * r_i)
- `m_rel`: complex refractive index ratio (sphere / medium)

# Returns
- `amn`: Matrix{ComplexF64} of size (neqns, 2), solution coefficients for polarizations q=1,2.
  Row ordering: for sphere i, p=1 block (mn=1..half_nblk_i) then p=2 block (mn=1..half_nblk_i),
  concatenated over all spheres in order i=1..N.
- `converged`: true if both polarizations converged within tolerance
- `n_iter`: maximum iteration count used across both polarizations
"""
function solve_tmatrix(
    positions::Matrix{Float64},
    radii::Vector{Float64},
    m_rel::ComplexF64;
    tol::Float64 = 1e-6,
    max_iter::Int = 200,
    normalize_error::Bool = true,
    use_fft::Bool = false
)::Tuple{Matrix{ComplexF64}, Bool, Int}

    N = length(radii)
    @assert size(positions, 1) == 3
    @assert size(positions, 2) == N

    # ── Step 1: Mie orders and offsets ──────────────────────────────────────
    nois       = Vector{Int}(undef, N)
    half_nblks = Vector{Int}(undef, N)
    nblks      = Vector{Int}(undef, N)
    offsets    = Vector{Int}(undef, N)   # 0-based offset in the global flat vector

    for i in 1:N
        nois[i]       = _mie_order(radii[i], m_rel)
        half_nblks[i] = nois[i] * (nois[i] + 2)
        nblks[i]      = 2 * half_nblks[i]
    end
    offsets[1] = 0
    for i in 2:N
        offsets[i] = offsets[i-1] + nblks[i-1]
    end
    neqns = offsets[N] + nblks[N]

    # ── Step 2: Mie coefficients ─────────────────────────────────────────────
    mie_vecs = Vector{Tuple{Vector{ComplexF64}, Vector{ComplexF64}}}(undef, N)
    for i in 1:N
        mie_vecs[i] = compute_mie_coefficients(radii[i], m_rel)
    end

    # ── Step 3: Incident plane wave coefficients ─────────────────────────────
    # Build global RHS vector for each incident polarization q=1,2.
    # Phase shift at sphere i: multiply by exp(im * k * z_i) = exp(im * z_i)
    # (positions already dimensionless, k_medium = 1)

    # Find nmax overall for p0 array (each sphere uses its own noi)
    # We generate p0 per sphere with that sphere's noi, then place into global vector.

    rhs = zeros(ComplexF64, neqns, 2)  # rhs[:,q] = incident coefficients for pol q

    for i in 1:N
        noi  = nois[i]
        hnb  = half_nblks[i]
        off  = offsets[i]
        z_i  = positions[3, i]
        phase = exp(im * z_i)

        p0_i = _genplanewavecoef_z0(noi)  # (hnb, 2, 2)

        for q in 1:2
            for p in 1:2
                p_blk_off = (p - 1) * hnb
                for mn in 1:hnb
                    rhs[off + p_blk_off + mn, q] += phase * p0_i[mn, p, q]
                end
            end
        end
    end

    # ── Step 5: Build operator functions ────────────────────────────────────

    # Temporary work arrays (reused to avoid allocation in hot loop)
    tmp_T  = zeros(ComplexF64, neqns)
    tmp_A  = zeros(ComplexF64, neqns)

    # FFT translation setup (if enabled)
    fft_data = nothing
    anode_buf = Array{ComplexF64}(undef, 0, 0, 0, 0, 0)
    gnode_buf = Array{ComplexF64}(undef, 0, 0, 0, 0, 0)
    if use_fft && N >= 2
        fft_data = init_fft_grid(positions, radii, nois)
        nx, ny, nz = fft_data.cell_dim
        nb = fft_data.nblk_node
        anode_buf = zeros(ComplexF64, nx, ny, nz, nb, 2)
        gnode_buf = zeros(ComplexF64, nx, ny, nz, nb, 2)
    end

    # Unified A operator: FFT or direct
    function _apply_A_unified!(out_vec, inp_vec)
        if fft_data !== nothing
            apply_A_fft!(out_vec, inp_vec, positions, fft_data,
                         offsets, half_nblks, nois, anode_buf, gnode_buf)
        else
            _apply_A!(out_vec, inp_vec, positions, offsets, half_nblks, nois)
        end
    end

    # apply_L!(out, inp): out = (I - T*A)*inp
    function apply_L!(out::Vector{ComplexF64}, inp::Vector{ComplexF64})
        fill!(tmp_A, zero(ComplexF64))
        _apply_A_unified!(tmp_A, inp)
        fill!(tmp_T, zero(ComplexF64))
        _apply_T!(tmp_T, tmp_A, mie_vecs, offsets, half_nblks, nois)
        @. out = inp - tmp_T
    end

    # apply_Lstar!(out, inp): out = (I - H^H * T^H)*inp
    # Algorithm:
    #   1. ain_t = conj(inp)
    #   2. ain_t = T * ain_t   (T = T^T for diagonal homogeneous spheres)
    #   3. Apply S to each sphere block in ain_t
    #   4. aout_t = H * ain_t  (standard forward translation)
    #   5. Apply S to each sphere block in aout_t
    #   6. out = inp - conj(aout_t)
    tmp_cT = zeros(ComplexF64, neqns)
    tmp_cA = zeros(ComplexF64, neqns)

    function apply_Lstar!(out::Vector{ComplexF64}, inp::Vector{ComplexF64})
        # Step 1-2: T * conj(inp)
        conj_inp = conj.(inp)
        fill!(tmp_cT, zero(ComplexF64))
        _apply_T!(tmp_cT, conj_inp, mie_vecs, offsets, half_nblks, nois)
        # Step 3: apply S in-place to tmp_cT
        for i in 1:N
            _apply_S!(tmp_cT, offsets[i], half_nblks[i], nois[i])
        end
        # Step 4: H * (S * T * conj(inp))
        fill!(tmp_cA, zero(ComplexF64))
        _apply_A_unified!(tmp_cA, tmp_cT)
        # Step 5: apply S in-place to tmp_cA
        for i in 1:N
            _apply_S!(tmp_cA, offsets[i], half_nblks[i], nois[i])
        end
        # Step 6: out = inp - conj(S*H*S * T * conj(inp)) = (I - H^H T^H) * inp
        @. out = inp - conj(tmp_cA)
    end

    # ── Step 6: Apply T to incident RHS: actual RHS = T * p_inc ─────────────
    T_rhs = zeros(ComplexF64, neqns, 2)
    for q in 1:2
        tmp_rhs_q = rhs[:, q]
        tmp_Trhs_q = zeros(ComplexF64, neqns)
        _apply_T!(tmp_Trhs_q, tmp_rhs_q, mie_vecs, offsets, half_nblks, nois)
        T_rhs[:, q] .= tmp_Trhs_q
    end

    # ── Step 7: Solve for each polarization ─────────────────────────────────
    amn = zeros(ComplexF64, neqns, 2)
    converged_both = true
    n_iter_max = 0

    for q in 1:2
        p_vec    = T_rhs[:, q]
        norm2_p  = real(dot(p_vec, p_vec))

        if norm2_p == 0.0
            # Trivial: zero RHS → zero solution
            amn[:, q] .= zero(ComplexF64)
            continue
        end

        x_sol, conv, iters = _solve_bicg(
            p_vec, apply_L!, apply_Lstar!, norm2_p, tol, max_iter
        )
        amn[:, q]      .= x_sol
        converged_both  = converged_both && conv
        n_iter_max      = max(n_iter_max, iters)
    end

    return (amn, converged_both, n_iter_max)
end
