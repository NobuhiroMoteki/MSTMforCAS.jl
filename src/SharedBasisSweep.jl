"""
    SharedBasisSweep

Sweep over the refractive index of one aggregate with one Krylov basis that all refractive
indices share.

The interaction equation is (I − T(m) A) x = T(m) p. The translation operator A and the
incident coefficients p do not depend on the refractive index m. T(m) is block diagonal with
the eigenvalues −aₙ(xᵢ, m) and −bₙ(xᵢ, m) per sphere i and order n.

# Method
1. The inverse Mie coefficients are fitted by an affine function of σ(m) = 1/(m² − 1),
   1/aₙ ≈ d + μ σ and 1/bₙ ≈ d + μ σ, per sphere and order. This gives
   T⁻¹(m) ≈ −(D₀ + σ M) and the shifted system (B + σ I) x = −c with
   B = M⁻¹(D₀ + A) and c = M⁻¹ p.
2. The Krylov space of B + σ I does not depend on σ. One Arnoldi basis V per incident
   polarization therefore serves all refractive indices. It costs one application of A per
   basis vector. The products W = A V are kept.
3. For each m the coefficients y of x = V y follow from a small shifted Hessenberg problem.
   The part of 1/aₙ, 1/bₙ that the affine fit leaves is added to that problem as a few
   separable terms.
4. The true residual of the interaction equation with the exact T(m) is
   r = T(m)(p + W y) − V y. It needs no application of A.
5. The outputs are linear in x up to the common-origin coefficients. The translation to the
   common origin is applied once to every basis vector.
6. Refractive indices whose true residual does not meet the tolerance are solved by CBICG
   with x = V y as the initial guess.

All quantities are dimensionless (lengths scaled by the wavenumber in the medium).

# References
- Frommer & Glässner (1998), SIAM J. Sci. Comput. 19:15 (shifted linear systems)
- Argentin, Inzhevatkin & Yurkin (2026), arXiv:2611.15909 (refractive-index sweeps in the DDA)
- docs/theory_note.pdf, section on the shared Krylov basis
"""

export sweep_refractive_index, shared_basis_range_issues

# ─────────────────────────────────────────────────────────────
# Block-diagonal operators with one truncation order for all spheres
#
# A block with the eigenvalues (α, β) for the coefficient types (a, b) is stored as
#   diag = (α + β)/2, off = (α − β)/2,
# the layout of the T-matrix in the lr basis (α = −aₙ, β = −bₙ give T).
# dvals, ovals have shape (order, N).
# ─────────────────────────────────────────────────────────────

function _apply_blockdiag!(
    out::AbstractVector{ComplexF64},
    inp::AbstractVector{ComplexF64},
    dvals::AbstractMatrix{ComplexF64},     # shape: (order, N)
    ovals::AbstractMatrix{ComplexF64},     # shape: (order, N)
    mn_to_n::Vector{Int},
    hnb::Int
)
    N = size(dvals, 2)
    @inbounds for i in 1:N
        off = (i - 1) * 2 * hnb
        for mn in 1:hnb
            n  = mn_to_n[mn]
            d  = dvals[n, i]
            o  = ovals[n, i]
            x1 = inp[off + mn]
            x2 = inp[off + hnb + mn]
            out[off + mn]       = d * x1 + o * x2
            out[off + hnb + mn] = o * x1 + d * x2
        end
    end
    return out
end

"""
    _blockvals!(dvals, ovals, alpha, beta)

Fill the block storage from the eigenvalues `alpha` (type a) and `beta` (type b), each a
vector over the channels in the order (sphere, order): index (i − 1) * order + n.
"""
function _blockvals!(
    dvals::Matrix{ComplexF64},
    ovals::Matrix{ComplexF64},
    alpha::AbstractVector{ComplexF64},
    beta::AbstractVector{ComplexF64}
)
    @inbounds for k in eachindex(alpha)
        dvals[k] = (alpha[k] + beta[k]) / 2
        ovals[k] = (alpha[k] - beta[k]) / 2
    end
    return nothing
end

"""
    _shifted_hessenberg_solve(H, beta, sigma, K, Cs, g) -> (y, relres)

Least-squares solution of the shifted model system in the first `K` basis vectors:
y = argmin |−beta e₁ − (H_K + sigma Ī + Σ_s g[s] Cs[s]) y| with H_K of shape (K+1, K).
`relres` is the residual of this small problem divided by `beta`.
"""
function _shifted_hessenberg_solve(
    H::Matrix{ComplexF64},
    beta::Float64,
    sigma::ComplexF64,
    K::Int,
    Cs::Vector{Matrix{ComplexF64}},
    g::AbstractVector{ComplexF64}
)
    Hs = H[1:K+1, 1:K]
    @inbounds for k in 1:K
        Hs[k, k] += sigma
    end
    for s in eachindex(Cs)
        @views Hs .+= g[s] .* Cs[s][1:K+1, 1:K]
    end
    rhs = zeros(ComplexF64, K + 1)
    rhs[1] = -beta
    y = qr(Hs) \ rhs
    return y, norm(rhs .- Hs * y) / beta
end

# Grow the column capacity of a basis matrix, keeping the first `ncol` columns.
function _grow_columns(V::Matrix{ComplexF64}, ncol::Int, newcap::Int)
    Vn = Matrix{ComplexF64}(undef, size(V, 1), newcap)
    @views Vn[:, 1:ncol] .= V[:, 1:ncol]
    return Vn
end

# ─────────────────────────────────────────────────────────────
# Range in which the shared basis was validated (theory note, section on the shared basis).
# Inside it the basis alone reached relative errors below 2e-4 in the forward amplitudes and
# below 1e-3 in the backward amplitudes for aggregates of touching spheres.
# ─────────────────────────────────────────────────────────────
const _SB_VALID_MREAL = (1.30, 3.00)    # real part of the relative refractive index
const _SB_VALID_MIMAG = (0.00, 1.65)    # imaginary part
const _SB_VALID_MX    = 1.6             # largest |m| × largest monomer size parameter

"""
    shared_basis_range_issues(m_rels, x_max) -> Vector{String}

Reasons why a sweep over the relative refractive indices `m_rels` of monomers with the
largest size parameter `x_max` lies outside the range in which the shared-basis sweep was
validated. Empty if the sweep is inside it.

The validated range is 1.30 ≤ Re m ≤ 3.00, 0 ≤ Im m ≤ 1.65 and max|m| × `x_max` ≤ 1.6.
Metallic refractive indices (Re m < 1) are outside it.
"""
function shared_basis_range_issues(m_rels::AbstractVector{ComplexF64}, x_max::Float64)::Vector{String}
    issues = String[]
    mr_lo, mr_hi = extrema(real, m_rels)
    mi_lo, mi_hi = extrema(imag, m_rels)
    m_abs = maximum(abs, m_rels)
    (mr_lo >= _SB_VALID_MREAL[1] - 1e-9 && mr_hi <= _SB_VALID_MREAL[2] + 1e-9) || push!(issues,
        @sprintf("Re m = %.3g to %.3g is outside %.2f to %.2f", mr_lo, mr_hi, _SB_VALID_MREAL...))
    (mi_lo >= _SB_VALID_MIMAG[1] - 1e-9 && mi_hi <= _SB_VALID_MIMAG[2] + 1e-9) || push!(issues,
        @sprintf("Im m = %.3g to %.3g is outside %.2f to %.2f", mi_lo, mi_hi, _SB_VALID_MIMAG...))
    m_abs * x_max <= _SB_VALID_MX || push!(issues,
        @sprintf("max|m| × largest size parameter = %.3g × %.3g = %.3g is above %.1f",
                 m_abs, x_max, m_abs * x_max, _SB_VALID_MX))
    return issues
end

"""
    _merge_basis_to_origin!(A1, A2, Vs, Ks, positions, r0, order, ntrani)

Common-origin coefficients of all basis vectors: the operation of `_merge_to_origin!`
applied to every column of the bases, with the translation matrix of each sphere computed
once. `A1[:, k, q]` and `A2[:, k, q]` receive the mode-1 and mode-2 coefficients of the
basis vector `Vs[q][:, k]` for k ≤ `Ks[q]`. All spheres have the truncation order `order`.
"""
function _merge_basis_to_origin!(
    A1::Array{ComplexF64,3},              # shape: (hnb0, Kmax, 2)
    A2::Array{ComplexF64,3},              # shape: (hnb0, Kmax, 2)
    Vs::Vector{Matrix{ComplexF64}},       # bases, shape (neqns, ≥ Ks[q]) each
    Ks::Vector{Int},
    positions::Matrix{Float64},           # shape: (3, N)
    r0::Vector{Float64},                  # common origin, shape (3,)
    order::Int,
    ntrani::Vector{Int}                   # translation order cap per sphere
)
    N    = size(positions, 2)
    hnb  = order * (order + 2)
    Kmax = maximum(Ks)
    max_hnb_i = maximum(n * (n + 2) for n in ntrani)
    T1 = Matrix{ComplexF64}(undef, max_hnb_i, Kmax)
    T2 = Matrix{ComplexF64}(undef, max_hnb_i, Kmax)
    r_trans = Vector{Float64}(undef, 3)
    for i in 1:N
        off   = (i - 1) * 2 * hnb
        hnb_i = ntrani[i] * (ntrani[i] + 2)
        r_trans[1] = r0[1] - positions[1, i]
        r_trans[2] = r0[2] - positions[2, i]
        r_trans[3] = r0[3] - positions[3, i]
        dist = sqrt(r_trans[1]^2 + r_trans[2]^2 + r_trans[3]^2)
        if dist < 1e-10
            # Zero translation: the coefficients are taken as they are
            n_copy = min(hnb, hnb_i)
            for q in 1:2
                @inbounds for k in 1:Ks[q], mn in 1:n_copy
                    v1 = Vs[q][off + mn, k]
                    v2 = Vs[q][off + hnb + mn, k]
                    A1[mn, k, q] += v1 + v2
                    A2[mn, k, q] += v1 - v2
                end
            end
        else
            H = compute_translation_matrix(r_trans, ComplexF64(1.0), order, ntrani[i];
                                           use_regular=true)       # shape: (hnb_i, hnb, 2)
            for q in 1:2
                K = Ks[q]
                t1 = view(T1, 1:hnb_i, 1:K)
                t2 = view(T2, 1:hnb_i, 1:K)
                mul!(t1, view(H, 1:hnb_i, 1:hnb, 1), view(Vs[q], off+1:off+hnb, 1:K))
                mul!(t2, view(H, 1:hnb_i, 1:hnb, 2), view(Vs[q], off+hnb+1:off+2hnb, 1:K))
                @inbounds for k in 1:K, mn in 1:hnb_i
                    A1[mn, k, q] += t1[mn, k] + t2[mn, k]
                    A2[mn, k, q] += t1[mn, k] - t2[mn, k]
                end
            end
        end
    end
    return nothing
end

"""
    sweep_refractive_index(positions, radii, m_rels; kwargs...) -> (results, info)

Scattering amplitudes and efficiencies of one aggregate for many refractive indices, with
one Krylov basis shared by all of them.

# Arguments
- `positions`: sphere centers, shape (3, N), dimensionless (k_medium × position)
- `radii`: size parameters xᵢ = k_medium × radius, shape (N,)
- `m_rels`: complex refractive indices relative to the medium, any order

# Keyword arguments
- `tol`: convergence threshold of |r|²/|b|², the criterion of the CBICG solver (default 1e-6)
- `max_iter`: maximum CBICG iterations of the fallback solves (default 200)
- `use_fft`: FFT-accelerated translation (default false)
- `truncation_order`: lower bound of the VSWF truncation order. All spheres and all
  refractive indices use one order: the larger of this value and the largest single-sphere
  Mie order over the spheres and `m_rels`.
- `precomputed_fft`: FFT grid data built for that order. If `nothing` with `use_fft=true`,
  the grid is built here.
- `fft_far_order`, `fft_near_factor`: far order of the FFT grid (see `solve_tmatrix`). The
  grid carries the orders up to `max(fft_far_order, Mie order)`, where the Mie order is the
  largest over the spheres and `m_rels`.
- `basis_tol`: the basis grows until the residual of the small shifted problem is below this
  value for every refractive index (default `0.03 * sqrt(tol)`)
- `max_basis`: largest number of basis vectors per incident polarization (default 100)
- `correction_terms`: number of separable terms that represent the remainder of the affine
  fit (default 2)
- `fallback`: solve the refractive indices that do not meet `tol` in the basis by CBICG,
  started from the basis solution (default true)
- `warn_range`: warn if the sweep lies outside the validated range (default true, see
  `shared_basis_range_issues`)
- `max_mie_order`: largest single-sphere Mie order over the spheres and `m_rels`, for callers
  that have computed it already. A larger value is allowed. Default `nothing`: it is
  determined here.

# Returns
- `results::Vector{ScatteringResult}`: one entry per element of `m_rels`. `converged` is true
  if |r|²/|b|² < `tol` for both incident polarizations, with the true residual r of the
  interaction equation. `n_iterations` is the basis size, plus the CBICG iterations where the
  fallback was used.
- `info`: named tuple with `method` (`:shared_basis` or `:independent`), `truncation_order`,
  `basis_size` (per polarization), `relres` (true relative residual norm in the basis, shape
  (length(m_rels), 2)), `n_fallback`, `model_residual` (largest residual of the small
  problem), `fit_singular_values` (of the remainder of the affine fit) and `timings`
  (seconds per stage: `translation_setup`, `mie_fit`, `basis`, `basis_outputs`,
  `solve_and_check`, the last one with the fallback solves).

# Validated range
The method was validated for 1.30 ≤ Re m ≤ 3.00, 0 ≤ Im m ≤ 1.65 and
max|m| × (largest monomer size parameter) ≤ 1.6. A warning is issued outside this range.
The affine fit is accurate for small monomers. The residual that the basis can reach grows
roughly as (|m| x)⁴ to (|m| x)⁶. With `fallback=true` every result still meets `tol`,
because refractive indices that the basis does not resolve are solved by CBICG. The gain in
time is then lost for them. With `fallback=false` such results are returned with
`converged == false`. Metallic refractive indices are outside the scope of the method.

With one sphere or fewer than three refractive indices every refractive index is solved
by `compute_scattering` (`info.method == :independent`).
"""
function sweep_refractive_index(
    positions::Matrix{Float64},
    radii::Vector{Float64},
    m_rels::AbstractVector{ComplexF64};
    tol::Float64 = 1e-6,
    max_iter::Int = 200,
    use_fft::Bool = false,
    truncation_order::Union{Int,Nothing} = nothing,
    precomputed_fft::Union{FFTGridData,Nothing} = nothing,
    fft_far_order::Union{Int,Nothing} = nothing,
    fft_near_factor::Float64 = 1.5,
    basis_tol::Union{Float64,Nothing} = nothing,
    max_basis::Int = 100,
    correction_terms::Int = 2,
    fallback::Bool = true,
    warn_range::Bool = true,
    max_mie_order::Union{Int,Nothing} = nothing
)
    N  = length(radii)
    nm = length(m_rels)
    size(positions) == (3, N) || throw(ArgumentError("positions must have shape (3, N)"))
    nm >= 1 || throw(ArgumentError("m_rels must not be empty"))
    tol > 0 || throw(ArgumentError("tol must be positive"))
    max_basis >= 2 || throw(ArgumentError("max_basis must be at least 2"))
    correction_terms >= 0 || throw(ArgumentError("correction_terms must be non-negative"))
    all(m -> isfinite(m) && abs(m^2 - 1) > 1e-12, m_rels) ||
        throw(ArgumentError("m_rels must be finite and different from ±1"))
    btol = basis_tol === nothing ? 0.03 * sqrt(tol) : basis_tol
    btol > 0 || throw(ArgumentError("basis_tol must be positive"))

    # ── One truncation order for all spheres and refractive indices ─────────
    max_mie = 0
    if max_mie_order === nothing
        for m in m_rels, i in 1:N
            max_mie = max(max_mie, _mie_order(radii[i], m))
        end
    else
        max_mie = max_mie_order
    end
    order = truncation_order === nothing ? max_mie : max(truncation_order, max_mie)

    # ── Cases without a shared basis ────────────────────────────────────────
    if N < 2 || nm < 3
        results = Vector{ScatteringResult}(undef, nm)
        prev_amn = nothing
        for j in 1:nm
            results[j], prev_amn = compute_scattering(positions, radii, m_rels[j];
                tol=tol, max_iter=max_iter, use_fft=use_fft && N >= 2, truncation_order=order,
                precomputed_fft=precomputed_fft, initial_amn=prev_amn,
                fft_far_order=fft_far_order, fft_near_factor=fft_near_factor)
        end
        info = (method = :independent, truncation_order = order, basis_size = (0, 0),
                relres = fill(NaN, nm, 2), n_fallback = 0, model_residual = NaN,
                fit_singular_values = Float64[], timings = NamedTuple())
        return results, info
    end

    if warn_range
        issues = shared_basis_range_issues(m_rels, maximum(radii))
        isempty(issues) || @warn(
            "sweep_refractive_index: outside the range in which the shared-basis sweep was " *
            "validated: " * join(issues, "; ") * ". " *
            (fallback ? "Results still meet tol through the CBICG fallback." :
                        "Check `converged` and `info.relres` of the results."))
    end

    t_stage = time()
    timings = Dict{Symbol,Float64}()
    lap! = function (name::Symbol)
        now = time()
        timings[name] = get(timings, name, 0.0) + (now - t_stage)
        t_stage = now
    end

    nois       = fill(order, N)
    hnb        = order * (order + 2)
    half_nblks = fill(hnb, N)
    offsets    = [(i - 1) * 2 * hnb for i in 1:N]
    neqns      = N * 2 * hnb
    mn_to_n    = _build_mn_to_n(order)

    # ── Translation operator ────────────────────────────────────────────────
    fft_data = nothing
    if use_fft
        if precomputed_fft !== nothing
            fft_data = precomputed_fft
        else
            far_nois = fft_far_order === nothing ? nois :
                       fill(min(order, max(fft_far_order, max_mie)), N)
            fft_data = init_fft_grid(positions, radii, nois;
                                     far_nois=far_nois, near_factor=fft_near_factor)
        end
    end
    local applyA!::Function
    if fft_data !== nothing
        nx, ny, nz = fft_data.cell_dim
        nb = fft_data.nblk_node
        anode = zeros(ComplexF64, nx, ny, nz, nb, 2)
        gnode = zeros(ComplexF64, nx, ny, nz, nb, 2)
        aft   = zeros(ComplexF64, 2nx, 2ny, 2nz, nb)
        gft   = zeros(ComplexF64, 2nx, 2ny, 2nz, nb)
        applyA! = (out, inp) -> begin
            fill!(out, zero(ComplexF64))
            apply_A_fft!(out, inp, positions, fft_data, offsets, half_nblks, nois,
                         anode, gnode, aft, gft)
        end
    else
        applyA! = (out, inp) -> begin
            fill!(out, zero(ComplexF64))
            _apply_A!(out, inp, positions, offsets, half_nblks, nois)
        end
    end

    lap!(:translation_setup)

    # ── Incident coefficients (as in solve_tmatrix) ─────────────────────────
    rhs = zeros(ComplexF64, neqns, 2)                 # shape: (neqns, 2)
    p0  = _genplanewavecoef_z0(order)                 # shape: (hnb, 2, 2)
    for i in 1:N
        phase = exp(im * positions[3, i])
        off = offsets[i]
        for q in 1:2, p in 1:2, mn in 1:hnb
            rhs[off + (p - 1) * hnb + mn, q] += phase * p0[mn, p, q]
        end
    end

    # ── Inverse Mie coefficients of all channels and refractive indices ─────
    # Channel index: (i − 1) * order + n. Finv_a, Finv_b have shape (order * N, nm).
    nch    = order * N
    Finv_a = Matrix{ComplexF64}(undef, nch, nm)
    Finv_b = Matrix{ComplexF64}(undef, nch, nm)
    for j in 1:nm, i in 1:N
        a, b = compute_mie_coefficients(radii[i], m_rels[j]; nmax=order)
        for n in 1:order
            Finv_a[(i - 1) * order + n, j] = 1 / a[n]
            Finv_b[(i - 1) * order + n, j] = 1 / b[n]
        end
    end
    (all(isfinite, Finv_a) && all(isfinite, Finv_b)) || error(
        "sweep_refractive_index: a Mie coefficient of order ≤ $order vanishes in double " *
        "precision. Use a lower truncation order.")

    # ── Affine fit in σ = 1/(m² − 1) and separable form of its remainder ────
    sig = ComplexF64[1 / (m^2 - 1) for m in m_rels]
    X   = hcat(ones(ComplexF64, nm), sig)             # shape: (nm, 2)
    Xp  = pinv(X)                                     # shape: (2, nm)
    Ca  = Finv_a * transpose(Xp)                      # shape: (nch, 2): d, μ of type a
    Cb  = Finv_b * transpose(Xp)
    (all(!iszero, view(Ca, :, 2)) && all(!iszero, view(Cb, :, 2))) || error(
        "sweep_refractive_index: the affine fit is degenerate for the given refractive indices")
    R = vcat((Finv_a .- Ca * transpose(X)) ./ view(Ca, :, 2),
             (Finv_b .- Cb * transpose(X)) ./ view(Cb, :, 2))   # shape: (2 nch, nm)
    Fsvd   = svd(R)
    nterms = min(correction_terms, nm - 2, length(Fsvd.S))
    gcor   = conj.(Fsvd.V[:, 1:nterms])               # shape: (nm, nterms)

    D0d = Matrix{ComplexF64}(undef, order, N); D0o = similar(D0d)
    Mid = similar(D0d);                        Mio = similar(D0d)
    _blockvals!(D0d, D0o, view(Ca, :, 1), view(Cb, :, 1))
    _blockvals!(Mid, Mio, 1 ./ view(Ca, :, 2), 1 ./ view(Cb, :, 2))
    Ed = [Matrix{ComplexF64}(undef, order, N) for _ in 1:nterms]
    Eo = [Matrix{ComplexF64}(undef, order, N) for _ in 1:nterms]
    for s in 1:nterms
        _blockvals!(Ed[s], Eo[s], Fsvd.U[1:nch, s] .* Fsvd.S[s],
                    Fsvd.U[nch+1:2nch, s] .* Fsvd.S[s])
    end

    lap!(:mie_fit)

    # ── Arnoldi bases, one per incident polarization ────────────────────────
    Vs    = Vector{Matrix{ComplexF64}}(undef, 2)      # shape: (neqns, ≥ K+1)
    Ws    = Vector{Matrix{ComplexF64}}(undef, 2)      # shape: (neqns, ≥ K), W = A V
    Hs    = Vector{Matrix{ComplexF64}}(undef, 2)      # shape: (≥ K+1, ≥ K)
    betas = zeros(2)
    Ks    = zeros(Int, 2)
    model_residual = 0.0
    no_corr = Matrix{ComplexF64}[]
    no_g    = ComplexF64[]
    tmp = zeros(ComplexF64, neqns)
    v   = zeros(ComplexF64, neqns)
    vk  = zeros(ComplexF64, neqns)
    wk  = zeros(ComplexF64, neqns)
    kmin = min(8, max_basis)
    worst_model = function (H, beta, k)
        wmax = 0.0
        for j in 1:nm
            wmax = max(wmax, _shifted_hessenberg_solve(H, beta, sig[j], k, no_corr, no_g)[2])
        end
        return wmax
    end
    for q in 1:2
        cap = min(max_basis, 32)
        V = Matrix{ComplexF64}(undef, neqns, cap + 1)
        W = Matrix{ComplexF64}(undef, neqns, cap)
        H = zeros(ComplexF64, max_basis + 1, max_basis)
        _apply_blockdiag!(v, view(rhs, :, q), Mid, Mio, mn_to_n, hnb)
        beta = norm(v)
        @views V[:, 1] .= v ./ beta
        K = 0
        worst = Inf
        checked_at = 0
        next_check = kmin
        for k in 1:max_basis
            K = k
            if k > cap
                cap = min(max_basis, 2 * cap)
                V = _grow_columns(V, k, cap + 1)
                W = _grow_columns(W, k - 1, cap)
            end
            @views vk .= V[:, k]                                    # the operators take a Vector
            applyA!(wk, vk)
            @views W[:, k] .= wk
            _apply_blockdiag!(tmp, vk, D0d, D0o, mn_to_n, hnb)     # D₀ v
            tmp .+= wk                                              # (D₀ + A) v
            _apply_blockdiag!(v, tmp, Mid, Mio, mn_to_n, hnb)      # B v
            Vk = view(V, :, 1:k)
            for pass in 1:2                                         # Gram–Schmidt, two passes
                h = Vk' * v
                mul!(v, Vk, h, -1.0, 1.0)
                @views H[1:k, k] .+= h
            end
            hnext = norm(v)
            H[k+1, k] = hnext
            if hnext <= 1e-14 * beta
                # The Krylov space is invariant: the model solution is exact in it.
                @views V[:, k+1] .= 0
                worst = 0.0
                checked_at = k
                break
            end
            @views V[:, k+1] .= v ./ hnext
            if k >= next_check
                worst = worst_model(H, beta, k)
                checked_at = k
                worst < btol && break
                next_check = k + max(2, k ÷ 8)
            end
        end
        checked_at == K || (worst = worst_model(H, beta, K))
        Vs[q] = V; Ws[q] = W; Hs[q] = H; betas[q] = beta; Ks[q] = K
        model_residual = max(model_residual, worst)
    end

    lap!(:basis)

    # ── Projections of the correction terms: C_s = V' (M⁻¹E_s) V[:, 1:K] ────
    Cs = [Matrix{ComplexF64}[] for _ in 1:2]
    for q in 1:2
        K = Ks[q]
        Vk1 = view(Vs[q], :, 1:K+1)
        EV = Matrix{ComplexF64}(undef, neqns, K)
        for s in 1:nterms
            for k in 1:K
                _apply_blockdiag!(view(EV, :, k), view(Vs[q], :, k), Ed[s], Eo[s], mn_to_n, hnb)
            end
            push!(Cs[q], Vk1' * EV)
        end
    end

    # ── Outputs of the basis vectors ────────────────────────────────────────
    x_eff    = _x_eff(radii)
    norm_fac = 2.0 / x_eff^2
    r0 = vec(sum(positions, dims=2)) ./ N
    ntrani = Vector{Int}(undef, N)
    for i in 1:N
        ntrani[i] = _tranordertest(sqrt(sum(abs2, r0 .- view(positions, :, i))), order, 1e-5)
    end
    nodrt = maximum(ntrani)
    hnb0  = nodrt * (nodrt + 2)
    Kmax  = maximum(Ks)
    A1 = zeros(ComplexF64, hnb0, Kmax, 2)             # mode-1 coefficients of V_q[:, k]
    A2 = zeros(ComplexF64, hnb0, Kmax, 2)             # mode-2 coefficients of V_q[:, k]
    _merge_basis_to_origin!(A1, A2, Vs, Ks, positions, r0, order, ntrani)
    Vp = [view(Vs[q], :, 1:Ks[q])' * view(rhs, :, q) for q in 1:2]   # V_q' p_q

    lap!(:basis_outputs)

    # ── Every refractive index in the basis ─────────────────────────────────
    results = Vector{ScatteringResult}(undef, nm)
    relres  = zeros(nm, 2)
    Td = Matrix{ComplexF64}(undef, order, N); To = similar(Td)
    u  = zeros(ComplexF64, neqns)
    w  = zeros(ComplexF64, neqns)
    c1 = zeros(ComplexF64, hnb0, 2)
    c2 = zeros(ComplexF64, hnb0, 2)
    n_fallback = 0
    for j in 1:nm
        _blockvals!(Td, To, -1 ./ view(Finv_a, :, j), -1 ./ view(Finv_b, :, j))   # T(m_j)
        ys = Vector{Vector{ComplexF64}}(undef, 2)
        for q in 1:2
            K = Ks[q]
            ys[q], _ = _shifted_hessenberg_solve(Hs[q], betas[q], sig[j], K, Cs[q], view(gcor, j, :))
            # True residual r = T (p + W y) − V y, relative to |T p|
            mul!(w, view(Ws[q], :, 1:K), ys[q])
            @views w .+= rhs[:, q]
            _apply_blockdiag!(u, w, Td, To, mn_to_n, hnb)
            mul!(u, view(Vs[q], :, 1:K), ys[q], -1.0, 1.0)
            _apply_blockdiag!(w, view(rhs, :, q), Td, To, mn_to_n, hnb)
            relres[j, q] = norm(u) / norm(w)
        end
        ok = relres[j, 1]^2 < tol && relres[j, 2]^2 < tol
        if ok || !fallback
            Q_ext_sum = 0.0
            for q in 1:2
                K = Ks[q]
                @views mul!(c1[:, q], A1[:, 1:K, q], ys[q])
                @views mul!(c2[:, q], A2[:, 1:K, q], ys[q])
                Q_ext_sum += real(dot(ys[q], Vp[q]))
            end
            Q_ext = -norm_fac * Q_ext_sum
            Q_sca = norm_fac * (sum(abs2, c1) + sum(abs2, c2)) / 2
            sa_fwd = _amplitude_from_mode_coefs(c1, c2, nodrt, true)
            sa_bwd = _amplitude_from_mode_coefs(c1, c2, nodrt, false)
            results[j] = ScatteringResult(ntuple(i -> -2 * sa_fwd[i], 4),
                                          ntuple(i -> -2 * sa_bwd[i], 4),
                                          Q_ext, Q_ext - Q_sca, Q_sca, ok, Kmax, order)
        else
            n_fallback += 1
            x0 = Matrix{ComplexF64}(undef, neqns, 2)
            for q in 1:2
                @views mul!(x0[:, q], Vs[q][:, 1:Ks[q]], ys[q])
            end
            r, _ = compute_scattering(positions, radii, m_rels[j]; tol=tol, max_iter=max_iter,
                                      use_fft=use_fft, truncation_order=order,
                                      precomputed_fft=fft_data, initial_amn=x0)
            results[j] = ScatteringResult(r.S_forward, r.S_backward, r.Q_ext, r.Q_abs, r.Q_sca,
                                          r.converged, Kmax + r.n_iterations, r.truncation_order)
        end
    end

    lap!(:solve_and_check)

    if !fallback
        n_bad = count(r -> !r.converged, results)
        n_bad == 0 || @warn(@sprintf(
            "sweep_refractive_index: %d of %d refractive indices do not meet tol = %.1e in the shared basis (largest relative residual %.1e) and fallback is off",
            n_bad, nm, tol, maximum(relres)))
    end

    info = (method = :shared_basis, truncation_order = order, basis_size = (Ks[1], Ks[2]),
            relres = relres, n_fallback = n_fallback, model_residual = model_residual,
            fit_singular_values = Fsvd.S[1:min(5, length(Fsvd.S))],
            timings = (; (k => v for (k, v) in timings)...))
    return results, info
end
