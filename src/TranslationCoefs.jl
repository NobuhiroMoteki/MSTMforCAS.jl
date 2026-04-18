"""
    TranslationCoefs

VSWF translation matrix coefficients for multi-sphere T-matrix method.

Implements the `gentranmatrix` algorithm from Mackowski's MSTM v4.0 Fortran code,
which computes the full translation matrix for each sphere pair using the
Clebsch-Gordan coefficient (`vcfunc`) approach.

# Algorithm reference
Mackowski & Mishchenko (2011), JQSRT 112:2182
Xu (1996), J. Comput. Phys. 127:285

# Index convention (model=1)
Source multipole index:  mn = n*(n+1)+m  for n=1,...,nmax_s, m=-n,...,n
Target multipole index:  kl = l*(l+1)+k  for l=1,...,nmax_t, k=-l,...,l
Both run from 1 to nmax*(nmax+2).

# Output
H[kl, mn, p] of size (nmax_t*(nmax_t+2), nmax_s*(nmax_s+2), 2)
where p=1,2 are the two incident polarization cases.
"""

export compute_translation_matrix, apply_translation_mvp!

# ─────────────────────────────────────────────────────────────
# Module-level lazy cache for precomputed arrays
# ─────────────────────────────────────────────────────────────

mutable struct _NumConstants
    nmax::Int
    fnr::Vector{Float64}   # fnr[n+1] = sqrt(n),  n = 0, 1, 2, ...
    bcof::Matrix{Float64}  # bcof[n+1, l+1] = sqrt((n+l)!/(n!l!))
end

mutable struct _TranCoefCache
    nmax::Int
    # tran_coef[mn, kl, w+1] : Real Fortran→ Julia translation
    # Stored as in Fortran gentrancoefconstants (source mn first, target kl second).
    # Accessed as tran_coef[kl, mn, w+1] in compute_translation_matrix,
    # which is equivalent because the array is symmetric in its first two indices.
    tran_coef::Array{Float64, 3}
end

const _NC  = Ref{_NumConstants}()
const _TCC = Ref{_TranCoefCache}()

# ─────────────────────────────────────────────────────────────
# bcof and fnr initialisation  (mirrors Fortran `init`)
# ─────────────────────────────────────────────────────────────

function _ensure_numconstants(notd::Int)
    if isassigned(_NC) && _NC[].nmax >= notd
        return _NC[]
    end
    nbc = 6*notd + 6
    fnr  = zeros(Float64, 2*nbc + 2)   # fnr[n+1] = sqrt(n)
    for n in 1:2*nbc
        fnr[n+1] = sqrt(Float64(n))
    end
    bcof = zeros(Float64, nbc+1, nbc+1)  # bcof[n+1, l+1] = sqrt(C(n+l,l))
    bcof[1,1] = 1.0
    for n in 0:nbc-1
        for l in n+1:nbc
            bcof[n+1, l+1] = fnr[n+l+1] * bcof[n+1, l] / fnr[l+1]
            bcof[l+1, n+1] = bcof[n+1, l+1]
        end
        bcof[n+2, n+2] = fnr[2*(n+1)+1] * fnr[2*(n+1)] * bcof[n+1, n+1] / fnr[n+2] / fnr[n+2]
    end
    _NC[] = _NumConstants(notd, fnr, bcof)
    return _NC[]
end

# ─────────────────────────────────────────────────────────────
# vcfunc – vector coupling (Gaunt/Clebsch-Gordan) coefficients
# Mirrors Fortran `vcfunc` + `vcfuncuprec`
# Returns vcn[1:n+l+1] where vcn[w+1] = VC coefficient at rank w
# ─────────────────────────────────────────────────────────────

# Safe bcof accessor: returns 0 when any index is out of bounds
# (physically: binomial coefficient is 0 for negative arguments)
@inline function _bcof_safe(bcof::Matrix{Float64}, a::Int, b::Int)::Float64
    (a < 1 || b < 1 || a > size(bcof,1) || b > size(bcof,2)) && return 0.0
    return @inbounds bcof[a, b]
end

function _vcfuncuprec!(vcn::Vector{Float64}, m::Int, n::Int, k::Int, l::Int,
                       wmax_up::Int, fnr, bcof)
    mk = m + k
    nl = abs(n - l)
    wmin = max(nl, abs(mk))
    w0   = wmin

    # Starting value
    if nl >= abs(mk)
        if n >= l
            m1, n1, l1, k1 = m, n, l, k
        else
            m1, n1, l1, k1 = k, l, m, n
        end
        b1 = _bcof_safe(bcof, l1+k1+1, w0-m1-k1+1)
        b2 = _bcof_safe(bcof, l1-k1+1, w0+m1+k1+1)
        b3 = _bcof_safe(bcof, 2l1+1,   2w0+2)
        vc1 = (b3 == 0.0) ? 0.0 : (-1)^(k1+l1) * b1 * b2 / b3
    else
        if mk >= 0
            b1 = _bcof_safe(bcof, n-l+w0+1, l-k+1)
            b2 = _bcof_safe(bcof, l-n+w0+1, n-m+1)
            b3 = _bcof_safe(bcof, 2w0+2,    n+l-w0+1)
            vc1 = (b3 == 0.0) ? 0.0 : (-1)^(n+m) * b1 * b2 / b3
        else
            b1 = _bcof_safe(bcof, n-l+w0+1, l+k+1)
            b2 = _bcof_safe(bcof, l-n+w0+1, n+m+1)
            b3 = _bcof_safe(bcof, 2w0+2,    n+l-w0+1)
            vc1 = (b3 == 0.0) ? 0.0 : (-1)^(l+k) * b1 * b2 / b3
        end
    end
    vcn[w0+1] = vc1

    wmax_up <= w0 && return

    # First upward step: w0 → w0+1
    w = w0 + 1
    if w <= wmax_up
        t1 = 2w * fnr[2w+2] * fnr[2w] /
             (fnr[w+mk+1] * fnr[w-mk+1] * fnr[n-l+w+1] *
              fnr[l-n+w+1] * fnr[n+l-w+2] * fnr[n+l+w+2])
        t2 = (w0 == 0) ? 0.5*(m-k) :
             Float64((m-k)*w*(w-1) - mk*n*(n+1) + mk*l*(l+1)) / Float64(2*w*(w-1))
        vcn[w+1] = t1 * t2 * vcn[w0+1]
    end

    # Continue upward
    for w in w0+2:wmax_up
        t1 = 2w * fnr[2w+2] * fnr[2w] /
             (fnr[w+mk+1] * fnr[w-mk+1] * fnr[n-l+w+1] *
              fnr[l-n+w+1] * fnr[n+l-w+2] * fnr[n+l+w+2])
        t2 = Float64((m-k)*w*(w-1) - mk*n*(n+1) + mk*l*(l+1)) / Float64(2*w*(w-1))
        t3 = fnr[w-mk] * fnr[w+mk] * fnr[l-n+w] * fnr[n-l+w] *
             fnr[n+l-w+3] * fnr[n+l+w+1] /
             (2.0*(w-1) * fnr[2w-2] * fnr[2w])
        vcn[w+1] = t1 * (t2*vcn[w] - t3*vcn[w-1+1])
    end
end

function _vcfunc(m::Int, n::Int, k::Int, l::Int, fnr, bcof)::Vector{Float64}
    wmax = n + l
    vcn  = zeros(Float64, wmax + 1)   # vcn[w+1] for w = 0,...,wmax
    mk   = m + k
    wmin = max(abs(n-l), abs(mk))
    wmin > wmax && return vcn

    # ── Starting values ──
    vcn[wmax+1] = bcof[n+m+1, l+k+1] * bcof[n-m+1, l-k+1] / bcof[2n+1, 2l+1]
    wmin == wmax && return vcn

    vcn[wmax] = vcn[wmax+1] * (l*m - k*n) * fnr[2*(n+l)] /
                (fnr[l+1] * fnr[n+1] * fnr[n+l+mk+1] * fnr[n+l-mk+1])
    wmin >= wmax-1 && return vcn

    # ── Downward recurrence from wmax to wmin+2 ──
    vcmax   = abs(vcn[wmax+1]) + abs(vcn[wmax])
    w_break = -1          # if early exit, upward recurrence fills from here
    for w in wmax:-1:wmin+2
        t1 = 2w * fnr[2w+2] * fnr[2w] /
             (fnr[w+mk+1] * fnr[w-mk+1] * fnr[n-l+w+1] *
              fnr[l-n+w+1] * fnr[n+l-w+2] * fnr[n+l+w+2])
        t2 = Float64((m-k)*w*(w-1) - mk*n*(n+1) + mk*l*(l+1)) / Float64(2*w*(w-1))
        t3 = fnr[w-mk] * fnr[w+mk] * fnr[l-n+w] * fnr[n-l+w] *
             fnr[n+l-w+3] * fnr[n+l+w+1] /
             (2.0*(w-1) * fnr[2w-2] * fnr[2w])
        vcn[w-1] = (t2*vcn[w] - vcn[w+1]/t1) / t3   # fills vcn[w-2+1]

        if (wmax - w) % 2 == 1
            vctest = abs(vcn[w-1]) + abs(vcn[w])
            vcmax  = max(vcmax, vctest)
            if vctest < 0.01 * vcmax
                w_break = w - 3
                break
            end
        end
    end

    # ── If early exit, switch to upward recurrence ──
    if w_break > wmin
        _vcfuncuprec!(vcn, m, n, k, l, w_break, fnr, bcof)
    end

    return vcn
end

# ─────────────────────────────────────────────────────────────
# gentrancoefconstants – precompute tran_coef array
# Mirrors Fortran subroutine of the same name
# ─────────────────────────────────────────────────────────────

function _init_tran_coef(nodrmax::Int, fnr, bcof)::Array{Float64, 3}
    ntot = nodrmax * (nodrmax + 2)
    tc   = zeros(Float64, 2*nodrmax + 1, ntot, ntot)  # (w+1, idx1, idx2) — stride-1 in w

    for l in 1:nodrmax
        for n in 1:nodrmax
            wmax = n + l
            vc2  = _vcfunc(-1, n, 1, l, fnr, bcof)
            c    = -(1im)^mod(n-l, 4) * fnr[2n+2] * fnr[2l+2]   # -i^{n-l}√(2n+1)√(2l+1)
            for k in -l:l
                kl = l*(l+1) + k
                for m in -n:n
                    mn  = n*(n+1) + m
                    m1m = iseven(m) ? 1 : -1
                    v   = k - m
                    vc1 = _vcfunc(-m, n, k, l, fnr, bcof)
                    wmin = max(abs(v), abs(n-l))
                    for w in wmin:wmax
                        a = (1im)^mod(w, 4) * c * m1m * vc1[w+1] * vc2[w+1]
                        if (wmax - w) % 2 == 0
                            tc[w+1, mn, kl] = real(a)
                        else
                            tc[w+1, mn, kl] = imag(a)
                        end
                    end
                end
            end
        end
    end
    return tc
end

function _ensure_tran_coef(nodrmax::Int, fnr, bcof)
    if isassigned(_TCC) && _TCC[].nmax >= nodrmax
        return _TCC[]
    end
    tc = _init_tran_coef(nodrmax, fnr, bcof)
    _TCC[] = _TranCoefCache(nodrmax, tc)
    return _TCC[]
end

# ─────────────────────────────────────────────────────────────
# Spherical Hankel functions h_n^(1)(z)
# Mirrors the inline computation in Fortran gentranmatrix
# Returns hn[n+1] = h_n^(1)(z) for n = 0,...,nmax
# ─────────────────────────────────────────────────────────────

function _hankel_upward(nmax::Int, z::ComplexF64)::Vector{ComplexF64}
    hn = Vector{ComplexF64}(undef, nmax + 1)
    hn[1] = -im * exp(im*z) / z                      # h_0^(1)
    nmax == 0 && return hn
    hn[2] = -exp(im*z) * (im + z) / z^2              # h_1^(1)
    for n in 2:nmax
        hn[n+1] = Float64(2n-1)/z * hn[n] - hn[n-1]
    end
    return hn
end

# In-place version (no allocation): fills buf[1..nmax+1]
function _hankel_upward!(buf::Vector{ComplexF64}, nmax::Int, z::ComplexF64)
    eiz = exp(im*z)
    buf[1] = -im * eiz / z
    nmax == 0 && return
    buf[2] = -eiz * (im + z) / z^2
    for n in 2:nmax
        buf[n+1] = Float64(2n-1)/z * buf[n] - buf[n-1]
    end
end

# Regular spherical Bessel jₙ(z) by Miller DOWNWARD recurrence (Wiscombe 1980).
#
# jₙ is the minimal (decaying) solution of the three-term recurrence
#   jₙ₋₁ + jₙ₊₁ = (2n+1)/z · jₙ,
# so the forward direction amplifies the growing solution yₙ at rate (2n-1)/z
# and destroys jₙ once n ≳ |z|.  In the translation matrix we evaluate
# jₙ(k·|rᵢⱼ|) at n up to ≈ nois[i] + ntrani[i] ≈ 2·truncation_order with
# k·|rᵢⱼ| ~ O(x_sphere), so for sub-wavelength aggregates the upward form
# is catastrophically unstable.  (See Bohren & Huffman 1983 App. A p.478,
# Wiscombe 1980 Applied Optics 19:1505; identical mechanism to the ψₙ fix
# in MieCoefficients.jl.)  Seed jₙstart+1 = 0, jₙstart = 1, recur down,
# normalise by j₀(z) = sin(z)/z.
function _bessel_upward(nmax::Int, z::ComplexF64)::Vector{ComplexF64}
    nstart = max(nmax, ceil(Int, abs(z))) + 16
    jn = Vector{ComplexF64}(undef, nstart + 2)
    _bessel_upward!(jn, nmax, z)
    resize!(jn, nmax + 1)
    return jn
end

# In-place Miller downward recurrence for jₙ(z).
# Caller must pre-allocate buf with length ≥ nmax + 18 (the extra 17 slots
# hold the downward-recurrence seeds and margin; see apply_translation_mvp!
# docstring).  On return, buf[n+1] = jₙ(z) for n = 0, 1, ..., nmax.
function _bessel_upward!(buf::Vector{ComplexF64}, nmax::Int, z::ComplexF64)
    nstart = max(nmax, ceil(Int, abs(z))) + 16
    @boundscheck length(buf) >= nstart + 2 ||
        throw(ArgumentError("_bessel_upward! needs buf length ≥ nmax+18"))
    @inbounds begin
        buf[nstart + 2] = zero(ComplexF64)   # jₙstart+1 seed
        buf[nstart + 1] = one(ComplexF64)    # jₙstart seed
        for n in nstart:-1:1
            # jₙ₋₁ = (2n+1)/z · jₙ - jₙ₊₁; buf[k] holds jₖ₋₁(z) (unnormalised).
            buf[n] = (Float64(2n + 1) / z) * buf[n + 1] - buf[n + 2]
        end
        # Normalise using the exact j₀(z) = sin(z)/z.
        scale = (sin(z) / z) / buf[1]
        for n in 1:(nmax + 1)
            buf[n] *= scale
        end
    end
    return
end

# ─────────────────────────────────────────────────────────────
# Normalised associated Legendre functions
# Mirrors Fortran `normalizedlegendre(ct, mmax, nmax, dc)`
# Returns ymn[v+mmax+1, w+1] for v = -mmax,...,mmax, w = 0,...,nmax
# ─────────────────────────────────────────────────────────────

function _normalizedlegendre(ct::Float64, mmax::Int, nmax::Int,
                              fnr, bcof)::Matrix{Float64}
    sbe  = sqrt(max(0.0, (1.0+ct)*(1.0-ct)))
    ymn  = zeros(Float64, 2*mmax+1, nmax+1)

    for m in 0:mmax
        mi = m + mmax + 1          # Julia index for azimuthal m
        # dc(m,m)
        ymn[mi, m+1] = (-1)^m * (0.5*sbe)^m * bcof[m+1, m+1]
        m == nmax && break
        # dc(m, m+1)
        ymn[mi, m+2] = fnr[2m+2] * ct * ymn[mi, m+1]
        # dc(m, n+1) for n = m+1,...,nmax-1
        for n in m+1:nmax-1
            ymn[mi, n+2] = (-fnr[n-m+1]*fnr[n+m+1]*ymn[mi, n] +
                             Float64(2n+1)*ct*ymn[mi, n+1]) /
                            (fnr[n+2-m] * fnr[n+2+m])
        end
    end
    # Negative m by symmetry: dc(-m,n) = (-1)^m * dc(m,n)
    for m in 1:mmax
        im_sign = iseven(m) ? 1 : -1
        for n in m:nmax
            ymn[-m+mmax+1, n+1] = im_sign * ymn[m+mmax+1, n+1]
        end
    end
    return ymn
end

# In-place version (no allocation): fills buf (must be size (2*mmax+1, nmax+1), zeroed on entry)
function _normalizedlegendre!(buf::Matrix{Float64}, ct::Float64, mmax::Int, nmax::Int, fnr, bcof)
    fill!(buf, 0.0)
    sbe = sqrt(max(0.0, (1.0+ct)*(1.0-ct)))
    for m in 0:mmax
        mi = m + mmax + 1
        buf[mi, m+1] = (-1)^m * (0.5*sbe)^m * bcof[m+1, m+1]
        m == nmax && break
        buf[mi, m+2] = fnr[2m+2] * ct * buf[mi, m+1]
        for n in m+1:nmax-1
            buf[mi, n+2] = (-fnr[n-m+1]*fnr[n+m+1]*buf[mi, n] +
                             Float64(2n+1)*ct*buf[mi, n+1]) /
                            (fnr[n+2-m] * fnr[n+2+m])
        end
    end
    for m in 1:mmax
        im_sign = iseven(m) ? 1 : -1
        for n in m:nmax
            buf[-m+mmax+1, n+1] = im_sign * buf[m+mmax+1, n+1]
        end
    end
end

# ─────────────────────────────────────────────────────────────
# Main public function: compute VSWF translation matrix
# ─────────────────────────────────────────────────────────────

"""
    compute_translation_matrix(r_ij, k_medium, nmax_s, nmax_t) -> H

Compute the VSWF translation matrix for re-expanding outgoing waves
centred on sphere j at sphere i.

# Arguments
- `r_ij::Vector{Float64}`: position vector from sphere j to sphere i (3-element),
  in physical units consistent with `k_medium` (i.e., `k_medium * r_ij` is dimensionless).
- `k_medium::ComplexF64`: wavenumber in the surrounding medium.
  For a lossless medium with n_medium, `k_medium = length_scale_factor`.
- `nmax_s::Int`: max multipole order of the SOURCE sphere j.
- `nmax_t::Int`: max multipole order of the TARGET sphere i.

# Returns
`H[kl_idx, mn_idx, p]` of type `Array{ComplexF64,3}`,
size `(nmax_t*(nmax_t+2), nmax_s*(nmax_s+2), 2)`, where:
- `kl_idx = l*(l+1)+k`  (target multipole, 1-indexed),
- `mn_idx = n*(n+1)+m`  (source multipole, 1-indexed),
- `p ∈ {1,2}` is the incident polarisation index.

For `r_ij = 0`, returns identity (itype=1 limit is not implemented here).
Use this function only for distinct spheres (`|r_ij| > 0`).
"""
function compute_translation_matrix(
    r_ij::Vector{Float64},
    k_medium::ComplexF64,
    nmax_s::Int,
    nmax_t::Int;
    use_regular::Bool = false
)::Array{ComplexF64, 3}

    nblks  = nmax_s * (nmax_s + 2)
    nblkt  = nmax_t * (nmax_t + 2)
    H      = zeros(ComplexF64, nblkt, nblks, 2)

    r2 = sum(abs2, r_ij)
    if r2 < 1e-24
        # Zero translation: identity (only for regular/itype=1, not used here)
        for n in 1:min(nblks, nblkt)
            H[n, n, 1] = 1.0
            H[n, n, 2] = 1.0
        end
        return H
    end
    r  = sqrt(r2)

    # Ensure precomputed constants are available
    nodrmax = max(nmax_s, nmax_t)
    nc  = _ensure_numconstants(nodrmax + 2)
    tcc = _ensure_tran_coef(nodrmax, nc.fnr, nc.bcof)

    wmax_global = nmax_s + nmax_t

    # ── Spherical coordinates of r_ij ──
    ct   = r_ij[3] / r       # cos θ
    if r_ij[1] == 0.0 && r_ij[2] == 0.0
        ephi = ComplexF64(1.0, 0.0)
    else
        ephi = ComplexF64(r_ij[1], r_ij[2]) / sqrt(r_ij[1]^2 + r_ij[2]^2)
    end

    # ── Azimuthal phases: ephim[m + wmax_global + 1] = e^{imφ} ──
    ephim = Vector{ComplexF64}(undef, 2*wmax_global + 1)
    ephim[wmax_global+1] = 1.0
    for m in 1:wmax_global
        ephim[m+wmax_global+1]   = ephi * ephim[m-1+wmax_global+1]
        ephim[-m+wmax_global+1]  = conj(ephim[m+wmax_global+1])
    end

    # ── Normalised Legendre: ymn[v+wmax+1, w+1] for v=-wmax:wmax, w=0:wmax ──
    ymn = _normalizedlegendre(ct, wmax_global, wmax_global, nc.fnr, nc.bcof)

    # ── Spherical Bessel/Hankel functions ──
    # use_regular=true (vswf_type=1): j_w(k*r)  — for translation to common origin
    # use_regular=false (vswf_type=2): h_w^(1)(k*r) — for T-matrix interaction
    z  = k_medium * r
    hn = use_regular ? _bessel_upward(wmax_global, z) : _hankel_upward(wmax_global, z)
    # For a homogeneous medium ri(1)=ri(2), so hn is the same for both polarizations

    tc = tcc.tran_coef   # alias

    # ── Main double loop: source (n,m), target (l,k) ──
    for n in 1:nmax_s
        for m in -n:n
            mn  = n*(n+1) + m
            for l in 1:nmax_t
                wmax = n + l
                for k in -l:l
                    kl   = l*(l+1) + k
                    v    = m - k
                    wmin = max(abs(v), abs(n-l))
                    a = zero(ComplexF64)   # even-parity sum
                    b = zero(ComplexF64)   # odd-parity sum
                    for w in wmax:-1:wmin
                        ywt = hn[w+1] * ymn[v+wmax_global+1, w+1] * tc[w+1, kl, mn]
                        if (wmax - w) % 2 == 0
                            a += ywt
                        else
                            b += ywt
                        end
                    end
                    ep = ephim[v+wmax_global+1]
                    H[kl, mn, 1] = ep * (a + im*b)
                    H[kl, mn, 2] = ep * (a - im*b)
                end
            end
        end
    end

    return H
end

# ─────────────────────────────────────────────────────────────
# Zero-allocation MVP: out += H_{ij} * inp
# Computes the matrix-vector product with the translation matrix without
# materializing H. Pre-allocated work buffers are passed in to avoid any
# heap allocation in the hot loop.
#
# Caller must pre-allocate:
#   ephim_buf : Vector{ComplexF64}, length ≥ 2*wmax_global+1
#   fn_buf    : Vector{ComplexF64}, length ≥ wmax_global+18  (17 scratch slots
#               are used by Miller-downward jₙ recurrence in _bessel_upward!;
#               _hankel_upward! only needs wmax_global+1, but the +18 contract
#               is uniform for both)
#   ymn_buf   : Matrix{Float64},   size (2*wmax_global+1, wmax_global+1)
#   fywt_buf  : Matrix{ComplexF64}, size (2*wmax_global+1, wmax_global+1)
# where wmax_global = nmax_s + nmax_t.
# ─────────────────────────────────────────────────────────────

"""
    apply_translation_mvp!(out, inp, off_t, hnb_t, off_s, hnb_s,
                            r_ij, nmax_s, nmax_t,
                            ephim_buf, fn_buf, ymn_buf, fywt_buf;
                            use_regular=false)

Compute `out += H_{ij} * inp` in-place without allocating the translation matrix H.
Uses pre-allocated work buffers `ephim_buf`, `fn_buf`, `ymn_buf` to avoid heap
allocation in the hot BiCG loop.

# Index conventions (same as `compute_translation_matrix`)
- Source (sphere j): multipole index mn = n*(n+1)+m, stored at `inp[off_s+mn]` (p=1)
  and `inp[off_s+hnb_s+mn]` (p=2).
- Target (sphere i): multipole index kl = l*(l+1)+k, accumulated to `out[off_t+kl]` (p=1)
  and `out[off_t+hnb_t+kl]` (p=2).
"""
function apply_translation_mvp!(
    out      ::Vector{ComplexF64},
    inp      ::Vector{ComplexF64},
    off_t    ::Int,
    hnb_t    ::Int,
    off_s    ::Int,
    hnb_s    ::Int,
    r_ij     ::Vector{Float64},
    nmax_s   ::Int,
    nmax_t   ::Int,
    ephim_buf::Vector{ComplexF64},
    fn_buf   ::Vector{ComplexF64},
    ymn_buf  ::Matrix{Float64},
    fywt_buf ::Matrix{ComplexF64};
    use_regular::Bool = false
)
    nodrmax     = max(nmax_s, nmax_t)
    nc          = _ensure_numconstants(nodrmax + 2)
    tcc         = _ensure_tran_coef(nodrmax, nc.fnr, nc.bcof)
    wmax_global = nmax_s + nmax_t

    r2  = r_ij[1]^2 + r_ij[2]^2 + r_ij[3]^2
    r   = sqrt(r2)
    ct  = r_ij[3] / r

    rxy = sqrt(r_ij[1]^2 + r_ij[2]^2)
    if rxy < 1e-14
        ephi = ComplexF64(1.0, 0.0)
    else
        ephi = ComplexF64(r_ij[1]/rxy, r_ij[2]/rxy)
    end

    # ── Azimuthal phases ──
    ephim_buf[wmax_global+1] = ComplexF64(1.0, 0.0)
    for mv in 1:wmax_global
        ephim_buf[mv+wmax_global+1]   = ephi * ephim_buf[mv-1+wmax_global+1]
        ephim_buf[-mv+wmax_global+1]  = conj(ephim_buf[mv+wmax_global+1])
    end

    # ── Legendre functions ──
    _normalizedlegendre!(ymn_buf, ct, wmax_global, wmax_global, nc.fnr, nc.bcof)

    # ── Radial functions ──
    z = ComplexF64(r)   # k_medium = 1 (positions already dimensionless)
    if use_regular
        _bessel_upward!(fn_buf, wmax_global, z)
    else
        _hankel_upward!(fn_buf, wmax_global, z)
    end

    # ── Pre-combine fn_buf * ymn_buf (pair-dependent, loop-independent) ──
    @inbounds for w in 0:wmax_global
        fnw = fn_buf[w+1]
        for vi in 1:(2*wmax_global+1)
            fywt_buf[vi, w+1] = fnw * ymn_buf[vi, w+1]
        end
    end

    tc = tcc.tran_coef

    # ── Fused translation + MVP (even/odd w split for branchless inner loops) ──
    @inbounds for n in 1:nmax_s
        for m in -n:n
            mn  = n*(n+1) + m
            v1  = inp[off_s + mn]           # p=1
            v2  = inp[off_s + hnb_s + mn]   # p=2
            for l in 1:nmax_t
                wmax_local = n + l
                for k in -l:l
                    kl   = l*(l+1) + k
                    v    = m - k
                    wmin = max(abs(v), abs(n-l))
                    vi   = v + wmax_global + 1
                    a    = zero(ComplexF64)
                    b    = zero(ComplexF64)
                    # Even terms: (wmax_local - w) even → w has same parity as wmax_local
                    for w in wmax_local:-2:wmin
                        a += fywt_buf[vi, w+1] * tc[w+1, kl, mn]
                    end
                    # Odd terms: (wmax_local - w) odd → w has opposite parity
                    for w in (wmax_local-1):-2:wmin
                        b += fywt_buf[vi, w+1] * tc[w+1, kl, mn]
                    end
                    ep = ephim_buf[vi]
                    h1 = ep * (a + im*b)
                    h2 = ep * (a - im*b)
                    out[off_t + kl]         += h1 * v1
                    out[off_t + hnb_t + kl] += h2 * v2
                end
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────
# Utility: Cartesian → spherical coordinates (kept from skeleton)
# ─────────────────────────────────────────────────────────────

"""
    cart_to_spherical(r) -> (r_mag, θ, φ)

Convert Cartesian coordinates to spherical coordinates.
θ ∈ [0, π], φ ∈ [0, 2π).
"""
function cart_to_spherical(r::Vector{Float64})::Tuple{Float64, Float64, Float64}
    x, y, z = r
    r_mag = sqrt(x^2 + y^2 + z^2)
    θ = r_mag > 0 ? acos(clamp(z / r_mag, -1.0, 1.0)) : 0.0
    φ = atan(y, x)
    φ < 0 && (φ += 2π)
    return (r_mag, θ, φ)
end
