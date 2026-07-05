# WignerRotation — general ZYZ Wigner rotation operators for VSWF coefficient
# vectors. Building block for cheap multi-orientation CAS-v2 evaluation
# (design note: docs/design_wigner_d_multiorientation.md, phase P1).
#
# Provides the real Wigner (small) d-matrix d^n_{m'm}(β) and the complex Wigner
# D-matrix D^n_{m'm}(α,β,γ) in the convention
#
#     D^n_{m'm}(α,β,γ) = exp(-i m' α) · d^n_{m'm}(β) · exp(-i m γ),
#
# with intrinsic ZYZ Euler angles (α,β,γ) in radians — matching the shared
# pcas_lut_schema orientation anchor (intrinsic ZYZ, scipy
# Rotation.from_euler('ZYZ',...)). The physical active/passive alignment to the
# MSTM/VIEM solvers is locked by cross-validation in design phase P4.
#
# Index convention: (m',m) with m',m ∈ -n..n maps to matrix entry [m'+n+1, m+n+1].
#
# Included directly into the MSTMforCAS module (relies on its `using LinearAlgebra`
# and `using SpecialFunctions`), following the plain-include style of the other
# src/*.jl files.

export wigner_d, wigner_D

# Single element d^n_{m'm}(β) from the explicit finite sum (Wigner's formula).
# Used only to seed at the extremal order n = max(|m'|,|m|), where the sum has a
# single term and is therefore exact (no cancellation).
function _wigner_d_explicit(n::Integer, m′::Integer, m::Integer, β::Real)
    c = cos(β / 2)
    s = sin(β / 2)
    logconst = 0.5 * (loggamma(n + m + 1) + loggamma(n - m + 1) +
                      loggamma(n + m′ + 1) + loggamma(n - m′ + 1))
    smin = max(0, m - m′)
    smax = min(n + m, n - m′)
    acc = 0.0
    for k in smin:smax
        p_c = 2n + m - m′ - 2k     # exponent of cos(β/2), ≥ 0 in range
        p_s = m′ - m + 2k          # exponent of sin(β/2), ≥ 0 in range
        logterm = logconst - (loggamma(n + m - k + 1) + loggamma(k + 1) +
                              loggamma(m′ - m + k + 1) + loggamma(n - m′ - k + 1))
        mag = exp(logterm) * c^p_c * s^p_s   # 0.0^0 == 1.0 in Julia
        acc += (iseven(m′ - m + k) ? mag : -mag)
    end
    return acc
end

# Legendre P_n(x) by upward recurrence (the m'=m=0 slice, d^n_{00}=P_n(cosβ)).
function _wigner_legendre_P(n::Integer, x::Real)
    n == 0 && return 1.0
    Pm1, P = 1.0, float(x)
    for k in 1:(n - 1)
        Pm1, P = P, ((2k + 1) * x * P - k * Pm1) / (k + 1)
    end
    return P
end

# d^N_{m'm}(β): exact extremal seed at n_min=max(|m'|,|m|), then the stable
# three-term upward recurrence in n. Avoids the explicit-sum cancellation, so it
# is accurate to ~machine precision across the n≲30 range needed here.
function _wigner_d_element(N::Integer, m′::Integer, m::Integer, β::Real, x::Real)
    nmin = max(abs(m′), abs(m))
    N < nmin && return 0.0
    (m′ == 0 && m == 0) && return _wigner_legendre_P(N, x)   # recurrence singular at nmin=0
    dcur = _wigner_d_explicit(nmin, m′, m, β)                # exact single-term seed
    N == nmin && return dcur
    dprev = 0.0                                              # d^{nmin-1} = 0
    for l in nmin:(N - 1)
        a  = l * sqrt(((l + 1)^2 - m′^2) * ((l + 1)^2 - m^2))
        b  = (2l + 1) * (l * (l + 1) * x - m′ * m)
        cc = (l + 1) * sqrt((l^2 - m′^2) * (l^2 - m^2))
        dnext = (b * dcur - cc * dprev) / a
        dprev, dcur = dcur, dnext
    end
    return dcur
end

"""
    wigner_d(n, β) -> Matrix{Float64}

Real Wigner (small) d-matrix `d^n_{m'm}(β)`, size `(2n+1, 2n+1)`, entry
`[m'+n+1, m+n+1]` = `d^n_{m'm}(β)`. `d^n(β)` is a real orthogonal matrix.

Each element uses an exact extremal seed (`n = max(|m'|,|m|)`, a single-term
Wigner sum) followed by the stable three-term upward recurrence in `n`, giving
~machine precision over the moderate orders (`n ≲ 30`) of CAS aggregate cluster
T-matrices.
"""
function wigner_d(n::Integer, β::Real)
    n >= 0 || throw(ArgumentError("n must be ≥ 0"))
    x = cos(β)
    d = zeros(Float64, 2n + 1, 2n + 1)
    for m′ in -n:n, m in -n:n
        d[m′ + n + 1, m + n + 1] = _wigner_d_element(n, m′, m, β, x)
    end
    return d
end

"""
    wigner_D(n, α, β, γ) -> Matrix{ComplexF64}

Complex Wigner D-matrix `D^n_{m'm}(α,β,γ) = e^{-i m' α} d^n_{m'm}(β) e^{-i m γ}`,
size `(2n+1, 2n+1)`, entry `[m'+n+1, m+n+1]`. `D^n` is unitary. Intrinsic ZYZ
Euler angles in radians.
"""
function wigner_D(n::Integer, α::Real, β::Real, γ::Real)
    d = wigner_d(n, β)
    D = Matrix{ComplexF64}(undef, 2n + 1, 2n + 1)
    for m′ in -n:n, m in -n:n
        i = m′ + n + 1
        j = m + n + 1
        D[i, j] = cis(-m′ * α) * d[i, j] * cis(-m * γ)
    end
    return D
end
