"""
    MieCoefficients

Lorenz-Mie scattering coefficients for homogeneous spheres.

Implements the Bohren-Huffman algorithm with Wiscombe's improved
downward recurrence for numerical stability at large size parameters.

# References
- Bohren & Huffman (1983), Chapter 4
- Wiscombe (1980), Applied Optics 19:1505
"""

export compute_mie_coefficients, mie_nmax

"""
    mie_nmax(x::Float64) -> Int

Upper bound on multipole order, using MSTM's starting formula: nint(x + 4x^{1/3}) + 5.
Use mie_nmax(x, m) for the converged (smaller) order.
"""
function mie_nmax(x::Float64)::Int
    return round(Int, x + 4.0 * x^(1/3)) + 5
end

"""
    mie_nmax(x::Float64, m::ComplexF64; eps::Float64=1e-6) -> Int

Converged multipole order using MSTM's mieoa criterion:
find smallest N such that |partial_Q_ext(N)/total_Q_ext - 1| < eps,
where partial_Q_ext(N) = (2/x²)Σ_{n=1}^N (2n+1)Re(aₙ+bₙ).

Matches MSTM's default mie_epsilon = 1e-6.
"""
function mie_nmax(x::Float64, m::ComplexF64; eps::Float64 = 1e-6)::Int
    nstop = mie_nmax(x)
    a, b = compute_mie_coefficients(x, m; nmax=nstop)
    qext_terms = [(2n+1) * real(a[n] + b[n]) for n in 1:nstop]
    qext_total = sum(qext_terms)
    if abs(qext_total) < 1e-30
        return 1
    end
    partial = 0.0
    for n in 1:nstop
        partial += qext_terms[n]
        if abs(1.0 - partial / qext_total) < eps
            return n
        end
    end
    return nstop
end

"""
    compute_mie_coefficients(x::Float64, m::ComplexF64; nmax::Union{Int,Nothing}=nothing)
        -> (a::Vector{ComplexF64}, b::Vector{ComplexF64})

Compute Mie scattering coefficients aₙ and bₙ for a homogeneous sphere.

# Arguments
- `x`: Size parameter in medium (k_medium * radius)
- `m`: Complex refractive index ratio (n_sphere / n_medium), BH83 convention
        (positive imaginary part = absorption)
- `nmax`: Maximum multipole order. If `nothing`, determined automatically.

# Returns
- `a`: Vector of aₙ coefficients, n = 1, ..., nmax
- `b`: Vector of bₙ coefficients, n = 1, ..., nmax

# Algorithm
Uses the logarithmic-derivative ratio method (Wiscombe 1980):
- `Dₙ(mx)` — downward recurrence (internal argument, complex `mx`)
- `ψₙ(x)`  — Miller downward recurrence (stable at all x, including x ≪ n;
             supersedes the upward recurrence used in BH83 Appendix A,
             which is conditionally stable only for x ≳ n and blows up
             for sub-wavelength particles because the growing solution
             χₙ = x·yₙ(x) contaminates ψₙ through rounding error)
- `χₙ(x)`  — upward recurrence (stable, since χₙ is the growing solution)

BH83 Eq. 4.53:
    aₙ = [Dₙ(mx)/m + n/x] ψₙ(x) - ψₙ₋₁(x)
         ─────────────────────────────────────────
         [Dₙ(mx)/m + n/x] ξₙ(x) - ξₙ₋₁(x)

    bₙ = [m·Dₙ(mx) + n/x] ψₙ(x) - ψₙ₋₁(x)
         ─────────────────────────────────────────
         [m·Dₙ(mx) + n/x] ξₙ(x) - ξₙ₋₁(x)

where Dₙ(z) = [d/dz ln(zψₙ(z))] = ψₙ'(z)/ψₙ(z) - 1/z  (logarithmic derivative).

# References
- Wiscombe (1980), Applied Optics 19:1505 — Miller algorithm for ψₙ
- Bohren & Huffman (1983), Appendix A (p. 478) — upward-recurrence caveat
"""
function compute_mie_coefficients(
    x::Float64,
    m::ComplexF64;
    nmax::Union{Int,Nothing} = nothing
)::Tuple{Vector{ComplexF64}, Vector{ComplexF64}}

    if nmax === nothing
        nmax = mie_nmax(x)
    end

    mx = m * x

    # --- Logarithmic derivative Dₙ(mx) by downward recurrence (Wiscombe 1980) ---
    # Number of terms for downward recurrence (must be > nmax)
    nmx = max(nmax, ceil(Int, abs(mx))) + 16
    D = zeros(ComplexF64, nmx + 1)
    # Downward recurrence: Dₙ₋₁ = n/z - 1/(Dₙ + n/z)
    for n in nmx:-1:1
        D[n] = n / mx - 1.0 / (D[n+1] + n / mx)
    end

    # --- Riccati-Bessel ψₙ(x) by Miller DOWNWARD recurrence (Wiscombe 1980) ---
    # ψₙ is the DECAYING solution of the three-term recurrence
    #     ψₙ₋₁ + ψₙ₊₁ = (2n+1)/x · ψₙ.
    # The upward direction (n ↑) is only conditionally stable (requires x ≳ n):
    # beyond that, the growing solution χₙ = x·yₙ(x) contaminates ψₙ through
    # rounding error and the recurrence blows up.  At x ≈ 0.3, n = 7 the
    # per-step amplification is (2n−1)/x ≈ 43, so the true ψ₇ ≈ 2.8·10⁻¹¹
    # is swamped by floating-point noise (≈ 10⁻¹⁰).  See BH83 Appendix A p.478
    # and Wiscombe 1980 (Applied Optics 19:1505).
    #
    # Miller's algorithm fixes this: start from N_start ≫ nmax with
    #   ψ_{N_start+1} = 0, ψ_{N_start} = 1 (arbitrary non-zero), recur
    # DOWNWARD, and normalise by the exact ψ₀(x) = sin(x).  In the downward
    # direction ψₙ dominates, so any spurious χₙ component is driven to
    # zero exponentially and the result is accurate to machine precision.
    # Margin of 16 terms matches the Dₙ(mx) seeding convention.
    nstart_psi = max(nmax, ceil(Int, x)) + 16
    psi = Vector{Float64}(undef, nstart_psi + 2)
    psi[nstart_psi + 2] = 0.0          # ψ_{N_start+1} seed
    psi[nstart_psi + 1] = 1.0          # ψ_{N_start}   seed (arbitrary scale)
    @inbounds for n in nstart_psi:-1:1
        # Identity: ψₙ₋₁ = (2n+1)/x · ψₙ - ψₙ₊₁, with psi[k] = ψ_{k-1}
        psi[n] = (2n + 1) / x * psi[n + 1] - psi[n + 2]
    end
    # Normalise using the exact ψ₀(x) = sin(x).
    scale = sin(x) / psi[1]
    @inbounds for n in 1:(nmax + 2)
        psi[n] *= scale
    end
    # After this, psi[n+1] = ψₙ(x) for n = 0, 1, ..., nmax  (to machine ε).

    # --- Riccati-Bessel χₙ(x) by upward recurrence (stable: χₙ is the
    #     growing solution, so the forward direction is well-conditioned) ---
    #   χ₀(x) = -cos(x),  χ₁(x) = -cos(x)/x - sin(x)
    chi_prev = -cos(x)               # χ₀
    chi_curr = -cos(x) / x - sin(x)  # χ₁

    a = Vector{ComplexF64}(undef, nmax)
    b = Vector{ComplexF64}(undef, nmax)

    @inbounds for n in 1:nmax
        if n > 1
            # Upward recurrence for χₙ only (ψₙ is read from the Miller buffer)
            factor = (2n - 1) / x
            chi_next = factor * chi_curr - chi_prev
            chi_prev = chi_curr
            chi_curr = chi_next
        end

        psi_curr = psi[n + 1]   # ψₙ(x)   — Miller-accurate
        psi_prev = psi[n]       # ψₙ₋₁(x)

        # ξₙ = ψₙ + i·χₙ  (BH83 convention)
        xi_curr = Complex(psi_curr, chi_curr)
        xi_prev = Complex(psi_prev, chi_prev)

        # BH83 Eq. 4.53;  D[1]=D₀, D[2]=D₁, ..., D[n+1]=Dₙ
        dn = D[n + 1]

        an_num = (dn / m + n / x) * psi_curr - psi_prev
        an_den = (dn / m + n / x) * xi_curr - xi_prev
        a[n] = an_num / an_den

        bn_num = (m * dn + n / x) * psi_curr - psi_prev
        bn_den = (m * dn + n / x) * xi_curr - xi_prev
        b[n] = bn_num / bn_den
    end

    return (a, b)
end
