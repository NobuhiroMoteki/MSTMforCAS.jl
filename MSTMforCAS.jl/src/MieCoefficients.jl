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

Compute the maximum multipole order needed for convergence.
Uses Wiscombe's criterion: n_max = x + 4x^{1/3} + 2.

# Arguments
- `x`: Size parameter (= k * radius, where k is wavenumber in medium)
"""
function mie_nmax(x::Float64)::Int
    return ceil(Int, x + 4.0 * x^(1/3) + 2.0)
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
Uses logarithmic derivative ratio method (Wiscombe 1980) with downward
recurrence to avoid numerical overflow for large size parameters.

BH83 Eq. 4.53:
    aₙ = [Dₙ(mx)/m + n/x] ψₙ(x) - ψₙ₋₁(x)
         ─────────────────────────────────────────
         [Dₙ(mx)/m + n/x] ξₙ(x) - ξₙ₋₁(x)

    bₙ = [m·Dₙ(mx) + n/x] ψₙ(x) - ψₙ₋₁(x)
         ─────────────────────────────────────────
         [m·Dₙ(mx) + n/x] ξₙ(x) - ξₙ₋₁(x)

where Dₙ(z) = [d/dz ln(zψₙ(z))] = ψₙ'(z)/ψₙ(z) - 1/z  (logarithmic derivative)
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

    # --- Riccati-Bessel functions ψₙ(x) and ξₙ(x) by upward recurrence ---
    # ψₙ(x) = x·jₙ(x),  ξₙ(x) = x·hₙ⁽¹⁾(x) = ψₙ(x) + i·χₙ(x)
    # Starting values:
    #   ψ₀ = sin(x),  ψ₋₁ = cos(x)  [i.e., ψ₁ = sin(x)/x - cos(x)]
    #   χ₀ = -cos(x), χ₋₁ = sin(x)

    psi_prev = sin(x)    # ψ₀
    psi_curr = sin(x) / x - cos(x)  # ψ₁
    chi_prev = -cos(x)   # χ₀
    chi_curr = -cos(x) / x - sin(x) # χ₁

    a = Vector{ComplexF64}(undef, nmax)
    b = Vector{ComplexF64}(undef, nmax)

    for n in 1:nmax
        if n > 1
            # Upward recurrence: fₙ = (2n-1)/x · fₙ₋₁ - fₙ₋₂
            factor = (2n - 1) / x
            psi_next = factor * psi_curr - psi_prev
            chi_next = factor * chi_curr - chi_prev
            psi_prev = psi_curr
            psi_curr = psi_next
            chi_prev = chi_curr
            chi_curr = chi_next
        end

        xi_curr = Complex(psi_curr, -chi_curr)  # ξₙ = ψₙ - i·χₙ  (BH83 convention)

        # BH83 Eq. 4.53
        dn = D[n+1]  # Dₙ(mx), index shifted because D[1] = D₀
        # Wait: D[n] from recurrence corresponds to Dₙ₋₁, need to check indexing
        # Actually D[n+1] should be Dₙ after the downward recurrence above
        # (D[nmx+1] = 0 starting point, recurrence fills D[nmx] down to D[1])
        # D[1] = D₀, D[2] = D₁, ..., D[n+1] = Dₙ ✓

        an_num = (dn / m + n / x) * psi_curr - psi_prev
        an_den = (dn / m + n / x) * xi_curr - Complex(psi_prev, -chi_prev)
        a[n] = an_num / an_den

        bn_num = (m * dn + n / x) * psi_curr - psi_prev
        bn_den = (m * dn + n / x) * xi_curr - Complex(psi_prev, -chi_prev)
        b[n] = bn_num / bn_den
    end

    return (a, b)
end
