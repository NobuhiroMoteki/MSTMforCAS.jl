"""
    TranslationCoefs

Vector Spherical Wave Function (VSWF) translation addition theorem coefficients.

Computes the coefficients needed to re-expand outgoing VSWFs centered on sphere j
as regular VSWFs centered on sphere i. This is the mathematical core of the
multi-sphere T-matrix method.

# Algorithm
Recursive scheme from Mackowski (1996), JOSA A 13:2266-2278.
Uses normalized coefficients related to Mackowski & Mishchenko (1996) by:
    J^{ij}_{mnp,mlq} = (E_{ml}/E_{mn})^{1/2} · ac(s, n, l(l+1)+m)
where E_{mn} = n(n+1)(n+m)! / ((2n+1)(n-m)!)

# References
- Mackowski (1996), JOSA A 13:2266
- Xu (1996), J. Comput. Phys. 127:285
"""

export compute_translation_matrix

"""
    compute_translation_matrix(
        r_ij::Vector{Float64},  # separation vector from sphere j to sphere i (3-element)
        k::ComplexF64,          # wavenumber in medium
        nmax_i::Int,            # max multipole order for sphere i
        nmax_j::Int,            # max multipole order for sphere j
        itype::Int              # 1 = regular (J), 3 = outgoing (H)
    ) -> (A::Array{ComplexF64,4}, B::Array{ComplexF64,4})

Compute VSWF translation coefficients A^{ij}_{mn,νμ} and B^{ij}_{mn,νμ}.

# Arguments
- `r_ij`: Position vector from sphere j to sphere i [x, y, z]
- `k`: Wavenumber in the surrounding medium
- `nmax_i`, `nmax_j`: Maximum multipole orders
- `itype`: Basis function type (1 = regular Bessel, 3 = outgoing Hankel)
           Use itype=3 for distinct spheres (kr > 0)

# Returns
- `A`, `B`: Translation coefficient arrays indexed as [m_offset, n, μ_offset, ν]
           where m_offset = m + nmax + 1 (to handle negative m indices)

# Notes
For the multi-sphere problem with non-overlapping spheres, always use itype=3.
itype=1 is only needed for concentric sphere (core-shell) geometries, which
are outside the scope of this package.
"""
function compute_translation_matrix(
    r_ij::Vector{Float64},
    k::ComplexF64,
    nmax_i::Int,
    nmax_j::Int,
    itype::Int
)::Tuple{Array{ComplexF64,4}, Array{ComplexF64,4}}

    # TODO: Implement Mackowski (1996) recursive scheme
    #
    # Implementation plan:
    #
    # 1. Convert r_ij to spherical coordinates (r, θ, φ)
    # 2. Compute spherical Bessel/Hankel functions h_p(kr) for p = 0, ..., nmax_i + nmax_j
    # 3. Compute normalized associated Legendre functions P_p^q(cos θ)
    # 4. Build the axial translation coefficients (for φ=0 case) using
    #    the Mackowski recursive scheme
    # 5. Apply rotation to account for general (θ, φ) orientation
    #
    # Key recursion (Mackowski 1996, Eqs. 20-23):
    #   The axial translation coefficients A^0_{n,ν}(kr) and B^0_{n,ν}(kr) are
    #   computed from Gaunt coefficients (integrals of three spherical harmonics)
    #   combined with spherical Bessel/Hankel functions.
    #
    # For non-axial translations, use Wigner rotation matrices to rotate
    # the axial result to the actual orientation of r_ij.

    error("TranslationCoefs.compute_translation_matrix: not yet implemented")
end

"""
    cart_to_spherical(r::Vector{Float64}) -> (r_mag, θ, φ)

Convert Cartesian coordinates to spherical coordinates.
θ ∈ [0, π], φ ∈ [0, 2π).
"""
function cart_to_spherical(r::Vector{Float64})::Tuple{Float64, Float64, Float64}
    x, y, z = r
    r_mag = sqrt(x^2 + y^2 + z^2)
    θ = r_mag > 0 ? acos(clamp(z / r_mag, -1.0, 1.0)) : 0.0
    φ = atan(y, x)
    if φ < 0
        φ += 2π
    end
    return (r_mag, θ, φ)
end
