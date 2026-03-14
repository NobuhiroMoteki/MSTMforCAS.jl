"""
    ScatteringAmplitude

Compute forward/backward scattering amplitude matrices and
polarization-averaged cross sections from the multi-sphere
scattered field expansion coefficients.

# Output convention (BH83 Eq. 3.12)

The amplitude scattering matrix relates incident and scattered fields:
    ⎡ E∥ˢ ⎤     exp(ikr)  ⎡ S₂  S₃ ⎤ ⎡ E∥ⁱ ⎤
    ⎢     ⎥ = ─────────── ⎢        ⎥ ⎢     ⎥
    ⎣ E⊥ˢ ⎦    -ikr       ⎣ S₄  S₁ ⎦ ⎣ E⊥ⁱ ⎦

At θ=0° (forward) and θ=180° (backward), the associated Legendre functions
take special values:
    πₙ(cos 0°) = τₙ(cos 0°) = n(n+1)/2
    πₙ(cos 180°) = -τₙ(cos 180°) = (-1)^(n+1) · n(n+1)/2

This greatly simplifies the amplitude computation.

# Cross sections

- C_ext via optical theorem: C_ext = (2π/k²) Σ_pol Re[S(0°)·ê_pol] (averaged over 2 polarizations)
- C_abs from internal field energy balance (per sphere)
- C_sca = C_ext - C_abs (energy conservation)

# References
- Bohren & Huffman (1983), Eqs. 3.12, 4.61, 4.62
- Mackowski & Mishchenko (2011), Section 3
"""

export compute_scattering_output

"""
    compute_scattering_output(
        positions::Matrix{Float64},    # [3, N] sphere positions
        radii::Vector{Float64},        # [N] sphere radii
        m_rel::ComplexF64,             # relative refractive index
        k::ComplexF64,                 # wavenumber in medium
        scattered_coeffs,              # from TMatrixSolver
        mie_a::Vector{Vector{ComplexF64}},  # Mie aₙ for each sphere
        mie_b::Vector{Vector{ComplexF64}}   # Mie bₙ for each sphere
    ) -> ScatteringResult

Compute the full scattering output from solved expansion coefficients.

# Returns
`ScatteringResult` containing:
- S_forward: (S₁, S₂, S₃, S₄) at θ=0° in BH83 convention
- S_backward: (S₁, S₂, S₃, S₄) at θ=180° in BH83 convention
- C_ext, C_abs, C_sca: polarization-averaged cross sections
"""
function compute_scattering_output(
    positions::Matrix{Float64},
    radii::Vector{Float64},
    m_rel::ComplexF64,
    k::ComplexF64,
    scattered_coeffs,
    mie_a::Vector{Vector{ComplexF64}},
    mie_b::Vector{Vector{ComplexF64}};
    converged::Bool = true,
    n_iterations::Int = 0
)::ScatteringResult

    # TODO: Implement
    #
    # 1. Compute S₁, S₂, S₃, S₄ at θ=0° and θ=180°
    #    using special values of πₙ(±1) and τₙ(±1)
    #
    # 2. C_ext via optical theorem:
    #    For x-polarized incidence: C_ext_x = (4π/k²) Re[S₂(0°)]  (BH83 Eq. 3.24 adapted)
    #    For y-polarized incidence: C_ext_y = (4π/k²) Re[S₁(0°)]
    #    C_ext = (C_ext_x + C_ext_y) / 2
    #
    #    NOTE: Need to run TMatrixSolver for BOTH polarizations to get
    #    polarization-averaged quantities. The T-matrix itself is
    #    polarization-independent, so only the incident field coefficients
    #    and the final amplitude assembly need to be repeated.
    #
    # 3. C_abs: For each sphere i, compute absorption from
    #    internal vs external field energy balance
    #    C_abs_i = -(1/k²) Σₙ Re[(2n+1)(aₙ*|cₙ|² + bₙ*|dₙ|²)] ... (simplified)
    #    Actually: use the standard formula involving Mie coefficients and
    #    scattered expansion coefficients (Mackowski 2011, Eq. 12-14)
    #
    # 4. C_sca = C_ext - C_abs

    error("ScatteringAmplitude.compute_scattering_output: not yet implemented")
end

"""
    optical_theorem_cext(k::ComplexF64, S_forward_par::ComplexF64, S_forward_perp::ComplexF64) -> Float64

Compute extinction cross section from forward scattering amplitudes
via the optical theorem (BH83 Eq. 3.24).

C_ext = (4π/k²) · (1/2) · [Re(S₂(0°)) + Re(S₁(0°))]

where S₂ corresponds to parallel polarization and S₁ to perpendicular.
The factor 1/2 is for polarization averaging.
"""
function optical_theorem_cext(
    k::ComplexF64,
    S2_forward::ComplexF64,
    S1_forward::ComplexF64
)::Float64
    k_real = real(k)  # for non-absorbing medium; generalize if needed
    return (2π / k_real^2) * (real(S2_forward) + real(S1_forward))
end
