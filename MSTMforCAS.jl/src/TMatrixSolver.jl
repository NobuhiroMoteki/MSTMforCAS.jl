"""
    TMatrixSolver

Solve the multi-sphere interaction equation using iterative methods.

The interaction equation for N spheres is:
    aᵢ = Tᵢ · (pᵢ + Σ_{j≠i} H_{ij} · aⱼ)   for i = 1, ..., N

where:
- aᵢ: scattered field expansion coefficients for sphere i
- Tᵢ: single-sphere T-matrix (diagonal, from Mie coefficients)
- pᵢ: incident field expansion coefficients at sphere i
- H_{ij}: translation operator from sphere j to sphere i

# Methods
- Order-of-scattering iteration (primary): simple, stable for well-separated spheres
- BiCGSTAB (fallback): for cases where order-of-scattering converges slowly

# References
- Mackowski (1996), JOSA A 13:2266, Section 4
- Mackowski & Mishchenko (2011), JQSRT 112:2182
"""

export solve_multi_sphere

"""
    solve_multi_sphere(
        positions::Matrix{Float64},      # [3, N_spheres] sphere center positions
        radii::Vector{Float64},          # [N_spheres] sphere radii
        m_rel::ComplexF64,               # relative refractive index (sphere/medium)
        k::ComplexF64,                   # wavenumber in medium
        incident_direction::Vector{Float64},  # unit vector [3]
        incident_polarization::Vector{ComplexF64};  # polarization vector [3]
        max_iterations::Int = 5000,
        convergence_epsilon::Float64 = 1e-6,
        method::Symbol = :order_of_scattering  # or :bicgstab
    ) -> (coefficients, converged, n_iterations)

Solve the multi-sphere interaction equation.

# Arguments
- `positions`: Sphere center coordinates, column per sphere [3 × N]
- `radii`: Sphere radii (same units as 1/k)
- `m_rel`: Complex refractive index ratio (sphere / medium)
- `k`: Wavenumber in the surrounding medium
- `incident_direction`: Propagation direction unit vector
- `incident_polarization`: Electric field polarization (complex, for circular pol.)
- `max_iterations`: Maximum iteration count
- `convergence_epsilon`: Relative convergence threshold
- `method`: Solver method

# Returns
- `coefficients`: Vector of scattered field coefficients for all spheres
- `converged::Bool`: Whether the iteration converged
- `n_iterations::Int`: Actual number of iterations used
"""
function solve_multi_sphere(
    positions::Matrix{Float64},
    radii::Vector{Float64},
    m_rel::ComplexF64,
    k::ComplexF64,
    incident_direction::Vector{Float64},
    incident_polarization::Vector{ComplexF64};
    max_iterations::Int = 5000,
    convergence_epsilon::Float64 = 1e-6,
    method::Symbol = :order_of_scattering
)
    N_spheres = size(positions, 2)
    @assert size(positions, 1) == 3
    @assert length(radii) == N_spheres

    # TODO: Implement
    #
    # 1. Compute Mie coefficients for each sphere
    # 2. Compute incident field expansion coefficients pᵢ for each sphere
    #    (plane wave expansion in VSWF basis, phase-shifted to sphere center)
    # 3. Pre-compute translation matrices H_{ij} for all sphere pairs
    # 4. Iterate:
    #    a^{(k+1)}_i = T_i · (p_i + Σ_{j≠i} H_{ij} · a^{(k)}_j)
    #    until max |a^{(k+1)} - a^{(k)}| / max |a^{(k+1)}| < ε
    # 5. Return converged coefficients

    error("TMatrixSolver.solve_multi_sphere: not yet implemented")
end

"""
    compute_incident_coefficients(
        k::ComplexF64,
        position::Vector{Float64},        # sphere center [3]
        direction::Vector{Float64},        # incident direction unit vector [3]
        polarization::Vector{ComplexF64},  # E-field polarization [3]
        nmax::Int
    ) -> Vector{ComplexF64}

Compute plane wave expansion coefficients in the VSWF basis
centered at the given sphere position.

BH83 Eq. 4.40: expansion of a plane wave in vector spherical harmonics.
The phase factor exp(ik·r_i) accounts for the sphere center offset.
"""
function compute_incident_coefficients(
    k::ComplexF64,
    position::Vector{Float64},
    direction::Vector{Float64},
    polarization::Vector{ComplexF64},
    nmax::Int
)::Vector{ComplexF64}

    # TODO: Implement
    error("TMatrixSolver.compute_incident_coefficients: not yet implemented")
end
