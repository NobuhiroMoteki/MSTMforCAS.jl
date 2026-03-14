"""
    MSTMforCAS

Multi-Sphere T-Matrix method specialized for Complex Amplitude Sensing (CAS).

Computes forward/backward complex scattering amplitudes and polarization-averaged
cross sections for aggregates of homogeneous spheres in an infinite homogeneous medium
under plane wave incidence.

# Modules
- `MieCoefficients`: Lorenz-Mie scattering coefficients for single spheres
- `TranslationCoefs`: Vector spherical wave function translation coefficients
- `TMatrixSolver`: Multi-sphere interaction equation solver
- `ScatteringAmplitude`: Forward/backward amplitudes and cross sections
- `AggregateIO`: Read aggregate geometry files (aggregate_generator_PTSA format)
- `ParameterSweep`: Orchestrate parameter sweeps with parallel execution

# References
- Bohren & Huffman (1983), Absorption and Scattering of Light by Small Particles
- Mackowski (1996), JOSA A 13:2266
- Mackowski & Mishchenko (2011), JQSRT 112:2182
- Moteki & Adachi (2024), Optics Express 32:36500
"""
module MSTMforCAS

using LinearAlgebra
using Printf
using SpecialFunctions

# Sub-modules — loaded in dependency order
include("MieCoefficients.jl")
include("TranslationCoefs.jl")
include("TMatrixSolver.jl")
include("ScatteringAmplitude.jl")
include("AggregateIO.jl")
include("ParameterSweep.jl")

# Re-export primary public API
export compute_mie_coefficients
export compute_scattering           # single aggregate, single refractive index
export run_parameter_sweep          # batch computation
export read_aggregate_file          # I/O
export ScatteringResult             # output data structure

"""
    ScatteringResult

Holds the scattering computation output for a single (aggregate, refractive index) pair.

# Fields
- `S_forward::NTuple{4, ComplexF64}`: S₁, S₂, S₃, S₄ at θ=0° (BH83 convention)
- `S_backward::NTuple{4, ComplexF64}`: S₁, S₂, S₃, S₄ at θ=180° (BH83 convention)
- `C_ext::Float64`: Polarization-averaged extinction cross section
- `C_abs::Float64`: Polarization-averaged absorption cross section
- `C_sca::Float64`: Polarization-averaged scattering cross section (= C_ext - C_abs)
- `converged::Bool`: Whether the T-matrix iteration converged
- `n_iterations::Int`: Number of iterations used
"""
struct ScatteringResult
    S_forward::NTuple{4, ComplexF64}
    S_backward::NTuple{4, ComplexF64}
    C_ext::Float64
    C_abs::Float64
    C_sca::Float64
    converged::Bool
    n_iterations::Int
end

end # module MSTMforCAS
