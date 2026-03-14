"""
    ParameterSweep

Orchestrate parameter sweeps over aggregate geometries and refractive indices.
Supports multi-threaded parallel execution via Julia's Threads.@threads.

# Typical usage

```julia
results = run_parameter_sweep(
    aggregate_dir = "path/to/aggregate_files/",
    refractive_indices = [1.5+0.0im, 1.5+0.01im, 1.5+0.1im, 1.75+0.0im],
    wavelength = 0.6328e-6,  # [m] He-Ne laser
    medium_refindex = 1.0,   # air
    monomer_radius = 50e-9,  # [m]
    output_file = "results.h5"
)
```
"""

export run_parameter_sweep, SweepConfig

"""
    SweepConfig

Configuration for a parameter sweep computation.

# Fields
- `aggregate_files::Vector{String}`: Paths to aggregate geometry files
- `refractive_indices::Vector{ComplexF64}`: Complex refractive indices to sweep
- `wavelength::Float64`: Incident wavelength (in physical units, e.g., meters)
- `medium_refindex::Float64`: Real refractive index of surrounding medium
- `monomer_radius::Float64`: Physical monomer radius (for scaling dimensionless aggregate files)
- `max_iterations::Int`: Max T-matrix solver iterations
- `convergence_epsilon::Float64`: Convergence threshold
- `n_threads::Int`: Number of threads (0 = use all available)
"""
Base.@kwdef struct SweepConfig
    aggregate_files::Vector{String}
    refractive_indices::Vector{ComplexF64}
    wavelength::Float64
    medium_refindex::Float64 = 1.0
    monomer_radius::Float64 = 1.0  # set to 1.0 if files already in physical units
    max_iterations::Int = 5000
    convergence_epsilon::Float64 = 1e-6
    n_threads::Int = 0  # 0 = Threads.nthreads()
end

"""
    run_parameter_sweep(config::SweepConfig; output_file::Union{String,Nothing}=nothing)
        -> DataFrame

Execute parameter sweep: for each (aggregate_file, refractive_index) pair,
compute the scattering result.

# Parallelism
The sweep is parallelized over parameter grid points using Julia Threads.
Start Julia with `julia -t auto` or `julia -t N` to enable multi-threading.

# Returns
DataFrame with columns:
- `aggregate_file`: source file name
- `n_monomers`: number of monomers in aggregate
- `m_real`, `m_imag`: refractive index components
- `S1_fwd_re`, `S1_fwd_im`, ...: forward amplitude components (8 columns)
- `S1_bwd_re`, `S1_bwd_im`, ...: backward amplitude components (8 columns)
- `C_ext`, `C_abs`, `C_sca`: cross sections
- `converged`: solver convergence flag
- `n_iterations`: iteration count
"""
function run_parameter_sweep(
    config::SweepConfig;
    output_file::Union{String,Nothing} = nothing
)
    # TODO: Implement
    #
    # 1. Build parameter grid: [(file, m) for file in files, m in indices]
    # 2. Parallel loop over grid (Threads.@threads)
    #    For each (file, m):
    #    a. read_aggregate_file(file; scale_factor=config.monomer_radius)
    #    b. k = 2π / config.wavelength * config.medium_refindex
    #    c. m_rel = m / config.medium_refindex
    #    d. compute_scattering(aggregate, m_rel, k)
    # 3. Collect results into DataFrame
    # 4. Optionally write to HDF5 or CSV
    #
    # Note: The T-matrix (Steps 2-3) depends on geometry and m_rel but NOT on
    # incident polarization. So for each (aggregate, m), solve once to get
    # all scattered coefficients, then assemble amplitudes for both polarizations
    # in Step 4 to get polarization-averaged cross sections.

    error("ParameterSweep.run_parameter_sweep: not yet implemented")
end

"""
    compute_scattering(agg::AggregateGeometry, m_rel::ComplexF64, k::ComplexF64;
                       kwargs...) -> ScatteringResult

High-level function: compute all scattering properties for a single
(aggregate, refractive index) pair.

This is the main entry point that chains Steps 1-4:
MieCoefficients → TMatrixSolver → ScatteringAmplitude
"""
function compute_scattering(
    agg::AggregateGeometry,
    m_rel::ComplexF64,
    k::ComplexF64;
    max_iterations::Int = 5000,
    convergence_epsilon::Float64 = 1e-6
)::ScatteringResult

    # TODO: Implement the full pipeline
    #
    # 1. For each sphere: x_i = real(k) * radii[i]
    #    (a_i, b_i) = compute_mie_coefficients(x_i, m_rel)
    #
    # 2-3. Solve multi-sphere interaction for x-polarization
    #      Solve multi-sphere interaction for y-polarization
    #      (only incident coefficients differ; translation matrices are shared)
    #
    # 4. Assemble forward/backward amplitudes from both polarization solutions
    #    Compute C_ext (optical theorem), C_abs, C_sca
    #
    # 5. Return ScatteringResult

    error("ParameterSweep.compute_scattering: not yet implemented")
end
