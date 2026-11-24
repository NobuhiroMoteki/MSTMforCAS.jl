using Test
using MSTMforCAS

@testset "MSTMforCAS.jl" begin
    include("test_mie_coefficients.jl")
    include("test_wigner_rotation.jl")     # Wigner-D kernel (design P1)
    include("test_cluster_tmatrix.jl")     # cluster T-matrix + rotation (design P2)
    include("test_truncation_order.jl")    # truncation order for near-touching monomers
    include("test_fft_translation.jl")     # FFT grid with far orders below the sphere orders
    include("test_parameter_sweep.jl")     # sweep wiring of truncation_eps and fft_far_order
    include("test_shared_basis.jl")        # refractive-index sweep with a shared Krylov basis
    include("test_tranordertest.jl")       # translation order test with cached tables
    include("test_fortran_unequal_radii.jl")  # Fortran reference for aggregates with unequal radii
    # include("test_translation_coefs.jl")   # uncomment as implemented
    # include("test_tmatrix_solver.jl")
    # include("test_scattering_amplitude.jl")
    # include("test_aggregate_io.jl")
    # include("test_end_to_end.jl")
end
