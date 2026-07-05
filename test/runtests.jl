using Test
using MSTMforCAS

@testset "MSTMforCAS.jl" begin
    include("test_mie_coefficients.jl")
    include("test_wigner_rotation.jl")     # Wigner-D kernel (design P1)
    include("test_cluster_tmatrix.jl")     # cluster T-matrix + rotation (design P2)
    # include("test_translation_coefs.jl")   # uncomment as implemented
    # include("test_tmatrix_solver.jl")
    # include("test_scattering_amplitude.jl")
    # include("test_aggregate_io.jl")
    # include("test_end_to_end.jl")
end
