using Test
using MSTMforCAS

@testset "AggregateIO" begin

    @testset "Read sample 3-monomer aggregate" begin
        filepath = joinpath(@__DIR__, "fixtures", "sample_aggregate_3.dat")
        agg = read_aggregate_file(filepath)

        @test agg.n_monomers == 3
        @test size(agg.positions) == (3, 3)
        @test length(agg.radii) == 3

        # Check first monomer at origin
        @test agg.positions[:, 1] ≈ [0.0, 0.0, 0.0]
        @test agg.radii[1] ≈ 1.0

        # Check second monomer
        @test agg.positions[:, 2] ≈ [2.0, 0.0, 0.0]
    end

    @testset "Scale factor" begin
        filepath = joinpath(@__DIR__, "fixtures", "sample_aggregate_3.dat")
        scale = 50e-9  # 50 nm monomer radius
        agg = read_aggregate_file(filepath; scale_factor=scale)

        @test agg.radii[1] ≈ 50e-9
        @test agg.positions[1, 2] ≈ 100e-9  # x of second monomer = 2.0 * 50e-9
    end

end
