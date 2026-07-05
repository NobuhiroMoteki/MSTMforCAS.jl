# Unit tests for the Wigner-D kernel (design phase P1).
# Self-contained: includes the module directly (no full-package / CUDA load).
#
# Run standalone:
#   julia --project=. test/test_wigner_rotation.jl

using Test
using LinearAlgebra
using SpecialFunctions   # provides `loggamma` used by the plain-include kernel

# WignerRotation.jl is a plain include (no module wrapper), matching the other
# src/*.jl files; include it directly so wigner_d/wigner_D land in this scope.
include(joinpath(@__DIR__, "..", "src", "WignerRotation.jl"))

# index helper
idx(n, m) = m + n + 1

# reference Legendre P_n(x) by recurrence (for the d^n_{00} check)
function legendre_P(n::Int, x::Float64)
    n == 0 && return 1.0
    n == 1 && return x
    Pm1, P = 1.0, x
    for k in 1:(n-1)
        Pm1, P = P, ((2k + 1) * x * P - k * Pm1) / (k + 1)
    end
    return P
end

# Cartesian rotation matrices (active, right-handed)
Rz(a) = [cos(a) -sin(a) 0.0; sin(a) cos(a) 0.0; 0.0 0.0 1.0]
Ry(b) = [cos(b) 0.0 sin(b); 0.0 1.0 0.0; -sin(b) 0.0 cos(b)]
Rzyz(a, b, g) = Rz(a) * Ry(b) * Rz(g)

# extract intrinsic ZYZ Euler angles from a rotation matrix (non-degenerate β)
function euler_zyz(R)
    β = acos(clamp(R[3, 3], -1.0, 1.0))
    α = atan(R[2, 3], R[1, 3])
    γ = atan(R[3, 2], -R[3, 1])
    return α, β, γ
end

@testset "WignerRotation" begin
    βs = [0.0, 0.1, 0.7, π/4, π/3, 1.9, π - 0.05]

    @testset "d^0 and identity at β=0" begin
        @test wigner_d(0, 0.7) == reshape([1.0], 1, 1)
        for n in 0:6
            @test wigner_d(n, 0.0) ≈ Matrix{Float64}(I, 2n+1, 2n+1) atol=1e-12
        end
    end

    @testset "n=1 explicit values" begin
        for β in βs
            d = wigner_d(1, β)
            c, s = cos(β), sin(β)
            ref = [ (1+c)/2   s/√2   (1-c)/2;      # rows m'=-1,0,+1 ; cols m=-1,0,+1
                    -s/√2     c      s/√2;
                    (1-c)/2  -s/√2   (1+c)/2 ]
            @test d ≈ ref atol=1e-12
        end
    end

    @testset "orthogonality d·dᵀ = I" begin
        maxerr = 0.0
        for n in 0:25, β in βs
            d = wigner_d(n, β)
            maxerr = max(maxerr, opnorm(d * transpose(d) - I))
        end
        @info "max ||d·dᵀ − I|| over n≤25" maxerr
        @test maxerr < 1e-11
    end

    @testset "d^n_{00} = P_n(cos β)" begin
        for n in 0:20, β in βs
            d = wigner_d(n, β)
            @test d[idx(n,0), idx(n,0)] ≈ legendre_P(n, cos(β)) atol=1e-12
        end
    end

    @testset "D unitary and D(0,0,0)=I" begin
        for n in 0:8
            @test wigner_D(n, 0.0, 0.0, 0.0) ≈ Matrix{ComplexF64}(I, 2n+1, 2n+1) atol=1e-12
            for (α, β, γ) in [(0.3,0.7,1.1), (2.0,1.3,0.4), (1.0,2.5,3.0)]
                D = wigner_D(n, α, β, γ)
                @test D * D' ≈ Matrix{ComplexF64}(I, 2n+1, 2n+1) atol=1e-9
            end
        end
    end

    @testset "group homomorphism (rep faithfulness, ZYZ-consistent)" begin
        Ω1 = (0.4, 0.9, 1.2)
        Ω2 = (1.7, 0.6, 0.3)
        R12 = Rzyz(Ω1...) * Rzyz(Ω2...)
        α, β, γ = euler_zyz(R12)
        # a valid rep satisfies one composition order; assert one holds for all n
        for n in 1:6
            D1 = wigner_D(n, Ω1...)
            D2 = wigner_D(n, Ω2...)
            Dc = wigner_D(n, α, β, γ)
            ok = isapprox(D1 * D2, Dc; atol=1e-8) || isapprox(D2 * D1, Dc; atol=1e-8)
            @test ok
        end
    end
end
