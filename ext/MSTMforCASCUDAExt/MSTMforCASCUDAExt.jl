"""
CUDA extension for MSTMforCAS.jl — GPU-accelerated batched BiCG solver.

Loaded automatically when the user calls `using CUDA` after `using MSTMforCAS`.
Registers a GPU batch solver that processes multiple refractive index values
simultaneously on a single GPU device.
"""
module MSTMforCASCUDAExt

using MSTMforCAS
using CUDA
using CUDA: CuArray, CuVector, CuMatrix
import CUDA.CUFFT
using LinearAlgebra: dot, norm

include("gpu_types.jl")
include("gpu_kernels.jl")
include("gpu_fft_translation.jl")
include("gpu_batch_solver.jl")

function __init__()
    if CUDA.functional()
        MSTMforCAS._gpu_batch_solve_ref[] = gpu_batch_solve_group
        dev = CUDA.device()
        mem_gb = round(CUDA.totalmem(dev) / 1024^3, digits=1)
        @info "MSTMforCAS CUDA extension loaded" device=CUDA.name(dev) memory="$(mem_gb) GB"
    else
        @warn "CUDA extension loaded but no functional GPU detected. GPU solver unavailable."
    end
end

end # module MSTMforCASCUDAExt
