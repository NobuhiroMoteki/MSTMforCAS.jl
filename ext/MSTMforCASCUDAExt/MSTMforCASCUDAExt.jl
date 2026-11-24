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
using LinearAlgebra: dot, norm, mul!

include("gpu_types.jl")
include("gpu_kernels.jl")
include("gpu_fft_translation.jl")
include("gpu_batch_solver.jl")

# CUDA.jl versions that the extension was written and tested for. `[compat]` of the package
# is not enforced when CUDA.jl comes from another environment of the load path, so the
# version is checked here.
const _SUPPORTED_CUDA_MAJORS = (5, 6)

function __init__()
    cuda_version = pkgversion(CUDA)
    if cuda_version !== nothing && !(cuda_version.major in _SUPPORTED_CUDA_MAJORS)
        @warn "MSTMforCAS was tested with CUDA.jl $(join(_SUPPORTED_CUDA_MAJORS, " and ")), but CUDA.jl $(cuda_version) is loaded. " *
              "The GPU solver is not available and sweeps run on the CPU. Install a supported version " *
              "in the default environment, for example: julia -e 'using Pkg; Pkg.add(name=\"CUDA\", version=\"$(last(_SUPPORTED_CUDA_MAJORS))\")'"
        return
    end
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
