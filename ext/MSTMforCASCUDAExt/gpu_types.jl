"""
GPU data structures and CPU↔GPU transfer functions for the batched BiCG solver.
"""

# ─────────────────────────────────────────────────────────────
# GPU mirror of FFTGridData (geometry-dependent, shared across all RIs)
# ─────────────────────────────────────────────────────────────

struct GPUFFTData
    cell_dim      ::NTuple{3, Int}
    d_cell        ::Float64
    node_order    ::Int
    nblk_node     ::Int
    N             ::Int                          # number of spheres

    # Sphere-node translation matrices, padded to uniform hnb_max
    # shape: (nblk_node, hnb_max, 2, N)
    sphere_node_H ::CuArray{ComplexF64, 4}
    node_sphere_H ::CuArray{ComplexF64, 4}

    # Cell-to-cell translation kernels in frequency domain
    # shape: (2nx, 2ny, 2nz, nblk_node, nblk_node)
    cell_tran_fft_p1 ::CuArray{ComplexF64, 5}
    cell_tran_fft_p2 ::CuArray{ComplexF64, 5}

    # Sphere-to-grid-cell mapping: (3, N) Int32
    sphere_node_idx ::CuArray{Int32, 2}

    # Near-field data in CSR-like format
    # For sphere i, neighbors are nearfield_col[nearfield_ptr[i]:nearfield_ptr[i+1]-1]
    nearfield_ptr    ::CuArray{Int32, 1}         # length N+1
    nearfield_col    ::CuArray{Int32, 1}         # total neighbor entries
    nearfield_H_flat ::CuArray{ComplexF64, 1}    # flattened H matrices
    nearfield_H_offsets ::CuArray{Int32, 1}      # offset into flat for each entry
    hnb_max          ::Int                        # padded half-block size
end

# ─────────────────────────────────────────────────────────────
# Batch buffers — pre-allocated for B simultaneous BiCG solves
# ─────────────────────────────────────────────────────────────

struct GPUBatchBuffers{CT<:Complex, RT<:AbstractFloat}
    B       ::Int
    neqns   ::Int
    noi_max ::Int

    # BiCG state vectors: (neqns, B) each — CT = ComplexF64 or ComplexF32
    x       ::CuMatrix{CT}
    r       ::CuMatrix{CT}
    q       ::CuMatrix{CT}
    pv      ::CuMatrix{CT}
    w       ::CuMatrix{CT}
    cap     ::CuMatrix{CT}
    caw     ::CuMatrix{CT}
    tmp_A   ::CuMatrix{CT}
    tmp_T   ::CuMatrix{CT}

    # Adjoint temporaries
    conj_buf ::CuMatrix{CT}

    # Per-RI scalars: (B,)
    sk      ::CuVector{CT}
    norm2_p ::CuVector{RT}
    err     ::CuVector{RT}
    converged ::CuVector{Int32}

    # T-matrix per RI: (noi_max, B) — always CT (downcast from FP64 if float32 mode)
    t_diag  ::CuMatrix{CT}
    t_off   ::CuMatrix{CT}

    # mn→n lookup: (hnb_max,) shared across B
    mn_to_n ::CuVector{Int32}

    # FFT work buffers — CT
    anode   ::CuArray{CT, 6}
    gnode   ::CuArray{CT, 6}
    aft     ::CuArray{CT, 5}
    gft     ::CuArray{CT, 5}

    # cuFFT plans
    fft_plan  ::Any
    ifft_plan ::Any
end

# ─────────────────────────────────────────────────────────────
# Upload FFT data to GPU
# ─────────────────────────────────────────────────────────────

function upload_fft_data(
    fft_data::MSTMforCAS.FFTGridData,
    noi_max::Int,
    N::Int
)::GPUFFTData
    nblk_node = fft_data.nblk_node
    hnb_max   = noi_max * (noi_max + 2)

    # Pad and stack sphere_node_H[i] into (nblk_node, hnb_max, 2, N)
    s2n = zeros(ComplexF64, nblk_node, hnb_max, 2, N)
    n2s = zeros(ComplexF64, hnb_max, nblk_node, 2, N)
    for i in 1:N
        hi = fft_data.sphere_node_H[i]   # (nblk_node, hnb_i, 2)
        hnb_i = size(hi, 2)
        s2n[:, 1:hnb_i, :, i] .= hi
        hi_r = fft_data.node_sphere_H[i] # (hnb_i, nblk_node, 2)
        n2s[1:hnb_i, :, :, i] .= hi_r
    end

    # Sphere grid indices
    sn_idx = Int32.(fft_data.sphere_node)  # (3, N)

    # Near-field CSR
    ptr = Int32[1]
    col = Int32[]
    h_flat = ComplexF64[]
    h_offsets = Int32[]
    flat_pos = Int32(1)
    for i in 1:N
        pairs_i = fft_data.neighbor_pairs[i]
        for (k, j) in enumerate(pairs_i)
            push!(col, Int32(j))
            push!(h_offsets, flat_pos)
            H_ij = fft_data.nearfield_H[i][k]  # (hnb_i, hnb_j, 2)
            hnb_i = size(H_ij, 1)
            hnb_j = size(H_ij, 2)
            # Pad to (hnb_max, hnb_max, 2) and flatten
            H_pad = zeros(ComplexF64, hnb_max, hnb_max, 2)
            H_pad[1:hnb_i, 1:hnb_j, :] .= H_ij
            append!(h_flat, vec(H_pad))
            flat_pos += Int32(hnb_max * hnb_max * 2)
        end
        push!(ptr, Int32(length(col) + 1))
    end

    return GPUFFTData(
        fft_data.cell_dim,
        fft_data.d_cell,
        fft_data.node_order,
        nblk_node,
        N,
        CuArray(s2n),
        CuArray(n2s),
        CuArray(fft_data.cell_tran_fft_p1),
        CuArray(fft_data.cell_tran_fft_p2),
        CuArray(sn_idx),
        CuArray(ptr),
        CuArray(col),
        CuArray(h_flat),
        CuArray(h_offsets),
        hnb_max,
    )
end

# ─────────────────────────────────────────────────────────────
# Allocate batch buffers
# ─────────────────────────────────────────────────────────────

function allocate_batch_buffers(
    neqns::Int,
    B::Int,
    noi_max::Int,
    gpu_fft::GPUFFTData;
    float32::Bool = false
)
    CT = float32 ? ComplexF32 : ComplexF64
    RT = float32 ? Float32 : Float64
    hnb_max   = noi_max * (noi_max + 2)
    nblk_node = gpu_fft.nblk_node
    nx, ny, nz = gpu_fft.cell_dim

    # BiCG vectors
    x       = CUDA.zeros(CT, neqns, B)
    r       = CUDA.zeros(CT, neqns, B)
    q       = CUDA.zeros(CT, neqns, B)
    pv      = CUDA.zeros(CT, neqns, B)
    w       = CUDA.zeros(CT, neqns, B)
    cap     = CUDA.zeros(CT, neqns, B)
    caw     = CUDA.zeros(CT, neqns, B)
    tmp_A   = CUDA.zeros(CT, neqns, B)
    tmp_T   = CUDA.zeros(CT, neqns, B)
    conj_buf = CUDA.zeros(CT, neqns, B)

    # Per-RI scalars
    sk      = CUDA.zeros(CT, B)
    norm2_p = CUDA.zeros(RT, B)
    err     = CUDA.zeros(RT, B)
    converged = CUDA.zeros(Int32, B)

    # T-matrix (will be downcast from FP64 at upload time)
    t_diag  = CUDA.zeros(CT, noi_max, B)
    t_off   = CUDA.zeros(CT, noi_max, B)

    # mn→n lookup
    mn_to_n_cpu = Vector{Int32}(undef, hnb_max)
    for n in 1:noi_max
        for m in -n:n
            mn = n * (n + 1) + m
            mn_to_n_cpu[mn] = Int32(n)
        end
    end
    mn_to_n = CuArray(mn_to_n_cpu)

    # FFT work buffers
    anode = CUDA.zeros(CT, nx, ny, nz, nblk_node, 2, B)
    gnode = CUDA.zeros(CT, nx, ny, nz, nblk_node, 2, B)
    aft   = CUDA.zeros(CT, 2nx, 2ny, 2nz, nblk_node, B)
    gft   = CUDA.zeros(CT, 2nx, 2ny, 2nz, nblk_node, B)

    # cuFFT plans
    fft_plan  = CUFFT.plan_fft!(aft, (1, 2, 3))
    ifft_plan = CUFFT.plan_ifft!(gft, (1, 2, 3))

    return GPUBatchBuffers(
        B, neqns, noi_max,
        x, r, q, pv, w, cap, caw, tmp_A, tmp_T, conj_buf,
        sk, norm2_p, err, converged,
        t_diag, t_off, mn_to_n,
        anode, gnode, aft, gft,
        fft_plan, ifft_plan,
    )
end

# ─────────────────────────────────────────────────────────────
# Upload Mie/T-matrix data for a batch of RIs
# ─────────────────────────────────────────────────────────────

function upload_batch_mie!(
    buf::GPUBatchBuffers,
    ri_batch::Vector{ComplexF64},
    radii_x::Vector{Float64},
    n_med::Float64,
    noi_max::Int
)
    B = length(ri_batch)
    N = length(radii_x)
    td_cpu = zeros(ComplexF64, noi_max, B)
    to_cpu = zeros(ComplexF64, noi_max, B)

    for b in 1:B
        m_rel = ri_batch[b] / n_med
        # Compute Mie coefficients for all spheres (same RI, different radii)
        # For CAS aggregates: all radii nearly identical, but we handle general case
        for i in 1:N
            a_v, b_v = MSTMforCAS.compute_mie_coefficients(radii_x[i], m_rel; nmax=noi_max)
            for n in 1:min(length(a_v), noi_max)
                # Accumulate: for uniform noi all spheres contribute identically
                # but store per-sphere average for correctness
                td_cpu[n, b] = -(a_v[n] + b_v[n]) / 2
                to_cpu[n, b] = -(a_v[n] - b_v[n]) / 2
            end
        end
    end

    # Downcast to buffer element type (ComplexF32 if float32 mode)
    CT = eltype(buf.t_diag)
    copyto!(buf.t_diag, CuArray(CT.(td_cpu)))
    copyto!(buf.t_off,  CuArray(CT.(to_cpu)))
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Determine optimal batch size from available GPU memory
# ─────────────────────────────────────────────────────────────

function determine_batch_size(
    neqns::Int,
    gpu_fft::GPUFFTData,
    n_ri::Int;
    float32::Bool = false
)::Int
    available = round(Int, Float64(CUDA.available_memory()) * 0.80)  # 80% safety margin

    nblk_node = gpu_fft.nblk_node
    nx, ny, nz = gpu_fft.cell_dim
    # BiCG working vectors use the solver precision; FFT buffers are always Float64
    bicg_elem = float32 ? sizeof(ComplexF32) : sizeof(ComplexF64)

    # Per-RI memory: BiCG vectors + FFT translation working buffers
    per_ri = (
        10 * neqns * bicg_elem +                                    # BiCG vectors (precision-aware)
        2 * nx * ny * nz * nblk_node * sizeof(ComplexF64) +         # anode/gnode per RI
        2 * 8 * nx * ny * nz * nblk_node * sizeof(ComplexF64) +     # aft/gft per RI
        2 * sizeof(ComplexF64) + 2 * sizeof(Float64)                # scalars
    )

    # No artificial cap — let available VRAM determine the batch size.
    # Capped at n_ri so we never allocate more than needed.
    B_max = max(1, available ÷ per_ri)
    return min(B_max, n_ri)
end
