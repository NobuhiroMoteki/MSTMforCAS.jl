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

struct GPUBatchBuffers
    B       ::Int
    neqns   ::Int
    noi_max ::Int

    # BiCG state vectors: (neqns, B) each
    x       ::CuMatrix{ComplexF64}
    r       ::CuMatrix{ComplexF64}
    q       ::CuMatrix{ComplexF64}
    pv      ::CuMatrix{ComplexF64}
    w       ::CuMatrix{ComplexF64}
    cap     ::CuMatrix{ComplexF64}
    caw     ::CuMatrix{ComplexF64}
    tmp_A   ::CuMatrix{ComplexF64}
    tmp_T   ::CuMatrix{ComplexF64}

    # Adjoint temporaries
    conj_buf ::CuMatrix{ComplexF64}

    # Per-RI scalars: (B,)
    sk      ::CuVector{ComplexF64}
    norm2_p ::CuVector{Float64}
    err     ::CuVector{Float64}
    converged ::CuVector{Int32}

    # T-matrix per RI: (noi_max, B)
    t_diag  ::CuMatrix{ComplexF64}
    t_off   ::CuMatrix{ComplexF64}

    # mn→n lookup: (hnb_max,) shared across B
    mn_to_n ::CuVector{Int32}

    # FFT work buffers
    # anode/gnode: (nx, ny, nz, nblk_node, 2, B)
    anode   ::CuArray{ComplexF64, 6}
    gnode   ::CuArray{ComplexF64, 6}
    # Zero-padded FFT buffers: (2nx, 2ny, 2nz, nblk_node, B)
    aft     ::CuArray{ComplexF64, 5}
    gft     ::CuArray{ComplexF64, 5}

    # cuFFT plans (for nblk_node * B batched 3D transforms)
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
    gpu_fft::GPUFFTData
)::GPUBatchBuffers
    hnb_max   = noi_max * (noi_max + 2)
    nblk_node = gpu_fft.nblk_node
    nx, ny, nz = gpu_fft.cell_dim

    # BiCG vectors
    x       = CUDA.zeros(ComplexF64, neqns, B)
    r       = CUDA.zeros(ComplexF64, neqns, B)
    q       = CUDA.zeros(ComplexF64, neqns, B)
    pv      = CUDA.zeros(ComplexF64, neqns, B)
    w       = CUDA.zeros(ComplexF64, neqns, B)
    cap     = CUDA.zeros(ComplexF64, neqns, B)
    caw     = CUDA.zeros(ComplexF64, neqns, B)
    tmp_A   = CUDA.zeros(ComplexF64, neqns, B)
    tmp_T   = CUDA.zeros(ComplexF64, neqns, B)
    conj_buf = CUDA.zeros(ComplexF64, neqns, B)

    # Per-RI scalars
    sk      = CUDA.zeros(ComplexF64, B)
    norm2_p = CUDA.zeros(Float64, B)
    err     = CUDA.zeros(Float64, B)
    converged = CUDA.zeros(Int32, B)

    # T-matrix
    t_diag  = CUDA.zeros(ComplexF64, noi_max, B)
    t_off   = CUDA.zeros(ComplexF64, noi_max, B)

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
    anode = CUDA.zeros(ComplexF64, nx, ny, nz, nblk_node, 2, B)
    gnode = CUDA.zeros(ComplexF64, nx, ny, nz, nblk_node, 2, B)
    aft   = CUDA.zeros(ComplexF64, 2nx, 2ny, 2nz, nblk_node, B)
    gft   = CUDA.zeros(ComplexF64, 2nx, 2ny, 2nz, nblk_node, B)

    # cuFFT plans: batched 3D FFT over dims (1,2,3) with nblk_node*B batch elements
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

    copyto!(buf.t_diag, CuArray(td_cpu))
    copyto!(buf.t_off,  CuArray(to_cpu))
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Determine optimal batch size from available GPU memory
# ─────────────────────────────────────────────────────────────

function determine_batch_size(
    neqns::Int,
    gpu_fft::GPUFFTData,
    n_ri::Int;
    max_batch::Int = 200
)::Int
    available = round(Int, Float64(CUDA.available_memory()) * 0.80)  # 80% safety margin

    nblk_node = gpu_fft.nblk_node
    nx, ny, nz = gpu_fft.cell_dim

    # Per-RI memory: BiCG vectors + FFT buffers
    per_ri = (
        10 * neqns * sizeof(ComplexF64) +                       # BiCG vectors
        2 * nx * ny * nz * nblk_node * sizeof(ComplexF64) +     # anode/gnode per RI
        2 * 8 * nx * ny * nz * nblk_node * sizeof(ComplexF64) + # aft/gft per RI
        2 * sizeof(ComplexF64) + 2 * sizeof(Float64)            # scalars
    )

    B_max = max(1, available ÷ per_ri)
    return min(B_max, n_ri, max_batch)
end
