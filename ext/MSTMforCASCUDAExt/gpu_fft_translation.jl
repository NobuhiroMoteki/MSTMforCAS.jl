"""
GPU-accelerated batched FFT translation operator.

Implements the batched A operator: for B input vectors simultaneously,
compute out[:,b] = A * inp[:,b] using shared geometry data.

Pipeline: Sphere→Node → FFT convolution → Node→Sphere → Near-field correction
"""

# ─────────────────────────────────────────────────────────────
# Sphere-to-node: scatter sphere coefficients to grid nodes
# ─────────────────────────────────────────────────────────────

# Per-sphere staging kernel: write each sphere's contribution to its own slot
# in a staging buffer (nblk_node, 2, N, B), then scatter-add to grid.
function _sphere_to_node_stage_kernel!(
    staging,        # (nblk_node, 2, N, B) — per-sphere output, no race
    inp,            # (neqns, B)
    s2n_H,          # (nblk_node, hnb_max, 2, N)
    hnb_max, nblk_node, N, B
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = nblk_node * B * N
    if idx > total
        return nothing
    end

    idx0 = idx - 1
    i  = idx0 ÷ (nblk_node * B) + 1
    rem1 = idx0 % (nblk_node * B)
    b  = rem1 ÷ nblk_node + 1
    kl = rem1 % nblk_node + 1

    blk = 2 * hnb_max
    off = (i - 1) * blk

    for p in Int32(1):Int32(2)
        p_off = (p - 1) * hnb_max
        acc = zero(eltype(staging))
        for mn in Int32(1):Int32(hnb_max)
            acc += s2n_H[kl, mn, p, i] * inp[off + p_off + mn, b]
        end
        staging[kl, p, i, b] = acc
    end

    return nothing
end

# Scatter-add: accumulate per-sphere staging into grid cells (sequential per cell, no race)
function _sphere_to_node_scatter_kernel!(
    anode,          # (nx, ny, nz, nblk_node, 2, B)
    staging,        # (nblk_node, 2, N, B)
    sphere_idx,     # (3, N)
    nblk_node, N, B
)
    # One thread per (kl, p, b) — loops over all spheres for this output element
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = nblk_node * 2 * B
    if idx > total
        return nothing
    end

    idx0 = idx - 1
    b  = idx0 ÷ (nblk_node * 2) + 1
    rem1 = idx0 % (nblk_node * 2)
    p  = rem1 ÷ nblk_node + 1
    kl = rem1 % nblk_node + 1

    # Accumulate contributions from all spheres
    for i in Int32(1):Int32(N)
        ix = sphere_idx[1, i]
        iy = sphere_idx[2, i]
        iz = sphere_idx[3, i]
        anode[ix, iy, iz, kl, p, b] += staging[kl, p, i, b]
    end

    return nothing
end

function gpu_sphere_to_node_batch!(buf::GPUBatchBuffers, inp, gpu_fft::GPUFFTData)
    buf.anode .= 0
    N = gpu_fft.N
    nblk = gpu_fft.nblk_node
    CT = eltype(buf.anode)

    # Stage: per-sphere MVP (no race)
    staging = CUDA.zeros(CT, nblk, 2, N, buf.B)
    total1 = nblk * buf.B * N
    threads = 256
    @cuda threads=threads blocks=cld(total1, threads) _sphere_to_node_stage_kernel!(
        staging, inp, gpu_fft.sphere_node_H,
        Int32(gpu_fft.hnb_max), Int32(nblk), Int32(N), Int32(buf.B))

    # Scatter: accumulate to grid (one thread per output element, sequential over spheres)
    total2 = nblk * 2 * buf.B
    @cuda threads=threads blocks=cld(total2, threads) _sphere_to_node_scatter_kernel!(
        buf.anode, staging, gpu_fft.sphere_node_idx,
        Int32(nblk), Int32(N), Int32(buf.B))

    return nothing
end

# ─────────────────────────────────────────────────────────────
# FFT convolution: shared kernel multiply for all B
# ─────────────────────────────────────────────────────────────

function _fft_kernel_multiply_kernel!(
    gft,        # (2nx, 2ny, 2nz, nblk, B)
    aft,        # (2nx, 2ny, 2nz, nblk, B)
    tran_fft,   # (2nx, 2ny, 2nz, nblk, nblk) — shared across B
    n2x, n2y, n2z, nblk, B
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = n2x * n2y * n2z * nblk * B
    if idx > total
        return nothing
    end

    idx0 = idx - 1
    b    = idx0 ÷ (n2x * n2y * n2z * nblk) + 1
    rem1 = idx0 % (n2x * n2y * n2z * nblk)
    l    = rem1 ÷ (n2x * n2y * n2z) + 1
    pt   = rem1 % (n2x * n2y * n2z) + 1

    acc = zero(eltype(gft))
    for n in Int32(1):Int32(nblk)
        acc += tran_fft[pt, l, n] * aft[pt, n, b]
    end
    gft[pt, l, b] += acc

    return nothing
end

function gpu_fft_convolution_batch!(buf::GPUBatchBuffers, gpu_fft::GPUFFTData)
    nx, ny, nz = gpu_fft.cell_dim
    n2x, n2y, n2z = 2nx, 2ny, 2nz
    nblk = gpu_fft.nblk_node
    B = buf.B
    n_spatial = n2x * n2y * n2z

    buf.gnode .= 0

    # Process each p-mode
    for (p, tran_fft) in enumerate((gpu_fft.cell_tran_fft_p1, gpu_fft.cell_tran_fft_p2))
        # Zero-pad anode into aft
        buf.aft .= 0
        # Copy anode[:,:,:,:,p,:] into aft[1:nx,1:ny,1:nz,:,:]
        copyto!(view(buf.aft, 1:nx, 1:ny, 1:nz, :, :),
                view(buf.anode, :, :, :, :, p, :))

        # Forward FFT (batched over nblk*B)
        buf.fft_plan * buf.aft

        # Kernel multiply: gft = tran_fft * aft (summed over source index n)
        buf.gft .= 0
        # Reshape for the kernel: treat spatial dims as flat
        tran_flat = reshape(tran_fft, n_spatial, nblk, nblk)
        aft_flat  = reshape(buf.aft, n_spatial, nblk, B)
        gft_flat  = reshape(buf.gft, n_spatial, nblk, B)

        total = n_spatial * nblk * B
        threads = 256
        blocks = cld(total, threads)
        @cuda threads=threads blocks=blocks _fft_kernel_multiply_kernel!(
            gft_flat, aft_flat, tran_flat,
            Int32(n_spatial), Int32(1), Int32(1), Int32(nblk), Int32(B))

        # Inverse FFT
        buf.ifft_plan * buf.gft

        # Extract to gnode: gnode[:,:,:,:,p,:] += gft[1:nx,1:ny,1:nz,:,:]
        view(buf.gnode, :, :, :, :, p, :) .+= view(buf.gft, 1:nx, 1:ny, 1:nz, :, :)
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────
# Node-to-sphere: gather from grid nodes to sphere coefficients
# ─────────────────────────────────────────────────────────────

function _node_to_sphere_kernel!(
    out,            # (neqns, B)
    gnode,          # (nx, ny, nz, nblk_node, 2, B)
    n2s_H,          # (hnb_max, nblk_node, 2, N)
    sphere_idx,     # (3, N)
    hnb_max, nblk_node, N, B
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = hnb_max * B * N
    if idx > total
        return nothing
    end

    idx0 = idx - 1
    i  = idx0 ÷ (hnb_max * B) + 1
    rem1 = idx0 % (hnb_max * B)
    b  = rem1 ÷ hnb_max + 1
    mn = rem1 % hnb_max + 1

    ix = sphere_idx[1, i]
    iy = sphere_idx[2, i]
    iz = sphere_idx[3, i]

    blk = 2 * hnb_max
    off = (i - 1) * blk

    for p in Int32(1):Int32(2)
        p_off = (p - 1) * hnb_max
        acc = zero(eltype(out))
        for kl in Int32(1):Int32(nblk_node)
            acc += n2s_H[mn, kl, p, i] * gnode[ix, iy, iz, kl, p, b]
        end
        out[off + p_off + mn, b] += acc
    end

    return nothing
end

function gpu_node_to_sphere_batch!(out, buf::GPUBatchBuffers, gpu_fft::GPUFFTData)
    N = gpu_fft.N
    hnb_max = gpu_fft.hnb_max
    nblk = gpu_fft.nblk_node
    total = hnb_max * buf.B * N
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks _node_to_sphere_kernel!(
        out, buf.gnode, gpu_fft.node_sphere_H, gpu_fft.sphere_node_idx,
        Int32(hnb_max), Int32(nblk), Int32(N), Int32(buf.B))
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Near-field correction: direct Hankel MVP for neighbor pairs
# ─────────────────────────────────────────────────────────────

# Nearfield: sphere-centric approach (no race condition)
# One thread per (sphere_i, mn_t, b) — loops over all neighbors of sphere i
function _nearfield_sphere_kernel!(
    out,            # (neqns, B)
    inp,            # (neqns, B)
    nf_ptr,         # (N+1,) CSR row pointers
    nf_col,         # neighbor column indices
    nf_H_flat,      # flattened H matrices
    nf_H_offsets,   # offsets into flat
    hnb_max, N, B
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = hnb_max * B * N
    if idx > total
        return nothing
    end

    idx0 = idx - 1
    i    = idx0 ÷ (hnb_max * B) + 1
    rem1 = idx0 % (hnb_max * B)
    b    = rem1 ÷ hnb_max + 1
    mn_t = rem1 % hnb_max + 1

    blk = 2 * hnb_max
    off_i = (i - 1) * blk
    h_stride = hnb_max * hnb_max

    # Loop over all neighbors of sphere i (sequential — no race on out[i])
    for pair_idx in nf_ptr[i]:(nf_ptr[i + 1] - Int32(1))
        j = nf_col[pair_idx]
        h_off = nf_H_offsets[pair_idx]
        off_j = (j - 1) * blk

        for p in Int32(1):Int32(2)
            p_off = (p - 1) * hnb_max
            h_p_off = (p - 1) * h_stride

            acc = zero(eltype(out))
            for mn_s in Int32(1):Int32(hnb_max)
                h_idx = h_off + h_p_off + (mn_s - 1) * hnb_max + (mn_t - 1)
                acc += nf_H_flat[h_idx] * inp[off_j + p_off + mn_s, b]
            end
            out[off_i + p_off + mn_t, b] += acc
        end
    end

    return nothing
end

function gpu_nearfield_batch!(out, inp, buf::GPUBatchBuffers, gpu_fft::GPUFFTData)
    N = gpu_fft.N
    hnb_max = gpu_fft.hnb_max
    n_pairs_cpu = Array(gpu_fft.nearfield_ptr)[N + 1] - 1
    if n_pairs_cpu == 0
        return nothing
    end
    total = hnb_max * buf.B * N
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks _nearfield_sphere_kernel!(
        out, inp, gpu_fft.nearfield_ptr, gpu_fft.nearfield_col,
        gpu_fft.nearfield_H_flat, gpu_fft.nearfield_H_offsets,
        Int32(hnb_max), Int32(N), Int32(buf.B))
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Full batched A operator
# ─────────────────────────────────────────────────────────────

function gpu_apply_A_batch!(out, inp, buf::GPUBatchBuffers, gpu_fft::GPUFFTData)
    out .= 0

    # 1. Sphere → Node
    gpu_sphere_to_node_batch!(buf, inp, gpu_fft)

    # 2. FFT convolution (node → node)
    gpu_fft_convolution_batch!(buf, gpu_fft)

    # 3. Node → Sphere
    gpu_node_to_sphere_batch!(out, buf, gpu_fft)

    # 4. Near-field correction
    gpu_nearfield_batch!(out, inp, buf, gpu_fft)

    return nothing
end
