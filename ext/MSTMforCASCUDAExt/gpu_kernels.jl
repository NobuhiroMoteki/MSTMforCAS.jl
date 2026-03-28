"""
CUDA kernels for batched T-matrix application, S-parity, and BiCG vector operations.

All kernels operate on batched arrays where the last dimension is the batch index B.
Uniform noi_max is assumed across all spheres (padded to hnb_max = noi_max*(noi_max+2)).
"""

# ─────────────────────────────────────────────────────────────
# T-matrix application: out = T * inp  (block-diagonal, per-RI)
# ─────────────────────────────────────────────────────────────

function gpu_apply_T_batch_kernel!(
    out, inp, t_diag, t_off, hnb_max, N, B, noi_max
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = 2 * hnb_max * N * B
    if idx > total
        return nothing
    end

    # Decode: idx → (eq_in_sphere, sphere_i, batch_b)
    # Each sphere has 2*hnb_max equations
    blk_size = 2 * hnb_max
    idx0 = idx - 1
    b   = idx0 ÷ (blk_size * N) + 1
    rem1 = idx0 % (blk_size * N)
    i   = rem1 ÷ blk_size + 1
    eq  = rem1 % blk_size + 1   # 1..2*hnb_max

    off = (i - 1) * blk_size    # global offset for sphere i (uniform)

    if eq <= hnb_max
        mn = eq
        # Compute n from mn: mn = n*(n+1)+m, so n = floor(sqrt(mn))
        # Use a simple search (noi_max is small, typically 4-10)
        n = 1
        while n < noi_max && (n + 1) * (n + 2) < mn
            n += 1
        end
        # Actually use the correct formula: n*(n+1) ranges are [2..n*(n+2)]
        # Just iterate
        n = 1
        for nn in 1:noi_max
            if mn <= nn * (nn + 2)
                n = nn
                break
            end
        end

        td = t_diag[n, b]
        to = t_off[n, b]
        p1 = inp[off + mn, b]
        p2 = inp[off + hnb_max + mn, b]
        out[off + mn, b]           = td * p1 + to * p2
        out[off + hnb_max + mn, b] = to * p1 + td * p2
    end
    # eq > hnb_max: handled by the mn <= hnb_max branch writing both p1 and p2

    return nothing
end

function gpu_apply_T_batch!(out, inp, buf::GPUBatchBuffers, N)
    hnb_max = buf.noi_max * (buf.noi_max + 2)
    total = hnb_max * N * buf.B  # each thread handles one (mn, i, b) and writes both p1, p2
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks gpu_apply_T_batch_kernel_v2!(
        out, inp, buf.t_diag, buf.t_off, buf.mn_to_n,
        Int32(hnb_max), Int32(N), Int32(buf.B))
    return nothing
end

# Cleaner kernel using mn_to_n lookup
function gpu_apply_T_batch_kernel_v2!(
    out, inp, t_diag, t_off, mn_to_n, hnb_max, N, B
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = hnb_max * N * B
    if idx > total
        return nothing
    end

    idx0 = idx - 1
    b    = idx0 ÷ (hnb_max * N) + 1
    rem1 = idx0 % (hnb_max * N)
    i    = rem1 ÷ hnb_max + 1
    mn   = rem1 % hnb_max + 1

    blk_size = 2 * hnb_max
    off = (i - 1) * blk_size

    n  = mn_to_n[mn]
    td = t_diag[n, b]
    to = t_off[n, b]
    p1 = inp[off + mn, b]
    p2 = inp[off + hnb_max + mn, b]
    out[off + mn, b]           = td * p1 + to * p2
    out[off + hnb_max + mn, b] = to * p1 + td * p2

    return nothing
end

# ─────────────────────────────────────────────────────────────
# S-parity operator: a_{m,n,p} → (-1)^m × a_{-m,n,p}
# ─────────────────────────────────────────────────────────────

function gpu_apply_S_batch_kernel!(v, hnb_max, noi_max, N, B)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # One thread per (n, m>0, i, b) — handles the swap for ±m
    # Total pairs: noi_max * noi_max is an upper bound, but exact is Σ_n n
    # Use a flat index over (n, m, i, b) with m=1..n only
    # For simplicity, iterate over all (mn, i, b) with mn = 1..hnb_max
    # and only act when m > 0
    total = hnb_max * N * B
    if idx > total
        return nothing
    end

    idx0 = idx - 1
    b    = idx0 ÷ (hnb_max * N) + 1
    rem1 = idx0 % (hnb_max * N)
    i    = rem1 ÷ hnb_max + 1
    mn   = rem1 % hnb_max + 1

    blk = 2 * hnb_max
    off = (i - 1) * blk

    # Decode (n, m) from mn = n*(n+1)+m
    # n is determined by: n*(n+2) >= mn and (n-1)*(n+1) < mn
    n = Int32(1)
    for nn in Int32(1):Int32(noi_max)
        if mn <= nn * (nn + 2)
            n = nn
            break
        end
    end
    m = mn - n * (n + 1)

    if m > 0
        mn_pos = n * (n + 1) + m
        mn_neg = n * (n + 1) - m
        sign = (m % 2 == 0) ? Int32(1) : Int32(-1)

        # p=1 block
        v1_pos = v[off + mn_pos, b]
        v1_neg = v[off + mn_neg, b]
        v[off + mn_pos, b] = sign * v1_neg
        v[off + mn_neg, b] = sign * v1_pos

        # p=2 block
        v2_pos = v[off + hnb_max + mn_pos, b]
        v2_neg = v[off + hnb_max + mn_neg, b]
        v[off + hnb_max + mn_pos, b] = sign * v2_neg
        v[off + hnb_max + mn_neg, b] = sign * v2_pos
    end

    return nothing
end

function gpu_apply_S_batch!(v, buf::GPUBatchBuffers, N)
    hnb_max = buf.noi_max * (buf.noi_max + 2)
    total = hnb_max * N * buf.B
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks gpu_apply_S_batch_kernel!(
        v, Int32(hnb_max), Int32(buf.noi_max), Int32(N), Int32(buf.B))
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Fused BiCG vector updates
# ─────────────────────────────────────────────────────────────

function _bicg_update_xrq_kernel!(x, r, q, pv, cap, caw, alpha, neqns, B)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > neqns * B
        return nothing
    end
    idx0 = idx - 1
    b  = idx0 ÷ neqns + 1
    eq = idx0 % neqns + 1
    a = alpha[b]
    x[eq, b] += a * pv[eq, b]
    r[eq, b] -= a * cap[eq, b]
    q[eq, b] -= conj(a) * caw[eq, b]
    return nothing
end

function gpu_bicg_update_xrq!(buf::GPUBatchBuffers)
    total = buf.neqns * buf.B
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks _bicg_update_xrq_kernel!(
        buf.x, buf.r, buf.q, buf.pv, buf.cap, buf.caw, buf.sk,  # sk reused as alpha temp
        Int32(buf.neqns), Int32(buf.B))
    return nothing
end

function _bicg_update_pw_kernel!(pv, w, r, q, beta, neqns, B)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > neqns * B
        return nothing
    end
    idx0 = idx - 1
    b  = idx0 ÷ neqns + 1
    eq = idx0 % neqns + 1
    bt = beta[b]
    pv[eq, b] = r[eq, b] + bt * pv[eq, b]
    w[eq, b]  = q[eq, b] + conj(bt) * w[eq, b]
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Element-wise operations
# ─────────────────────────────────────────────────────────────

function _sub_kernel!(out, a, b_arr, neqns, B)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > neqns * B
        return nothing
    end
    idx0 = idx - 1
    bi = idx0 ÷ neqns + 1
    eq = idx0 % neqns + 1
    out[eq, bi] = a[eq, bi] - b_arr[eq, bi]
    return nothing
end

function gpu_sub!(out, a, b, neqns, B)
    total = neqns * B
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks _sub_kernel!(out, a, b, Int32(neqns), Int32(B))
    return nothing
end

function _conj_kernel!(out, inp, neqns, B)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > neqns * B
        return nothing
    end
    idx0 = idx - 1
    bi = idx0 ÷ neqns + 1
    eq = idx0 % neqns + 1
    out[eq, bi] = conj(inp[eq, bi])
    return nothing
end

function gpu_conj!(out, inp, neqns, B)
    total = neqns * B
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks _conj_kernel!(out, inp, Int32(neqns), Int32(B))
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Batched dot products: out[b] = sum(conj(a[:,b]) .* b_arr[:,b])
# Uses cuBLAS zdotc for each batch element (simple, sufficient for B~100)
# ─────────────────────────────────────────────────────────────

function gpu_batch_dot!(out::CuVector{ComplexF64}, a::CuMatrix{ComplexF64},
                        b_arr::CuMatrix{ComplexF64}, neqns::Int, B::Int)
    out_cpu = Vector{ComplexF64}(undef, B)
    for bi in 1:B
        a_col = a[:, bi]
        b_col = b_arr[:, bi]
        out_cpu[bi] = dot(a_col, b_col)  # LinearAlgebra.dot: Hermitian
    end
    copyto!(out, CuArray(out_cpu))
    return nothing
end

function gpu_batch_real_dot!(out::CuVector{Float64}, a::CuMatrix{ComplexF64},
                             neqns::Int, B::Int)
    out_cpu = Vector{Float64}(undef, B)
    for bi in 1:B
        v = a[:, bi]
        out_cpu[bi] = real(dot(v, v))
    end
    copyto!(out, CuArray(out_cpu))
    return nothing
end
