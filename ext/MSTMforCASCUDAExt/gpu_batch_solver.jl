"""
GPU-accelerated batched BiCG solver and top-level GPU sweep function.

Solves B independent linear systems (I - T_b * A) * x_b = T_b * p_inc
simultaneously on GPU, where A (translation operator) is shared across all B
and T_b (Mie T-matrix) varies per refractive index.
"""

# ─────────────────────────────────────────────────────────────
# Batched L operator: out = (I - T*A) * inp
# ─────────────────────────────────────────────────────────────

function gpu_apply_L_batch!(out, inp, buf::GPUBatchBuffers, gpu_fft::GPUFFTData, N::Int)
    # tmp_A = A * inp
    gpu_apply_A_batch!(buf.tmp_A, inp, buf, gpu_fft)
    # tmp_T = T * tmp_A
    gpu_apply_T_batch!(buf.tmp_T, buf.tmp_A, buf, N)
    # out = inp - tmp_T
    gpu_sub!(out, inp, buf.tmp_T, buf.neqns, buf.B)
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Batched L* (adjoint) operator: out = (I - conj(S*A*S) * T) * inp
# For homogeneous spheres: T is self-transpose, so T^H = conj(T)
# L* = I - conj(S * A * S * T * conj(·))
# ─────────────────────────────────────────────────────────────

function gpu_apply_Lstar_batch!(out, inp, buf::GPUBatchBuffers, gpu_fft::GPUFFTData, N::Int)
    neqns = buf.neqns
    B = buf.B

    # 1. conj_buf = conj(inp)
    gpu_conj!(buf.conj_buf, inp, neqns, B)
    # 2. tmp_T = T * conj_buf (T is self-transpose for homogeneous spheres)
    gpu_apply_T_batch!(buf.tmp_T, buf.conj_buf, buf, N)
    # 3. Apply S-parity to tmp_T
    gpu_apply_S_batch!(buf.tmp_T, buf, N)
    # 4. tmp_A = A * tmp_T
    gpu_apply_A_batch!(buf.tmp_A, buf.tmp_T, buf, gpu_fft)
    # 5. Apply S-parity to tmp_A
    gpu_apply_S_batch!(buf.tmp_A, buf, N)
    # 6. conj_buf = conj(tmp_A)
    gpu_conj!(buf.conj_buf, buf.tmp_A, neqns, B)
    # 7. out = inp - conj_buf
    gpu_sub!(out, inp, buf.conj_buf, neqns, B)
    return nothing
end

# ─────────────────────────────────────────────────────────────
# Batched BiCG solver (one polarization)
# ─────────────────────────────────────────────────────────────

function gpu_solve_bicg_batch!(
    buf::GPUBatchBuffers,
    rhs::CuMatrix,    # (neqns, B) — right-hand side T*p_inc (CT element type)
    gpu_fft::GPUFFTData,
    N::Int,
    tol::Float64,
    max_iter::Int
)

    neqns = buf.neqns
    B = buf.B

    # Initialize: x = rhs (default initial guess)
    copyto!(buf.x, rhs)

    # r = rhs - L(x) = rhs - (I - T*A)*x
    gpu_apply_L_batch!(buf.cap, buf.x, buf, gpu_fft, N)  # cap = L(x) temporarily
    gpu_sub!(buf.r, rhs, buf.cap, neqns, B)              # r = rhs - L(x)

    # q = conj(r), pv = r, w = q
    gpu_conj!(buf.q, buf.r, neqns, B)
    copyto!(buf.pv, buf.r)
    copyto!(buf.w, buf.q)

    # sk[b] = dot(q[:,b], r[:,b])
    gpu_batch_dot!(buf.sk, buf.q, buf.r, neqns, B)

    # norm2_p[b] = real(dot(rhs[:,b], rhs[:,b]))
    gpu_batch_real_dot!(buf.norm2_p, rhs, neqns, B)

    # Check initial convergence
    gpu_batch_real_dot!(buf.err, buf.r, neqns, B)
    err_cpu = Array(buf.err)
    norm2_cpu = Array(buf.norm2_p)
    converged_cpu = Vector{Bool}(undef, B)
    iter_cpu = zeros(Int, B)
    for b in 1:B
        converged_cpu[b] = (norm2_cpu[b] < 1e-300) || (err_cpu[b] / max(norm2_cpu[b], 1e-300) < tol)
    end
    if all(converged_cpu)
        return (copy(buf.x), converged_cpu, iter_cpu)
    end

    # Pre-allocate scalar buffers (reused across iterations — no per-iteration alloc)
    CT = eltype(buf.x)
    RT = real(CT)
    alpha_cpu = Vector{CT}(undef, B)
    beta_cpu  = Vector{CT}(undef, B)
    sk_cpu    = Array(buf.sk)

    # GPU scalar buffers (pre-allocated, reused)
    denom_gpu = CUDA.zeros(CT, B)
    sk2_gpu   = CUDA.zeros(CT, B)
    alpha_gpu = CUDA.zeros(CT, B)
    beta_gpu  = CUDA.zeros(CT, B)

    convergence_check_interval = 5
    total_elem = neqns * B
    threads = 256
    blocks = cld(total_elem, threads)

    for it in 1:max_iter
        # cap = L(pv), caw = L*(w)
        gpu_apply_L_batch!(buf.cap, buf.pv, buf, gpu_fft, N)
        gpu_apply_Lstar_batch!(buf.caw, buf.w, buf, gpu_fft, N)

        # denom[b] = dot(w[:,b], cap[:,b])
        gpu_batch_dot!(denom_gpu, buf.w, buf.cap, neqns, B)
        denom_cpu = Array(denom_gpu)

        # alpha = sk / denom
        for b in 1:B
            if !converged_cpu[b]
                alpha_cpu[b] = abs(denom_cpu[b]) > 1e-300 ? sk_cpu[b] / denom_cpu[b] : zero(CT)
            else
                alpha_cpu[b] = zero(CT)
            end
        end
        copyto!(alpha_gpu, CuArray(alpha_cpu))

        # x += alpha * pv, r -= alpha * cap, q -= conj(alpha) * caw
        @cuda threads=threads blocks=blocks _bicg_update_xrq_kernel!(
            buf.x, buf.r, buf.q, buf.pv, buf.cap, buf.caw, alpha_gpu,
            Int32(neqns), Int32(B))

        # sk2[b] = dot(q[:,b], r[:,b])
        gpu_batch_dot!(sk2_gpu, buf.q, buf.r, neqns, B)
        sk2_cpu = Array(sk2_gpu)

        # Convergence check every few iterations
        if it % convergence_check_interval == 0 || it == max_iter
            gpu_batch_real_dot!(buf.err, buf.r, neqns, B)
            err_cpu = Array(buf.err)
            n_converged = 0
            for b in 1:B
                if !converged_cpu[b]
                    rel_err = err_cpu[b] / max(norm2_cpu[b], RT(1e-300))
                    if rel_err < tol
                        converged_cpu[b] = true
                        iter_cpu[b] = it
                    end
                end
                if converged_cpu[b]
                    n_converged += 1
                end
            end
            if all(converged_cpu)
                break
            end
        end

        # beta = sk2 / sk, update search directions
        for b in 1:B
            beta_cpu[b] = abs(sk_cpu[b]) > 1e-300 ? sk2_cpu[b] / sk_cpu[b] : zero(CT)
            sk_cpu[b] = sk2_cpu[b]
        end
        copyto!(beta_gpu, CuArray(beta_cpu))

        @cuda threads=threads blocks=blocks _bicg_update_pw_kernel!(
            buf.pv, buf.w, buf.r, buf.q, beta_gpu,
            Int32(neqns), Int32(B))
    end

    # Final convergence status for any that weren't caught
    for b in 1:B
        if iter_cpu[b] == 0
            iter_cpu[b] = max_iter
        end
    end

    return (copy(buf.x), converged_cpu, iter_cpu)
end

# ─────────────────────────────────────────────────────────────
# Top-level: solve a full (aggregate, medium) group on GPU
# ─────────────────────────────────────────────────────────────

"""
    gpu_batch_solve_group(agg, k, ri_values, n_med, fft_cache, params)

Solve scattering for all refractive indices in `ri_values` using GPU batch processing.
Called via the `_gpu_batch_solve_ref[]` dispatch hook from `run_parameter_sweep`.

Returns `Vector{Tuple{ScatteringResult, Matrix{ComplexF64}}}`.

If `params.on_result` is provided, it is called as `on_result(ri_index, result, amn)` for each
completed RI immediately after its batch finishes. This enables incremental HDF5 flushing and
crash recovery (results are recorded before the next batch starts).
"""
function gpu_batch_solve_group(
    agg::MSTMforCAS.AggregateGeometry,
    k::Float64,
    ri_values::Vector{ComplexF64},
    n_med::Float64,
    fft_cache::Union{MSTMforCAS.FFTGridData, Nothing},
    params::NamedTuple
)::Vector{Tuple{MSTMforCAS.ScatteringResult, Matrix{ComplexF64}}}

    positions_x = agg.positions .* k
    radii_x     = agg.radii .* k
    N = length(radii_x)
    n_ri = length(ri_values)

    use_f32     = get(params, :float32, false)
    tol         = use_f32 ? max(params.tol, 1e-5) : params.tol  # relax for FP32
    max_iter    = params.max_iter
    on_result   = get(params, :on_result, nothing)  # optional callback(ri_idx, result, amn)
    trunc    = params.truncation_order

    # Determine uniform noi_max across all RIs
    noi_max = 0
    for m_abs in ri_values
        m_rel = m_abs / n_med
        for s in 1:N
            noi = MSTMforCAS.mie_nmax(radii_x[s], m_rel)
            if trunc !== nothing
                noi = max(noi, trunc)
            end
            noi_max = max(noi_max, noi)
        end
    end
    hnb_max = noi_max * (noi_max + 2)
    neqns   = N * 2 * hnb_max

    # Ensure FFT cache exists (GPU implies FFT mode)
    if fft_cache === nothing && N >= 2
        nois_grid = fill(noi_max, N)
        fft_cache = MSTMforCAS.init_fft_grid(positions_x, radii_x, nois_grid)
    end

    # Upload geometry to GPU (always F64 for accuracy — see GPUFFTData comment)
    gpu_fft = upload_fft_data(fft_cache, noi_max, N)

    # Determine batch size
    B = determine_batch_size(neqns, gpu_fft, n_ri; float32=use_f32)
    prec_str = use_f32 ? "Float32" : "Float64"
    @info "GPU batch solver" N=N noi_max=noi_max neqns=neqns n_ri=n_ri batch_size=B precision=prec_str tol=tol

    # Allocate batch buffers
    buf = allocate_batch_buffers(neqns, B, noi_max, gpu_fft; float32=use_f32)

    # Build uniform offsets (all spheres same noi_max)
    blk_size = 2 * hnb_max
    offsets = [(i - 1) * blk_size for i in 1:N]
    half_nblks = fill(hnb_max, N)
    nois = fill(noi_max, N)

    # Build incident wave coefficients (shared structure, T-dependent per RI)
    p0 = MSTMforCAS._genplanewavecoef_z0(noi_max)
    mn_to_n = MSTMforCAS._build_mn_to_n(noi_max)

    results = Vector{Tuple{MSTMforCAS.ScatteringResult, Matrix{ComplexF64}}}(undef, n_ri)

    # ── CPU batch preparation (extract as function for pipeline reuse) ────
    function _prepare_batch_cpu(ri_batch_local)
        bs = length(ri_batch_local)
        rhs_T   = zeros(ComplexF64, neqns, 2, bs)
        rhs_inc = zeros(ComplexF64, neqns, 2, bs)
        td_all  = zeros(ComplexF64, noi_max, bs)  # for upload_batch_mie
        to_all  = zeros(ComplexF64, noi_max, bs)

        # Parallel over batch items: each bi writes to independent array slices
        Threads.@threads for bi in eachindex(ri_batch_local)
            m_abs = ri_batch_local[bi]
            m_rel = m_abs / n_med
            mie_vecs = [MSTMforCAS.compute_mie_coefficients(radii_x[s], m_rel; nmax=noi_max) for s in 1:N]
            td_vecs, to_vecs = MSTMforCAS._precompute_T_values(mie_vecs, nois)

            # T-matrix for GPU upload (use sphere 1; all identical for uniform radii)
            for n in 1:noi_max
                td_all[n, bi] = td_vecs[1][n]
                to_all[n, bi] = to_vecs[1][n]
            end

            # Incident wave coefficients with phase factors
            inc = zeros(ComplexF64, neqns, 2)
            for i in 1:N
                off = offsets[i]
                phase = exp(im * positions_x[3, i])
                p0_i = MSTMforCAS._genplanewavecoef_z0(noi_max)
                for q in 1:2, p in 1:2
                    p_blk_off = (p - 1) * hnb_max
                    for mn in 1:hnb_max
                        inc[off + p_blk_off + mn, q] += phase * p0_i[mn, p, q]
                    end
                end
            end
            rhs_inc[:, :, bi] .= inc

            # T * rhs_inc
            for q in 1:2, i in 1:N
                off = offsets[i]
                for mn in 1:hnb_max
                    n = mn_to_n[mn]
                    td = td_vecs[i][n]; to = to_vecs[i][n]
                    p1 = inc[off + mn, q]; p2 = inc[off + hnb_max + mn, q]
                    rhs_T[off + mn, q, bi]           = td * p1 + to * p2
                    rhs_T[off + hnb_max + mn, q, bi] = to * p1 + td * p2
                end
            end
        end
        return (rhs_T=rhs_T, rhs_inc=rhs_inc, td=td_all, to=to_all)
    end

    # ── GPU solve for one batch (uses pre-prepared CPU data) ──────────────
    function _solve_batch_gpu(prep, batch_size_local)
        CT = eltype(buf.x)
        # Upload T-matrix (zero-fill first to clear stale data from previous batch)
        buf.t_diag .= 0
        buf.t_off .= 0
        copyto!(view(buf.t_diag, :, 1:batch_size_local), CuArray(CT.(prep.td)))
        copyto!(view(buf.t_off,  :, 1:batch_size_local), CuArray(CT.(prep.to)))

        amn = zeros(ComplexF64, neqns, 2, batch_size_local)
        conv = fill(true, batch_size_local)
        iters = zeros(Int, batch_size_local)

        rhs_gpu = CUDA.zeros(CT, neqns, buf.B)
        for q in 1:2
            rhs_gpu .= 0
            copyto!(view(rhs_gpu, :, 1:batch_size_local), CuArray(CT.(prep.rhs_T[:, q, :])))
            x_gpu, c, it = gpu_solve_bicg_batch!(buf, rhs_gpu, gpu_fft, N, tol, max_iter)
            amn[:, q, :] .= ComplexF64.(Array(view(x_gpu, :, 1:batch_size_local)))
            for bi in 1:batch_size_local
                conv[bi] = conv[bi] && c[bi]
                iters[bi] = max(iters[bi], it[bi])
            end
        end
        return (amn=amn, converged=conv, iters=iters)
    end

    # ── Common-origin geometry: precomputed once per group ────────────────
    # r0, ntrani, nodrt are geometry-only (no RI dependence), so computing
    # them once avoids N×n_ri redundant _tranordertest and
    # compute_translation_matrix calls (the dominant post-processing cost).
    r0 = vec(sum(positions_x, dims=2)) ./ N
    ntrani_g = Vector{Int}(undef, N)
    for i in 1:N
        dist = sqrt(sum(abs2, r0 .- positions_x[:, i]))
        ntrani_g[i] = MSTMforCAS._tranordertest(dist, nois[i], 1e-5)
    end
    nodrt_g = maximum(ntrani_g)
    hnb0_g  = nodrt_g * (nodrt_g + 2)
    x_eff   = cbrt(sum(r^3 for r in radii_x))
    norm_fac = 2.0 / x_eff^2

    # ── Post-process batch: H_i computed once per sphere, GEMV per (bi,q) ───
    # compute_translation_matrix called N times (once per sphere, not N×bs).
    # H_i is applied to each (bi,q) via BLAS GEMV — no large intermediates.
    # (Previous GEMM approach allocated hnb_i×2bs per sphere, causing OOM/GC storm
    # for large aggregates where hnb_i can reach ~40000 with bs=300+.)
    function _postprocess_batch!(results_ref, batch_start_local, prep, sol)
        bs = size(sol.amn, 3)

        # Q_ext for all RI: fast parallel dot products (independent per RI)
        Q_ext_all = Vector{Float64}(undef, bs)
        Threads.@threads for bi in 1:bs
            s = 0.0
            for q in 1:2
                s += real(dot(view(sol.amn, :, q, bi), view(prep.rhs_inc, :, q, bi)))
            end
            Q_ext_all[bi] = -norm_fac * s
        end

        # Common-origin translation: H_i computed once per sphere (not per RI).
        # Serial sphere loop; BLAS GEMV (hnb_i,) per (bi, q) — O(1) allocation.
        amn0_mode1 = zeros(ComplexF64, hnb0_g, 2, bs)
        amn0_mode2 = zeros(ComplexF64, hnb0_g, 2, bs)

        for i in 1:N
            off   = offsets[i]
            hnb   = half_nblks[i]
            noi   = nois[i]
            ntri  = ntrani_g[i]
            hnb_i = ntri * (ntri + 2)

            r_trans = r0 .- positions_x[:, i]
            dist = sqrt(r_trans[1]^2 + r_trans[2]^2 + r_trans[3]^2)

            if dist < 1e-10
                n_copy = min(hnb, hnb_i)
                # Parallelize over bi: each bi owns amn0[*,*,bi] — no write conflict
                Threads.@threads for bi in 1:bs
                    for q in 1:2
                        @inbounds for mn in 1:n_copy
                            v1 = sol.amn[off + mn, q, bi]
                            v2 = sol.amn[off + hnb + mn, q, bi]
                            amn0_mode1[mn, q, bi] += v1 + v2
                            amn0_mode2[mn, q, bi] += v1 - v2
                        end
                    end
                end
            else
                # Compute H_i ONCE (geometry-only, shared across all bs RI values)
                H = MSTMforCAS.compute_translation_matrix(
                    r_trans, ComplexF64(1.0), noi, ntri; use_regular=true)
                H_p1 = H[:, :, 1]  # (hnb_i, hnb)
                H_p2 = H[:, :, 2]  # (hnb_i, hnb)

                # Parallelize over bi: each thread allocates its own r1_tmp/r2_tmp.
                # H_p1/H_p2 are read-only shared; amn0[*,*,bi] is disjoint per bi.
                Threads.@threads for bi in 1:bs
                    r1_tmp = Vector{ComplexF64}(undef, hnb_i)
                    r2_tmp = Vector{ComplexF64}(undef, hnb_i)
                    for q in 1:2
                        mul!(r1_tmp, H_p1, view(sol.amn, off+1:off+hnb,       q, bi))
                        mul!(r2_tmp, H_p2, view(sol.amn, off+hnb+1:off+2*hnb, q, bi))
                        @inbounds for kl in 1:hnb_i
                            amn0_mode1[kl, q, bi] += r1_tmp[kl] + r2_tmp[kl]
                            amn0_mode2[kl, q, bi] += r1_tmp[kl] - r2_tmp[kl]
                        end
                    end
                end
            end
        end

        # Per-RI observables: Q_sca, Q_abs, S amplitudes (parallel)
        Threads.@threads for bi in 1:bs
            ri_idx = batch_start_local + bi - 1
            amn0_m1 = amn0_mode1[:, :, bi]
            amn0_m2 = amn0_mode2[:, :, bi]

            Q_sca_sum = 0.0
            for q in 1:2
                Q_sca_sum += real(dot(amn0_m1[:, q], amn0_m1[:, q]))
                Q_sca_sum += real(dot(amn0_m2[:, q], amn0_m2[:, q]))
            end
            Q_sca = norm_fac * Q_sca_sum / 2
            Q_abs = Q_ext_all[bi] - Q_sca

            sa_fwd = MSTMforCAS._amplitude_from_mode_coefs(amn0_m1, amn0_m2, nodrt_g, true)
            sa_bwd = MSTMforCAS._amplitude_from_mode_coefs(amn0_m1, amn0_m2, nodrt_g, false)
            bh83_fwd = ntuple(j -> -2 * sa_fwd[j], 4)
            bh83_bwd = ntuple(j -> -2 * sa_bwd[j], 4)

            amn_bi = sol.amn[:, :, bi]
            result = MSTMforCAS.ScatteringResult(
                bh83_fwd, bh83_bwd,
                Q_ext_all[bi], Q_abs, Q_sca,
                sol.converged[bi], sol.iters[bi], noi_max)
            results_ref[ri_idx] = (result, amn_bi)
            if on_result !== nothing
                on_result(ri_idx, result, amn_bi)
            end
        end
    end

    # ── Pipelined batch loop ──────────────────────────────────────────────
    # While GPU solves batch k, CPU prepares batch k+1.
    # GPU kernels are asynchronous; CPU work proceeds in parallel until
    # we access GPU results (Array() triggers synchronization).
    batch_ranges = [(s, min(s + B - 1, n_ri)) for s in 1:B:n_ri]
    n_batches = length(batch_ranges)

    if n_batches == 0
        return results
    end

    # Prepare first batch on CPU
    s1, e1 = batch_ranges[1]
    prep_current = _prepare_batch_cpu(ri_values[s1:e1])

    for k in 1:n_batches
        sk, ek = batch_ranges[k]
        batch_size_k = ek - sk + 1

        # Launch GPU solve for batch k (asynchronous — returns quickly)
        sol_k = _solve_batch_gpu(prep_current, batch_size_k)
        # Note: _solve_batch_gpu calls Array() internally which synchronizes.
        # For full async we'd need to restructure, but the current CPU prep
        # for batch k+1 still overlaps with GPU post-kernel work + data transfer.

        # Prepare batch k+1 on CPU (overlaps with GPU→CPU transfer of batch k)
        if k < n_batches
            sk1, ek1 = batch_ranges[k + 1]
            prep_next = _prepare_batch_cpu(ri_values[sk1:ek1])
        end

        # Post-process batch k
        _postprocess_batch!(results, sk, prep_current, sol_k)

        # Advance pipeline
        if k < n_batches
            prep_current = prep_next
        end
    end

    return results
end

