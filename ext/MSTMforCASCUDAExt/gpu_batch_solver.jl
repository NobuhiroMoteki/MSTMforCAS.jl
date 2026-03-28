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
    rhs::CuMatrix{ComplexF64},    # (neqns, B) — right-hand side T*p_inc
    gpu_fft::GPUFFTData,
    N::Int,
    tol::Float64,
    max_iter::Int
)::Tuple{CuMatrix{ComplexF64}, Vector{Bool}, Vector{Int}}

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

    # Allocate CPU-side scalar buffers
    alpha_cpu = Vector{ComplexF64}(undef, B)
    beta_cpu  = Vector{ComplexF64}(undef, B)
    sk_cpu    = Array(buf.sk)

    convergence_check_interval = 5

    for it in 1:max_iter
        # cap = L(pv), caw = L*(w)
        gpu_apply_L_batch!(buf.cap, buf.pv, buf, gpu_fft, N)
        gpu_apply_Lstar_batch!(buf.caw, buf.w, buf, gpu_fft, N)

        # denom[b] = dot(w[:,b], cap[:,b])
        denom_buf = CUDA.zeros(ComplexF64, B)
        gpu_batch_dot!(denom_buf, buf.w, buf.cap, neqns, B)
        denom_cpu = Array(denom_buf)

        # alpha = sk / denom
        for b in 1:B
            if !converged_cpu[b]
                alpha_cpu[b] = abs(denom_cpu[b]) > 1e-300 ? sk_cpu[b] / denom_cpu[b] : zero(ComplexF64)
            else
                alpha_cpu[b] = zero(ComplexF64)
            end
        end
        alpha_gpu = CuArray(alpha_cpu)

        # x += alpha * pv, r -= alpha * cap, q -= conj(alpha) * caw
        total = neqns * B
        threads = 256
        blocks = cld(total, threads)
        @cuda threads=threads blocks=blocks _bicg_update_xrq_kernel!(
            buf.x, buf.r, buf.q, buf.pv, buf.cap, buf.caw, alpha_gpu,
            Int32(neqns), Int32(B))

        # sk2[b] = dot(q[:,b], r[:,b])
        sk2_gpu = CUDA.zeros(ComplexF64, B)
        gpu_batch_dot!(sk2_gpu, buf.q, buf.r, neqns, B)
        sk2_cpu = Array(sk2_gpu)

        # Convergence check every few iterations
        if it % convergence_check_interval == 0 || it == max_iter
            gpu_batch_real_dot!(buf.err, buf.r, neqns, B)
            err_cpu = Array(buf.err)
            n_converged = 0
            for b in 1:B
                if !converged_cpu[b]
                    rel_err = err_cpu[b] / max(norm2_cpu[b], 1e-300)
                    if rel_err < tol
                        converged_cpu[b] = true
                        iter_cpu[b] = it
                    end
                end
                if converged_cpu[b]
                    n_converged += 1
                end
            end
            # Early exit if >= 80% converged
            if n_converged >= ceil(Int, 0.8 * B) || all(converged_cpu)
                # Mark remaining as done at max iterations
                for b in 1:B
                    if !converged_cpu[b]
                        iter_cpu[b] = it
                    end
                end
                break
            end
        end

        # beta = sk2 / sk, update search directions
        for b in 1:B
            beta_cpu[b] = abs(sk_cpu[b]) > 1e-300 ? sk2_cpu[b] / sk_cpu[b] : zero(ComplexF64)
            sk_cpu[b] = sk2_cpu[b]
        end
        beta_gpu = CuArray(beta_cpu)

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

    tol      = params.tol
    max_iter = params.max_iter
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

    # Upload geometry to GPU
    gpu_fft = upload_fft_data(fft_cache, noi_max, N)

    # Determine batch size
    B = determine_batch_size(neqns, gpu_fft, n_ri)
    @info "GPU batch solver" N=N noi_max=noi_max neqns=neqns n_ri=n_ri batch_size=B

    # Allocate batch buffers
    buf = allocate_batch_buffers(neqns, B, noi_max, gpu_fft)

    # Build uniform offsets (all spheres same noi_max)
    blk_size = 2 * hnb_max
    offsets = [(i - 1) * blk_size for i in 1:N]
    half_nblks = fill(hnb_max, N)
    nois = fill(noi_max, N)

    # Build incident wave coefficients (shared structure, T-dependent per RI)
    p0 = MSTMforCAS._genplanewavecoef_z0(noi_max)
    mn_to_n = MSTMforCAS._build_mn_to_n(noi_max)

    results = Vector{Tuple{MSTMforCAS.ScatteringResult, Matrix{ComplexF64}}}(undef, n_ri)

    # Process in batches
    for batch_start in 1:B:n_ri
        batch_end = min(batch_start + B - 1, n_ri)
        batch_size = batch_end - batch_start + 1
        ri_batch = ri_values[batch_start:batch_end]

        # Upload Mie/T-matrix data for this batch
        upload_batch_mie!(buf, ri_batch, radii_x, n_med, noi_max)

        # Build RHS for each RI on CPU, then upload
        # CPU solver does: (1) rhs_inc = phase * p0_i per sphere, (2) T_rhs = T * rhs_inc
        # BiCG solves (I - T*A)*x = T_rhs. Q_ext uses dot(amn, rhs_inc) (NOT T_rhs).
        rhs_T_cpu   = zeros(ComplexF64, neqns, 2, batch_size)  # T * rhs_inc (for BiCG)
        rhs_inc_cpu = zeros(ComplexF64, neqns, 2, batch_size)  # raw rhs_inc (for Q_ext)

        for (bi, m_abs) in enumerate(ri_batch)
            m_rel = m_abs / n_med
            mie_vecs = [MSTMforCAS.compute_mie_coefficients(radii_x[s], m_rel; nmax=noi_max) for s in 1:N]
            td_vecs, to_vecs = MSTMforCAS._precompute_T_values(mie_vecs, nois)

            # Step 1: Build incident wave rhs_inc = Σ_i phase_i * p0_i (per sphere)
            rhs_inc = zeros(ComplexF64, neqns, 2)
            for i in 1:N
                off = offsets[i]
                z_i = positions_x[3, i]
                phase = exp(im * z_i)
                p0_i = MSTMforCAS._genplanewavecoef_z0(noi_max)
                for q in 1:2, p in 1:2
                    p_blk_off = (p - 1) * hnb_max
                    for mn in 1:hnb_max
                        rhs_inc[off + p_blk_off + mn, q] += phase * p0_i[mn, p, q]
                    end
                end
            end
            rhs_inc_cpu[:, :, bi] .= rhs_inc

            # Step 2: T_rhs = T * rhs_inc (same as CPU Step 6)
            for q in 1:2
                for i in 1:N
                    off = offsets[i]
                    for mn in 1:hnb_max
                        n = mn_to_n[mn]
                        td = td_vecs[i][n]
                        to = to_vecs[i][n]
                        p1 = rhs_inc[off + mn, q]
                        p2 = rhs_inc[off + hnb_max + mn, q]
                        rhs_T_cpu[off + mn, q, bi]           = td * p1 + to * p2
                        rhs_T_cpu[off + hnb_max + mn, q, bi] = to * p1 + td * p2
                    end
                end
            end
        end

        # Solve for each polarization
        amn_cpu = zeros(ComplexF64, neqns, 2, batch_size)
        converged_all = fill(true, batch_size)
        iter_all = zeros(Int, batch_size)

        for q in 1:2
            # Upload RHS for this polarization — pad to full B if batch_size < B
            rhs_gpu = CUDA.zeros(ComplexF64, neqns, buf.B)
            copyto!(view(rhs_gpu, :, 1:batch_size), CuArray(rhs_T_cpu[:, q, :]))

            # Run batched BiCG
            x_gpu, conv, iters = gpu_solve_bicg_batch!(buf, rhs_gpu, gpu_fft, N, tol, max_iter)

            # Download solution
            x_cpu = Array(view(x_gpu, :, 1:batch_size))
            amn_cpu[:, q, :] .= x_cpu

            for bi in 1:batch_size
                converged_all[bi] = converged_all[bi] && conv[bi]
                iter_all[bi] = max(iter_all[bi], iters[bi])
            end
        end

        # Post-process each RI on CPU (amplitudes, cross sections)
        for bi in 1:batch_size
            ri_idx = batch_start + bi - 1
            amn_bi = amn_cpu[:, :, bi]
            rhs_bi = rhs_inc_cpu[:, :, bi]  # raw incident coefficients (NOT T*rhs)

            # Compute ScatteringResult using existing CPU functions
            result = _postprocess_scattering(
                amn_bi, rhs_bi, positions_x, radii_x,
                nois, offsets, half_nblks, noi_max,
                converged_all[bi], iter_all[bi])

            results[ri_idx] = (result, amn_bi)
        end
    end

    return results
end

# ─────────────────────────────────────────────────────────────
# Post-processing: compute Q and S from solved amn (CPU)
# ─────────────────────────────────────────────────────────────

function _postprocess_scattering(
    amn::Matrix{ComplexF64},
    rhs::Matrix{ComplexF64},
    positions::Matrix{Float64},
    radii::Vector{Float64},
    nois::Vector{Int},
    offsets::Vector{Int},
    half_nblks::Vector{Int},
    noi_max::Int,
    converged::Bool,
    n_iter::Int
)::MSTMforCAS.ScatteringResult

    N = length(radii)
    x_eff = cbrt(sum(r^3 for r in radii))
    norm_fac = 2.0 / x_eff^2

    # Q_ext (optical theorem)
    Q_ext_sum = 0.0
    for q in 1:2
        Q_ext_sum += real(dot(view(amn, :, q), view(rhs, :, q)))
    end
    Q_ext = -norm_fac * Q_ext_sum

    # Common-origin translation
    r0 = vec(sum(positions, dims=2)) ./ N
    ntrani = Vector{Int}(undef, N)
    for i in 1:N
        dist = sqrt(sum(abs2, r0 .- positions[:, i]))
        ntrani[i] = MSTMforCAS._tranordertest(dist, nois[i], 1e-5)
    end
    nodrt = maximum(ntrani)

    hnb0 = nodrt * (nodrt + 2)
    amn0_mode1 = zeros(ComplexF64, hnb0, 2)
    amn0_mode2 = zeros(ComplexF64, hnb0, 2)
    MSTMforCAS._merge_to_origin!(amn0_mode1, amn0_mode2, amn, positions, r0,
                                  nois, offsets, half_nblks, nodrt, ntrani)

    # Q_sca
    Q_sca_sum = 0.0
    for q in 1:2
        Q_sca_sum += real(dot(amn0_mode1[:, q], amn0_mode1[:, q]))
        Q_sca_sum += real(dot(amn0_mode2[:, q], amn0_mode2[:, q]))
    end
    Q_sca = norm_fac * Q_sca_sum / 2
    Q_abs = Q_ext - Q_sca

    # S amplitudes
    sa_fwd = MSTMforCAS._amplitude_from_mode_coefs(amn0_mode1, amn0_mode2, nodrt, true)
    sa_bwd = MSTMforCAS._amplitude_from_mode_coefs(amn0_mode1, amn0_mode2, nodrt, false)
    bh83_fwd = ntuple(i -> -2 * sa_fwd[i], 4)
    bh83_bwd = ntuple(i -> -2 * sa_bwd[i], 4)

    return MSTMforCAS.ScatteringResult(
        bh83_fwd, bh83_bwd,
        Q_ext, Q_abs, Q_sca,
        converged, n_iter, noi_max)
end
