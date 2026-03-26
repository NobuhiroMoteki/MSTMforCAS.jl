"""
    ScatteringAmplitude

Compute forward/backward scattering amplitudes and cross sections from
multi-sphere scattered-field expansion coefficients (lr_tran basis).

# Conventions
- BH83 = Bohren & Huffman (1983) amplitude matrix convention
- lr_tran basis: p=1 block (a_n Mie), p=2 block (b_n Mie)
- Q_ext/abs/sca are dimensionless efficiencies (cross section / π a_eff²)
"""

export compute_scattering

# ─────────────────────────────────────────────────────────────
# Volume-equivalent sphere radius (for efficiency normalisation)
# ─────────────────────────────────────────────────────────────
function _x_eff(radii::Vector{Float64})::Float64
    # a_eff = (Σ r_i³)^{1/3}, x_eff = k_medium * a_eff = a_eff (dimensionless)
    return cbrt(sum(r^3 for r in radii))
end

# ─────────────────────────────────────────────────────────────
# Translation order determination — mirrors Fortran tranordertest.
#
# Computes the minimum truncation order n such that the translation
# addition theorem from a source of order `lmax` at distance `r`
# has converged to within `eps`.
#
# Algorithm: accumulate coupling-weighted spherical Bessel sum until
# |1 - partial_sum| < eps  (completeness criterion for the translation matrix).
#
# Reference: Mackowski & Mishchenko MSTM v4.0, subroutine tranordertest.
# ─────────────────────────────────────────────────────────────

function _tranordertest(r::Float64, lmax::Int, eps::Float64 = 1e-5)::Int
    r < 1e-10 && return lmax

    nlim = 200
    notd = nlim + lmax
    nbc  = 2 * notd + 4   # max bcof index needed ≤ 2*notd+1; +4 safety margin

    # fnr[j] = sqrt(j-1)  (1-indexed, mirrors Fortran fnr(0..2*nbc) = sqrt(0..2*nbc))
    fnr = Vector{Float64}(undef, 2 * nbc + 2)
    fnr[1] = 0.0
    for j in 2:length(fnr)
        fnr[j] = sqrt(Float64(j - 1))
    end
    # Convenience: access fnr as if 0-indexed
    F(k) = fnr[k + 1]   # F(k) = sqrt(k)

    # bcof[a+1, b+1] = sqrt((a+b)! / (a! * b!))  (Fortran bcof(0:nbc, 0:nbc))
    bcof = zeros(Float64, nbc + 1, nbc + 1)
    bcof[1, 1] = 1.0
    for n in 0:nbc-1
        for l in n+1:nbc
            bcof[n+1, l+1] = F(n + l) * bcof[n+1, l] / F(l)
            bcof[l+1, n+1] = bcof[n+1, l+1]
        end
        bcof[n+2, n+2] = F(2n+2) * F(2n+1) * bcof[n+1, n+1] / F(n+1)^2
    end
    B(a, b) = (a >= 0 && b >= 0 && a <= nbc && b <= nbc) ? bcof[a+1, b+1] : 0.0

    # ── Inner function: vcfunc(m=-1, n, k=1, l) ───────────────────────────────
    # Returns coupling coefficient array vc[w+1] for w = 0..n+l.
    # m = -1, k = +1, m+k = mk = 0, m-k = -2.
    function _vcfunc!(vc::AbstractVector{Float64}, n::Int, l::Int)
        fill!(vc, 0.0)
        m = -1; k = 1; mk = 0
        wmax = n + l
        wmin = abs(n - l)   # max(|n-l|, |mk|) = |n-l| since mk=0

        # Starting value at wmax
        vc[wmax+1] = B(n-1, l+1) * B(n+1, l-1) / B(2n, 2l)

        if wmin == wmax; return; end

        # Second value at wmax-1
        # (l*m - k*n) * F(2(l+n)-1) / F(l) / F(n) / F(n+l+mk)^2
        # with n+l+mk = n+l+0 = n+l
        denom_nl = F(n + l)
        vc[wmax] = vc[wmax+1] * Float64(-l - n) * F(2*(l+n) - 1) /
                   F(l) / F(n) / denom_nl / denom_nl

        if wmin == wmax - 1; return; end

        # Downward recurrence: compute vc[w-2+1] from vc[w-1+1] and vc[w+1]
        # mk=0 simplifies: t2 = (m-k)*w*(w-1)/(2w(w-1)) = -2w(w-1)/(2w(w-1)) = -1
        vcmax = abs(vc[wmax+1]) + abs(vc[wmax])
        w_break = wmin   # if we switch to upward recurrence, start from here

        for w in wmax:-1:wmin+2
            d1 = F(w)^2 * F(n-l+w) * F(l-n+w) * F(n+l-w+1) * F(n+l+w+1)
            if d1 == 0.0; w_break = w - 3; break; end
            t1 = 2*Float64(w) * F(2w+1) * F(2w-1) / d1

            t3_num = F(w-1)^2 * F(l-n+w-1) * F(n-l+w-1) * F(n+l-w+2) * F(n+l+w)
            t3_den = 2.0 * Float64(w-1) * F(2w-3) * F(2w-1)
            if t3_den == 0.0; w_break = w - 3; break; end
            t3 = t3_num / t3_den

            vc[w-1] = (-vc[w] - vc[w+1] / t1) / t3   # t2 = -1

            if isodd(wmax - w)
                vctest = abs(vc[w-1]) + abs(vc[w])
                vcmax  = max(vcmax, vctest)
                if vctest / vcmax < 0.01
                    w_break = w - 3
                    break
                end
            end
        end

        # Upward recurrence from wmin to w_break (if downward stopped early)
        if w_break > wmin
            # Starting value at w = wmin = |n-l|
            w0 = wmin
            if n >= l
                # m1=-1, n1=n, l1=l, k1=1: w0=n-l
                # vc1 = (-1)^(1+l) * B(l+1, w0+1) * B(l-1, w0-1) / B(2l, 2*w0+1)
                # Note: w0-m1-k1 = w0+1-1 = w0, w0+m1+k1 = w0-1+1 = w0... wait
                # vcfuncuprec: m1=m=-1, k1=k=1
                # w-m1-k1 = w0-(-1)-1 = w0, w+m1+k1 = w0+(-1)+1 = w0
                vc0 = (-1)^(1+l) * B(l+1, w0) * B(l-1, w0) / B(2l, 2*w0+1)
            else
                # m1=k=1, n1=l, l1=n, k1=m=-1
                # vc1 = (-1)^(-1+n) * B(n+1, w0) * B(n-1, w0) / B(2n, 2*w0+1)
                # w-m1-k1 = w0-1-(-1) = w0, w+m1+k1 = w0+1+(-1) = w0
                vc0 = (-1)^(n-1) * B(n+1, w0) * B(n-1, w0) / B(2n, 2*w0+1)
            end
            vc[w0+1] = vc0
            w_end = min(w_break, n + l)

            if w_end > w0
                # w = w0+1 (special case when w0=0: t2 = 0.5*(m-k) = -1)
                w1 = w0 + 1
                d1 = F(w1)^2 * F(n-l+w1) * F(l-n+w1) * F(n+l-w1+1) * F(n+l+w1+1)
                if d1 != 0.0
                    t1_up = 2*Float64(w1) * F(2w1+1) * F(2w1-1) / d1
                    t2_up = (w0 == 0) ? -1.0 :
                        Float64((-2)*w1*(w1-1)) / Float64(2*w1*(w1-1))
                    vc[w1+1] = t1_up * t2_up * vc[w0+1]
                end
            end

            for w in w0+2:w_end
                d1 = F(w)^2 * F(n-l+w) * F(l-n+w) * F(n+l-w+1) * F(n+l+w+1)
                if d1 == 0.0; break; end
                t1_up = 2*Float64(w) * F(2w+1) * F(2w-1) / d1
                t2_up = Float64((-2)*w*(w-1)) / Float64(2*w*(w-1))
                t3_num_up = F(w-1)^2 * F(l-n+w-1) * F(n-l+w-1) * F(n+l-w+2) * F(n+l+w)
                t3_den_up = 2.0 * Float64(w-1) * F(2w-3) * F(2w-1)
                if t3_den_up == 0.0; break; end
                t3_up = t3_num_up / t3_den_up
                vc[w+1] = t1_up * (t2_up * vc[w] - t3_up * vc[w-1])
            end
        end
    end

    # ── Main tranordertest convergence loop ────────────────────────────────────
    vc_buf = zeros(Float64, nlim + 2 * lmax + 2)  # reuse buffer
    total_sum = 0.0
    result_n  = nlim

    # Pre-compute spherical Bessel j_l(r) for l = 0..nlim+lmax using ratio method
    # (stable downward recurrence for ratios, then forward reconstruction).
    # This mirrors the Fortran cricbessel approach used in MieCoefficients.jl.
    norder_max = nlim + lmax
    nstart = norder_max + max(15, round(Int, 4 * norder_max^(1/3)))
    # Downward recurrence for ratios: R_l = j_l / j_{l-1}
    # R_{nstart} ≈ 0, R_l = 1 / ((2l+1)/r - R_{l+1})
    ratios = Vector{Float64}(undef, nstart + 1)  # ratios[l+1] = R_l for l=1..nstart
    ratios[nstart+1] = 0.0
    for l in nstart-1:-1:1
        ratios[l+1] = 1.0 / (Float64(2l + 1) / r - ratios[l+2])
    end
    # Forward reconstruction: j_0 = sin(r)/r, j_l = R_l * j_{l-1}
    j_buf = Vector{Float64}(undef, norder_max + 1)
    j_buf[1] = sin(r) / r
    for l in 1:norder_max
        j_buf[l+1] = ratios[l+1] * j_buf[l]
    end

    for n in 1:nlim
        norder = n + lmax

        j = view(j_buf, 1:norder+1)

        # Coupling coefficients vcfunc(-1, n, 1, lmax)
        wmax_vc = n + lmax
        if wmax_vc + 1 > length(vc_buf)
            resize!(vc_buf, wmax_vc + 2)
        end
        vc_view = view(vc_buf, 1:wmax_vc+1)
        _vcfunc!(vc_view, n, lmax)

        # c = F(2n+1) * F(2*lmax+1) * i^(n-lmax)
        c = F(2n+1) * F(2*lmax+1) * im^mod(n - lmax, 4)

        wmin = abs(n - lmax)

        a = zero(ComplexF64)
        b = zero(ComplexF64)
        for w in wmin:wmax_vc
            alnw  = vc_buf[w+1]^2
            xi_w  = j[w+1] * im^mod(w, 4)   # j_w(r) * i^w
            if iseven(n + lmax + w)
                a += alnw * xi_w
            else
                b += alnw * xi_w
            end
        end
        a = c * a
        b = c * b

        total_sum += abs2(a) + abs2(b)

        if abs(1.0 - total_sum) < eps
            result_n = n
            break
        end
    end

    return max(result_n, lmax)
end

# ─────────────────────────────────────────────────────────────
# Translate per-sphere lr-coefficients to a common origin and
# accumulate into single-origin mode-indexed arrays.
#
# amn0_mode1[mn, q] += Σ_p (H_p * amn_i_p + direct if zero-dist)  (mode-1 = lr_p1+lr_p2)
# amn0_mode2[mn, q] +=                                              (mode-2 = lr_p1-lr_p2)
#
# ntrani[i] = translation order cap for sphere i (from tranordertest).
#             Only output modes 1..ntrani[i]*(ntrani[i]+2) are populated for sphere i.
# ─────────────────────────────────────────────────────────────
function _merge_to_origin!(
    amn0_mode1::Matrix{ComplexF64},   # (nodrt*(nodrt+2), 2) mode-1 coefficients
    amn0_mode2::Matrix{ComplexF64},   # (nodrt*(nodrt+2), 2) mode-2 coefficients
    amn::Matrix{ComplexF64},          # (neqns, 2) all-sphere flat lr-coefficients
    positions::Matrix{Float64},       # (3, N)
    r0::Vector{Float64},              # (3,) common origin
    nois::Vector{Int},
    offsets::Vector{Int},
    half_nblks::Vector{Int},
    nodrt::Int,
    ntrani::Vector{Int}               # per-sphere translation order cap
)
    N = length(nois)

    # Pre-allocate translation work buffers (reused per sphere)
    max_hnb_i = maximum(ntrani[k] * (ntrani[k] + 2) for k in 1:N)
    trans_p1 = zeros(ComplexF64, max_hnb_i)
    trans_p2 = zeros(ComplexF64, max_hnb_i)
    r_trans  = Vector{Float64}(undef, 3)

    for i in 1:N
        off  = offsets[i]
        hnb  = half_nblks[i]
        noi  = nois[i]
        ntri = ntrani[i]
        hnb_i = ntri * (ntri + 2)     # output modes for this sphere

        # per-sphere lr blocks
        amn_p1 = @view amn[off+1:off+hnb, :]          # (hnb, 2) p=1 block
        amn_p2 = @view amn[off+hnb+1:off+2hnb, :]     # (hnb, 2) p=2 block

        r_trans[1] = r0[1] - positions[1, i]
        r_trans[2] = r0[2] - positions[2, i]
        r_trans[3] = r0[3] - positions[3, i]
        dist = sqrt(r_trans[1]^2 + r_trans[2]^2 + r_trans[3]^2)

        if dist < 1e-10
            # Zero translation: accumulate directly (pad with zeros if noi < ntri)
            n_copy = min(hnb, hnb_i)
            @inbounds for q in 1:2
                for mn in 1:n_copy
                    amn0_mode1[mn, q] += amn_p1[mn, q] + amn_p2[mn, q]
                    amn0_mode2[mn, q] += amn_p1[mn, q] - amn_p2[mn, q]
                end
            end
        else
            H = compute_translation_matrix(r_trans, ComplexF64(1.0), noi, ntri; use_regular=true)

            @inbounds for q in 1:2
                fill!(view(trans_p1, 1:hnb_i), zero(ComplexF64))
                fill!(view(trans_p2, 1:hnb_i), zero(ComplexF64))
                for kl in 1:hnb_i
                    s1 = zero(ComplexF64)
                    s2 = zero(ComplexF64)
                    for mn in 1:hnb
                        s1 += H[kl, mn, 1] * amn_p1[mn, q]
                        s2 += H[kl, mn, 2] * amn_p2[mn, q]
                    end
                    trans_p1[kl] = s1
                    trans_p2[kl] = s2
                end
                for mn in 1:hnb_i
                    amn0_mode1[mn, q] += trans_p1[mn] + trans_p2[mn]
                    amn0_mode2[mn, q] += trans_p1[mn] - trans_p2[mn]
                end
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────
# Compute raw scattering amplitudes (sa1..sa4) at θ=0° or θ=180°
#
# Uses the scatteringmatrix formula from Mackowski & Mishchenko (2011).
# For BH83 convention: S_i = -2 * sa_i_raw (with normalize_s11=.false.)
#
# Input: amn0_mode1, amn0_mode2 = single-origin mode-transformed coefficients
# q=1 incident polarization used for a; q=2 used for b in BH83 sa formula.
#
# At θ=0°:  tau(1,n,1) = tau(1,n,2) = -fnm (m=+1)
#            tau(n+1,1,1) = +fnm, tau(n+1,1,2) = -fnm (m=-1)
# At θ=180°: tau(1,n,1) = -fnm*(-1)^n, tau(1,n,2) = fnm*(-1)^n (m=+1)
#             tau(n+1,1,1) = fnm*(-1)^n, tau(n+1,1,2) = fnm*(-1)^n (m=-1)
# ─────────────────────────────────────────────────────────────
function _amplitude_from_mode_coefs(
    amn0_mode1::Matrix{ComplexF64},
    amn0_mode2::Matrix{ComplexF64},
    nodrt::Int,
    forward::Bool    # true → θ=0°, false → θ=180°
)::NTuple{4, ComplexF64}

    sa1 = zero(ComplexF64)
    sa2 = zero(ComplexF64)
    sa3 = zero(ComplexF64)
    sa4 = zero(ComplexF64)

    @inbounds for n in 1:nodrt
        fnm   = sqrt((2n + 1) / 2.0) / 4.0
        cin   = (-im)^n                # (-i)^n

        mn_p1 = n * (n + 1) + 1   # m = +1
        mn_m1 = n * (n + 1) - 1   # m = -1

        # tau values × phase factor
        # Forward (θ=0°):  tau(+1,n,1) = tau(+1,n,2) = -fnm (m=+1)
        #                  tau(-1,n,1) = +fnm, tau(-1,n,2) = -fnm (m=-1)
        # Backward (θ=180°): all multiplied by (-1)^n, but signs of p1 vs p2 differ:
        #   tau(+1,n,1) = -cf, tau(+1,n,2) = +cf,
        #   tau(-1,n,1) = +cf, tau(-1,n,2) = +cf
        if forward
            tau_p1_p1 = -fnm;  tau_p1_p2 = -fnm
            tau_m1_p1 = +fnm;  tau_m1_p2 = -fnm
        else
            cf = (-1)^n * fnm
            tau_p1_p1 = -cf;   tau_p1_p2 = +cf
            tau_m1_p1 = +cf;   tau_m1_p2 = +cf
        end

        # Read mode coefs: mode1 = lr_p1+lr_p2, mode2 = lr_p1-lr_p2
        # Incident pol q=1 → a; q=2 (with sign flip) → b = -amn0(p,2)
        # In Fortran: a = amn0(m1,n1,p,q=1), b = -amn0(m1,n1,p,q=2)
        a1_p1 = amn0_mode1[mn_p1, 1];  a1_p2 = amn0_mode2[mn_p1, 1]
        a2_p1 = amn0_mode1[mn_m1, 1];  a2_p2 = amn0_mode2[mn_m1, 1]
        b1_p1 = -amn0_mode1[mn_p1, 2]; b1_p2 = -amn0_mode2[mn_p1, 2]
        b2_p1 = -amn0_mode1[mn_m1, 2]; b2_p2 = -amn0_mode2[mn_m1, 2]

        # sa(2) += ci * cin * tau(m1,n1,p) * a * ephim(m)  [ephim=1 for phi=0]
        # m=+1, p=1: ci*cin*tau_p1_p1*a1_p1
        # m=+1, p=2: ci*cin*tau_p1_p2*a1_p2
        # m=-1, p=1: ci*cin*tau_m1_p1*a2_p1
        # m=-1, p=2: ci*cin*tau_m1_p2*a2_p2
        sa2 += im * cin * (tau_p1_p1 * a1_p1 + tau_p1_p2 * a1_p2 +
                           tau_m1_p1 * a2_p1 + tau_m1_p2 * a2_p2)

        # sa(1) += cin * tau(m1,n1,3-p) * b * ephim(m)
        # 3-p: p=1 → tau for p=2 index, p=2 → tau for p=1 index
        sa1 += cin * (tau_p1_p2 * b1_p1 + tau_p1_p1 * b1_p2 +
                      tau_m1_p2 * b2_p1 + tau_m1_p1 * b2_p2)

        # sa(3) += ci * cin * tau(m1,n1,p) * b * ephim(m)
        sa3 += im * cin * (tau_p1_p1 * b1_p1 + tau_p1_p2 * b1_p2 +
                           tau_m1_p1 * b2_p1 + tau_m1_p2 * b2_p2)

        # sa(4) += cin * tau(m1,n1,3-p) * a * ephim(m)
        sa4 += cin * (tau_p1_p2 * a1_p1 + tau_p1_p1 * a1_p2 +
                      tau_m1_p2 * a2_p1 + tau_m1_p1 * a2_p2)
    end

    return (sa1, sa2, sa3, sa4)
end

# ─────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────


"""
    compute_scattering(
        positions::Matrix{Float64},
        radii::Vector{Float64},
        m_rel::ComplexF64;
        tol::Float64 = 1e-6,
        max_iter::Int = 200,
        use_fft::Bool = false,
        truncation_order::Union{Int,Nothing} = nothing
    ) -> ScatteringResult

Compute multi-sphere scattering amplitudes and cross sections.

# Arguments
- `positions`: [3, N] sphere centers (dimensionless: k_medium × physical_pos)
- `radii`: [N] sphere size parameters x_i = k_medium × r_i
- `m_rel`: complex refractive index m_sphere / m_medium
- `truncation_order`: if specified, use this VSWF truncation order for all spheres

# Returns
`ScatteringResult` with BH83 amplitudes S₁–S₄ at forward/backward and Q efficiencies.
"""
function compute_scattering(
    positions::Matrix{Float64},
    radii::Vector{Float64},
    m_rel::ComplexF64;
    tol::Float64 = 1e-6,
    max_iter::Int = 200,
    use_fft::Bool = false,
    truncation_order::Union{Int,Nothing} = nothing,
    precomputed_fft::Union{FFTGridData, Nothing} = nothing
)::ScatteringResult

    N = length(radii)

    # ── Solve T-matrix interaction equation ──────────────────────────────────
    amn, converged, n_iter, noi_max, nois, offsets, half_nblks, rhs = solve_tmatrix(
        positions, radii, m_rel; tol=tol, max_iter=max_iter, use_fft=use_fft,
        truncation_order=truncation_order, precomputed_fft=precomputed_fft
    )

    # ── Q_ext (optical theorem) ───────────────────────────────────────────────
    x_eff = _x_eff(radii)
    norm_fac = 2.0 / x_eff^2

    Q_ext_sum = 0.0
    for q in 1:2
        Q_ext_sum += real(dot(amn[:, q], rhs[:, q]))
    end
    Q_ext = -norm_fac * Q_ext_sum

    # ── Merge all spheres to common origin ───────────────────────────────────
    r0 = vec(sum(positions, dims=2)) ./ N   # centroid

    # Translation order per sphere: determined by _tranordertest (matches Fortran)
    ntrani = Vector{Int}(undef, N)
    for i in 1:N
        r_trans = r0 .- positions[:, i]
        dist    = sqrt(sum(abs2, r_trans))
        ntrani[i] = _tranordertest(dist, nois[i], 1e-5)
    end
    nodrt = maximum(ntrani)

    hnb0 = nodrt * (nodrt + 2)
    amn0_mode1 = zeros(ComplexF64, hnb0, 2)
    amn0_mode2 = zeros(ComplexF64, hnb0, 2)
    _merge_to_origin!(amn0_mode1, amn0_mode2, amn, positions, r0,
                      nois, offsets, half_nblks, nodrt, ntrani)

    # ── Q_sca from common-origin coherent sum ──────────────────────────────
    Q_sca_sum = 0.0
    for q in 1:2
        Q_sca_sum += real(dot(amn0_mode1[:, q], amn0_mode1[:, q]))
        Q_sca_sum += real(dot(amn0_mode2[:, q], amn0_mode2[:, q]))
    end
    Q_sca = norm_fac * Q_sca_sum / 2
    Q_abs = Q_ext - Q_sca

    # ── S amplitudes from common-origin ────────────────────────────────────
    sa_fwd = _amplitude_from_mode_coefs(amn0_mode1, amn0_mode2, nodrt, true)
    sa_bwd = _amplitude_from_mode_coefs(amn0_mode1, amn0_mode2, nodrt, false)

    bh83_fwd = ntuple(i -> -2 * sa_fwd[i], 4)
    bh83_bwd = ntuple(i -> -2 * sa_bwd[i], 4)

    return ScatteringResult(
        bh83_fwd,
        bh83_bwd,
        Q_ext,
        Q_abs,
        Q_sca,
        converged,
        n_iter,
        noi_max
    )
end
