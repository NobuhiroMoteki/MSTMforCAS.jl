# _tranordertest with cached tables against the version that built the tables on every call.
# The reference below is the function as it was up to v1.1.1, unchanged except for its name.

function tranordertest_reference(r::Float64, lmax::Int, eps::Float64 = 1e-5)::Int
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

@testset "Translation order test with cached tables" begin
    rs = vcat([1e-12, 1e-3, 0.01], exp.(range(log(0.05), log(150.0), length = 45)),
              [0.3731, 1.0, 2.5, 7.123456789, 31.4159, 99.99])       # distances (dimensionless)
    @testset "same result as the reference" begin
        for lmax in 1:12, eps in (1e-5, 1e-3)
            for r in rs
                @test MSTMforCAS._tranordertest(r, lmax, eps) == tranordertest_reference(r, lmax, eps)
            end
        end
        @test MSTMforCAS._tranordertest(3.0, 4) == tranordertest_reference(3.0, 4)   # default eps
    end
    @testset "same result from several threads" begin
        cases = [(r, lmax) for lmax in (2, 3, 5, 8, 11) for r in rs]
        serial = [tranordertest_reference(r, lmax, 1e-5) for (r, lmax) in cases]
        threaded = Vector{Int}(undef, length(cases))
        Threads.@threads for i in eachindex(cases)
            threaded[i] = MSTMforCAS._tranordertest(cases[i][1], cases[i][2], 1e-5)
        end
        @test threaded == serial
    end
end
