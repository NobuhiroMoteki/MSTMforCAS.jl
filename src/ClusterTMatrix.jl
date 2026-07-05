# ClusterTMatrix — build the aggregate cluster T-matrix and evaluate CAS-v2
# forward amplitudes at arbitrary orientation via a Wigner-D rotation (design
# note: docs/design_wigner_d_multiorientation.md, phase P2, strategy R2).
#
# The cluster T-matrix T^(cl) maps regular VSWF incident coefficients about the
# aggregate common origin to outgoing VSWF scattered coefficients about the same
# origin, in the lr_tran (p=1,p=2) basis. It depends on (geometry, m_rel, size)
# but NOT orientation, so it is built once and reused for every orientation.
#
# Origin-mode indexing used here: a coefficient (kl, p) with kl = l(l+1)+k
# (k=-l..l, l=1..nodrt) and p ∈ {1,2} maps to the stacked vector index
# (p-1)*hnb0 + kl, where hnb0 = nodrt*(nodrt+2).
#
# Plain-include style, part of the MSTMforCAS module (relies on its internal
# functions and `using LinearAlgebra`).

export build_cluster_tmatrix, cluster_forward_amplitudes

# Apply the block-diagonal VSWF rotation D(α,β,γ) (or its adjoint) to an
# origin-mode vector/matrix `v` of size (2*hnb0, Ncols). Rotation is block
# diagonal in n and identical on both p-blocks (lr_tran); it couples only m
# within each n. `hnb0 = nodrt*(nodrt+2)`.
function apply_origin_rotation(v::AbstractMatrix{ComplexF64}, nodrt::Int,
                               α::Real, β::Real, γ::Real; dagger::Bool=false)
    hnb0 = nodrt * (nodrt + 2)
    ncols = size(v, 2)
    out = zeros(ComplexF64, size(v))
    for n in 1:nodrt
        Dn = wigner_D(n, α, β, γ)            # (2n+1, 2n+1), rows m', cols m
        R = dagger ? adjoint(Dn) : Dn
        base = n * (n + 1)                    # kl = base + m, m=-n..n
        for p in 0:1
            poff = p * hnb0
            for c in 1:ncols
                for m′ in -n:n
                    acc = zero(ComplexF64)
                    for m in -n:n
                        acc += R[m′ + n + 1, m + n + 1] * v[poff + base + m, c]
                    end
                    out[poff + base + m′, c] = acc
                end
            end
        end
    end
    return out
end

"""
    build_cluster_tmatrix(positions, radii, m_rel; tol, max_iter, solver, truncation_order)
        -> (T_cl, nodrt, r0, converged)

Assemble the aggregate cluster T-matrix `T_cl` of size `(2*hnb0, 2*hnb0)` with
`hnb0 = nodrt*(nodrt+2)`, in the origin-mode (kl,p) basis described above.

Column `(kl0,p0)` is the scattered origin coefficients produced by a unit regular
VSWF incident mode `(kl0,p0)` at the common origin: the mode is regular-translated
to each sphere, the interaction equation is solved (via `solve_tmatrix`'s custom
`rhs_ext`), and the solution is merged back to the origin.

Uses the direct A path (`use_fft=false`), as required by `rhs_ext` until P3.
"""
function build_cluster_tmatrix(positions::Matrix{Float64}, radii::Vector{Float64},
                               m_rel::ComplexF64;
                               tol::Float64=1e-8, max_iter::Int=400,
                               solver::Symbol=:cbicg,
                               truncation_order::Union{Int,Nothing}=nothing)
    N = length(radii)
    @assert size(positions, 2) == N

    # per-sphere multipole orders / flat-vector layout (mirrors solve_tmatrix Step 1)
    nois       = Vector{Int}(undef, N)
    half_nblks = Vector{Int}(undef, N)
    offsets    = Vector{Int}(undef, N)
    for i in 1:N
        na = _mie_order(radii[i], m_rel)
        nois[i]       = truncation_order !== nothing ? max(truncation_order, na) : na
        half_nblks[i] = nois[i] * (nois[i] + 2)
    end
    offsets[1] = 0
    for i in 2:N
        offsets[i] = offsets[i-1] + 2 * half_nblks[i-1]
    end
    neqns = offsets[N] + 2 * half_nblks[N]

    # common origin (centroid) and per-sphere translation orders
    r0 = vec(sum(positions, dims=2)) ./ N
    ntrani = Vector{Int}(undef, N)
    for i in 1:N
        d = sqrt(sum(abs2, r0 .- positions[:, i]))
        ntrani[i] = _tranordertest(d, nois[i], 1e-5)
    end
    nodrt = maximum(ntrani)
    hnb0  = nodrt * (nodrt + 2)
    Ncol  = 2 * hnb0

    # ── incident RHS matrix: one column per origin mode (kl0,p0) ─────────────
    # For sphere i, regular-translate origin modes -> sphere modes:
    #   H_inc[mn(noi_i), kl(nodrt), p] = compute_translation_matrix(r_i - r0, ...)
    rhs = zeros(ComplexF64, neqns, Ncol)
    for i in 1:N
        r_ij = positions[:, i] .- r0          # origin(source) -> sphere i(target)
        hnb  = half_nblks[i]
        off  = offsets[i]
        if sqrt(sum(abs2, r_ij)) < 1e-10
            # sphere at the origin: identity on the shared modes
            ncopy = min(hnb, hnb0)
            for p0 in 1:2, kl0 in 1:ncopy
                col = (p0 - 1) * hnb0 + kl0
                rhs[off + (p0 - 1) * hnb + kl0, col] = 1.0
            end
        else
            Hinc = compute_translation_matrix(r_ij, ComplexF64(1.0), nodrt, nois[i];
                                              use_regular=true)  # [mn(noi_i), kl(nodrt), p]
            for p0 in 1:2
                for kl0 in 1:hnb0
                    col = (p0 - 1) * hnb0 + kl0
                    for mn in 1:hnb
                        rhs[off + (p0 - 1) * hnb + mn, col] = Hinc[mn, kl0, p0]
                    end
                end
            end
        end
    end

    # ── solve all columns with the shared operator ───────────────────────────
    amn, converged, _, _, nois2, offsets2, half_nblks2, _ = solve_tmatrix(
        positions, radii, m_rel; tol=tol, max_iter=max_iter, use_fft=false,
        truncation_order=truncation_order, solver=solver, rhs_ext=rhs)

    # ── merge each solved column to the origin, assemble T_cl in (kl,p) basis ─
    T_cl = zeros(ComplexF64, Ncol, Ncol)
    mode1 = zeros(ComplexF64, hnb0, 2)
    mode2 = zeros(ComplexF64, hnb0, 2)
    for cpair in 1:2:Ncol
        fill!(mode1, 0); fill!(mode2, 0)
        _merge_to_origin!(mode1, mode2, amn[:, cpair:cpair+1], positions, r0,
                          nois, offsets, half_nblks, nodrt, ntrani)
        for (j, col) in enumerate((cpair, cpair + 1))
            for kl in 1:hnb0
                p1 = (mode1[kl, j] + mode2[kl, j]) / 2   # lr p=1
                p2 = (mode1[kl, j] - mode2[kl, j]) / 2   # lr p=2
                T_cl[kl, col]        = p1
                T_cl[hnb0 + kl, col] = p2
            end
        end
    end

    return (T_cl, nodrt, r0, converged)
end

"""
    cluster_forward_amplitudes(T_cl, nodrt, r0, α, β, γ) -> NTuple{4,ComplexF64}

BH83 forward scattering amplitudes `(S1,S2,S3,S4)` at particle orientation
(intrinsic ZYZ Euler `α,β,γ`), evaluated from the cluster T-matrix by strategy
R2: rotate the origin plane-wave incident vector by `D(Ω)^†`, apply `T_cl`, rotate
back by `D(Ω)`, and read the `m=±1` forward amplitude. `Ω=(0,0,0)` reproduces the
fixed +z-incidence result.

`r0` is the common origin (aggregate centroid) returned by `build_cluster_tmatrix`;
the incident plane wave is phase-referenced to the lab plane `z=0` (matching
`compute_scattering`) by the factor `exp(i·r0[3])` (medium wavenumber = 1).
"""
function cluster_forward_amplitudes(T_cl::Matrix{ComplexF64}, nodrt::Int,
                                    r0::AbstractVector{Float64},
                                    α::Real=0.0, β::Real=0.0, γ::Real=0.0)
    hnb0 = nodrt * (nodrt + 2)

    # origin incident plane-wave coefficients (z), phase-referenced to lab z=0
    # via exp(i·r0[3]) so the absolute phase matches compute_scattering.
    phase0 = cis(r0[3])
    p0 = _genplanewavecoef_z0(nodrt)          # (hnb0, 2, 2)  [kl, p, q]
    pinc = zeros(ComplexF64, 2 * hnb0, 2)
    for q in 1:2, p in 1:2, kl in 1:hnb0
        pinc[(p - 1) * hnb0 + kl, q] = phase0 * p0[kl, p, q]
    end

    rotate = !(α == 0.0 && β == 0.0 && γ == 0.0)
    pin = rotate ? apply_origin_rotation(pinc, nodrt, α, β, γ; dagger=true) : pinc
    a0 = T_cl * pin
    a0 = rotate ? apply_origin_rotation(a0, nodrt, α, β, γ; dagger=false) : a0

    # to mode1/mode2 (hnb0, 2) and extract forward amplitude
    mode1 = zeros(ComplexF64, hnb0, 2)
    mode2 = zeros(ComplexF64, hnb0, 2)
    for q in 1:2, kl in 1:hnb0
        p1 = a0[kl, q]
        p2 = a0[hnb0 + kl, q]
        mode1[kl, q] = p1 + p2
        mode2[kl, q] = p1 - p2
    end
    sa = _amplitude_from_mode_coefs(mode1, mode2, nodrt, true)
    return ntuple(i -> -2 * sa[i], 4)         # BH83
end
