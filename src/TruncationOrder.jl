"""
    TruncationOrder

Choose the VSWF truncation order of near-touching monomers from a preliminary
two-sphere calculation.

The single-sphere Mie criterion (`mie_nmax`) resolves the field scattered by an
isolated monomer. It does not resolve the near field between two monomers that
almost touch. For small monomers the order needed for that coupling is set by the
relative refractive index and by the contact geometry. It is nearly independent of
the size parameter.

The preliminary calculation solves a cluster of two equal spheres whose axis is
perpendicular to the incident direction. One incident polarization then has its
electric field along the axis, which is the slowly converging configuration. The
forward amplitudes S₁(0°), S₂(0°) are computed for increasing order N. The
truncation error at order N is estimated as the difference to the highest computed
order plus a bound on the remaining tail. The tail bound assumes that the increments
decay at least algebraically, |S(N) − S(N−1)| ∝ N^(−p), with p taken from the last
computed orders.

# References
- Mackowski (1994), JOSA A 11:2851 (convergence of multipole expansions for touching spheres)
"""

export contact_truncation_order, contact_order_scan

"""
    _dimer_forward(x, m_rel, gap, order; solver_tol) -> NTuple{2, ComplexF64}

Forward amplitudes (S₁(0°), S₂(0°)), BH83 convention, of two equal spheres with size
parameter `x` whose centers lie on the x-axis. Incidence is along +z. `gap` is the
surface separation divided by the sum of the radii (dimensionless).
"""
function _dimer_forward(
    x::Float64,
    m_rel::ComplexF64,
    gap::Float64,
    order::Int;
    solver_tol::Float64 = 1e-16
)::NTuple{2, ComplexF64}
    d = 2x * (1 + gap)                    # center distance (dimensionless, k_medium × length)
    positions = zeros(Float64, 3, 2)      # shape: (3, 2)
    positions[1, 1] = -d / 2
    positions[1, 2] =  d / 2
    result, _ = compute_scattering(positions, [x, x], m_rel;
                                   tol = solver_tol, max_iter = 2000,
                                   use_fft = false, truncation_order = order)
    result.converged ||
        @warn "two-sphere preliminary solve did not converge" x m_rel gap order
    return (result.S_forward[1], result.S_forward[2])
end

"""
    contact_order_scan(x, m_rel; eps=1e-2, gap=0.0, ref_fraction=0.25, n_cap=40)
        -> (order, orders, errors, tail, reference_ok)

Scan the truncation order of a near-touching two-sphere cluster and estimate the
truncation error of the forward amplitudes at each order.

# Arguments
- `x`: monomer size parameter k_medium × radius (dimensionless)
- `m_rel`: complex refractive index of the monomers relative to the medium
- `eps`: target relative error of S₁(0°) and S₂(0°). The error is the modulus of the
  complex difference divided by the modulus of the amplitude, so it covers both
  amplitude and phase.
- `gap`: surface separation divided by the sum of the radii (dimensionless).
  The default 0 (point contact) is the slowest-converging case.
- `ref_fraction`: the scan stops once the tail bound at the highest computed order
  is below `ref_fraction × eps`.
- `n_cap`: largest order tried.

# Returns
A named tuple:
- `order`: smallest order from which the estimated error stays below `eps`
- `orders`: orders that were computed, starting at the single-sphere Mie order
- `errors`: estimated relative truncation error at each computed order
- `tail`: relative tail bound at the highest computed order
- `reference_ok`: `false` if `n_cap` was reached before the tail bound met its target.
  `order` is then a lower bound only.
"""
function contact_order_scan(
    x::Float64,
    m_rel::ComplexF64;
    eps::Float64 = 1e-2,
    gap::Float64 = 0.0,
    ref_fraction::Float64 = 0.25,
    n_cap::Int = 40
)
    eps > 0 || throw(ArgumentError("eps must be positive"))
    gap >= 0 || throw(ArgumentError("gap must be non-negative"))
    0 < ref_fraction < 1 || throw(ArgumentError("ref_fraction must be in (0, 1)"))

    n_start = mie_nmax(x, m_rel)
    orders = Int[]
    amps   = NTuple{2, ComplexF64}[]      # (S1, S2) forward at each computed order
    incr   = Float64[]                    # relative increment ‖S(N) − S(N−1)‖ / ‖S(N)‖
    tail   = Inf                          # relative tail bound, per-component scale
    reference_ok = false

    n = n_start
    while n <= max(n_cap, n_start)
        s = _dimer_forward(x, m_rel, gap, n)
        push!(orders, n)
        push!(amps, s)
        k = length(orders)
        norm_s = hypot(abs(s[1]), abs(s[2]))
        if k >= 2
            sp = amps[k-1]
            push!(incr, hypot(abs(s[1] - sp[1]), abs(s[2] - sp[2])) / norm_s)
        end
        # Tail bound from the last three increments: with |d_N| ∝ N^(-p),
        # Σ_{j>N} |d_j| ≤ |d_N| N / (p − 1). Use the smaller of the last two p estimates.
        if k >= 4
            d2, d1, d0 = incr[end-2], incr[end-1], incr[end]
            if d0 == 0.0
                tail = 0.0
            elseif d0 < d1 < d2
                p_new = log(d1 / d0) / log(n / (n - 1))
                p_old = log(d2 / d1) / log((n - 1) / (n - 2))
                p = min(p_new, p_old)
                tail = p > 1 ? d0 * n / (p - 1) * norm_s / min(abs(s[1]), abs(s[2])) : Inf
            else
                tail = Inf                # increments not yet decreasing monotonically
            end
            if tail <= ref_fraction * eps
                reference_ok = true
                break
            end
        end
        n += 1
    end

    s_top = amps[end]
    errors = Vector{Float64}(undef, length(orders))
    for k in eachindex(orders)
        s = amps[k]
        diff = max(abs(s[1] - s_top[1]) / abs(s_top[1]), abs(s[2] - s_top[2]) / abs(s_top[2]))
        errors[k] = diff + tail
    end

    # Smallest order from which the estimated error stays below eps.
    k_last_bad = findlast(e -> !(e < eps), errors)
    order = k_last_bad === nothing ? orders[1] :
            k_last_bad == length(orders) ? orders[end] : orders[k_last_bad + 1]

    reference_ok ||
        @warn "contact_order_scan: tail bound not below ref_fraction*eps at n_cap; returned order is a lower bound" x m_rel eps gap n_cap tail
    return (order = order, orders = orders, errors = errors, tail = tail, reference_ok = reference_ok)
end

"""
    contact_truncation_order(x, m_rel; eps=1e-2, gap=0.0, kwargs...) -> Int

VSWF truncation order needed to resolve the coupling between near-touching monomers
with size parameter `x` (dimensionless) and relative refractive index `m_rel`, such
that the forward amplitudes S₁(0°), S₂(0°) of a two-sphere cluster are converged to a
relative error `eps`.

The result is never smaller than the single-sphere Mie order `mie_nmax(x, m_rel)`.
Pass it as `truncation_order` to `compute_scattering` or `solve_tmatrix`.
Suggested values: `eps = 1e-2` for routine use, `eps = 1e-3` for precise calculations.

See `contact_order_scan` for the keyword arguments and for the per-order error table.
"""
function contact_truncation_order(
    x::Float64,
    m_rel::ComplexF64;
    eps::Float64 = 1e-2,
    gap::Float64 = 0.0,
    kwargs...
)::Int
    return contact_order_scan(x, m_rel; eps = eps, gap = gap, kwargs...).order
end
