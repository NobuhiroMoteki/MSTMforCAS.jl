#!/usr/bin/env julia
"""
run_doublet_sweep.jl

Compute forward-scattering amplitudes (CAS-v2 observables) for PS doublets
on a 4D grid of (r_v, q, cos_theta_opt, phi_opt), and write results to HDF5.

This script constructs 2-sphere contact geometries, rotates them to desired
optical-frame orientations, and calls MSTMforCAS.compute_scattering() directly.

Usage:
    julia -t auto --project=. scripts/run_doublet_sweep.jl \\
        --wavelength 0.638 --nominal-diameter 303 \\
        [--output doublet_sweep_wl638nm_PS_303nm.h5] \\
        [--n-rv 20] [--n-q 6] [--n-ct 16] [--n-phi 20] \\
        [--use-fft] [--gpu]

The r_v grid range is automatically computed from the nominal diameter
and the particle size distribution sigma_p.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MSTMforCAS
using HDF5
using LinearAlgebra
using Printf

# =========================================================================
# PS refractive index table (same as PCAS_Bayes_PSstandard/config.py)
# Zhang et al., Appl. Opt. 59, 2337-2344 (2020)
# =========================================================================
const PS_REFINDEX = Dict(
    0.453 => 1.610,
    0.638 => 1.585,
    0.834 => 1.576,
)

# =========================================================================
# PS single-sphere size distribution sigma [um] by nominal diameter [nm]
# Source: Thermo Fisher Nanosphere datasheets (same as config.py)
# =========================================================================
const SIGMA_P_TABLE = Dict(
    240  => 0.0037,
    303  => 0.0047,
    345  => 0.0065,
    401  => 0.0050,
    453  => 0.0079,
    510  => 0.0092,
    600  => 0.0100,
    702  => 0.0049,
    803  => 0.0056,
    994  => 0.0100,
)

# =========================================================================
# SWEEP PARAMETERS — defaults (overridable via CLI)
# =========================================================================

# Radius ratio q = r2/r1 grid
const Q_MIN = 0.90
const Q_MAX = 1.00
const N_Q_DEFAULT = 6

# Optical polar angle: cos(theta_opt) in [-1, 1] (full domain, no mirroring)
const N_CT_DEFAULT = 21

# Optical azimuthal angle: phi_opt in [0, 2*pi) (periodic, ghost-pad later)
const N_PHI_DEFAULT = 20

# Number of r_v grid points
const N_RV_DEFAULT = 20

# r_v margin: ±N_SIGMA_MARGIN * sigma_rv beyond the nominal doublet r_v range
const N_SIGMA_MARGIN = 7.0

# Medium refractive index
const M_MEDIUM = 1.0

# MSTM solver settings
const MAX_ITERATIONS = 200
const CONVERGENCE_TOL = 1e-6
const USE_FFT_DEFAULT = false


# =========================================================================
# Rotation matrix for polar/azimuthal angles
# =========================================================================

"""
    rotation_matrix(theta, phi) -> Matrix{Float64}

Construct 3x3 rotation matrix that rotates the z-axis unit vector
to the direction (theta, phi) in spherical coordinates.

R maps (0, 0, 1) -> (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
"""
function rotation_matrix(theta::Float64, phi::Float64)::Matrix{Float64}
    ct, st = cos(theta), sin(theta)
    cp, sp = cos(phi), sin(phi)
    # R = R_z(phi) * R_y(theta)
    return [ct*cp  -sp  st*cp;
            ct*sp   cp  st*sp;
            -st     0.0  ct]
end


# =========================================================================
# Doublet geometry construction
# =========================================================================

"""
    make_doublet(r_v, q) -> (positions, radii)

Construct a contact doublet geometry aligned along the z-axis.

Parameters:
- r_v: volume-equivalent radius [um]
- q: radius ratio r2/r1 (0 < q <= 1)

Returns:
- positions: [3, 2] matrix of sphere centres [um]
- radii: [2] vector of sphere radii [um]

The doublet is centred at the origin with sphere 1 at z < 0 and
sphere 2 at z > 0. The z-axis is the optical axis in the
reference orientation (theta_opt=0).
"""
function make_doublet(r_v::Float64, q::Float64)
    # r_v^3 = r1^3 + r2^3 = r1^3 * (1 + q^3)
    r1 = r_v / (1.0 + q^3)^(1.0/3.0)
    r2 = q * r1

    # Contact: centre-to-centre distance = r1 + r2
    d = r1 + r2

    # Place symmetrically about origin
    z1 = -d / 2.0
    z2 =  d / 2.0

    positions = [0.0  0.0;
                 0.0  0.0;
                 z1   z2]  # [3, 2]
    radii = [r1, r2]

    return positions, radii
end


# =========================================================================
# BH83 -> MI02 -> CAS conversion
# =========================================================================

"""
    bh83_to_cas(S_fwd, k) -> (S_s, S_p)

Convert BH83-convention forward scattering amplitudes to CAS observables.

S_fwd = (S1, S2, S3, S4) from ScatteringResult.S_forward

MI02 convention (Mishchenko 2002):
    S11 = S2 / (-ik)
    S22 = S1 / (-ik)
    S12 = S3 / (ik)
    S21 = S4 / (ik)

CAS observables:
    S_s = S11 + i * S12  (s-polarization)
    S_p = S22 - i * S21  (p-polarization)
"""
function bh83_to_cas(S_fwd::NTuple{4, ComplexF64}, k::Float64)
    ik = im * k
    S11 = S_fwd[2] / (-ik)
    S22 = S_fwd[1] / (-ik)
    S12 = S_fwd[3] / (ik)
    S21 = S_fwd[4] / (ik)

    S_s = S11 + im * S12
    S_p = S22 - im * S21

    return S_s, S_p
end


# =========================================================================
# Main sweep
# =========================================================================

"""
    compute_rv_range(nominal_diameter_nm) -> (rv_min, rv_max)

Compute the r_v grid range from the nominal single-sphere diameter.

The doublet r_v varies with q:
  r_v(q) = r_single * (1 + q^3)^{1/3}
  r_v_min_nominal = r_v(q=Q_MIN)  (smallest doublet)
  r_v_max_nominal = r_v(q=Q_MAX)  (largest doublet, q=1 -> 2^{1/3} * r_single)

We add a margin of ±N_SIGMA_MARGIN * sigma_rv, where sigma_rv is the
volume-equivalent radius uncertainty propagated from the single-sphere
size distribution sigma_p.
"""
function compute_rv_range(nominal_diameter_nm::Int)
    r_single = nominal_diameter_nm / 2000.0  # [um]
    sigma_p = SIGMA_P_TABLE[nominal_diameter_nm]  # single-sphere sigma [um]

    # Doublet r_v range from q variation
    rv_at_qmin = r_single * (1.0 + Q_MIN^3)^(1.0/3.0)
    rv_at_qmax = r_single * (1.0 + Q_MAX^3)^(1.0/3.0)

    # Propagate sigma_p to sigma_rv: sigma_rv ≈ sigma_p/2 * 2^{1/3}
    # (half because sigma_p is diameter; factor 2^{1/3} for doublet volume)
    sigma_rv = (sigma_p / 2.0) * 2.0^(1.0/3.0)

    margin = N_SIGMA_MARGIN * sigma_rv
    rv_min = rv_at_qmin - margin
    rv_max = rv_at_qmax + margin

    # Ensure positive
    rv_min = max(rv_min, 0.01)

    return rv_min, rv_max
end


function run_sweep(;
        wavelength::Float64,
        nominal_diameter_nm::Int,
        output_h5::String,
        n_rv::Int = N_RV_DEFAULT,
        n_q::Int = N_Q_DEFAULT,
        n_ct::Int = N_CT_DEFAULT,
        n_phi::Int = N_PHI_DEFAULT,
        use_fft::Bool = USE_FFT_DEFAULT,
        use_gpu::Bool = false)

    # --- Grid setup ---
    rv_min, rv_max = compute_rv_range(nominal_diameter_nm)
    q_min, q_max = Q_MIN, Q_MAX

    rv_grid  = range(rv_min, rv_max; length=n_rv)
    q_grid   = range(q_min, q_max; length=n_q)
    ct_grid  = range(-1.0, 1.0; length=n_ct)
    phi_grid = range(0.0, 2π * (1 - 1/n_phi); length=n_phi)  # [0, 2π) without endpoint

    # PS refractive index
    m_real_PS = PS_REFINDEX[wavelength]
    m_imag_PS = 0.0
    m_rel = ComplexF64(m_real_PS / M_MEDIUM, m_imag_PS / M_MEDIUM)
    k = 2π * M_MEDIUM / wavelength  # wavenumber [1/um]

    n_total = n_rv * n_q * n_ct * n_phi
    @printf("Doublet sweep: PS %dnm, wl=%.3f um, m_PS=%.3f\n",
            nominal_diameter_nm, wavelength, m_real_PS)
    @printf("  r_v range: [%.5f, %.5f] um\n", rv_min, rv_max)
    @printf("Grid: %d rv × %d q × %d cos_θ × %d φ = %d total\n",
            n_rv, n_q, n_ct, n_phi, n_total)

    # --- Pre-allocate result arrays ---
    res_rv      = Vector{Float64}(undef, n_total)
    res_q       = Vector{Float64}(undef, n_total)
    res_ct      = Vector{Float64}(undef, n_total)
    res_phi     = Vector{Float64}(undef, n_total)
    res_Ss_re   = Vector{Float64}(undef, n_total)
    res_Ss_im   = Vector{Float64}(undef, n_total)
    res_Sp_re   = Vector{Float64}(undef, n_total)
    res_Sp_im   = Vector{Float64}(undef, n_total)
    res_conv    = Vector{Int8}(undef, n_total)

    # --- Sweep ---
    idx = 0
    n_done = 0

    for (i_rv, rv) in enumerate(rv_grid)
        for (i_q, q_val) in enumerate(q_grid)
            # Construct reference doublet (z-axis aligned)
            pos_ref, radii = make_doublet(rv, q_val)

            for (i_ct, ct) in enumerate(ct_grid)
                theta_opt = acos(ct)

                for (i_phi, phi) in enumerate(phi_grid)
                    idx += 1

                    # Rotate doublet to desired optical-frame orientation
                    R = rotation_matrix(theta_opt, phi)
                    pos_rot = R * pos_ref  # [3, 2]

                    # Scale to dimensionless (size parameter)
                    pos_x = pos_rot .* k
                    radii_x = radii .* k

                    # Compute scattering
                    result, _ = compute_scattering(
                        pos_x, radii_x, m_rel;
                        tol=CONVERGENCE_TOL,
                        max_iter=MAX_ITERATIONS,
                        use_fft=use_fft)

                    # Convert to CAS observables
                    S_s, S_p = bh83_to_cas(result.S_forward, k)

                    res_rv[idx]    = rv
                    res_q[idx]     = q_val
                    res_ct[idx]    = ct
                    res_phi[idx]   = phi
                    res_Ss_re[idx] = real(S_s)
                    res_Ss_im[idx] = imag(S_s)
                    res_Sp_re[idx] = real(S_p)
                    res_Sp_im[idx] = imag(S_p)
                    res_conv[idx]  = result.converged ? Int8(1) : Int8(0)

                    n_done += 1
                    if n_done % 100 == 0 || n_done == n_total
                        @printf("\r  Progress: %d / %d (%.1f%%)",
                                n_done, n_total, 100.0 * n_done / n_total)
                    end
                end
            end
        end
    end
    println()

    # --- Report convergence ---
    n_fail = count(==(Int8(0)), res_conv)
    if n_fail > 0
        @printf("WARNING: %d / %d computations did not converge\n",
                n_fail, n_total)
    else
        println("All computations converged.")
    end

    # --- Write HDF5 ---
    h5open(output_h5, "w") do f
        # Metadata
        attrs(f)["wavelength"]           = wavelength
        attrs(f)["medium_refindex"]      = M_MEDIUM
        attrs(f)["m_real_PS"]            = m_real_PS
        attrs(f)["m_imag_PS"]            = m_imag_PS
        attrs(f)["nominal_diameter_nm"]  = Int64(nominal_diameter_nm)

        # Grid specification (for rebuild by Python LUT builder)
        attrs(f)["rv_min"]  = Float64(rv_min)
        attrs(f)["rv_max"]  = Float64(rv_max)
        attrs(f)["n_rv"]    = Int64(n_rv)
        attrs(f)["q_min"]   = Float64(q_min)
        attrs(f)["q_max"]   = Float64(q_max)
        attrs(f)["n_q"]     = Int64(n_q)
        attrs(f)["ct_min"]  = -1.0
        attrs(f)["ct_max"]  = 1.0
        attrs(f)["n_ct"]    = Int64(n_ct)
        attrs(f)["n_phi"]   = Int64(n_phi)

        # Grid arrays
        f["r_v_grid"]          = collect(rv_grid)
        f["q_grid"]            = collect(q_grid)
        f["cos_theta_opt_grid"] = collect(ct_grid)
        f["phi_opt_grid"]      = collect(phi_grid)

        # Result datasets (flat 1D arrays, row-major order matching the loop)
        f["r_v"]      = res_rv
        f["q"]        = res_q
        f["cos_theta_opt"] = res_ct
        f["phi_opt"]  = res_phi
        f["S_s_re"]   = res_Ss_re
        f["S_s_im"]   = res_Ss_im
        f["S_p_re"]   = res_Sp_re
        f["S_p_im"]   = res_Sp_im
        f["converged"] = res_conv
    end

    @printf("Saved %d results to: %s\n", n_total, output_h5)
end


# =========================================================================
# CLI
# =========================================================================

function main()
    wavelength = 0.638
    nominal_diameter_nm = 0  # required
    output_h5 = nothing
    n_rv = N_RV_DEFAULT
    n_q = N_Q_DEFAULT
    n_ct = N_CT_DEFAULT
    n_phi = N_PHI_DEFAULT
    use_fft = USE_FFT_DEFAULT
    use_gpu = false

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--wavelength"
            i += 1
            wavelength = parse(Float64, ARGS[i])
        elseif arg == "--nominal-diameter"
            i += 1
            nominal_diameter_nm = parse(Int, ARGS[i])
        elseif arg == "--output"
            i += 1
            output_h5 = ARGS[i]
        elseif arg == "--n-rv"
            i += 1
            n_rv = parse(Int, ARGS[i])
        elseif arg == "--n-q"
            i += 1
            n_q = parse(Int, ARGS[i])
        elseif arg == "--n-ct"
            i += 1
            n_ct = parse(Int, ARGS[i])
        elseif arg == "--n-phi"
            i += 1
            n_phi = parse(Int, ARGS[i])
        elseif arg == "--use-fft"
            use_fft = true
        elseif arg == "--gpu"
            use_gpu = true
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    if nominal_diameter_nm == 0
        error("--nominal-diameter is required (e.g. --nominal-diameter 303)")
    end
    if !haskey(SIGMA_P_TABLE, nominal_diameter_nm)
        error("No sigma_p entry for $(nominal_diameter_nm)nm. " *
              "Available: $(sort(collect(keys(SIGMA_P_TABLE))))")
    end

    if output_h5 === nothing
        wl_nm = round(Int, wavelength * 1000)
        # Output directly to the doublet project's data directory
        doublet_project = expanduser(
            "~/Python_in_WSL/PCAS_Bayes_PSstandard_doublet")
        output_h5 = joinpath(doublet_project, "data", "wl_$(wl_nm)nm",
                             "doublet_sweep_wl$(wl_nm)nm_PS_$(nominal_diameter_nm)nm.h5")
    end

    mkpath(dirname(output_h5))

    run_sweep(;
        wavelength=wavelength,
        nominal_diameter_nm=nominal_diameter_nm,
        output_h5=output_h5,
        n_rv=n_rv, n_q=n_q, n_ct=n_ct, n_phi=n_phi,
        use_fft=use_fft,
        use_gpu=use_gpu)
end


main()
