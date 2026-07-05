# pcas_lut_schema producer: multi-orientation CAS-v2 sampled table for a sphere
# aggregate, via the cluster T-matrix + Wigner-D rotation (design P2/P4).
#
# Thin CLI:
#     julia --project=. scripts/run_sampled_lut.jl --preset {air,liquid} --output out.h5
#
# Builds the aggregate cluster T-matrix once per (geometry, material, wavelength),
# then evaluates the forward CAS-v2 amplitudes S_s(0)/S_p(0) at many orientations
# cheaply by rotation. Writes a contract-conformant `lut_sampled` HDF5 (columnar,
# species_kind="aggregate"), one row per (wavelength, m_p, orientation), grouped
# by geometry_id. Provenance uses the vendored contract canonical config hash.
#
# The example uses a tiny built-in aggregate and coarse material/orientation grids
# so it runs in seconds; edit the SETTINGS for a real sweep (budget accordingly).

using MSTMforCAS
using MSTMforCAS: build_cluster_tmatrix, cluster_forward_amplitudes
using SHA
using Printf
using HDF5

# ── vendored canonical hash (pcas_lut_schema/reference/canonical_hash.jl, v0.1.0) ──
_ch_enc(::Nothing) = UInt8['n', ';']
_ch_enc(v::Bool) = v ? UInt8['b', '1', ';'] : UInt8['b', '0', ';']
_ch_enc(v::Integer) = vcat(UInt8('i'), Vector{UInt8}(string(v)), UInt8(';'))
function _ch_enc(v::AbstractFloat)
    bits = hton(reinterpret(UInt64, Float64(v)))
    vcat(UInt8('f'), reinterpret(UInt8, [bits]))
end
function _ch_enc(v::AbstractString)
    b = Vector{UInt8}(String(v))
    vcat(UInt8('s'), Vector{UInt8}(string(length(b))), UInt8(':'), b)
end
function _ch_enc(v::Union{AbstractVector,Tuple})
    parts = UInt8[]
    for x in v; append!(parts, _ch_enc(x)); end
    vcat(UInt8('l'), Vector{UInt8}(string(length(v))), UInt8(':'), parts)
end
function _ch_enc(v::AbstractDict)
    ks = sort(collect(keys(v)); by = string)
    parts = UInt8[]
    for k in ks; append!(parts, _ch_enc(string(k))); append!(parts, _ch_enc(v[k])); end
    vcat(UInt8('d'), Vector{UInt8}(string(length(ks))), UInt8(':'), parts)
end

function _provenance_attrs(config::AbstractDict)
    repo = dirname(@__DIR__)   # scripts/ -> package root
    _git(args...) = try
        strip(read(Cmd(String["git", "-C", repo, args...]), String))
    catch
        ""
    end
    return [
        "producer_repo"    => "MSTMforCAS.jl",
        "producer_version" => "0.3.0",
        "git_sha"          => _git("rev-parse", "HEAD"),
        "git_dirty"        => !isempty(_git("status", "--porcelain", "--untracked-files=no")),
        "contract_version" => "0.1.0",
        "config_hash"      => bytes2hex(sha256(_ch_enc(config))),
        "created_utc"      => string(time()),   # epoch seconds (no Dates dep here)
    ]
end

# ── CLI / preset ─────────────────────────────────────────────────────────────
function _cli_arg(flag, default)
    i = findfirst(==(flag), ARGS)
    (i === nothing || i == length(ARGS)) ? default : ARGS[i + 1]
end
const PRESET = lowercase(_cli_arg("--preset", get(ENV, "MSTM_SWEEP_PRESET", "liquid")))
const MEDIUM_CONDITIONS =
    PRESET == "air"    ? [(0.638, 1.0), (0.834, 1.0)] :
    PRESET == "liquid" ? [(0.637, 1.3315), (0.773, 1.3300)] :
    error("unknown preset $(PRESET) (use 'air' or 'liquid')")
const OUT_FILE = _cli_arg("--output", joinpath(@__DIR__, "mstm_sampled_$(PRESET).h5"))

# ── SETTINGS (tiny example; edit for a real sweep) ───────────────────────────
# built-in example aggregate: 4 monomers, radius R_MON [um], non-overlapping
const R_MON = 0.03
const POSITIONS_PHYS = [  0.00  0.06  0.03  0.03;    # x [um]
                          0.00  0.00  0.052 0.017;   # y
                          0.00  0.00  0.00  0.049 ]  # z  (3 x 4)
const N_MON = size(POSITIONS_PHYS, 2)
const RADII_PHYS = fill(R_MON, N_MON)

# black-carbon-like material grid (absolute m_p = m_real + i*m_imag)
const M_REAL_VALS = [1.8, 2.0]
const M_IMAG_VALS = [0.6, 0.8]

# orientation set (intrinsic ZYZ Euler), coarse example
const ORIENTATIONS = [
    (0.0, 0.0, 0.0), (0.0, 0.6, 0.0), (0.7, 1.1, 0.0),
    (1.3, 0.8, 2.0), (2.5, 1.4, 0.5), (0.4, 0.3, 1.7),
]

const CONFIG = Dict{String,Any}(
    "preset"            => PRESET,
    "medium_conditions" => [[float(wl), float(mm)] for (wl, mm) in MEDIUM_CONDITIONS],
    "n_mon"             => N_MON,
    "r_mon"             => float(R_MON),
    "positions_phys"    => [float(POSITIONS_PHYS[i, j]) for j in 1:N_MON for i in 1:3],
    "m_real_vals"       => Float64.(M_REAL_VALS),
    "m_imag_vals"       => Float64.(M_IMAG_VALS),
    "n_orient"          => length(ORIENTATIONS),
)

println("MSTM sampled-table producer  preset=$(PRESET)  → $(OUT_FILE)")
@printf("  aggregate: %d monomers (R=%.3f um), %d materials, %d orientations, %d wavelengths\n",
        N_MON, R_MON, length(M_REAL_VALS) * length(M_IMAG_VALS),
        length(ORIENTATIONS), length(MEDIUM_CONDITIONS))

# ── sweep ────────────────────────────────────────────────────────────────────
row_id = Int64[]; geometry_id = Int64[]
wl_c = Float64[]; mm_c = Float64[]; mre = Float64[]; mim = Float64[]
ea = Float64[]; eb = Float64[]; eg = Float64[]
ssr = Float64[]; ssi = Float64[]; spr = Float64[]; spi = Float64[]
conv_c = Int8[]; trunc_c = Int64[]
rid = 0; gid = 0

for (wl, m_m) in MEDIUM_CONDITIONS
    k = 2π * m_m / wl                       # dimensional wavenumber [1/um]
    pos_dl = k .* POSITIONS_PHYS            # dimensionless
    rad_dl = k .* RADII_PHYS
    ik = im * k
    for mr in M_REAL_VALS, mi in M_IMAG_VALS
        global gid += 1
        m_rel = ComplexF64(mr, mi) / m_m
        T_cl, nodrt, r0, converged = build_cluster_tmatrix(pos_dl, rad_dl, m_rel)
        @printf("  [g%d] wl=%.3f m_p=%.2f+%.2fi  nodrt=%d conv=%s\n", gid, wl, mr, mi, nodrt, converged)
        for (α, β, γ) in ORIENTATIONS
            S1, S2, S3, S4 = cluster_forward_amplitudes(T_cl, nodrt, r0, α, β, γ)
            # BH83 -> MI02 (dimensional) -> CAS-v2 (Moteki & Adachi 2024)
            S11 = S2 / (-ik); S22 = S1 / (-ik); S12 = S3 / ik; S21 = S4 / ik
            Ss = S11 + im * S12
            Sp = S22 - im * S21
            global rid += 1
            push!(row_id, rid); push!(geometry_id, gid)
            push!(wl_c, wl); push!(mm_c, m_m); push!(mre, mr); push!(mim, mi)
            push!(ea, α); push!(eb, β); push!(eg, γ)
            push!(ssr, real(Ss)); push!(ssi, imag(Ss))
            push!(spr, real(Sp)); push!(spi, imag(Sp))
            push!(conv_c, converged ? Int8(1) : Int8(0)); push!(trunc_c, nodrt)
        end
    end
end

# ── write lut_sampled HDF5 ───────────────────────────────────────────────────
h5open(OUT_FILE, "w") do f
    attrs(f)["species_kind"] = "aggregate"
    attrs(f)["sampling"] = "grid(m_p) x fixed orientation set; one geometry"
    write_dataset(f, "row_id", row_id)
    write_dataset(f, "geometry_id", geometry_id)
    write_dataset(f, "wl_0", wl_c);   write_dataset(f, "m_m", mm_c)
    write_dataset(f, "m_real", mre);  write_dataset(f, "m_imag", mim)
    write_dataset(f, "N_mon", fill(Int64(N_MON), length(row_id)))
    write_dataset(f, "R_mon_mean", fill(float(R_MON), length(row_id)))
    write_dataset(f, "euler_alpha", ea); write_dataset(f, "euler_beta", eb); write_dataset(f, "euler_gamma", eg)
    write_dataset(f, "S_s_re", ssr); write_dataset(f, "S_s_im", ssi)
    write_dataset(f, "S_p_re", spr); write_dataset(f, "S_p_im", spi)
    write_dataset(f, "converged", conv_c); write_dataset(f, "truncation_order", trunc_c)
    pg = create_group(f, "provenance")
    for (kk, vv) in _provenance_attrs(CONFIG)
        attrs(pg)[kk] = vv
    end
end

@printf("Wrote %s  (%d rows, %d geometries)\n", OUT_FILE, length(row_id), gid)
