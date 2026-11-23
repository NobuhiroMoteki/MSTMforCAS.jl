"""
Regenerate the sweep behind the black-carbon LUT
(`results_fullsweep_agg20260323_00_ext.h5`: 50 Np x 22 m_real x 30 m_imag x 13 Df x 10 seeds
x 3 wavelengths) with the corrected GPU path.

The rows of that file with Np > 500 came from a GPU path that used the T-matrix of sphere 1
for every sphere. This script recomputes the grid with one uniform truncation order, so that
CPU rows and GPU rows are consistent. Results are written incrementally. The run can be
interrupted and resumed.

Two refractive index grids are available (`--preset`):
- `core` (default): a window per wavelength around the published posteriors, in steps of 0.10.
  Wavelengths are processed in the order 834, 638, 453 nm.
      834 nm: m_real 1.65..2.55 (10) x m_imag 0.55..1.45 (10)
      638 nm: m_real 1.55..2.55 (11) x m_imag 0.55..1.55 (11)
      453 nm: m_real 1.35..2.25 (10) x m_imag 0.75..1.55 (9)
- `ext`: the complement of `core` in a wider grid with the same step of 0.10. It adds the low
  m_imag values down to 0.15 and one more column at the two edges where the published
  posteriors come closest to the core window. `core` and `ext` together give
      834 nm: m_real 1.65..2.65 (11) x m_imag 0.15..1.45 (14)
      638 nm: m_real 1.55..2.55 (11) x m_imag 0.15..1.65 (16)
      453 nm: m_real 1.35..2.25 (10) x m_imag 0.15..1.55 (15)
- `ext2`: more m_real columns above `core` + `ext`, over the whole m_imag range. Added because
  the re-analysis with the corrected LUT puts the fullerene-soot solution at the upper m_real
  edge. `core`, `ext` and `ext2` together give
      834 nm: m_real 1.65..2.95 (14) x m_imag 0.15..1.45 (14)
      638 nm: m_real 1.55..2.75 (13) x m_imag 0.15..1.65 (16)
      453 nm: m_real 1.35..2.55 (13) x m_imag 0.15..1.55 (15)
- `full`: m_real 1.55..2.60 x m_imag 0.15..1.60 for all wavelengths, step `--ri-step`.

Run with:
  julia -t auto --project=. scripts/run_sweep_h5_regen.jl [options]

Options:
  --gpu                : GPU batched solver for groups with Np > 500 (requires CUDA.jl)
  --float32            : Float32 GPU iterations (implies --gpu)
  --truncation-order N : uniform VSWF truncation order, a lower bound for every sphere (default 3,
                         the order that the GPU rows of the original file used)
  --shared-basis       : one shared Krylov basis per aggregate and wavelength instead of one
                         CBICG solve per refractive index (CPU only). Use a tag of its own.
  --truncation-eps EPS : truncation order from the two-sphere calculation (see run_sweep_h5.jl)
  --fft-far-order F    : far order of the FFT grid (see run_sweep_h5.jl)
  --preset core|ext|ext2|full : refractive index grid, see above (default core)
  --ri-step STEP       : with --preset full: 0.05 (default) = the 22 x 30 grid, 0.10 = every
                         second value (11 x 15). The coarse values are a subset of the fine
                         grid, so a later run with 0.05 into the same file adds the rest.
  --wl LIST            : comma-separated wavelengths in um to process (default: all three)
  --np-min N, --np-max N : range of monomer numbers (default 350 to 2800)
  --df-min X, --df-max X : range of fractal dimensions (default 2.35 to 2.95)
  --part I/N           : process only the aggregates with index ≡ I-1 (mod N). Use it to split
                         the work between machines. Each part writes its own output file.
  --tag TAG            : tag in the output file name (default "regen")

Output: data/results/results_fullsweep_agg<ID>_<TAG>[_partIofN].h5
"""

using MSTMforCAS, Printf

function _arg(name::String, default)
    for (i, arg) in enumerate(ARGS)
        arg == name && i < length(ARGS) && return ARGS[i+1]
    end
    return default
end

gpu_float32 = "--float32" in ARGS
use_gpu = "--gpu" in ARGS || gpu_float32
if use_gpu
    @info "GPU mode requested — loading CUDA.jl..."
    using CUDA
end

truncation_order = parse(Int, _arg("--truncation-order", "3"))
shared_basis = "--shared-basis" in ARGS
shared_basis && use_gpu && error("--shared-basis is not available with --gpu or --float32")
truncation_eps = (v = _arg("--truncation-eps", nothing); v === nothing ? nothing : parse(Float64, v))
fft_far_order  = (v = _arg("--fft-far-order", nothing); v === nothing ? nothing : parse(Int, v))
preset  = _arg("--preset", "core")
ri_step = parse(Float64, _arg("--ri-step", "0.05"))
wl_sel  = parse.(Float64, split(_arg("--wl", "0.834,0.638,0.453"), ","))
df_min  = parse(Float64, _arg("--df-min", "2.35"))
df_max  = parse(Float64, _arg("--df-max", "2.95"))
np_min  = parse(Int, _arg("--np-min", "350"))
np_max  = parse(Int, _arg("--np-max", "2800"))
tag     = _arg("--tag", "regen")
part_i, part_n = let p = _arg("--part", "1/1")
    a, b = parse.(Int, split(p, "/"))
    1 <= a <= b || error("--part I/N needs 1 <= I <= N")
    (a, b)
end

# ─── Refractive index grid per wavelength ────────────────────────────────────
# ri_grids[wavelength in um] = list of rectangles ((m_real min, max, n), (m_imag min, max, n))
ri_grids =
    if preset == "core"
        Dict(0.834 => [((1.65, 2.55, 10), (0.55, 1.45, 10))],
             0.638 => [((1.55, 2.55, 11), (0.55, 1.55, 11))],
             0.453 => [((1.35, 2.25, 10), (0.75, 1.55, 9))])
    elseif preset == "ext"
        Dict(0.834 => [((2.65, 2.65, 1), (0.55, 1.45, 10)),      # one more m_real column
                       ((1.65, 2.65, 11), (0.15, 0.45, 4))],     # low m_imag block
             0.638 => [((1.55, 2.55, 11), (1.65, 1.65, 1)),      # one more m_imag row
                       ((1.55, 2.55, 11), (0.15, 0.45, 4))],
             0.453 => [((1.35, 2.25, 10), (0.15, 0.65, 6))])
    elseif preset == "ext2"
        Dict(0.834 => [((2.75, 2.95, 3), (0.15, 1.45, 14))],
             0.638 => [((2.65, 2.75, 2), (0.15, 1.65, 16))],
             0.453 => [((2.35, 2.55, 3), (0.15, 1.55, 15))])
    elseif preset == "full"
        g = isapprox(ri_step, 0.05) ? ((1.55, 2.60, 22), (0.15, 1.60, 30)) :
            isapprox(ri_step, 0.10) ? ((1.55, 2.55, 11), (0.15, 1.55, 15)) :
            error("--ri-step must be 0.05 or 0.10")
        Dict(0.834 => [g], 0.638 => [g], 0.453 => [g])
    else
        error("--preset must be core, ext, ext2 or full")
    end
all(wl -> haskey(ri_grids, wl), wl_sel) || error("--wl values must be among 0.834, 0.638, 0.453")
n_ri(wl) = sum(r[1][3] * r[2][3] for r in ri_grids[wl])

# ─── Aggregate data files ────────────────────────────────────────────────────
agg_dir  = joinpath(@__DIR__, "..", "data", "aggregates")
h5_file  = joinpath(agg_dir, "aggregates_20260323_00.h5")
csv_file = joinpath(agg_dir, "catalog_20260323_00.csv")
agg_id = let m = match(r"(\d{8}_\d{2})", basename(h5_file))
    m !== nothing ? m[1] : "unknown"
end

println("Reading aggregate catalog:")
println("  H5:  $h5_file")
println("  CSV: $csv_file")
aggregates = read_aggregate_catalog(h5_file, csv_file; Np_range=(np_min, np_max),
                                    Df_range=(df_min - 1e-6, df_max + 1e-6))
n_all = length(aggregates)
aggregates = aggregates[part_i:part_n:end]
println("  Loaded $n_all aggregates with Np in [$np_min, $np_max], Df in [$df_min, $df_max]; " *
        "part $part_i/$part_n has $(length(aggregates))")
println()

# ─── Output path ─────────────────────────────────────────────────────────────
results_dir = joinpath(@__DIR__, "..", "data", "results")
mkpath(results_dir)
part_tag = part_n == 1 ? "" : "_part$(part_i)of$(part_n)"
output_h5 = joinpath(results_dir, "results_fullsweep_agg$(agg_id)_$(tag)$(part_tag).h5")
println("Output: $output_h5")
isfile(output_h5) && println("  (existing file found — will resume from where left off)")
println()

# ─── Run one sweep per wavelength ────────────────────────────────────────────
println("Preset: $preset, threads: $(Threads.nthreads()), use_gpu: $use_gpu, gpu_float32: $gpu_float32, " *
        "truncation_order: $truncation_order, truncation_eps: $(repr(truncation_eps)), " *
        "fft_far_order: $(repr(fft_far_order)), shared_basis: $shared_basis")
n_total = sum(length(aggregates) * n_ri(wl) for wl in wl_sel)
println("Total jobs: $n_total")
println()

t0 = time()
for wl in wl_sel, (m_real_range, m_imag_range) in ri_grids[wl]
    config = SweepConfig(
        medium_conditions = [(wl, 1.0)],          # (wavelength [μm], n_medium)
        m_real_range = m_real_range,
        m_imag_range = m_imag_range,
        max_iterations      = 200,
        convergence_epsilon = 1e-6,
        use_fft             = true,
        truncation_order    = truncation_order,
        truncation_eps      = truncation_eps,
        fft_far_order       = fft_far_order,
        shared_basis        = shared_basis,
        use_gpu             = use_gpu,
        gpu_float32         = gpu_float32,
        gpu_np_threshold    = 500,
    )
    n_rect = m_real_range[3] * m_imag_range[3]
    @printf("=== wavelength %.3f um: m_real %s × m_imag %s → %d RI values, %d jobs ===\n",
            wl, string(m_real_range), string(m_imag_range), n_rect, length(aggregates) * n_rect)
    run_parameter_sweep(aggregates, config; output_h5=output_h5)
end
@printf("Done in %.1f s\nResults saved to: %s\n", time() - t0, output_h5)
