"""
Merge sweep result files into one HDF5 file, row by row.

Run with:
  julia --project=. scripts/merge_sweep_h5.jl OUT.h5 IN1.h5 [IN2.h5 ...] [--core]

- All input files must have the datasets of `_SWEEP_COLS` in src/ParameterSweep.jl.
- `m_real` and `m_imag` are rounded to 2 decimals, so that values from different
  `range()` calls give one clean grid.
- `--core` keeps only the rows on the "core" grid of scripts/run_sweep_h5_regen.jl
  (a refractive index window per wavelength in steps of 0.10). Use it to cut a full-grid
  file down to that grid, for a LUT that differs from a regenerated one in the data only.
- Rows with the same (source, wavelength, m_real, m_imag) in more than one place are an
  error.

The output file is written in one piece. Root attributes are not copied.
"""

using HDF5, Printf

const H5_CHUNK = 10_000

# (m_real min, max), (m_imag min, max) per wavelength [um]; values in steps of 0.10
const CORE_WINDOWS = Dict(
    834 => ((1.65, 2.55), (0.55, 1.45)),
    638 => ((1.55, 2.55), (0.55, 1.55)),
    453 => ((1.35, 2.25), (0.75, 1.55)),
)

"""`true` if `x` (rounded to 2 decimals) lies on the grid lo, lo+0.10, ..., hi."""
function _on_grid(x::Float64, lo::Float64, hi::Float64)::Bool
    k = round(Int, 100 * (x - lo))
    return k >= 0 && x <= hi + 1e-9 && k % 10 == 0
end

function _core_mask(wl::Vector{Float64}, mr::Vector{Float64}, mi::Vector{Float64})::BitVector
    mask = falses(length(wl))
    for i in eachindex(wl)
        key = round(Int, 1000 * wl[i])
        haskey(CORE_WINDOWS, key) || continue
        (r_lo, r_hi), (i_lo, i_hi) = CORE_WINDOWS[key]
        mask[i] = _on_grid(mr[i], r_lo, r_hi) && _on_grid(mi[i], i_lo, i_hi)
    end
    return mask
end

function main()
    args = filter(a -> !startswith(a, "--"), ARGS)
    core = "--core" in ARGS
    length(args) >= 2 || error("usage: merge_sweep_h5.jl OUT.h5 IN1.h5 [IN2.h5 ...] [--core]")
    out = args[1]
    inputs = args[2:end]
    isfile(out) && error("$out exists. Remove it first.")
    foreach(p -> isfile(p) || error("missing input $p"), inputs)

    # ── Row selection per input and duplicate check ──────────────────────────
    masks = BitVector[]
    seen = Set{Tuple{String,Int,Int,Int}}()
    for p in inputs
        h5open(p, "r") do f
            wl = read(f["wavelength"])
            mr = round.(read(f["m_real"]); digits = 2)
            mi = round.(read(f["m_imag"]); digits = 2)
            src = read(f["source"])
            mask = core ? _core_mask(wl, mr, mi) : trues(length(wl))
            for i in eachindex(wl)
                mask[i] || continue
                key = (src[i], round(Int, 1000 * wl[i]), round(Int, 100 * mr[i]), round(Int, 100 * mi[i]))
                key in seen && error("duplicate row $key in $p")
                push!(seen, key)
            end
            push!(masks, mask)
            @printf("%-70s rows %9d, kept %9d\n", basename(p), length(wl), count(mask))
        end
    end
    n_total = sum(count, masks)

    # ── Copy dataset by dataset ──────────────────────────────────────────────
    names = h5open(f -> sort(collect(keys(f))), inputs[1], "r")
    h5open(out, "w") do fo
        for name in names
            parts = Any[]
            for (p, mask) in zip(inputs, masks)
                v = h5open(f -> read(f[name]), p, "r")
                name in ("m_real", "m_imag") && (v = round.(v; digits = 2))
                push!(parts, v[mask])
            end
            col = reduce(vcat, parts)
            length(col) == n_total || error("length mismatch in $name")
            fo[name, chunk = (min(H5_CHUNK, max(n_total, 1)),)] = col
        end
        HDF5.attrs(fo)["merged_from"] = join(basename.(inputs), ";")
        HDF5.attrs(fo)["core_subset"] = core ? 1 : 0
    end
    @printf("wrote %s: %d rows, %d datasets\n", out, n_total, length(names))
end

main()
