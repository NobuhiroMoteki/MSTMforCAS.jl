"""
Merge the original LUT with the extension LUT into a unified LUT.

  original  : data/results/results_fullsweep_agg20260323_00.h5
  extension : data/results/results_fullsweep_agg20260323_00_extension.h5
  output    : data/results/results_fullsweep_agg20260323_00_ext.h5  (22×30 RI grid)

The merge is a simple row-wise concatenation of every dataset, with one
normalization step: `m_real` and `m_imag` are rounded to 2 decimal places
(step 0.05) so that the combined file presents a clean unique-value grid
(22 unique m_real values × 30 unique m_imag values) regardless of tiny
Float64 drift between the two sweeps' `range()` calls.

Run with:
  julia --project=. scripts/merge_extension_h5.jl
"""

using HDF5, Printf

const ORIGINAL_H5  = joinpath(@__DIR__, "..", "data", "results", "results_fullsweep_agg20260323_00.h5")
const EXTENSION_H5 = joinpath(@__DIR__, "..", "data", "results", "results_fullsweep_agg20260323_00_extension.h5")
const MERGED_H5    = joinpath(@__DIR__, "..", "data", "results", "results_fullsweep_agg20260323_00_ext.h5")

const H5_CHUNK = 10_000

# ─── Dataset schema (must match _SWEEP_COLS in src/ParameterSweep.jl) ───────
const SWEEP_COLS = Tuple{String,DataType}[
    ("source",           String),
    ("mean_rp",          Float64),
    ("rel_std_rp",       Float64),
    ("k_f",              Float64),
    ("Df",               Float64),
    ("n_monomers",       Int64),
    ("agg_num",          Int64),
    ("R_ve",             Float64),
    ("R_g",              Float64),
    ("eps_agg",          Float64),
    ("wavelength",       Float64),
    ("medium_refindex",  Float64),
    ("m_real",           Float64),
    ("m_imag",           Float64),
    ("S1_fwd_re",        Float64), ("S1_fwd_im",  Float64),
    ("S2_fwd_re",        Float64), ("S2_fwd_im",  Float64),
    ("S3_fwd_re",        Float64), ("S3_fwd_im",  Float64),
    ("S4_fwd_re",        Float64), ("S4_fwd_im",  Float64),
    ("S11_fwd_re",       Float64), ("S11_fwd_im", Float64),
    ("S22_fwd_re",       Float64), ("S22_fwd_im", Float64),
    ("S12_fwd_re",       Float64), ("S12_fwd_im", Float64),
    ("S21_fwd_re",       Float64), ("S21_fwd_im", Float64),
    ("S1_bwd_re",        Float64), ("S1_bwd_im",  Float64),
    ("S2_bwd_re",        Float64), ("S2_bwd_im",  Float64),
    ("S3_bwd_re",        Float64), ("S3_bwd_im",  Float64),
    ("S4_bwd_re",        Float64), ("S4_bwd_im",  Float64),
    ("S11_bwd_re",       Float64), ("S11_bwd_im", Float64),
    ("S22_bwd_re",       Float64), ("S22_bwd_im", Float64),
    ("S12_bwd_re",       Float64), ("S12_bwd_im", Float64),
    ("S21_bwd_re",       Float64), ("S21_bwd_im", Float64),
    ("Q_ext",            Float64),
    ("Q_abs",            Float64),
    ("Q_sca",            Float64),
    ("converged",        Int8),
    ("n_iterations",     Int64),
    ("truncation_order", Int64),
]

# ─── Expected final grid (for verification and attribute metadata) ──────────
const M_REAL_RANGE_NEW = (1.55, 2.60, 22)
const M_IMAG_RANGE_NEW = (0.15, 1.60, 30)

function main()
    for p in (ORIGINAL_H5, EXTENSION_H5)
        isfile(p) || error("Input file not found: $p")
    end
    if isfile(MERGED_H5)
        error("Output file already exists (refusing to overwrite): $MERGED_H5")
    end

    println("Merging:")
    println("  original  : $ORIGINAL_H5")
    println("  extension : $EXTENSION_H5")
    println("  output    : $MERGED_H5")
    println()

    # Read row counts to allocate final extents
    n_orig, n_ext = HDF5.h5open(ORIGINAL_H5, "r") do f
        size(f["source"], 1)
    end, HDF5.h5open(EXTENSION_H5, "r") do f
        size(f["source"], 1)
    end
    n_total = n_orig + n_ext
    @printf("Rows: original=%d, extension=%d, total=%d\n\n", n_orig, n_ext, n_total)

    HDF5.h5open(MERGED_H5, "w") do fout
        # ─── File-level attributes ─────────────────────────────────────────
        HDF5.h5open(ORIGINAL_H5, "r") do forig
            # Preserve medium conditions from the original file
            wl_vals   = HDF5.attrs(forig)["medium_conditions_wavelength"]
            nmed_vals = HDF5.attrs(forig)["medium_conditions_refindex"]
            HDF5.attrs(fout)["medium_conditions_wavelength"] = wl_vals
            HDF5.attrs(fout)["medium_conditions_refindex"]   = nmed_vals
        end
        HDF5.attrs(fout)["m_real_range"] = Float64[M_REAL_RANGE_NEW...]
        HDF5.attrs(fout)["m_imag_range"] = Float64[M_IMAG_RANGE_NEW...]

        # ─── Concatenate each column ───────────────────────────────────────
        HDF5.h5open(ORIGINAL_H5, "r") do forig
            HDF5.h5open(EXTENSION_H5, "r") do fext
                for (col, T) in SWEEP_COLS
                    print("  $col ... ")
                    v_orig = read(forig[col])
                    v_ext  = read(fext[col])
                    @assert length(v_orig) == n_orig "orig length mismatch for $col"
                    @assert length(v_ext)  == n_ext  "ext  length mismatch for $col"

                    # Normalize m_real and m_imag to canonical step-0.05 values
                    if col == "m_real" || col == "m_imag"
                        v_orig = round.(v_orig, digits=2)
                        v_ext  = round.(v_ext,  digits=2)
                    end

                    if T === String
                        # Strings: create simple dataset (non-extendable)
                        vcat_vals = vcat(v_orig, v_ext)
                        HDF5.create_dataset(fout, col, HDF5.datatype(T),
                            HDF5.dataspace((n_total,)))
                        fout[col][:] = vcat_vals
                    else
                        # Numeric: extendable dataset, chunk-aligned
                        ds = HDF5.create_dataset(fout, col, HDF5.datatype(T),
                            HDF5.dataspace((n_total,), (-1,)); chunk=(H5_CHUNK,))
                        ds[1:n_orig] = v_orig
                        ds[(n_orig+1):n_total] = v_ext
                    end
                    println("done")
                end
            end
        end
    end

    println()
    println("Verifying merged file…")
    HDF5.h5open(MERGED_H5, "r") do f
        n = size(f["source"], 1)
        @assert n == n_total "total row count mismatch"
        m_real = read(f["m_real"])
        m_imag = read(f["m_imag"])
        u_mr = sort(unique(m_real))
        u_mi = sort(unique(m_imag))
        @printf("  total rows        = %d (expected %d)\n", n, n_total)
        @printf("  unique m_real     = %d (expected 22)\n", length(u_mr))
        @printf("  unique m_imag     = %d (expected 30)\n", length(u_mi))
        println("  m_real values: ", u_mr)
        println("  m_imag values: ", u_mi)
    end

    @printf("\nMerge complete: %s\n", MERGED_H5)
end

main()
