"""
    FFTTranslation

FFT-accelerated translation for the multi-sphere T-matrix solver.
Replaces O(N²) direct sphere-to-sphere translation with O(N + M log M) grid-based
convolution following Mackowski (2014).

Algorithm:
1. Map spheres to uniform 3D grid cells
2. Precompute FFT of cell-to-cell translation matrices (excluding near-field neighbors)
3. Per solver iteration:
   a. Sphere → Node: translate sphere coefficients to nearest grid node (J-type Bessel)
   b. FFT convolution: forward FFT → element-wise multiply → inverse FFT
   c. Node → Sphere: translate grid node results back to sphere positions
   d. Near-field correction: direct Hankel translation for excluded neighbor pairs
"""

using FFTW

# ─────────────────────────────────────────────────────────────
# Helper: round up to nearest n = 2^a × 3^b × 5^c
# ─────────────────────────────────────────────────────────────

"""
    _correctn235(n) -> Int

Round `n` up to the nearest integer of the form 2^a × 3^b × 5^c,
suitable for efficient FFT computation.
"""
function _correctn235(n::Int)::Int
    n <= 1 && return 1
    candidates = Int[]
    max_val = 4 * n  # generous upper bound
    a = 1
    while a <= max_val
        b = a
        while b <= max_val
            c = b
            while c <= max_val
                c >= n && push!(candidates, c)
                c *= 5
            end
            b *= 3
        end
        a *= 2
    end
    return minimum(candidates)
end

# ─────────────────────────────────────────────────────────────
# Grid setup data structure
# ─────────────────────────────────────────────────────────────

struct FFTGridData
    cell_dim      ::NTuple{3, Int}          # grid dimensions (nx, ny, nz)
    d_cell        ::Float64                  # uniform cell spacing
    cell_boundary ::Vector{Float64}          # (3,) lower corner of grid
    node_order    ::Int                      # multipole order per grid node
    nblk_node     ::Int                      # node_order * (node_order + 2)
    sphere_node   ::Matrix{Int}             # (3, N) grid cell index per sphere
    neighbor_offsets ::Vector{NTuple{3,Int}} # neighbor cell offsets (model=2)
    # Per-sphere neighbor lists: neighbor_pairs[i] = [j1, j2, ...] for j > i
    neighbor_pairs ::Vector{Vector{Int}}
    # Pre-FFT'd translation matrices: (2nx, 2ny, 2nz, nblk, nblk) per p-mode
    cell_tran_fft_p1 ::Array{ComplexF64, 5}
    cell_tran_fft_p2 ::Array{ComplexF64, 5}
    # Cached sphere-to-node J-type translation matrices
    # sphere_node_H[i] = H[nblk_node, hnb_sphere_i, 2]
    sphere_node_H ::Vector{Array{ComplexF64, 3}}
    # Cached node-to-sphere reverse translation matrices
    # node_sphere_H[i] = H[hnb_sphere_i, nblk_node, 2]
    node_sphere_H ::Vector{Array{ComplexF64, 3}}
    # FFTW plans
    fft_plan  ::Any   # pre-planned forward FFT
    ifft_plan ::Any   # pre-planned inverse FFT
end

# ─────────────────────────────────────────────────────────────
# Grid setup
# ─────────────────────────────────────────────────────────────

"""
    _build_neighbor_offsets(model::Int) -> Vector{NTuple{3,Int}}

Build neighbor cell offsets for given model:
- model=0: self only (1 cell)
- model=1: self + face-adjacent (7 cells)
- model=2: self + face + edge adjacent (19 cells)
"""
function _build_neighbor_offsets(model::Int)::Vector{NTuple{3,Int}}
    offsets = NTuple{3,Int}[]
    for iz in -1:1, iy in -1:1, ix in -1:1
        ir = ix^2 + iy^2 + iz^2
        if model == 0
            ir > 0 && continue
        elseif model == 1
            ir > 1 && continue
        elseif model == 2
            ir > 2 && continue
        end
        push!(offsets, (ix, iy, iz))
    end
    return offsets
end

"""
    _setup_fft_grid(positions, radii, nois, max_noi; fv_target=0.2, neighbor_model=2)

Create uniform 3D grid, assign spheres to cells, and build neighbor lists.
"""
function _setup_fft_grid(
    positions::Matrix{Float64},
    radii::Vector{Float64},
    nois::Vector{Int},
    max_noi::Int;
    fv_target::Float64 = 0.2,
    neighbor_model::Int = 2
)
    N = size(positions, 2)

    # Bounding box
    pos_min = vec(minimum(positions, dims=2))
    pos_max = vec(maximum(positions, dims=2))
    extent  = pos_max .- pos_min

    # Cell size from per-sphere volume fraction (Fortran formula):
    # d_cell^3 = (4π/3) × r_mean^3 / fv_target
    # This gives the cell volume such that one average sphere occupies fv_target of the cell.
    r_mean = sum(radii) / N
    d_cell = r_mean * (4π / 3 / fv_target)^(1/3)

    # Grid dimensions (round up to 2^a × 3^b × 5^c)
    cell_dim = ntuple(3) do m
        if extent[m] < 1e-10
            1
        else
            _correctn235(ceil(Int, extent[m] / d_cell))
        end
    end

    # Recompute d_cell to exactly cover bounding box
    d_cell_candidates = [extent[m] / cell_dim[m] for m in 1:3 if cell_dim[m] > 1]
    if !isempty(d_cell_candidates)
        d_cell = maximum(d_cell_candidates)
    end
    d_cell = max(d_cell, 0.1)  # safety floor

    # Cell boundary (center the grid around the spheres)
    cell_boundary = Vector{Float64}(undef, 3)
    for m in 1:3
        grid_extent = d_cell * cell_dim[m]
        center = 0.5 * (pos_min[m] + pos_max[m])
        cell_boundary[m] = center - 0.5 * grid_extent
    end

    # Node order: determined from cell size, must be >= max Mie order
    node_order = max(ceil(Int, d_cell), max_noi)

    # Assign spheres to grid cells
    sphere_node = Matrix{Int}(undef, 3, N)
    for i in 1:N
        for m in 1:3
            idx = floor(Int, (positions[m, i] - cell_boundary[m]) / d_cell) + 1
            sphere_node[m, i] = clamp(idx, 1, cell_dim[m])
        end
    end

    # Build neighbor offsets
    neighbor_offsets = _build_neighbor_offsets(neighbor_model)

    # Build cell-to-sphere lookup
    cell_spheres = Dict{NTuple{3,Int}, Vector{Int}}()
    for i in 1:N
        key = (sphere_node[1,i], sphere_node[2,i], sphere_node[3,i])
        if !haskey(cell_spheres, key)
            cell_spheres[key] = Int[]
        end
        push!(cell_spheres[key], i)
    end

    # Build per-sphere neighbor lists (all pairs (i,j) in neighbor cells)
    neighbor_pairs = [Int[] for _ in 1:N]
    for i in 1:N
        ni = (sphere_node[1,i], sphere_node[2,i], sphere_node[3,i])
        for off in neighbor_offsets
            nj = (ni[1] + off[1], ni[2] + off[2], ni[3] + off[3])
            # Bounds check
            (1 <= nj[1] <= cell_dim[1] && 1 <= nj[2] <= cell_dim[2] && 1 <= nj[3] <= cell_dim[3]) || continue
            spheres_in_cell = get(cell_spheres, nj, Int[])
            for j in spheres_in_cell
                j == i && continue
                push!(neighbor_pairs[i], j)
            end
        end
    end

    return cell_dim, d_cell, cell_boundary, node_order, sphere_node,
           neighbor_offsets, neighbor_pairs
end

# ─────────────────────────────────────────────────────────────
# FFT translation matrix precomputation
# ─────────────────────────────────────────────────────────────

"""
    _precompute_fft_translation(cell_dim, d_cell, node_order, neighbor_offsets)

Precompute the FFT of cell-to-cell translation matrices.
Returns two 5D arrays (for p=1, p=2 modes) of shape (2nx, 2ny, 2nz, nblk, nblk).
"""
function _precompute_fft_translation(
    cell_dim::NTuple{3,Int},
    d_cell::Float64,
    node_order::Int,
    neighbor_offsets::Vector{NTuple{3,Int}}
)
    nx, ny, nz = cell_dim
    nblk = node_order * (node_order + 2)
    k_medium = ComplexF64(1.0)  # positions are already in size parameter units

    # Allocate translation matrices on 2× grid (for zero-padding)
    tran_p1 = zeros(ComplexF64, 2nx, 2ny, 2nz, nblk, nblk)
    tran_p2 = zeros(ComplexF64, 2nx, 2ny, 2nz, nblk, nblk)

    # Neighbor hole set for quick lookup
    hole_set = Set(neighbor_offsets)

    # Compute translation matrix for each grid offset.
    # Offsets range from -(dim-1) to +(dim-1). We iterate non-negative offsets and
    # use sign combinations, skipping duplicate signs when offset=0 on an axis.
    for izoff in 0:nz-1, iyoff in 0:ny-1, ixoff in 0:nx-1
        for isz in (izoff == 0 ? (1,) : (1, -1)),
            isy in (iyoff == 0 ? (1,) : (1, -1)),
            isx in (ixoff == 0 ? (1,) : (1, -1))

            dx = isx * ixoff
            dy = isy * iyoff
            dz = isz * izoff

            # Skip zero offset (self-interaction handled separately)
            dx == 0 && dy == 0 && dz == 0 && continue

            # Check if in neighbor hole
            (dx, dy, dz) in hole_set && continue

            # Wrapped 2× grid coordinates (1-indexed)
            n1 = dx >= 0 ? dx + 1 : dx + 2nx + 1
            n2 = dy >= 0 ? dy + 1 : dy + 2ny + 1
            n3 = dz >= 0 ? dz + 1 : dz + 2nz + 1

            # Physical translation vector
            r_vec = Float64[dx * d_cell, dy * d_cell, dz * d_cell]

            # Compute Hankel translation matrix (outgoing wave, use_regular=false)
            H = compute_translation_matrix(r_vec, k_medium, node_order, node_order;
                                           use_regular=false)
            # H is (nblk, nblk, 2) where dim 3 = p-mode

            tran_p1[n1, n2, n3, :, :] .= H[:, :, 1]
            tran_p2[n1, n2, n3, :, :] .= H[:, :, 2]
        end
    end

    # Forward FFT along spatial dimensions (first 3 dims) for each (l, n) pair
    for l in 1:nblk, n in 1:nblk
        slice1 = @view tran_p1[:, :, :, l, n]
        slice1 .= fft(slice1)
        slice2 = @view tran_p2[:, :, :, l, n]
        slice2 .= fft(slice2)
    end

    return tran_p1, tran_p2
end

# ─────────────────────────────────────────────────────────────
# Sphere ↔ Node translation matrix caching
# ─────────────────────────────────────────────────────────────

"""
    _precompute_sphere_node_translations(positions, sphere_node, cell_boundary,
                                         d_cell, nois, node_order)

Precompute J-type (regular Bessel) translation matrices for sphere↔node mapping.
"""
function _precompute_sphere_node_translations(
    positions::Matrix{Float64},
    sphere_node::Matrix{Int},
    cell_boundary::Vector{Float64},
    d_cell::Float64,
    nois::Vector{Int},
    node_order::Int
)
    N = size(positions, 2)
    k_medium = ComplexF64(1.0)

    sphere_to_node_H = Vector{Array{ComplexF64, 3}}(undef, N)
    node_to_sphere_H = Vector{Array{ComplexF64, 3}}(undef, N)

    for i in 1:N
        # Translation vector: node center → sphere position
        # Node center: cell_boundary + d_cell * (node_idx - 0.5)
        rtran = Float64[
            d_cell * (sphere_node[m, i] - 0.5) + cell_boundary[m] - positions[m, i]
            for m in 1:3
        ]

        noi = nois[i]

        # Sphere→Node: translate from sphere (order noi) to node (order node_order)
        # Regular Bessel (J-type) since sphere is inside the node region
        sphere_to_node_H[i] = compute_translation_matrix(
            rtran, k_medium, noi, node_order; use_regular=true)

        # Node→Sphere: translate from node (order node_order) to sphere (order noi)
        # Reverse direction: -rtran
        node_to_sphere_H[i] = compute_translation_matrix(
            -rtran, k_medium, node_order, noi; use_regular=true)
    end

    return sphere_to_node_H, node_to_sphere_H
end

# ─────────────────────────────────────────────────────────────
# Full FFT grid initialization
# ─────────────────────────────────────────────────────────────

"""
    init_fft_grid(positions, radii, nois) -> FFTGridData

Initialize the FFT translation grid for the given sphere configuration.
"""
function init_fft_grid(
    positions::Matrix{Float64},
    radii::Vector{Float64},
    nois::Vector{Int}
)::FFTGridData
    N = size(positions, 2)
    max_noi = maximum(nois)

    # Grid setup
    cell_dim, d_cell, cell_boundary, node_order, sphere_node,
        neighbor_offsets, neighbor_pairs = _setup_fft_grid(
            positions, radii, nois, max_noi)

    nblk_node = node_order * (node_order + 2)

    # Precompute FFT of translation matrices
    tran_p1, tran_p2 = _precompute_fft_translation(
        cell_dim, d_cell, node_order, neighbor_offsets)

    # Precompute sphere↔node translation matrices
    sphere_to_node_H, node_to_sphere_H = _precompute_sphere_node_translations(
        positions, sphere_node, cell_boundary, d_cell, nois, node_order)

    # Create FFTW plans for the convolution step
    nx, ny, nz = cell_dim
    plan_buf = zeros(ComplexF64, 2nx, 2ny, 2nz)
    fft_plan  = plan_fft!(plan_buf)
    ifft_plan = plan_ifft!(plan_buf)

    return FFTGridData(
        cell_dim, d_cell, cell_boundary, node_order, nblk_node,
        sphere_node, neighbor_offsets, neighbor_pairs,
        tran_p1, tran_p2,
        sphere_to_node_H, node_to_sphere_H,
        fft_plan, ifft_plan
    )
end

# ─────────────────────────────────────────────────────────────
# Per-iteration FFT translation operator
# ─────────────────────────────────────────────────────────────

"""
    _sphere_to_node!(anode, inp, fft_data, offsets, half_nblks, nois)

Translate sphere expansion coefficients to grid nodes (J-type Bessel).
`anode` is (nx, ny, nz, nblk_node, 2) for p=1,2 modes.
"""
function _sphere_to_node!(
    anode::Array{ComplexF64, 5},
    inp::Vector{ComplexF64},
    fft_data::FFTGridData,
    offsets::Vector{Int},
    half_nblks::Vector{Int},
    nois::Vector{Int}
)
    fill!(anode, zero(ComplexF64))
    N = length(offsets)
    nblk_node = fft_data.nblk_node

    @inbounds for i in 1:N
        off = offsets[i]
        hnb = half_nblks[i]
        H = fft_data.sphere_node_H[i]  # (nblk_node, hnb, 2)
        ix = fft_data.sphere_node[1, i]
        iy = fft_data.sphere_node[2, i]
        iz = fft_data.sphere_node[3, i]

        for p in 1:2
            p_off_s = (p - 1) * hnb  # source p-block offset within sphere
            for kl in 1:nblk_node
                val = zero(ComplexF64)
                for mn in 1:hnb
                    val += H[kl, mn, p] * inp[off + p_off_s + mn]
                end
                anode[ix, iy, iz, kl, p] += val
            end
        end
    end
end

"""
    _node_to_sphere!(out, gnode, fft_data, offsets, half_nblks, nois)

Translate grid node results back to sphere positions (reverse J-type Bessel).
`gnode` is (nx, ny, nz, nblk_node, 2).
"""
function _node_to_sphere!(
    out::Vector{ComplexF64},
    gnode::Array{ComplexF64, 5},
    fft_data::FFTGridData,
    offsets::Vector{Int},
    half_nblks::Vector{Int},
    nois::Vector{Int}
)
    N = length(offsets)
    nblk_node = fft_data.nblk_node

    @inbounds for i in 1:N
        off = offsets[i]
        hnb = half_nblks[i]
        H = fft_data.node_sphere_H[i]  # (hnb, nblk_node, 2)
        ix = fft_data.sphere_node[1, i]
        iy = fft_data.sphere_node[2, i]
        iz = fft_data.sphere_node[3, i]

        for p in 1:2
            p_off_t = (p - 1) * hnb  # target p-block offset
            for mn in 1:hnb
                val = zero(ComplexF64)
                for kl in 1:nblk_node
                    val += H[mn, kl, p] * gnode[ix, iy, iz, kl, p]
                end
                out[off + p_off_t + mn] += val
            end
        end
    end
end

"""
    _fft_node_to_node!(gnode, anode, fft_data, aft, gft, ifft_buf)

Perform FFT-based node-to-node translation (convolution in frequency domain).
Work buffers `aft`, `gft`, `ifft_buf` should be pre-allocated for reuse across iterations.
"""
function _fft_node_to_node!(
    gnode::Array{ComplexF64, 5},
    anode::Array{ComplexF64, 5},
    fft_data::FFTGridData,
    aft::Array{ComplexF64, 3},
    gft::Array{ComplexF64, 4},
    ifft_buf::Array{ComplexF64, 3}
)
    nx, ny, nz = fft_data.cell_dim
    nblk = fft_data.nblk_node

    fill!(gnode, zero(ComplexF64))

    for p in 1:2
        tran = p == 1 ? fft_data.cell_tran_fft_p1 : fft_data.cell_tran_fft_p2
        fill!(gft, zero(ComplexF64))

        for n in 1:nblk
            # Zero-pad input: copy cell_dim block into 2× grid
            fill!(aft, zero(ComplexF64))
            @inbounds aft[1:nx, 1:ny, 1:nz] .= anode[1:nx, 1:ny, 1:nz, n, p]

            # Forward FFT
            fft_data.fft_plan * aft

            # Multiply by pre-FFT'd translation matrix and accumulate
            @inbounds for l in 1:nblk
                tran_ln = @view tran[:, :, :, l, n]
                gft_l = @view gft[:, :, :, l]
                @. gft_l += tran_ln * aft
            end
        end

        # Inverse FFT for each target multipole (use temp buffer for FFTW compatibility)
        for l in 1:nblk
            @inbounds ifft_buf .= @view gft[:, :, :, l]
            fft_data.ifft_plan * ifft_buf

            # Extract first cell_dim block (discard zero-padding)
            @inbounds gnode[1:nx, 1:ny, 1:nz, l, p] .+= ifft_buf[1:nx, 1:ny, 1:nz]
        end
    end
end

"""
    _nearfield_direct!(out, inp, positions, fft_data, offsets, half_nblks, nois)

Direct sphere-to-sphere Hankel translation for neighbor pairs excluded from FFT.
"""
function _nearfield_direct!(
    out::Vector{ComplexF64},
    inp::Vector{ComplexF64},
    positions::Matrix{Float64},
    fft_data::FFTGridData,
    offsets::Vector{Int},
    half_nblks::Vector{Int},
    nois::Vector{Int}
)
    N = length(offsets)
    nodrmax     = maximum(nois)
    wmax_global = 2 * nodrmax

    # Work buffers
    ephim_buf = Vector{ComplexF64}(undef, 2*wmax_global + 1)
    fn_buf    = Vector{ComplexF64}(undef, wmax_global + 1)
    ymn_buf   = Matrix{Float64}(undef, 2*wmax_global + 1, wmax_global + 1)
    fywt_buf  = Matrix{ComplexF64}(undef, 2*wmax_global + 1, wmax_global + 1)
    r_ij      = Vector{Float64}(undef, 3)

    for i in 1:N
        off_i = offsets[i]
        hnb_i = half_nblks[i]
        noi_i = nois[i]

        for j in fft_data.neighbor_pairs[i]
            off_j = offsets[j]
            hnb_j = half_nblks[j]
            noi_j = nois[j]

            r_ij[1] = positions[1, i] - positions[1, j]
            r_ij[2] = positions[2, i] - positions[2, j]
            r_ij[3] = positions[3, i] - positions[3, j]

            apply_translation_mvp!(
                out, inp, off_i, hnb_i, off_j, hnb_j,
                r_ij, noi_j, noi_i,
                ephim_buf, fn_buf, ymn_buf, fywt_buf
            )
        end
    end
end

# ─────────────────────────────────────────────────────────────
# Combined FFT translation operator (replaces _apply_A!)
# ─────────────────────────────────────────────────────────────

"""
    apply_A_fft!(out, inp, positions, fft_data, offsets, half_nblks, nois,
                 anode_buf, gnode_buf, fft_aft, fft_gft, fft_ifft_buf)

FFT-accelerated translation operator. Drop-in replacement for `_apply_A!`.
Work buffers `fft_aft`, `fft_gft`, `fft_ifft_buf` are pre-allocated for reuse.
"""
function apply_A_fft!(
    out::Vector{ComplexF64},
    inp::Vector{ComplexF64},
    positions::Matrix{Float64},
    fft_data::FFTGridData,
    offsets::Vector{Int},
    half_nblks::Vector{Int},
    nois::Vector{Int},
    anode_buf::Array{ComplexF64, 5},
    gnode_buf::Array{ComplexF64, 5},
    fft_aft::Array{ComplexF64, 3},
    fft_gft::Array{ComplexF64, 4},
    fft_ifft_buf::Array{ComplexF64, 3}
)
    # Step 1: Sphere → Node
    _sphere_to_node!(anode_buf, inp, fft_data, offsets, half_nblks, nois)

    # Step 2: FFT node-to-node convolution
    _fft_node_to_node!(gnode_buf, anode_buf, fft_data, fft_aft, fft_gft, fft_ifft_buf)

    # Step 3: Node → Sphere
    _node_to_sphere!(out, gnode_buf, fft_data, offsets, half_nblks, nois)

    # Step 4: Near-field direct correction
    _nearfield_direct!(out, inp, positions, fft_data, offsets, half_nblks, nois)
end
