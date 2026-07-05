# Design: Cheap multi-orientation CAS-v2 via cluster T-matrix + Wigner-D rotation

**Status:** design pass (no code yet). Target: MSTMforCAS.jl.
**Author:** prepared with the assistance of Claude (Anthropic); the author assumes full responsibility.

## 1. Motivation and current state

CAS-v2 simulation-based inference for aggregates (black carbon) needs the forward
amplitudes `S_s(0°), S_p(0°)` at **many particle orientations** sampled from a
shear-flow orientation distribution `p(Ω|κ)` (orientation marginalization). Today
MSTMforCAS.jl solves a **fixed +z incidence** only (2 RHS), forms **no cluster
T-matrix**, and has **no rotation machinery** — so each orientation means rotating the
geometry and re-solving from scratch (`scripts/run_doublet_sweep.jl` pattern). That is
prohibitively expensive for the ≳10²–10⁴ orientations marginalization needs.

**Goal.** Build the aggregate **cluster T-matrix once** per (geometry, `m_rel`, size), then
evaluate every orientation by a cheap **Wigner-D rotation** + the existing forward-amplitude
extraction. This regains an efficient multi-orientation capability (analogous to the
block-Krylov orientation amortization of block-DDA_Py / block-VIEM.jl, but via a different
mechanism).

## 2. Key physics that makes this cheap

- **Forward-only survival.** At θ=0 only the `m=±1` common-origin coefficients contribute
  to `S(0)` (theory Eq. 24). Rotation couples all `m`, so the *full* cluster T-matrix is
  still required — but the final amplitude read only touches `m=±1` rows.
- **lr_tran rotation is block-diagonal.** In the left/right-circular basis a rigid rotation
  acts **identically on the p=1 and p=2 half-blocks** and is **block-diagonal in n** (couples
  only `m` within a fixed `n`). So the rotation operator is a set of small `(2n+1)×(2n+1)`
  Wigner-`d^n(β)` blocks plus two phase diagonals — an `O(Σ_n (2n+1)²) = O(d^{3/2})`-ish
  cheap apply, not a dense `d×d` multiply.

## 3. What already exists vs. what must be added

| Piece | Status | Location |
|---|---|---|
| Single-sphere lr_tran T (Eq. 10) | EXISTS | `src/TMatrixSolver.jl` `_apply_T!` |
| Matrix-free translation `A` (direct + FFT), S-parity adjoint | EXISTS | `TMatrixSolver.jl` `_apply_A!`, `FFTTranslation.jl` |
| Reusable operator `L=(I−T·A)`, CBICG/GMRES | EXISTS | `TMatrixSolver.jl` `apply_L!`, `_solve_bicg`, Krylov.jl |
| Regular (Bessel `J`) translation kernel | EXISTS | `TranslationCoefs.jl` `compute_translation_matrix(...; use_regular=true)` |
| Common-origin re-expansion `a_0=Σ J_{0i}a^{(i)}` (Eq. 21–22) | EXISTS | `ScatteringAmplitude.jl` `_merge_to_origin!` |
| Common-origin order `nodrt` (`tranordertest`) | EXISTS | `ScatteringAmplitude.jl` `_tranordertest` |
| Forward amplitude from common-origin coeffs (Eq. 24) → S₁–S₄ → MI02 → CAS | EXISTS | `ScatteringAmplitude.jl` `_amplitude_from_mode_coefs`, `ParameterSweep.jl` |
| **General ZYZ Wigner-D operator** `D^n_{m'm}(α,β,γ)` + block apply | **ADD** | new, e.g. `src/WignerRotation.jl` |
| **Arbitrary-origin-mode incident RHS** (unit VSWF at origin → `J_{i,0}` to spheres) | **ADD** | new RHS generator (reuses regular translation) |
| **Block / multi-RHS solve** to build the T-matrix efficiently | **ADD** | generalize solver (FFT buffers hardwired to width 2) |
| Cluster T-matrix assembly driver | **ADD** | new |

Only **two genuinely new numerical components**: (A) a general ZYZ Wigner-D kernel, and
(B) an arbitrary-mode RHS + block solve to assemble the cluster T-matrix. The regular
translation, common-origin merge, order selection, and amplitude extraction are reused
unchanged — and all of them already consume the common-origin coefficient vector, so a
**rotated** coefficient vector plugs straight in.

## 4. Cluster T-matrix construction

The cluster T-matrix `T^(cl)` maps regular VSWF incident coefficients about the aggregate
common origin to outgoing VSWF scattered coefficients about the same origin; size
`d = 2·nodrt(nodrt+2)` (both polarization half-blocks). Column `(k,l,p)` is obtained by:

1. Set a unit regular VSWF coefficient `e_{klp}` at the common origin.
2. Distribute to each sphere as a **regular** incident field `p_inc^{(i)} = J_{i,0} e_{klp}`
   (origin→sphere Bessel translation — existing kernel).
3. Solve `(I−T·A) x = T·p_inc` with the existing reusable operator.
4. Merge the solution to the common origin, `a_0 = Σ_i J_{0i} x^{(i)}` (existing
   `_merge_to_origin!`) → that merged vector **is** column `(k,l,p)` of `T^(cl)`.

This is `d` right-hand sides sharing **one** operator `L`. Recommended: a **block-Krylov**
solve (block-CBICG or block-GMRES) that shares a single Krylov subspace across all columns —
the same amortization strategy already used in block-DDA_Py / block-VIEM.jl. Engineering
note: the FFT work buffers (`anode_buf/gnode_buf`) are currently hardwired to width 2
(`TMatrixSolver.jl:497-500`) and must be generalized to block width B.

`T^(cl)` depends on (geometry, `m_rel`, size) but **not** orientation → it is a natural
**cacheable intermediate** (content-addressed on `geometry_id + m_rel + size`, per the
`pcas_lut_schema` cache; optional manifest kind `cluster_tmatrix`).

## 5. Per-orientation evaluation — recommended strategy R2

Fix the lab incidence along +z (as the code does) and place the particle at orientation Ω.

- **R1 (rotate the T-matrix):** `T(Ω) = D(Ω) T^(cl) D(Ω)^†`, then
  `a_0(Ω) = T(Ω) p^{inc}_{0,z}`. Cost per orientation: two dense `d×d` multiplies, `O(d³)`.
  Yields the full rotated T (useful only if other observables are ever needed).
- **R2 (rotate the coefficient vectors) — RECOMMENDED:** compute
  `a_0(Ω) = D(Ω)·[ T^(cl) · ( D(Ω)^† p^{inc}_{0,z} ) ]`. Steps per orientation:
  (i) rotate the incident vector `p̃ = D(Ω)^† p^{inc}_{0,z}` (block Wigner-D apply),
  (ii) one dense mat-vec `T^(cl)·p̃` (`O(d²)`),
  (iii) rotate back, computing **only the `m=±1` rows** of `a_0(Ω)` (all that the forward
  amplitude needs). Cost `O(d²)` per orientation, no `O(d³)`.

R2 also covers the backward OCBS amplitude `S_bak` (also `m=±1` only). For `d~500`,
`O(d²)~10⁵` complex flops per orientation → microseconds; 10⁴ orientations is trivial.
**Choose R2.** R1 remains available if a full rotated T is ever required.

## 6. Euler-convention alignment (critical, contract-anchored)

The rotation must match the shared `pcas_lut_schema` orientation anchor: **intrinsic ZYZ,
active particle rotation, scipy `Rotation.from_euler('ZYZ',[α,β,γ])`** (already implemented
by block-DDA_Py and block-VIEM.jl). The VSWF rotation operator is
`D^n_{m'm}(α,β,γ) = e^{-i m' α} d^n_{m'm}(β) e^{-i m γ}` (Mishchenko/Mackowski convention);
the mapping of its (active/passive, sign) conventions to the contract ZYZ **must be pinned
by validation**, not assumed. The existing translation code carries only the `m'=0` column
of `d^n` (normalized Legendre, `TranslationCoefs.jl:316-366`); the **full** `d^n_{m'm}(β)`
recurrence is new. Reuse the existing numeric-constant infrastructure (`_ensure_numconstants`,
`fnr`/`bcof`).

## 7. Validation ladder

1. **Identity:** at Ω=0 the T-matrix route reproduces the existing direct z-incidence
   `S_s(0),S_p(0)` to solver tolerance.
2. **Rotation operator:** `D(0)=I`, unitarity `D(Ω)D(Ω)^†=I`, group law
   `D(Ω₁)D(Ω₂)=D(Ω₁∘Ω₂)`; small-`n` blocks vs. an independent Wigner-d (e.g. scipy/`sympy`).
3. **Rotate-and-resolve:** for several Ω, rotate the geometry and re-solve
   (`run_doublet_sweep.jl` pattern) vs. cluster-T + Wigner-D → agree to solver tolerance.
4. **Cross-solver (convention detector):** a **non-sintered** aggregate,
   MSTM(Wigner-D per orientation) vs. **block-VIEM.jl** (multi-orientation) → agree; any ZYZ
   mismatch shows up here.
5. **Orientation average:** random-orientation average of `Q_ext/Q_sca` vs. known
   orientation-averaged values.

## 8. Cost model

- **Build `T^(cl)`:** `d = 2·nodrt(nodrt+2)` RHS on one shared operator. For BC aggregates
  (Np~100–300, per-monomer `noi∈{1..4}`), `nodrt~10–30` → `d~250–1900`,
  `T^(cl)` size `d²` complex ≈ 4–30 MB. Block-Krylov makes the build ≈ a small multiple of a
  single solve (operator apply amortized across the block).
- **Per orientation (R2):** one `O(d²)` mat-vec + block Wigner-D apply → microseconds.
- **Break-even:** the cluster-T route wins once the orientation count exceeds roughly
  (RHS-to-build)/(block-speedup × iters-per-solve); for the 10²–10⁴ orientations CAS
  marginalization needs, it wins decisively. `T^(cl)` is cached and reused across orientation
  grids (wavelengths handled as separate `T^(cl)`).
- **Consequence for the inference design:** with orientation now cheap, the aggregate
  orientation integral `∫ p(Ω|κ) dΩ` can be done by **dense quadrature** (like spheroids);
  the curse of dimensionality for aggregates is confined to the **shape** parameters
  (`Df, N_mon, R_mon, sintering`), which remain the NLE/emulator target.

## 9. Contract integration

- The MSTM **sampled-table** output gains explicit orientation columns
  (`euler_alpha/beta/gamma`) — this completes the `pcas_lut_schema` "MSTM orientation column"
  item. Rows sharing one geometry/`T^(cl)` share a `geometry_id` (cheap many-orientation
  nesting).
- `T^(cl)` is an optional cacheable artifact (`kind: cluster_tmatrix`, keyed on
  `geometry_id + m_rel + size`).

## 10. Phased implementation plan

- **P1 — Wigner-D kernel (standalone).** General `d^n_{m'm}(β)` recurrence + block-diagonal
  `D^n(α,β,γ)` apply on an lr_tran coefficient vector. Unit tests: identity, unitarity, group
  law, small-`n` vs. reference. No solver changes. *(Lowest risk; do first.)*
- **P2 — Cluster T-matrix (single-RHS).** Arbitrary-origin-mode RHS generator (reuse regular
  translation) + assembly by looping RHS (no block yet) + validation steps 1 & 3.
- **P3 — Block/multi-RHS solve.** Generalize FFT buffers (width 2 → B) and add block-CBICG or
  block-GMRES for T-matrix build performance.
- **P4 — Multi-orientation CAS output + caching.** Wire orientation columns into the sampled
  table, cross-solver validation (step 4) vs. block-VIEM, and `T^(cl)` caching.

## 11. Risks / open questions

- **Block-Krylov generalization** of the width-2-hardwired FFT buffers is the main
  engineering risk.
- **`nodrt` growth** for lacunar (Df≈1.8) clusters inflates `d` and the `T^(cl)` build cost;
  monitor and cap via `truncation_order` if needed.
- **Convention mapping** VSWF-`D` ↔ scipy ZYZ must be locked by validation step 4.
- **Dependencies:** the Wigner-d recurrence is implemented from scratch reusing existing
  numeric-constant infrastructure — expected **no new external dependency**. To be confirmed.
