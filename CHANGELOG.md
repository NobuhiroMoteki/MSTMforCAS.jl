# Changelog

Changes to MSTMforCAS.jl, newest first. Dates are year-month-day.

The first section lists the corrections that change numerical results. Read it if you
have results from an earlier version. The sections after it list all changes by version.

## Corrections that change results

| Fixed in | Affected versions | What was wrong | Results that change | What to do |
|---|---|---|---|---|
| 1.2.1 | 1.1.0 to 1.2.0 | On the GPU path the truncation order of a group came from the refractive indices that remained in the run, not from the whole grid. | Rows of a GPU sweep with `truncation_eps` that were computed after an interruption and a resume. They can have a lower truncation order than the rows before the interruption (in a test order 3 against 7, 0.8 % in `Q_ext`). Uninterrupted sweeps and sweeps without `truncation_eps` are not affected. | Check the column `truncation_order` of such files: within one (aggregate, medium) group it must have one value. Recompute the groups where it has not. |
| 1.2.1 | all before 1.2.1 | A sweep did not write its row buffer after the last thread had ended. A row that was recorded after the last write was lost. | No value changes. Rows can be missing at the end of a sweep, most likely on the GPU path (one row in 1 of 24 small test sweeps). | Compare the number of rows of a file with the number of jobs. Running the same sweep again with the same output file computes the missing rows. |
| 1.1.0 | 1.0.1, 0.4.1 to 0.4.5 | The GPU path applied the Mie T-matrix of the first sphere to every sphere in the iteration operator. | All outputs of the GPU path (`use_gpu = true`) for aggregates of unequal spheres. For monomer radii with a relative standard deviation of 0.1, the forward amplitudes of single aggregates were off by about 5 to 20 %. The CPU path and aggregates of equal spheres are not affected. | Recompute the GPU results for aggregates of unequal spheres. In the hybrid mode these are the groups with more monomers than `gpu_np_threshold` (default 500). |
| 1.1.0 | all before 1.1.0 | The backward amplitude of the OCBS signal had the factor `1/√2` of the printed Eq. (28) of Moteki and Adachi (2024). The corrected equation, given in the erratum to that paper, has `1/2`. | `S_bak` from `scripts/read_results_h5_for_cas.jl` and from the code of the Python technical note. The old value is `√2` times the correct one, with the same phase. | Divide old `S_bak` values by `√2`, or compute `S_bak = (-S11_bwd + S22_bwd) / 2` again. The HDF5 files store `S11_bwd` and `S22_bwd` and need no change. |
| 0.4.4 | all before 0.4.4 | The regular spherical Bessel function `j_n(z)` in the translation matrix was computed by an unstable upward recurrence. | Forward and backward amplitudes of aggregates of sub-wavelength spheres when the truncation order is well above the size parameter (for example order 8 or more at x ≈ 0.3). The values were too large by orders of magnitude. `Q_ext` and the solver were not affected. | Recompute such cases. Calculations with the automatic truncation order used low orders at small x. |
| 0.4.2 | all before 0.4.2 | The Riccati–Bessel function `ψ_n(x)` of the Mie coefficients was computed by an unstable upward recurrence for x < 1. | Mie coefficients of high order for sub-wavelength spheres (for example `a_7` at x ≈ 0.3), and the amplitudes computed with a truncation order of 7 or more for such spheres. | Recompute such cases. |

## 1.2.1 (2026-11-24)

No result of an uninterrupted sweep changes. One case of a resumed GPU sweep changes, and a
sweep can no longer lose rows at its end (see the table above). CUDA.jl is no longer installed
with the package.
The release is archived on Zenodo (DOI 10.5281/zenodo.24526613, published 2026-11-25).

### Changed

- CUDA.jl is a weak dependency (`[weakdeps]` in `Project.toml`). It was a regular dependency
  before, so every installation downloaded it. A CPU-only installation no longer needs it.
  To use the GPU solver, install CUDA.jl in the default environment
  (`julia -e 'using Pkg; Pkg.add("CUDA")'`). Do not add it with `--project=.`.
- The GPU extension works with CUDA.jl 5 and 6 (`[compat]` `CUDA = "5, 6"`, tested with 5.11.0
  and 6.5.1). It used a function that CUDA.jl 6 removed. The extension checks the version when
  it loads and, for another major version, issues a warning and leaves the GPU solver off.
- GPU path: the truncation order of an (aggregate, medium condition) group now comes from the
  whole refractive index grid. Before, it came from the refractive indices that remained in the
  run. A sweep with `truncation_eps` that was interrupted and resumed could therefore compute
  the remaining rows at a lower order than an uninterrupted sweep (in a test, order 3 against
  7 and a difference of 0.8 % in `Q_ext`). Rows of uninterrupted sweeps do not change.
- `gpu_float32 = true` without `use_gpu = true` now issues a warning. It had and has no effect.

### Fixed

- `run_parameter_sweep` could lose rows at the very end of a sweep. Rows wait in a buffer and are
  written at intervals, at the end of a group and when the last job is counted. When several
  threads finished at almost the same time, the thread that counted the last job could write
  the buffer before another thread had stored its row, and nothing wrote that row afterwards.
  The sweep now writes the buffer once more after all threads have ended. The values of the
  rows that were written are not affected. The loss was seen on the GPU path, where the results
  of a batch are recorded by several threads (1 of 24 small test sweeps lost one row). It can
  also occur in a multi-threaded CPU sweep. To check an existing file, compare its number of
  rows with the number of jobs. Running the same sweep again with the same output file computes
  the missing rows.

### Added

- `test/test_gpu_solver.jl` (run separately, needs a GPU): tests of the GPU path against the CPU
  path for an aggregate of 100 monomers with unequal radii, for a far order of the FFT grid,
  for Float32, and for the parameter sweep with the hybrid dispatch and a resumed run.
- `test/test_fortran_unequal_radii.jl`: regression test against the Fortran code for aggregates
  with unequal monomer radii (20 and 100 monomers, two refractive indices, direct translation,
  tolerance 1e-12), with the fixtures `test/fixtures/aggregate_unequal_radii_Np*.ptsa`. The
  three test cases of the Fortran distribution have equal spheres only.

### Documents

- Theory note (`docs/theory_note.pdf`) revised on 2026-11-24.
  - New parts: a table of notation, the derivation of the T-matrix block with its eigenvalues
    and inverse, the basis of the affine form behind the shared-basis sweep, the algorithm of
    the shared-basis sweep as a table, and a section on the validation against the Fortran code.
  - The shared-basis sweep is now a section of its own. The incident coefficients moved to the
    section on the interaction equation. The flow summary and the implementation mapping are
    one section.
  - The section on the translation matrix was rewritten. The package uses the one-step form of
    the addition theorem. The earlier text named the rotation, axial translation, rotation
    factorisation, which the package does not use. Both forms give the same matrix.
  - Corrections of the text: mode 1 is the TM mode and mode 2 the TE mode (the earlier text had
    them reversed), the sign in the definition of the Riccati–Bessel function of the second
    kind, the recurrence used for the first kind (Miller's downward recurrence since 0.4.2),
    the convergence test of CBICG (Euclidean norm), and the references (Mackowski & Mishchenko
    1996, Mackowski & Kolokolova 2022).
- Validation recorded in the theory note. With direct translation the amplitudes and `Q_ext`
  agree with the Fortran output to its printed precision, for the three test cases and for
  aggregates with unequal monomer radii (20 to 504 monomers, relative standard deviation of the
  radii 0.08 to 0.10). Two points to note for users:
  - `Q_abs` is computed as `Q_ext - Q_sca` and carries the absolute error of `Q_ext`. For weakly
    absorbing aggregates the default tolerance leaves an error of up to a few times 1e-3 in
    `Q_abs`, and the FFT translation an error of about 1e-2. Use a smaller tolerance and direct
    translation when `Q_abs` of such aggregates is needed to better than a percent.
  - For an aggregate of 20 monomers the FFT translation differs from direct translation by up to
    1.7e-3 in the forward and 6.7e-3 in the backward amplitudes.
- A comment in `src/TMatrixSolver.jl` named the modes of the Mie coefficients the wrong way
  round. Only the comment changed.
- README brought in line with the theory note: contents of the note by section, validation
  (test cases, unequal radii, notes on accuracy), references. The output table and the docstring
  of `ScatteringResult` said that `Q_sca` is `Q_ext - Q_abs`. The package computes `Q_sca` from
  the expansion about the common origin and `Q_abs` as `Q_ext - Q_sca`. Only the descriptions
  changed.

## 1.2.0 (2026-11-23)

No result of an existing option changes. The new method is off by default.

### Added

- `sweep_refractive_index`: all refractive indices of one aggregate at one wavelength with one
  Krylov basis that they share. The translation operator is applied once per basis vector, not
  once per iteration and refractive index. For about 200 refractive indices of aggregates with
  1500 to 2800 monomers this was 92 to 146 times faster than one CBICG solve per refractive
  index, on one thread. Every result meets the convergence criterion of the CBICG solver: the
  true residual is checked, and refractive indices that the basis does not resolve are solved
  by CBICG from the basis solution.
- `SweepConfig.shared_basis` (default `false`) and `--shared-basis` in
  `scripts/run_sweep_h5.jl` and `scripts/run_sweep_h5_regen.jl`: the parameter sweep uses the
  shared basis for each (aggregate, medium condition) group. All spheres and refractive
  indices of a group then use one truncation order. Not available together with `use_gpu`.
  The output file records the option in the attribute `shared_basis`, and rows of the two
  methods cannot share a file.
- `shared_basis_range_issues` and a warning when a sweep with the shared basis lies outside
  the validated range (1.30 ≤ Re m ≤ 3.00, 0 ≤ Im m ≤ 1.65, max|m| × largest monomer size
  parameter ≤ 1.6). Metallic refractive indices are outside the scope of the method.
- `scripts/run_sweep_h5_regen.jl`: options `--truncation-eps` and `--fft-far-order`.
- `test/test_shared_basis.jl`, `test/test_tranordertest.jl`.

### Changed

- `_tranordertest` (translation order to the common origin) takes its coefficient tables from
  a cache. It built them again on every call before. The return values are identical: the
  test compares 1298 inputs with the earlier code. The post-processing of every solve is
  faster, by about 0.9 s per refractive index for an aggregate of 1500 monomers.

### Documents

- Theory note: new subsection on the shared Krylov basis with the measured accuracy, times
  and limits. Contact orders for several ranges of the refractive index, including the
  finding that metal spheres in point contact do not converge below order 30. A paragraph on
  the error that the default tolerance of CBICG leaves in the outputs.
- README: usage of `sweep_refractive_index` and `shared_basis`, and the measured times.

## 1.1.1 (2026-11-20)

Same code as 1.1.0. The documents are brought up to date.
The release is archived on Zenodo (DOI 10.5281/zenodo.24526547, published 2026-11-25).
The DOI of all archived versions is 10.5281/zenodo.20690115.

- Theory note (`docs/theory_note.pdf`): new sections on the truncation order for touching
  spheres, the far order of the FFT grid, and the cluster T-matrix with the Wigner-D
  rotation. The implementation table names functions that exist.
- README: the note on particle orientation describes the cluster T-matrix path. The project
  structure lists all source files, scripts and documents. The throughput table has the
  rates measured with 1.1.0.
- Python technical note: the attributes that the sweep writes.
- The Markdown version of the theory note (`docs/theory_note.md`, state of 0.3.0) is
  removed. The PDF is the maintained version.
- This changelog is added.

## 1.1.0 (2026-11-20)

### Fixed

- GPU solver: the iteration operator uses the T-matrix of each sphere. See the first row of
  the table above. `test/test_gpu_solver.jl` has a test with unequal radii.
- `S_bak` follows the corrected Eq. (28) of Moteki and Adachi (2024). See the second row of
  the table above.

### Added

- `truncation_eps`: truncation order from a preliminary two-sphere calculation
  (`contact_truncation_order`, `contact_order_scan`). The single-sphere criterion does not
  resolve the near field between monomers that touch. Default off.
- `fft_far_order`, `fft_near_factor`: the FFT grid carries the orders up to a far order.
  Higher orders couple near pairs only, by direct translation. Default off. The default
  operator is identical to the earlier one.
- `scripts/run_sweep_h5_regen.jl`: regeneration of the black-carbon LUT sweep with the
  refractive index grids `core`, `ext` and `ext2`.
- `scripts/merge_sweep_h5.jl`: merge of sweep files row by row, with `--core` and
  `--df-min`, `--df-max`.

### Added since 0.4.5 without a version of their own

- Wigner d and D matrices (`wigner_d`, `wigner_D`) and the cluster T-matrix
  (`build_cluster_tmatrix`, `cluster_forward_amplitudes`): forward amplitudes at any
  particle orientation from one T-matrix. Direct translation only (2026-07-05).
- `scripts/run_sampled_lut.jl`: table of forward amplitudes sampled over orientation
  (2026-07-05).
- `scripts/run_sweep_h5_extension.jl`, `scripts/merge_extension_h5.jl`: sweep of an extended
  refractive index range and its merge with the base file (2026-05-05).
- `docs/installation_guide.md` (2026-04-26) and
  `docs/design_wigner_d_multiorientation.md` (2026-07-05).

### Version numbers

The release 1.0.1 was made on 2026-03-28. The versions 0.4.1 to 0.4.5 came after it but
carry smaller numbers. Version 1.1.0 supersedes all of them. The number 0.5.0 was used for
a few commits on 2026-11-20 and was replaced by 1.1.0.

## 0.4.5 (2026-04-19)

- Documents only: the technical note is renamed to the theory note, the README shows the
  version, internal references are removed.

## 0.4.4 (2026-04-18)

### Fixed

- `j_n(z)` in `compute_translation_matrix` is computed by Miller's downward recurrence. See
  the third row of the table above.

## 0.4.3 (2026-04-18)

### Fixed

- `solve_tmatrix` requested too few Mie coefficients when `truncation_order` exceeded the
  single-sphere order. The call then stopped with a `BoundsError`, for example at order 9 or
  more for x ≈ 0.3. No wrong results were returned.

## 0.4.2 (2026-04-18)

### Fixed

- `ψ_n(x)` is computed by Miller's downward recurrence. See the fourth row of the table
  above.

## 0.4.1 (2026-03-28 to 2026-04-09)

### Added

- Snake-path traversal of the refractive index grid, for the continuation method.
- GPU batched BiCG solver as a CUDA.jl package extension (`use_gpu`), with a Float32 option
  (`gpu_float32`), incremental recording of the results, and a hybrid mode in which CPU
  threads compute the small aggregates (`gpu_np_threshold`).
- FFT translation is switched on automatically when an aggregate has more than 20 monomers.
- `scripts/run_doublet_sweep.jl`: two-sphere sweep over orientation.

### Fixed during the development of the GPU path

- Missing phase factor in the right-hand side and a wrong term in `Q_ext` (2026-03-28).
- Race conditions in the sphere-to-node and near-field kernels (2026-03-28).
- A rule that stopped a batch before all refractive indices had converged (2026-03-28).
- Memory use of the post-processing, which stalled large batches (2026-03-31).
- Float32 mode ran the FFT part in Float64. The batch size is limited to 250 refractive
  indices (2026-03-31).

The error of the GPU iteration operator for unequal spheres was present from the first GPU
commit and was corrected in 1.1.0.

## 1.0.1 (2026-03-28)

Tag of the release "paper-production version", archived on Zenodo
(DOI 10.5281/zenodo.20690116). It marks a state within the 0.4.1 series: the GPU batched
solver with the hybrid mode, before the later changes of that series. The package was numbered
1.0.1 at that commit and returned to 0.4.1 afterwards. Both corrections of 1.1.0 and the
fixes of 0.4.2 to 0.4.4 apply to this release.

## 0.4.0 (2026-03-28)

- Continuation method: the solution at the previous refractive index is the initial guess of
  the next one.

## 0.3.0 (2026-03-26)

- Faster solver: layout of the translation coefficients, precomputed T-matrix, batched FFT
  plans, cached near-field translation matrices, FFT setup shared within a geometry group.
- GMRES solver as an option (`solver = :gmres`, Krylov.jl).
- Dynamic thread scheduling with the large aggregates first. Progress display with the
  remaining time.

## 0.2.0 (2026-03-23)

- Results are written to HDF5 step by step, and an interrupted sweep resumes.
- Progress display. Lower memory use of the sweep.

## 0.1.1 (2026-03-16)

- Aggregate catalogs in HDF5 with CSV. Amplitudes in the MI02 convention and the
  volume-equivalent radius in the output.
- FFT-accelerated translation, validated against the Fortran reference (testcase3).
- Fixed on 2026-03-15: the translation order test used an unstable recurrence for `j_l(r)`
  and returned far too high orders at small distances, which exhausted the memory.

## 0.1.0 (2026-03-14)

- Initial project skeleton.
