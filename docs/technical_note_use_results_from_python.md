# Using MSTMforCAS results from Python

MSTMforCAS outputs HDF5 files (`.h5`) that are language-agnostic. To use the results in a Python project, copy the `.h5` file and read it with `h5py`. No Julia dependency is needed.

## Requirements

```
pip install h5py numpy
```

## HDF5 file structure

Each column of the sweep results is stored as a 1-D dataset (length = number of jobs). Root-level attributes store sweep configuration metadata.

### Datasets (per-row arrays)

| Dataset | dtype | Description |
|---------|-------|-------------|
| `source` | string | Aggregate source identifier (h5_key or file path) |
| `wavelength` | float64 | Vacuum wavelength [same unit as coordinates] |
| `medium_refindex` | float64 | Medium refractive index |
| `agg_num` | int | Aggregate index |
| `n_monomers` | int | Number of monomers |
| `Df` | float64 | Fractal dimension |
| `mean_rp` | float64 | Mean monomer radius |
| `rel_std_rp` | float64 | Relative std dev of monomer radius |
| `k_f` | float64 | Fractal prefactor |
| `R_ve` | float64 | Volume-equivalent sphere radius |
| `R_g` | float64 | Radius of gyration |
| `eps_agg` | float64 | Aggregate porosity |
| `m_real` | float64 | Real part of absolute refractive index |
| `m_imag` | float64 | Imaginary part of absolute refractive index |
| `S{1,2,3,4}_fwd_re`, `S{1,2,3,4}_fwd_im` | float64 | BH83 forward amplitudes (dimensionless) |
| `S{1,2,3,4}_bwd_re`, `S{1,2,3,4}_bwd_im` | float64 | BH83 backward amplitudes (dimensionless) |
| `S{11,22,12,21}_fwd_re`, `S{11,22,12,21}_fwd_im` | float64 | MI02 forward amplitudes [length] |
| `S{11,22,12,21}_bwd_re`, `S{11,22,12,21}_bwd_im` | float64 | MI02 backward amplitudes [length] |
| `Q_ext`, `Q_abs`, `Q_sca` | float64 | Efficiency factors (dimensionless) |
| `converged` | bool | Solver convergence flag |
| `n_iterations` | int | Number of solver iterations |
| `truncation_order` | int | Maximum VSWF truncation order used |

### Root attributes (sweep config)

| Attribute | dtype | Description |
|-----------|-------|-------------|
| `medium_conditions_wavelength` | float64[] | Wavelength values for each medium condition |
| `medium_conditions_refindex` | float64[] | Medium refindex values for each medium condition |
| `m_real_range` | float64[3] | (min, max, n_grid) for Re(m) |
| `m_imag_range` | float64[3] | (min, max, n_grid) for Im(m) |
| `n_jobs` | int | Total number of result rows |

## CAS-v2 observable quantities

The CAS-v2 directly observed quantities are derived from the MI02 scattering amplitude matrix elements:

| Observable | Formula | Description |
|------------|---------|-------------|
| `Ss_fwd` | `S11_fwd + S12_fwd * 1j` | s-polarization forward complex scattering amplitude |
| `Sp_fwd` | `S22_fwd - S21_fwd * 1j` | p-polarization forward complex scattering amplitude |
| `S_bak` | `(-S11_bwd + S22_bwd) / sqrt(2)` | Backward scattering amplitude (depolarization-sensitive) |

where `S11`, `S12`, `S21`, `S22` are complex MI02-convention amplitudes with dimension of length (same unit as wavelength).

## Python implementation

### Basic reading

```python
import h5py
import numpy as np

with h5py.File("results_fullsweep_agg20260316_00.h5", "r") as f:
    # All datasets are 1-D arrays of the same length
    wavelength = f["wavelength"][:]
    medium_refindex = f["medium_refindex"][:]
    m_real = f["m_real"][:]
    m_imag = f["m_imag"][:]
    Q_ext = f["Q_ext"][:]

    # Reconstruct complex MI02 amplitudes
    S11_fwd = f["S11_fwd_re"][:] + 1j * f["S11_fwd_im"][:]
    S22_fwd = f["S22_fwd_re"][:] + 1j * f["S22_fwd_im"][:]
    S12_fwd = f["S12_fwd_re"][:] + 1j * f["S12_fwd_im"][:]
    S21_fwd = f["S21_fwd_re"][:] + 1j * f["S21_fwd_im"][:]

    # CAS-v2 observables (vectorized over all rows)
    Ss_fwd = S11_fwd + S12_fwd * 1j
    Sp_fwd = S22_fwd - S21_fwd * 1j
```

### Parameter-indexed lookup (equivalent to `read_results_h5_for_cas.jl`)

```python
import h5py
import numpy as np


def read_cas_results(h5path, i_medium=0, i_agg=0, i_Np=0, i_Df=0,
                     i_m_real=0, i_m_imag=0):
    """
    Read CAS-v2 observables from MSTMforCAS HDF5 results.

    All indices are 0-based (Python convention).

    Parameters
    ----------
    h5path : str
        Path to HDF5 results file.
    i_medium : int
        Index into unique (wavelength, medium_refindex) pairs.
    i_agg : int
        Index into unique agg_num values.
    i_Np : int
        Index into unique n_monomers values.
    i_Df : int
        Index into unique Df values.
    i_m_real : int
        Index into unique m_real values.
    i_m_imag : int
        Index into unique m_imag values.

    Returns
    -------
    dict with keys: wavelength, medium_refindex, agg_num, n_monomers, Df,
        m_real, m_imag, R_ve, Q_ext, Q_abs, Q_sca, Ss_fwd, Sp_fwd, S_bak.
        Scalar values for the selected parameters; arrays for the result
        quantities (length = number of matching rows, typically 1).
    """
    with h5py.File(h5path, "r") as f:
        wl   = f["wavelength"][:]
        nmed = f["medium_refindex"][:]
        agg  = f["agg_num"][:]
        Np   = f["n_monomers"][:]
        Df   = f["Df"][:]
        mr   = f["m_real"][:]
        mi   = f["m_imag"][:]

        # Build unique sorted parameter values
        medium_pairs = sorted(set(zip(wl.tolist(), nmed.tolist())))
        u_agg  = np.unique(agg)
        u_Np   = np.unique(Np)
        u_Df   = np.unique(Df)
        u_mr   = np.unique(mr)
        u_mi   = np.unique(mi)

        # Print available values (like the Julia script)
        print(f"Available parameter values:")
        print(f"  medium   ({len(medium_pairs)}): {medium_pairs}")
        print(f"  agg_num  ({len(u_agg)}): {u_agg.tolist()}")
        print(f"  Np       ({len(u_Np)}): {u_Np.tolist()}")
        print(f"  Df       ({len(u_Df)}): {u_Df.tolist()}")
        print(f"  m_real   ({len(u_mr)}): {u_mr.tolist()}")
        print(f"  m_imag   ({len(u_mi)}): {u_mi.tolist()}")

        # Resolve indices to values
        v_wl, v_nmed = medium_pairs[i_medium]
        v_agg  = u_agg[i_agg]
        v_Np   = u_Np[i_Np]
        v_Df   = u_Df[i_Df]
        v_mr   = u_mr[i_m_real]
        v_mi   = u_mi[i_m_imag]

        # Find matching rows
        idx = np.where(
            (wl == v_wl) & (nmed == v_nmed) &
            (agg == v_agg) & (Np == v_Np) & (Df == v_Df) &
            (mr == v_mr) & (mi == v_mi)
        )[0]

        if len(idx) == 0:
            raise ValueError("No matching row found for the specified parameters.")

        # Read MI02 amplitudes
        S11_fwd = f["S11_fwd_re"][:] + 1j * f["S11_fwd_im"][:]
        S12_fwd = f["S12_fwd_re"][:] + 1j * f["S12_fwd_im"][:]
        S21_fwd = f["S21_fwd_re"][:] + 1j * f["S21_fwd_im"][:]
        S22_fwd = f["S22_fwd_re"][:] + 1j * f["S22_fwd_im"][:]
        S11_bwd = f["S11_bwd_re"][:] + 1j * f["S11_bwd_im"][:]
        S22_bwd = f["S22_bwd_re"][:] + 1j * f["S22_bwd_im"][:]

        # CAS-v2 observables
        Ss_fwd = S11_fwd + S12_fwd * 1j
        Sp_fwd = S22_fwd - S21_fwd * 1j
        S_bak  = (-S11_bwd + S22_bwd) / np.sqrt(2)

        return {
            "wavelength": v_wl,
            "medium_refindex": v_nmed,
            "agg_num": v_agg,
            "n_monomers": v_Np,
            "Df": v_Df,
            "m_real": v_mr,
            "m_imag": v_mi,
            "R_ve":   f["R_ve"][:][idx],
            "Q_ext":  f["Q_ext"][:][idx],
            "Q_abs":  f["Q_abs"][:][idx],
            "Q_sca":  f["Q_sca"][:][idx],
            "Ss_fwd": Ss_fwd[idx],
            "Sp_fwd": Sp_fwd[idx],
            "S_bak":  S_bak[idx],
        }


# Example usage
if __name__ == "__main__":
    result = read_cas_results(
        "results_fullsweep_agg20260316_00.h5",
        i_medium=0, i_agg=0, i_Np=1, i_Df=0, i_m_real=1, i_m_imag=1,
    )
    print(f"\nSelected: wl={result['wavelength']}, n_med={result['medium_refindex']}, "
          f"Np={result['n_monomers']}, Df={result['Df']}, "
          f"m={result['m_real']}+{result['m_imag']}i")
    print(f"  R_ve    = {result['R_ve']}")
    print(f"  Q_ext   = {result['Q_ext']}")
    print(f"  Q_abs   = {result['Q_abs']}")
    print(f"  Q_sca   = {result['Q_sca']}")
    print(f"  Ss_fwd  = {result['Ss_fwd']}")
    print(f"  Sp_fwd  = {result['Sp_fwd']}")
    print(f"  S_bak   = {result['S_bak']}")
```

### Bulk loading into pandas

```python
import h5py
import pandas as pd

with h5py.File("results_fullsweep_agg20260316_00.h5", "r") as f:
    # Load all scalar/real columns into a DataFrame
    columns = ["wavelength", "medium_refindex", "agg_num", "n_monomers",
               "Df", "m_real", "m_imag", "R_ve", "Q_ext", "Q_abs", "Q_sca",
               "converged", "n_iterations", "truncation_order"]
    df = pd.DataFrame({col: f[col][:] for col in columns})

    # Add complex MI02 amplitudes as separate columns if needed
    for name in ["S11_fwd", "S22_fwd", "S12_fwd", "S21_fwd"]:
        df[name] = f[f"{name}_re"][:] + 1j * f[f"{name}_im"][:]
```

## Index convention note

- **Julia** (`read_results_h5_for_cas.jl`): 1-based indices
- **Python** (`read_cas_results`): 0-based indices

The parameter listing output is identical; only the starting index differs.
