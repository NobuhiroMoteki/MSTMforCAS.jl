# Installation Guide

Step-by-step setup of `MSTMforCAS.jl` from scratch on:

- **Windows 11** with **WSL2 + Ubuntu**
- **Linux** (native **Ubuntu** 22.04 / 24.04)

The runtime requirements and the Julia-side setup are identical on both
platforms; only the OS prerequisites differ. After completing your
platform's section, jump to [Common: Julia and project setup](#3-common-julia-and-project-setup).

---

## 1. Windows 11 — WSL2 + Ubuntu

### 1.1 Enable WSL2 and install Ubuntu

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu-24.04
```

This enables the WSL2 feature, downloads the Ubuntu 24.04 image, and
launches it. Reboot if Windows asks. On first launch, Ubuntu will
prompt you to create a UNIX username and password.

If WSL is already installed and you only need Ubuntu:

```powershell
wsl --list --online        # show available distros
wsl --install -d Ubuntu-24.04
wsl --set-default-version 2
```

Verify the distro is on WSL2:

```powershell
wsl -l -v
```

The `VERSION` column must read `2`.

### 1.2 Optional: configure WSL2 resource limits

Large parameter sweeps may need more memory than the default WSL2 cap
(50 % of host RAM). Create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=24GB
processors=12
swap=8GB
```

Then in PowerShell: `wsl --shutdown` to apply.

### 1.3 Optional: NVIDIA GPU passthrough

To use the GPU-accelerated solver from inside WSL2:

1. Install the latest **NVIDIA driver for Windows** (≥ 535) from
   <https://www.nvidia.com/Download/index.aspx>. Do **not** install a
   Linux-side NVIDIA driver inside WSL2 — Windows handles the driver,
   and WSL exposes the device through the `/usr/lib/wsl/lib/` shim.
2. From inside Ubuntu, verify GPU visibility:

   ```bash
   nvidia-smi
   ```

   The header should report a CUDA Version. CUDA toolkit itself is
   **not** required system-wide: `CUDA.jl` will fetch a matching
   toolkit at first use.

### 1.4 Update Ubuntu and install build tools

Inside the Ubuntu shell:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl ca-certificates
```

`build-essential` provides `gcc`/`make`, needed by some Julia binary
package fallbacks.

Continue to [§3 Common: Julia and project setup](#3-common-julia-and-project-setup).

---

## 2. Linux (native Ubuntu)

Tested on Ubuntu 22.04 LTS and 24.04 LTS.

### 2.1 Update and install build tools

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl ca-certificates
```

### 2.2 Optional: NVIDIA GPU driver

```bash
ubuntu-drivers devices             # find the recommended driver
sudo ubuntu-drivers autoinstall    # install it
sudo reboot
```

After reboot:

```bash
nvidia-smi
```

A CUDA Version line confirms the driver is loaded. As with WSL2, the
CUDA toolkit itself is fetched by `CUDA.jl` automatically; no
system-wide CUDA install is needed.

Continue to [§3 Common: Julia and project setup](#3-common-julia-and-project-setup).

---

## 3. Common: Julia and project setup

### 3.1 Install Julia via juliaup

`juliaup` is the official Julia version manager. It is the recommended
way to install Julia on both WSL2 Ubuntu and native Ubuntu:

```bash
curl -fsSL https://install.julialang.org | sh
```

Accept the default install location (`~/.juliaup`). After install,
restart your shell or:

```bash
source ~/.bashrc
```

Verify:

```bash
juliaup --version
julia --version          # should report 1.10 or newer
```

`MSTMforCAS.jl` requires Julia ≥ 1.10 (see `Project.toml`). If a
specific version is needed:

```bash
juliaup add 1.11
juliaup default 1.11
```

### 3.2 Clone the repository

Choose a workspace directory (e.g. `~/Julia/`):

```bash
mkdir -p ~/Julia && cd ~/Julia
git clone https://github.com/NobuhiroMoteki/MSTMforCAS.jl.git
cd MSTMforCAS.jl
```

### 3.3 Instantiate the project environment

This installs every dependency at the exact version pinned in
`Manifest.toml`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The first run downloads BinaryBuilder artifacts (HDF5, FFTW, OpenBLAS,
…) — expect 1–3 minutes on a fresh machine.

### 3.4 Run the test suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

All tests should pass. If a test fails, see [§5 Troubleshooting](#5-troubleshooting).

### 3.5 Validate against reference test cases (optional)

The detailed validation workflow against the Fortran reference output
(testcase1: 2-sphere dimer; testcase2: 1000-sphere aggregate) is
described in `CLAUDE.md`. You do not need to run these to use the
package — `Pkg.test()` already exercises the core code paths.

---

## 4. Optional: GPU verification

If you set up a GPU in §1.3 or §2.2, confirm Julia can reach it:

```bash
julia --project=. -e '
  using CUDA
  println("functional: ", CUDA.functional())
  println("device:     ", CUDA.name(CUDA.device()))
  println("memory:     ", round(CUDA.totalmem(CUDA.device())/2^30, digits=1), " GiB")
'
```

The first invocation triggers `CUDA.jl` to download a matching CUDA
toolkit (~2 GB). Subsequent runs are instantaneous.

To run a sweep on the GPU, pass `use_gpu=true` (see the README "Usage"
section). The GPU path is auto-disabled when `CUDA.functional()` is
`false`, so the same script works on CPU-only machines.

---

## 5. Troubleshooting

### `Pkg.instantiate()` fails behind a proxy

Set `JULIA_PKG_SERVER` and `https_proxy`:

```bash
export https_proxy=http://your.proxy:port
export JULIA_PKG_SERVER=https://pkg.julialang.org
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### HDF5 / FFTW shared-library errors

The pinned `Manifest.toml` ships matching JLL artifacts, so no system
HDF5/FFTW install is required. If you previously set
`JULIA_HDF5_PATH` or `JULIA_FFTW_PROVIDER` to use a system library,
unset them:

```bash
unset JULIA_HDF5_PATH
unset JULIA_FFTW_PROVIDER
julia --project=. -e 'using Pkg; Pkg.build()'
```

### `nvidia-smi` works but `CUDA.functional()` returns `false`

- Confirm the WSL distro is on WSL2 (not WSL1): `wsl -l -v` from
  PowerShell.
- Ensure no Linux-side NVIDIA driver is installed inside WSL2
  (Windows-only driver is correct).
- Update `CUDA.jl`: `julia --project=. -e 'using Pkg; Pkg.update("CUDA")'`.
- Check the artifact toolkit versus driver version:

  ```bash
  julia --project=. -e 'using CUDA; CUDA.versioninfo()'
  ```

### Out-of-memory during large sweeps

- Increase WSL2 memory allocation (§1.2).
- Reduce the number of Julia threads: `julia -t 4 --project=. ...`.
- Enable `use_fft=true` for aggregates with `Np > 20` (auto in recent
  versions).

### Building from a fork / development install

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path=".")'
```

Then edits in `src/` are picked up immediately on the next `using
MSTMforCAS`.

---

## Reference: minimum versions

| Component | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04 (native or WSL2) | Other Linux distros work but are not tested. |
| Julia | 1.10 | Pinned in `Project.toml`. 1.11 is recommended. |
| NVIDIA driver | 535 | Required only for GPU. |
| RAM | 8 GiB | 16+ GiB recommended for sweeps with `Np > 1000`. |
| Disk | 5 GiB | Includes Julia + artifacts + repo. |
