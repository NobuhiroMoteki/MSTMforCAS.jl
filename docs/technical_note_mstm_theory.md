---

# Multi-Sphere T-Matrix Theory: Mathematical Formulation

This document describes the physical problem, mathematical formulation, numerical solution methods, and output quantities implemented in MSTMforCAS.jl.  The implementation follows the multi-sphere superposition formulation developed by Mackowski (1991, 1994, 1996) and the addition theorem treatment by Xu (1996), with the Complex BiConjugate Gradient (CBICG) solver of Mackowski & Mishchenko (2011) and the FFT-accelerated translation algorithm of Mackowski & Kolokolova (2022).  Scattering amplitude conventions follow Bohren & Huffman (1983, hereafter BH83).

---

## 1. Problem Setup

### Geometry and incident field

Consider \(N_S\) homogeneous, non-overlapping spheres embedded in an infinite, lossless, homogeneous medium of real refractive index \(n_\mathrm{med}\).  Sphere \(i\) (\(i = 1, \dots, N_S\)) has centre position \(\mathbf{r}_i\), radius \(a_i\), and complex refractive index \(m_\mathrm{sphere}\) (common to all spheres in the parameter-sweep mode of this code).  A time-harmonic plane wave (\(e^{-i\omega t}\) convention, following BH83) propagates along the \(+z\) axis in the medium with vacuum wavelength \(\lambda_0\) and wavenumber in the medium

```math
k = \frac{2\pi \, n_\mathrm{med}}{\lambda_0}.
```

A positive imaginary part of \(m_\mathrm{sphere}\) corresponds to absorption.  All lengths are non-dimensionalised by \(k\): positions become \(k\mathbf{r}_i\), radii become size parameters \(x_i = k a_i\), and the relative refractive index entering all scattering formulas is \(m_\mathrm{rel} = m_\mathrm{sphere} / n_\mathrm{med}\).  Tildes are suppressed hereafter; all equations are in dimensionless units unless stated otherwise.

The volume-equivalent sphere radius \(a_\mathrm{eff}\) and its size parameter \(x_\mathrm{eff}\) are defined as

```math
a_\mathrm{eff} = \left(\sum_{i=1}^{N_S} a_i^3\right)^{\!1/3}, \qquad x_\mathrm{eff} = k\,a_\mathrm{eff}.
```

These appear in the normalisation of cross-section efficiencies (Section 5).

### Vector spherical wave function expansion

The electromagnetic field in each region is expanded in vector spherical wave functions (VSWFs) \(\mathbf{M}_{mn}^{(\nu)}\), \(\mathbf{N}_{mn}^{(\nu)}\) of degree \(m\) and order \(n\) (\(n = 1, 2, \dots\); \(m = -n, \dots, n\)), where \(\nu = 1\) denotes regular (Bessel \(j_n\)) waves and \(\nu = 3\) denotes outgoing (Hankel \(h_n^{(1)}\)) waves. The normalisation convention for these VSWFs follows that of Mishchenko et al. (2002) and Mackowski (1994). The scattered field from sphere \(i\) is expanded in outgoing VSWFs centred at \(\mathbf{r}_i\):

```math
\mathbf{E}_\mathrm{sca}^{(i)}(\mathbf{r}) = \sum_{n=1}^{N_i} \sum_{m=-n}^{n} \left[ a_{mn}^{(i)}\,\mathbf{M}_{mn}^{(3)}(\mathbf{r} - \mathbf{r}_i) + b_{mn}^{(i)}\,\mathbf{N}_{mn}^{(3)}(\mathbf{r} - \mathbf{r}_i) \right]
```

where \(N_i\) is the truncation order for sphere \(i\).  The field exciting sphere \(i\) — the incident plane wave plus the scattered fields from all other spheres — is likewise expanded in regular VSWFs \(\mathbf{M}_{mn}^{(1)}, \mathbf{N}_{mn}^{(1)}\) centred at \(\mathbf{r}_i\).

### Left–right circular (lr_tran) basis

Following Mackowski (1996, §2), the code works in the left–right circular VSWF basis obtained by the linear combinations

```math
\tilde{\mathbf{N}}_{mn,L}^{(\nu)} = \mathbf{N}_{mn,1}^{(\nu)} + \mathbf{N}_{mn,2}^{(\nu)}, \qquad \tilde{\mathbf{N}}_{mn,R}^{(\nu)} = \mathbf{N}_{mn,1}^{(\nu)} - \mathbf{N}_{mn,2}^{(\nu)}
```

where the subscripts 1, 2 refer to the TE and TM modes. The fundamental mathematical advantage of this basis is that it completely decouples the translation matrix \(\mathbf{H}_{ij}\) between the two polarisation blocks. While the single-sphere T-matrix is diagonal in the combined multipole index \(mn = n(n+1) + m\) in both the standard TE/TM and lr_tran bases, in the lr_tran basis it couples the two polarisation blocks (labeled \(p = 1\) for left and \(p = 2\) for right). This structure greatly simplifies the multi-sphere interaction equations, as the computationally expensive translation operations can be performed independently for each polarisation block.

---

## 2. Single-Sphere Scattering: Lorenz–Mie Theory

For a homogeneous sphere with size parameter \(x\) and relative refractive index \(m_\mathrm{rel}\), the Lorenz–Mie scattering coefficients \(a_n\) (TM/electric) and \(b_n\) (TE/magnetic) are given by (BH83, Eq. 4.88):

```math
a_n = \frac{\left[\dfrac{D_n(m_\mathrm{rel}\,x)}{m_\mathrm{rel}} + \dfrac{n}{x}\right] \psi_n(x) - \psi_{n-1}(x)}{\left[\dfrac{D_n(m_\mathrm{rel}\,x)}{m_\mathrm{rel}} + \dfrac{n}{x}\right] \xi_n(x) - \xi_{n-1}(x)}
```

```math
b_n = \frac{\left[m_\mathrm{rel}\,D_n(m_\mathrm{rel}\,x) + \dfrac{n}{x}\right] \psi_n(x) - \psi_{n-1}(x)}{\left[m_\mathrm{rel}\,D_n(m_\mathrm{rel}\,x) + \dfrac{n}{x}\right] \xi_n(x) - \xi_{n-1}(x)}
```

where:

* \(\psi_n(z) = z\,j_n(z)\) is the Riccati–Bessel function (\(j_n\): spherical Bessel function of the first kind)
* \(\xi_n(z) = z\,h_n^{(1)}(z) = \psi_n(z) + i\,\chi_n(z)\) is the Riccati–Hankel function
* \(\chi_n(z) = -z\,y_n(z)\) (\(y_n\): spherical Bessel function of the second kind)
* \(D_n(z) = \frac{d}{dz}\ln[z\,j_n(z)] = \frac{\psi_n'(z)}{\psi_n(z)}\) is the logarithmic derivative

The logarithmic derivative \(D_n(y)\) with \(y = m_\mathrm{rel}\,x\) is computed by downward recurrence (Wiscombe 1980):

```math
D_{n-1}(y) = \frac{n}{y} - \frac{1}{D_n(y) + n/y},
```

starting from a sufficiently large \(n_\mathrm{start} \geq \max(N_\mathrm{max}, |y|) + 16\) where \(D_{n_\mathrm{start}} \approx 0\).  This recurrence is unconditionally stable for any complex \(y\).  The Riccati–Bessel functions \(\psi_n(x)\) and \(\chi_n(x)\) (for real argument \(x\)) are computed by upward recurrence:

```math
f_n(x) = \frac{2n - 1}{x}\,f_{n-1}(x) - f_{n-2}(x)
```

with starting values \(\psi_0 = \sin x\), \(\psi_1 = \sin x / x - \cos x\), \(\chi_0 = -\cos x\), \(\chi_1 = -\cos x / x - \sin x\).

The multipole expansion is truncated at order \(N_i\) determined by the criterion of Wiscombe (1980) as implemented in the MSTM code:

```math
N_i = \mathrm{round}\!\left(x_i + 4\,x_i^{1/3}\right) + 5,
```

refined by a convergence check: starting from this upper bound, \(N_i\) is reduced until the cumulative partial extinction efficiency \(Q_\mathrm{ext}^{(N_i)}\) deviates from the total by more than \(10^{-6}\).

In the lr_tran basis, the single-sphere T-matrix for multipole order \(n\) takes the \(2 \times 2\) form coupling the \(p = 1\) and \(p = 2\) blocks (Mackowski 1996, derived from the mieoa routine in MSTM v4.0):

```math
\mathbf{T}_n = -\frac{1}{2}\begin{pmatrix} a_n + b_n & a_n - b_n \\ a_n - b_n & a_n + b_n \end{pmatrix}
```

This matrix is diagonal in the combined index \(mn\) (it does not mix different \((m,n)\) pairs) but couples the two circular polarisation blocks.  For an isotropic sphere, \(a_n \neq b_n\) in general, so the off-diagonal coupling is physically meaningful and arises from the difference in electric and magnetic multipole scattering.

---

## 3. Multi-Sphere Interaction Equation

### Coupled field equations

The field that excites sphere \(i\) consists of the external incident wave plus the scattered fields from all other spheres, translated to the coordinate origin of sphere \(i\) using the VSWF translation addition theorem (Mackowski 1991, 1996; Xu 1996):

```math
\mathbf{p}_\mathrm{exc}^{(i)} = \mathbf{p}_\mathrm{inc}^{(i)} + \sum_{j \neq i} \mathbf{H}_{ij}\,\mathbf{a}^{(j)}
```

where \(\mathbf{H}_{ij}\) is the translation matrix that re-expands outgoing VSWFs centred at \(\mathbf{r}_j\) into regular VSWFs centred at \(\mathbf{r}_i\) (Hankel-to-Bessel type), and \(\mathbf{a}^{(j)}\) is the vector of scattered-field expansion coefficients of sphere \(j\).

The T-matrix relates the exciting field to the scattered field:

```math
\mathbf{a}^{(i)} = \mathbf{T}_i\,\mathbf{p}_\mathrm{exc}^{(i)}.
```

Substituting and rearranging into a single global linear system for \(\mathbf{x} = (\mathbf{a}^{(1)}, \dots, \mathbf{a}^{(N_S)})^\top\) gives (Mackowski 1994, Eq. 6):

```math
(\mathbf{I} - \mathbf{T}\,\mathbf{A})\,\mathbf{x} = \mathbf{T}\,\mathbf{p}_\mathrm{inc}
```

where \(\mathbf{T} = \mathrm{diag}(\mathbf{T}_1, \dots, \mathbf{T}_{N_S})\) is the block-diagonal T-matrix and \(\mathbf{A}\) is the off-diagonal translation operator with blocks \(\mathbf{H}_{ij}\) (\(i \neq j\)) and zeros on the diagonal.  The dimension of this system is \(\sum_i 2\,N_i(N_i+2)\) — twice the multipole block size because each sphere carries both \(p = 1\) and \(p = 2\) polarisation blocks.

### Translation matrix via rotation–axial translation–rotation

The translation matrix \(\mathbf{H}_{ij}\) is computed via the factorisation into rotation, axial translation, and inverse rotation (Mackowski 1996, §3; Xu 1996, §4), which avoids the direct evaluation of Gaunt integrals and is more efficient:

1. **Rotation.** Compute the spherical coordinates \((d_{ij}, \theta_{ij}, \phi_{ij})\) of the translation vector \(\mathbf{r}_i - \mathbf{r}_j\).  The azimuthal phase factors \(e^{im\phi_{ij}}\) and the normalised associated Legendre functions \(P_n^m(\cos\theta_{ij})\) provide the rotation from the laboratory frame to a frame where the translation vector is along \(z\).
2. **Axial translation coefficients.** In the rotated frame where the two origins are separated along \(z\) by distance \(d_{ij}\), the translation coefficients factorise.  The key quantity involves a sum over intermediate order \(w\):
```math
C_{mn,kl} = \sum_{w=w_\mathrm{min}}^{n+l} c_w(m,n,k,l) \; h_w^{(1)}(d_{ij}) \; P_w^{m-k}(\cos\theta_{ij})
```

where \(w_\mathrm{min} = \max(|n-l|, |m-k|)\) and \(c_w(m,n,k,l)\) are products of vector coupling (Gaunt) coefficients.  These coupling coefficients are related to Clebsch–Gordan coefficients and are evaluated using a combined downward–upward recurrence algorithm (Xu 1996, §3; Mackowski 1996, Appendix) that avoids explicit factorial computation and maintains numerical stability even at high multipole orders.
3. **Polarisation coupling in the lr_tran basis.** In this basis the translation matrix has a particularly simple structure: the even-\(w\) and odd-\(w\) contributions (relative to \(n + l\)) combine to give the two polarisation blocks independently.  Specifically, denoting the even- and odd-\(w\) partial sums as \(A\) and \(B\):
```math
H_{kl,mn}^{(p=1)} = e^{i(m-k)\phi_{ij}} \left(A + iB\right), \qquad H_{kl,mn}^{(p=2)} = e^{i(m-k)\phi_{ij}} \left(A - iB\right).
```

This decoupling is the fundamental advantage of the lr_tran basis (Mackowski 1996, Eq. 18): the two circular polarisation modes interact with each other only through the T-matrix, not through translation.

The coupling coefficients are precomputed once and cached as a 3D array `tran_coef[mn, kl, w]` (Fortran routine `gentrancoefconstants`), while the distance-dependent Hankel functions and Legendre functions are evaluated per sphere pair.

### FFT-accelerated translation

For large numbers of spheres (\(N_S \gtrsim 100\)), the direct pairwise evaluation of \(\mathbf{A}\,\mathbf{v}\) scales as \(O(N_S^2)\) per iteration and dominates the computation time.  Mackowski & Kolokolova (2022) introduced a discrete Fourier convolution (DFC) scheme into MSTM v4.0 that reduces this to approximately \(O(N_S \ln N_S)\) scaling.  The algorithm proceeds as follows:

1. **Grid construction.** A uniform cubic grid with cell spacing \(d\) is mapped onto the target volume.  The cell spacing is chosen from the mean monomer radius and a target volume fraction \(f_v\) as \(d = \bar{a} \left(4\pi / 3 f_v\right)^{1/3}\), and each grid dimension is rounded up to the nearest integer of the form \(2^a \times 3^b \times 5^c\) for efficient FFT computation (Temperton's generalised prime-factor FFT).
2. **Sphere–node association.** Each sphere is assigned to the grid cell containing its centre.  A *neighbour hole* region is defined around each cell: for all cell pairs within this exclusion region (typically \(|i - j| \leq \sqrt{2}\), i.e., self + face-adjacent + edge-adjacent cells), sphere–sphere interactions are computed by direct Hankel-type translation, preserving accuracy at short inter-sphere distances.
3. **Sphere → Node.** For each sphere, the scattered-field expansion coefficients are translated to its associated grid node using a regular (Bessel \(j_n\)) translation matrix \(\mathbf{J}\).  This is valid because the sphere centre is within the grid cell (distance less than \(d/2\)), well within the convergence region of the regular addition theorem.
4. **Node-to-node convolution.** The grid-based translation operator is a discrete convolution: the translation matrix from node \(\mathbf{i}\) to node \(\mathbf{j}\) depends only on the displacement \(\mathbf{j} - \mathbf{i}\).  This convolution is evaluated in the frequency domain:
```math
\hat{g}_\mu(\mathbf{k}) = \sum_\nu \hat{H}_{\mu\nu}(\mathbf{k})\,\hat{a}_\nu(\mathbf{k})
```

where \(\mu\) and \(\nu\) represent the combined multipole indices. \(\hat{H}_{\mu\nu}(\mathbf{k})\) is the pre-computed 3D FFT of the cell-to-cell translation matrix (excluding the neighbour-hole cells), \(\hat{a}_\nu(\mathbf{k})\) is the FFT of the zero-padded node coefficient field, and the product is taken element-wise per index pair \((\mu, \nu)\).  The use of a doubled grid (\(2N_x \times 2N_y \times 2N_z\)) with zero-padding avoids circular convolution artefacts.
5. **Node → Sphere.** The resulting node-level exciting field is translated back to each sphere position via the reverse regular translation.
6. **Near-field correction.** Direct sphere-to-sphere Hankel-type translation is applied for all sphere pairs within the neighbour-hole region (step 2), adding these contributions to the FFT-based far-field result.

The total cost per iteration becomes \(O(N_S + M \log M)\) where \(M\) is the number of active grid cells, compared with \(O(N_S^2 N_\mathrm{max}^4)\) for the direct algorithm.  The memory overhead consists of the pre-FFT'd translation matrices, which scale as \(O(M \times N_\mathrm{node}^2)\) where \(N_\mathrm{node}\) is the node multipole order.  Mackowski & Kolokolova (2022) report speed improvements exceeding two orders of magnitude for systems of \(10^3\)–\(10^4\) spheres.

---

## 4. Iterative Solution: Complex BiConjugate Gradient

The linear system \(\mathbf{L}\,\mathbf{x} = \mathbf{b}\) where \(\mathbf{L} = \mathbf{I} - \mathbf{T}\,\mathbf{A}\) and \(\mathbf{b} = \mathbf{T}\,\mathbf{p}_\mathrm{inc}\) is solved by the Complex BiConjugate Gradient (CBICG) method (Mackowski & Mishchenko 2011, §2.3).

### Algorithm

The CBICG iteration requires both the forward operator \(\mathbf{L}\,\mathbf{v} = \mathbf{v} - \mathbf{T}\,\mathbf{A}\,\mathbf{v}\) and the adjoint (conjugate transpose) operator \(\mathbf{L}^\dagger\,\mathbf{v} = \mathbf{v} - \mathbf{A}^\dagger\,\mathbf{T}^\dagger\,\mathbf{v}\).  The forward operator is computed by applying the translation \(\mathbf{A}\) (direct or FFT-accelerated), followed by the block-diagonal T-matrix application \(\mathbf{T}\).

The adjoint of the translation operator \(\mathbf{A}\) is the computationally expensive part.  Mackowski & Mishchenko (2011) exploit the S-parity symmetry of the VSWF translation matrix to avoid explicitly forming \(\mathbf{A}^\dagger\):

```math
\mathbf{A}^\dagger = \left(\mathbf{S}\,\mathbf{A}\,\mathbf{S}\right)^*
```

where \(\mathbf{S}\) is the parity operator acting on each sphere's coefficient block as \(a_{m,n,p} \mapsto (-1)^m\,a_{-m,n,p}\), and \({}^*\) denotes element-wise complex conjugation.  This profound mathematical identity fundamentally arises from the spatial inversion symmetry and reciprocity of the electromagnetic translation operator. This means that the adjoint translation can be evaluated by: (i) applying \(\mathbf{S}\), (ii) performing the same forward translation \(\mathbf{A}\), (iii) applying \(\mathbf{S}\) again, and (iv) complex-conjugating.  This reuses the forward translation code path entirely, including the FFT-accelerated version, at no additional implementation cost.

The system is solved independently for two orthogonal incident linear polarisations (\(q = 1\) corresponding to \(x\)-polarisation, and \(q = 2\) corresponding to \(y\)-polarisation).  The convergence criterion is:

```math
\frac{\mathrm{Re}(\mathbf{r}^\top \mathbf{r})}{\|\mathbf{b}\|^2} < \varepsilon
```

where \(\mathbf{r}\) is the bilinear (not sesquilinear) residual and \(\varepsilon = 10^{-6}\) is the default tolerance.

### Incident plane wave coefficients

For \(z\)-axis incidence (\(\beta = 0\), \(\alpha = 0\)), only the \(m = \pm 1\) multipole components are non-zero due to the azimuthal symmetry of a circularly decomposed plane wave.  In the lr_tran basis, the incident field coefficients for sphere \(i\) are:

```math
p_{mn,p=1,q=1}^{(i)} = i^{\,n+1}\sqrt{\frac{2n+1}{2}}\;\delta_{m,+1}\;e^{iz_i}, \qquad p_{mn,p=2,q=1}^{(i)} = -i^{\,n+1}\sqrt{\frac{2n+1}{2}}\;\delta_{m,-1}\;e^{iz_i}
```

for the first incident polarisation (\(q = 1\)), and

```math
p_{mn,p=1,q=2}^{(i)} = -i^{\,n+2}\sqrt{\frac{2n+1}{2}}\;\delta_{m,+1}\;e^{iz_i}, \qquad p_{mn,p=2,q=2}^{(i)} = -i^{\,n+2}\sqrt{\frac{2n+1}{2}}\;\delta_{m,-1}\;e^{iz_i}
```

for \(q = 2\).  The phase factor \(e^{iz_i}\) accounts for the spatial position of sphere \(i\) along the propagation direction.

---

## 5. Output Quantities

### Translation to a common origin

After solving for the per-sphere scattered-field coefficients \(\mathbf{a}^{(i)}\), all coefficients are re-expanded about a common origin \(\mathbf{r}_0\) (typically the aggregate centroid) using regular (Bessel \(j_n\)) translation:

```math
\mathbf{a}_0 = \sum_{i=1}^{N_S} \mathbf{J}_{0i}\,\mathbf{a}^{(i)}
```

where \(\mathbf{J}_{0i}\) is the regular translation matrix from sphere \(i\) to the common origin.  The translation order for each sphere is determined by a convergence test on the translation addition theorem (Fortran routine `tranordertest`), ensuring that the re-expansion captures the scattered field to a specified accuracy.

The common-origin lr_tran coefficients are then transformed from the \((p=1, p=2)\) basis to mode-indexed form:

```math
a_{mn}^{(\text{mode}\,1)} = a_{mn}^{(p=1)} + a_{mn}^{(p=2)}, \qquad a_{mn}^{(\text{mode}\,2)} = a_{mn}^{(p=1)} - a_{mn}^{(p=2)}.
```

These mode-1 and mode-2 coefficients correspond to the standard TE and TM VSWF expansion of the total scattered field about the common origin (Mackowski 1994, §3).

### Scattering amplitude matrix (BH83 convention)

The far-field scattered wave is described by the \(2 \times 2\) amplitude matrix (BH83, Eq. 3.12):

```math
\begin{pmatrix} E_\parallel^\mathrm{sca} \\ E_\perp^\mathrm{sca} \end{pmatrix} = \frac{e^{ikr}}{-ikr} \begin{pmatrix} S_2 & S_3 \\ S_4 & S_1 \end{pmatrix} \begin{pmatrix} E_\parallel^\mathrm{inc} \\ E_\perp^\mathrm{inc} \end{pmatrix}
```

where \(S_1, S_2, S_3, S_4\) are dimensionless complex functions of the scattering angles \((\theta, \phi)\).

For the forward (\(\theta = 0°\)) and backward (\(\theta = 180°\)) directions only the \(m = \pm 1\) terms of the common-origin expansion survive.  The amplitudes are computed from the mode coefficients as

```math
S_j = -2 \sum_{n=1}^{N_\mathrm{max}} \sum_{m = \pm 1} \sum_{p = 1}^{2} F_j(n, m, p)\;\tau_{m,n,p}(\theta)
```

where \(F_j\) selects the appropriate combination of mode coefficients and incident polarisation (mode-1 from \(q=1\), mode-2 from \(q=1\) for \(S_2\), \(S_4\); similarly with sign change from \(q=2\) for \(S_1\), \(S_3\)), and \(\tau_{m,n,p}(\theta)\) are angular functions derived from the Wigner \(d\)-matrix at \(\cos\theta\).

At \(\theta = 0°\) the angular functions simplify to (Mackowski & Mishchenko 2011, `taufunc` routine):

| \(m\) | \(p\) | \(\tau_{m,n,p}(0°)\) |
| --- | --- | --- |
| \(+1\) | 1 | \(-f_n\) |
| \(+1\) | 2 | \(-f_n\) |
| \(-1\) | 1 | \(+f_n\) |
| \(-1\) | 2 | \(-f_n\) |

where \(f_n = \frac{1}{4}\sqrt{(2n+1)/2}\).

At \(\theta = 180°\) an additional factor of \((-1)^n\) appears, and the signs of the \(p = 1\) vs \(p = 2\) contributions differ:

| \(m\) | \(p\) | \(\tau_{m,n,p}(180°)\) |
| --- | --- | --- |
| \(+1\) | 1 | \(-(-1)^n f_n\) |
| \(+1\) | 2 | \(+(-1)^n f_n\) |
| \(-1\) | 1 | \(+(-1)^n f_n\) |
| \(-1\) | 2 | \(+(-1)^n f_n\) |

**Note on \(S_3\) and \(S_4\):** For a single sphere or a symmetric dimer with the symmetry axis along \(z\), \(S_3 = S_4 = 0\) by symmetry.  For general aggregates lacking such symmetry, all four elements are non-zero.

### Cross-section efficiencies

The dimensionless efficiency factors are defined as \(Q = C / (\pi\,a_\mathrm{eff}^2)\) where \(C\) is the corresponding cross section, and computed for unpolarised incidence by averaging over the two incident polarisations \(q = 1, 2\).

**Extinction efficiency** (optical theorem, Mackowski 1994, Eq. 12):

```math
Q_\mathrm{ext} = -\frac{2}{x_\mathrm{eff}^2} \sum_{q=1}^{2} \mathrm{Re}\!\left(\mathbf{a}_q^\dagger \,\mathbf{p}_{\mathrm{inc},q}\right)
```

where the inner product \(\mathbf{a}_q^\dagger \mathbf{p}_{\mathrm{inc},q}\) is evaluated in the per-sphere basis (before common-origin translation).  This is the multi-sphere generalisation of the forward-scattering (optical) theorem: the extinction is determined entirely by the projection of the scattered-field coefficients onto the incident-field pattern. The factor of 2 (instead of 4) in the prefactor arises from averaging the cross section over the two incident orthogonal polarisation states (\(q=1, 2\)).

**Scattering efficiency** (from the coherent common-origin expansion, Mackowski 1994, Eq. 13):

```math
Q_\mathrm{sca} = \frac{1}{x_\mathrm{eff}^2} \sum_{q=1}^{2} \left(\|\mathbf{a}_{0,q}^{(\text{mode}\,1)}\|^2 + \|\mathbf{a}_{0,q}^{(\text{mode}\,2)}\|^2 \right)
```

where the norm is the sum of squared magnitudes of all common-origin expansion coefficients for incident polarisation \(q\).  This requires the common-origin re-expansion (Section 5, first subsection), which coherently accounts for the relative phases of all spheres.

**Absorption efficiency:**

```math
Q_\mathrm{abs} = Q_\mathrm{ext} - Q_\mathrm{sca}
```

following from energy conservation.

### MI02 scattering amplitudes

The MI02 (Mishchenko, Travis & Lacis 2002) amplitude matrix \(\mathbf{S}^\mathrm{MI02}\) has dimension of length and is related to the dimensionless BH83 amplitudes by:

```math
S_{11}^\mathrm{MI02} = \frac{S_2^\mathrm{BH83}}{-ik}, \quad S_{22}^\mathrm{MI02} = \frac{S_1^\mathrm{BH83}}{-ik}, \quad S_{12}^\mathrm{MI02} = \frac{S_3^\mathrm{BH83}}{ik}, \quad S_{21}^\mathrm{MI02} = \frac{S_4^\mathrm{BH83}}{ik}
```

where \(k = 2\pi\,n_\mathrm{med}/\lambda_0\) is the (dimensional) wavenumber in the medium.  Note the index mapping: the MI02 \(S_{11}\) element corresponds to BH83 \(S_2\) (not \(S_1\)), and vice versa.

### CAS-v2 observable amplitudes

The Complex Amplitude Sensing (CAS-v2) protocol (Moteki & Adachi 2024) directly measures three complex amplitudes derived from the MI02 matrix elements:

```math
S_s^\mathrm{fwd} = S_{11}^\mathrm{MI02} + i\,S_{12}^\mathrm{MI02}
```

```math
S_p^\mathrm{fwd} = S_{22}^\mathrm{MI02} - i\,S_{21}^\mathrm{MI02}
```

```math
S^\mathrm{bak} = \frac{-S_{11}^\mathrm{MI02,bwd} + S_{22}^\mathrm{MI02,bwd}}{\sqrt{2}}
```

where \(S_s^\mathrm{fwd}\) and \(S_p^\mathrm{fwd}\) are the forward scattering amplitudes for \(s\)- and \(p\)-polarised incident light, respectively, and \(S^\mathrm{bak}\) is the depolarisation-sensitive backward scattering amplitude.

---

## 6. Computational Flow Summary

```text
Input: sphere positions r_i, radii a_i, m_sphere, λ₀, n_med

1. Non-dimensionalise: x_i = k·a_i, r_i → k·r_i, m_rel = m_sphere/n_med
                                                       [ParameterSweep.jl]

2. Mie coefficients (a_n, b_n) for each sphere          [MieCoefficients.jl]

3. Build T-matrix (lr_tran basis) per sphere             [TMatrixSolver.jl]

4. Compute incident-field expansion p_inc for each sphere
   (only m = ±1 terms for z-axis incidence)              [TMatrixSolver.jl]

5. Solve (I − T·A)·x = T·p_inc by CBICG iteration      [TMatrixSolver.jl]
   - A·v: pairwise translation (direct or FFT)
     [TranslationCoefs.jl / FFTTranslation.jl]

6. Translate solved coefficients to common origin        [ScatteringAmplitude.jl]

7. Compute S₁–S₄ (BH83) at θ = 0° and θ = 180°         [ScatteringAmplitude.jl]

8. Compute Q_ext (optical theorem), Q_sca, Q_abs         [ScatteringAmplitude.jl]

9. Convert to MI02 amplitudes and CAS-v2 observables     [ParameterSweep.jl]

```

---

## References

* Bohren, C.F. & Huffman, D.R. (1983). *Absorption and Scattering of Light by Small Particles*. Wiley. [BH83]
* Mackowski, D.W. (1991). Analysis of radiative scattering for multiple sphere configurations. *Proc. R. Soc. Lond. A*, 433, 599–614.
* Mackowski, D.W. (1994). Calculation of total cross sections of multiple-sphere clusters. *JOSA A*, 11(11), 2851–2861.
* Mackowski, D.W. (1996). Calculation of the total scattering cross section and the total absorbed energy for the multiple sphere clusters. *JOSA A*, 13(11), 2266–2278.
* Mackowski, D.W. (2014). A general superposition solution for electromagnetic scattering by multiple spherical domains of optically active media. *JQSRT*, 133, 264–270.
* Mackowski, D.W. & Kolokolova, L. (2022). Application of the multiple sphere superposition solution to large-scale systems of spheres via an accelerated algorithm. *JQSRT*, 280, 108075.
* Mackowski, D.W. & Mishchenko, M.I. (2011). A multiple sphere T-matrix Fortran code for use on parallel computer clusters. *JQSRT*, 112(13), 2182–2192.
* Mishchenko, M.I., Travis, L.D. & Lacis, A.A. (2002). *Scattering, Absorption, and Emission of Light by Small Particles*. Cambridge University Press. [MI02]
* Moteki, N. & Adachi, K. (2024). CAS-v2 protocol. *Optics Express*, 32(21), 36500–36522.
* Wiscombe, W.J. (1980). Improved Mie scattering algorithms. *Applied Optics*, 19(9), 1505–1509.
* Xu, Y.-l. (1996). Calculation of the addition coefficients in electromagnetic multisphere-scattering theory. *J. Comput. Phys.*, 127(2), 285–298.

---
