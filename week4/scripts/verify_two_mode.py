#!/usr/bin/env python3
"""Independent numpy reference for the two-mode nonlinear interaction.

Cross-implementation check for fluid.design.toml: builds the Hermitian
two-mode state (k1 = (1, 2) amplitude 1, k2 = (3, 1) amplitude 2i) and
computes the pseudospectral nonlinear term N = -(u . grad(omega)) with
pairing A (psi_hat = +omega_hat/kappa^2, u_x = i ky psi, u_y = -i kx psi)
directly in frequency space, using numpy's FFT for the physical products.
The resulting spectrum is the contract asserted by tests/fluid.rs.

Run from week4/scripts/:

    /tmp/lineplotenv/bin/python verify_two_mode.py
"""

import numpy as np

n = 32
K = np.arange(n)
K = np.where(K <= n // 2, K, K - n)
KX, KY = np.meshgrid(K, K)
kappa2 = KX**2 + KY**2

wh = np.zeros((n, n), dtype=complex)
wh[2, 1] = 1.0          # k1 = (kx=1, ky=2)
wh[n - 2, n - 1] = 1.0  # conjugate
wh[1, 3] = 2j           # k2 = (kx=3, ky=1)
wh[n - 1, n - 3] = -2j  # conjugate

psi = np.where(kappa2 > 0, wh / np.where(kappa2 > 0, kappa2, 1), 0)
ux = 1j * KY * psi
uy = -1j * KX * psi
ux[n // 2, :] = 0  # per-axis Nyquist no-travel
uy[:, n // 2] = 0
wx = 1j * KX * wh
wy = 1j * KY * wh

uxp = np.fft.ifft2(ux) * n**2
uyp = np.fft.ifft2(uy) * n**2
wxp = np.fft.ifft2(wx) * n**2
wyp = np.fft.ifft2(wy) * n**2
prod = (uxp * wxp + uyp * wyp).real
Nl = -np.fft.fft2(prod) / n**2
Nl = np.where((np.abs(KX) <= n // 3) & (np.abs(KY) <= n // 3), Nl, 0)

print("nonzero modes of the two-mode nonlinear term:")
for ky in range(n):
    for kx in range(n):
        if abs(Nl[ky, kx]) > 1e-12:
            print(f"  (kx={KX[ky, kx]:3d}, ky={KY[ky, kx]:3d}): N = {Nl[ky, kx]}")
