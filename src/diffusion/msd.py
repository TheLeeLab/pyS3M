"""Mean-squared-displacement (MSD) analysis of single-particle trajectories.

``MSD.DSigma2_OLSF`` estimates the diffusion coefficient D and the localisation variance
sigma^2 of one trajectory by an optimal least-squares fit (OLSF) of its MSD curve, choosing the
number of fitted MSD points as in Michalet, Phys. Rev. E 82, 041914 (2010) and Michalet &
Berglund, Phys. Rev. E 85, 061916 (2012).

Model (Michalet & Berglund 2012, Eq. 10), for a d-dimensional trajectory with frame time dT,
motion-blur coefficient R and localisation variance sigma^2 per axis::

    rho(i dT) = 2d (sigma^2 - 2 R D dT) + 2d D dT i

so a straight-line fit rho = a + b i gives D = b / (2 d dT) and
sigma^2 = (a + 4 d R D dT) / (2 d).

Units: D is in [coordinate unit]^2 / [time unit] and sigma^2 in [coordinate unit]^2, for
coordinates and dT in whatever units are passed in (e.g. nm and s give nm^2/s and nm^2).

Copied from pyDiffusion_LeeLab (``src/MSDFunctions.py``, 4159f52, 2025-05-30), so that the
tracking analysis needs no other package. D and ``msd_fft`` are computed exactly as there;
sigma^2 is corrected (see ``DSigma2_OLSF``).

pyDiffusion_LeeLab licence:

    MIT License

    Copyright (c) 2024 The Lee Lab

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
from typing import Tuple

import numpy as np
from numba import jit


class MSD:
    """MSD curve and optimal least-squares-fit estimates of D and sigma^2."""

    def DSigma2_OLSF(
        self,
        coordinates: np.ndarray,
        dT: float,
        R: float = 1.0 / 6,
        n_d: int = 1,
        maxiter: int = 100,
        min_points: int = 10,
        supress_warning: bool = False,
    ) -> Tuple[float, float]:
        """Estimate D and the localisation variance sigma^2 with the OLSF MSD approach.

        The MSD curve is fitted twice, iterating the number of fitted points to its optimum
        separately for the intercept (``p_a``) and the slope (``p_b``): D comes from the
        slope of the ``p_b`` fit, sigma^2 from the intercept of the ``p_a`` fit (Michalet &
        Berglund 2012, Eq. 10 and Sec. IV)::

            D       = b_b / (2 n_d dT)
            sigma^2 = (a_a + 4 n_d R D_a dT) / (2 n_d),   D_a = b_a / (2 n_d dT)

        Changes from pyDiffusion_LeeLab, which give sigma^2 only (D is unchanged):

        * the blur term is ``4 n_d R D dT`` (was ``4 R D dT``, too small by n_d for 2D/3D data);
        * sigma^2 comes from the intercept fit only; the slope fit used to overwrite it with
          ``(b_b + 4 R D dT) / (2 n_d)``, which is not a variance;
        * it is labelled sigma^2 (a variance, [coordinate unit]^2), not sigma.

        Args:
            coordinates: Trajectory, shape (N,) for n_d = 1 or (N, n_d), in any length unit.
            dT: Frame time, in any time unit.
            R: Motion-blur coefficient (1/6 for exposure = frame time, 0 for instantaneous
                exposure; Michalet & Berglund 2012, Eq. 5).
            n_d: Number of dimensions; ``coordinates.shape[1]`` must equal it if > 1.
            maxiter: Maximum number of iterations.
            min_points: Minimum number of trajectory points for an estimate.
            supress_warning: Do not print when the fit stops without converging.

        Returns:
            (D, sigma2): D in [length unit]^2 / [time unit], sigma^2 in [length unit]^2;
            both NaN if the trajectory is too short or the iteration does not converge.
        """
        if n_d > 1:
            try:
                if coordinates.shape[1] != n_d:
                    raise Exception(
                        "Dimension of coordinates and n_d are inconsistent."
                    )
            except Exception as error:
                print("Caught this error: " + repr(error))
                return np.nan, np.nan

        try:
            if coordinates.shape[0] < min_points:
                raise Exception(
                    """Not enough points to calculate a reliable estimate of diffusion coefficient.
                                Please see section V of Michalet and Berglund, Optimal Diffusion Coefficient Estimation
                                in Single-Particle Tracking. Phys. Rev. E 2012, 85 (6), 061916. https://doi.org/10.1103/PhysRevE.85.061916."""
                )
        except Exception as error:
            print("Caught this error: " + repr(error))
            return np.nan, np.nan

        NXM = len(coordinates.ravel())
        pa = np.array([int(np.floor(NXM / 10))])
        pb = np.array([int(np.floor(NXM / 10))])
        donea = 0
        doneb = 0
        iter = 1
        rho = []
        sigma2 = np.nan
        while iter <= maxiter and (not donea or not doneb):
            if iter == maxiter:
                donea = True
                doneb = True
                D = np.nan
                sigma2 = np.nan
                print("OLSF did not converge in {} iterations".format(maxiter))
            else:
                if np.max(np.hstack([pa, pb])) > len(rho):
                    rho = self.msd_fft(coordinates)
                rho = rho[: np.max(np.hstack([pa, pb]))]
                if not donea:
                    A = np.vstack((np.ones(pa[-1]), np.arange(1, pa[-1] + 1))).T
                    B = np.asarray(rho[: A.shape[0]])
                    if B.shape[0] != A.shape[0]:
                        if supress_warning == False:
                            print(
                                "OLSF did not converge in {} iterations".format(maxiter)
                            )
                        return np.nan, np.nan
                    X = np.linalg.lstsq(A, B, rcond=None)[0]
                    aa, ba = X
                    xa = 0 if aa < 0 else np.inf if ba < 0 else aa / ba
                    newpa = self.PMin_XM(xa, NXM)
                    if np.any(np.isin(newpa, pa)):
                        Da = ba / (2 * dT * n_d)
                        # Eq. 10: intercept = 2 n_d (sigma^2 - 2 R D dT)
                        sigma2 = (aa + 4 * n_d * Da * R * dT) / (2 * n_d)
                        donea = True
                    pa = np.hstack([pa, newpa])
                if not doneb:
                    A = np.vstack((np.ones(pb[-1]), np.arange(1, pb[-1] + 1))).T
                    B = np.asarray(rho[: int(pb[-1])])
                    if B.shape[0] != A.shape[0]:
                        if supress_warning == False:
                            print(
                                "OLSF did not converge in {} iterations".format(maxiter)
                            )
                        return np.nan, np.nan
                    X = np.linalg.lstsq(A, B, rcond=None)[0]
                    ab, bb = X
                    xb = 0 if ab < 0 else np.inf if bb < 0 else ab / bb
                    newpb = self.PMin_XM(xb, NXM)
                    if np.any(np.isin(newpb, pb)):
                        # Eq. 10: slope = 2 n_d D dT
                        D = bb / (2 * dT * n_d)
                        doneb = True
                    pb = np.hstack([pb, newpb])
                iter += 1

        return D, sigma2

    def autocorrFFT(self, x: np.ndarray) -> np.ndarray:
        """Autocorrelation of a 1D signal by FFT, normalised by the number of overlapping
        points at each lag.

        Args:
            x: Signal, shape (N,).

        Returns:
            Autocorrelation at lags 0..N-1, in the units of x squared.
        """
        N = len(x)
        F = np.fft.fft(x, n=2 * N)  # 2*N because of zero-padding
        PSD = F * F.conjugate()
        res = np.fft.ifft(PSD)
        res = (res[:N]).real  # now we have the autocorrelation in convention B
        n = N * np.ones(N) - np.arange(0, N)  # divide res(m) by (N-m)
        return res / n  # this is the autocorrelation in convention A

    def msd_fft(self, r: np.ndarray) -> np.ndarray:
        """Time-averaged MSD of one trajectory, by FFT.

        Args:
            r: Trajectory, shape (N, n_d), one row per frame, in any length unit.

        Returns:
            MSD at lags 1..N-1 frames, shape (N - 1,), in [length unit]^2.
        """
        N = len(r)
        D = np.square(r).sum(axis=1)
        D = np.append(D, 0)
        S2 = sum([self.autocorrFFT(r[:, i]) for i in range(r.shape[1])])
        Q = 2 * D.sum()
        S1 = np.zeros(N)
        for m in range(N):
            Q = Q - D[m - 1] - D[N - m]
            S1[m] = Q / (N - m)
        S = S1 - 2 * S2
        return S[1:]

    @staticmethod
    @jit(nopython=True)
    def PMin_XM(x, N):
        """Optimal numbers of MSD points to fit, from the reduced localisation error.

        Formulae from Michalet, Phys. Rev. E 82, 041914 (2010),
        https://doi.org/10.1103/PhysRevE.82.041914.

        Args:
            x (float): Reduced square localisation error, intercept / slope of the MSD fit.
            N (int): Number of trajectory values (points x dimensions).

        Returns:
            (pa, pb): Optimal number of points for the intercept (pa) and the slope (pb).
        """
        fa = 2 + 1.6 * (x**0.51)
        La = 3 + ((4.5 * (N**0.4) - 8.5) ** 1.2)

        fb = 2 + 1.35 * x**0.6
        Lb = 0.8 + (0.564 * N)

        if np.isinf(x):
            pa = int(np.floor(La))
            pb = int(np.floor(Lb))
        else:
            pa = int(np.floor(fa * La / (fa**3 + La**3) ** 0.33))
            pb = min(
                int(np.floor(Lb)), int(np.floor(fb * Lb / (fb**3 + Lb**3) ** 0.33))
            )

        # Make sure nothing is zero
        pa = max(2, pa)
        pb = max(2, pb)

        return pa, pb
