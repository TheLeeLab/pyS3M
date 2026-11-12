"""``ImageAnalysisFunctions._leastsq``: the covariance of a fit whose sqrt-space parameter sits at 0.

Amplitudes and backgrounds are fitted as sqrt-values (b = p^2). A good fit whose background in one
colour channel is zero has p ~ 0, where the model no longer depends on p: leastsq calls J^T J
singular and returns ``cov_x = None``, and every error -- the position errors too -- used to become
NaN, so the fit was dropped (bright beads behind a single-band filter, 2026-11-12). ``_leastsq``
then uses the pseudo-inverse of J^T J: the determined parameters get their usual variances.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import leastsq

from pyS3M.ImageAnalysisFunctions import _leastsq

X = np.linspace(-1.0, 1.0, 41)
RNG = np.random.default_rng(0)
Y = 2.0 + 0.5 * X + 0.01 * RNG.standard_normal(X.size)   # no quadratic term


def _residuals(p, x, y):
    """y = a + c x + q^2 x^2, the quadratic weight fitted in sqrt-space as the backgrounds are."""
    a, c, q = p
    return a + c * x + q ** 2 * x ** 2 - y


def _residuals_flat(p, x, y):
    """As _residuals, but q's term has no support in the data (x never exceeds 2): the model does
    not depend on q at all, as for a sqrt-space background pinned at zero, so J^T J is singular."""
    a, c, q = p
    return a + c * x + q ** 2 * np.maximum(x - 2.0, 0.0) - y


def test_plain_leastsq_has_no_covariance_here():
    """The case being fixed: the model does not depend on q, so leastsq returns no covariance."""
    _, cov, _, _, ier = leastsq(_residuals_flat, [1.0, 0.0, 0.3], args=(X, Y), full_output=True)
    assert ier in (1, 2, 3, 4)
    assert cov is None


def test_pseudo_inverse_gives_the_determined_parameters_their_variances():
    pfit, cov, _, _, ier = _leastsq(_residuals_flat, [1.0, 0.0, 0.3], args=(X, Y))
    assert ier in (1, 2, 3, 4)
    assert cov is not None and np.all(np.isfinite(cov))
    # a and c are a straight-line fit: (J^T J)^-1 of the columns [1, x]
    j = np.column_stack([np.ones_like(X), X])
    expected = np.linalg.inv(j.T @ j)
    assert cov[0, 0] == pytest.approx(expected[0, 0], rel=1e-6)
    assert cov[1, 1] == pytest.approx(expected[1, 1], rel=1e-6)
    assert pfit[0] == pytest.approx(2.0, abs=0.01)
    assert pfit[1] == pytest.approx(0.5, abs=0.01)


def test_fits_with_a_covariance_are_unchanged():
    p0 = [1.0, 0.0, 0.3]
    y = 2.0 + 0.5 * X + 0.2 * X ** 2
    ref = leastsq(_residuals, p0, args=(X, y), full_output=True)
    got = _leastsq(_residuals, p0, args=(X, y))
    assert ref[1] is not None
    np.testing.assert_array_equal(got[0], ref[0])
    np.testing.assert_array_equal(got[1], ref[1])


def test_failed_fits_get_no_covariance():
    """Only successful fits (ier 1-4) get the pseudo-inverse; a failed one keeps None."""
    _, cov, _, _, ier = _leastsq(_residuals_flat, [1.0, 0.0, 0.3], args=(X, Y), maxfev=1)
    assert ier not in (1, 2, 3, 4)
    assert cov is None
