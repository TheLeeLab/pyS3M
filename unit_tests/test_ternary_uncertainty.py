#!/usr/bin/env python3
"""Tests for TernaryPlotMixin.plot_ternary_credible_regions / plot_ternary_errorbars."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mpltern  # noqa: F401 -- registers the ternary projection
import numpy as np
import pytest
from matplotlib.path import Path

from pyS3M.PlottingBase import PublicationPlotter


@pytest.fixture
def plotter():
    return PublicationPlotter()


@pytest.fixture
def ax():
    fig = plt.figure()
    yield fig.add_subplot(projection="ternary")
    plt.close(fig)


def _dirichlet(mean, concentration, n=3000, seed=0):
    return np.random.default_rng(seed).dirichlet(np.asarray(mean) * concentration, n)


def _fraction_inside(ax, polygons, comp):
    xy = ax.transProjection.transform(comp)
    inside = np.zeros(len(comp), dtype=bool)
    for poly in polygons:
        inside |= Path(poly.get_xy()).contains_points(xy)
    return inside.mean()


class TestILR:
    def test_roundtrip(self, plotter):
        comp = _dirichlet([0.2, 0.5, 0.3], 30, n=50)
        np.testing.assert_allclose(plotter._ilr_inv(plotter._ilr(comp)), comp, atol=1e-12)


class TestCredibleRegions:
    @pytest.mark.parametrize("mean,level", [([0.2, 0.6, 0.2], 0.95), ([0.05, 0.15, 0.8], 0.68)])
    def test_region_contains_level_of_samples(self, plotter, ax, mean, level):
        samples = _dirichlet(mean, 40)
        polys = plotter.plot_ternary_credible_regions(ax, samples, levels=level)
        assert _fraction_inside(ax, polys, samples) == pytest.approx(level, abs=0.03)

    def test_region_stays_inside_triangle(self, plotter, ax):
        polys = plotter.plot_ternary_credible_regions(ax, _dirichlet([0.03, 0.07, 0.9], 30))
        for poly in polys:
            tlr = ax.transProjection.inverted().transform(poly.get_xy())
            assert (tlr > -1e-9).all()

    def test_nested_levels_and_multiple_points(self, plotter, ax):
        samples = [_dirichlet([0.2, 0.6, 0.2], 40, seed=1), _dirichlet([0.5, 0.3, 0.2], 40, seed=2)]
        polys = plotter.plot_ternary_credible_regions(
            ax, samples, levels=(0.5, 0.95), colors=["red", "blue"]
        )
        assert len(polys) >= 4
        assert {p.get_facecolor()[:3] for p in polys} == {(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)}

    def test_percent_input_matches_fractions(self, plotter):
        samples = _dirichlet([0.3, 0.4, 0.3], 40)
        regions = []
        for scale in (1, 100):
            fig = plt.figure()
            ax = fig.add_subplot(projection="ternary")
            regions.append(plotter.plot_ternary_credible_regions(ax, samples * scale, levels=0.9)[0].get_xy())
            plt.close(fig)
        np.testing.assert_allclose(regions[0], regions[1], atol=1e-9)

    @pytest.mark.parametrize("kwargs,match", [
        (dict(levels=1.0), "levels"),
        (dict(colors=["red", "blue"]), "one per point"),
    ])
    def test_bad_arguments(self, plotter, ax, kwargs, match):
        with pytest.raises(ValueError, match=match):
            plotter.plot_ternary_credible_regions(ax, _dirichlet([0.3, 0.4, 0.3], 40), **kwargs)

    def test_bad_sample_shape(self, plotter, ax):
        with pytest.raises(ValueError, match="n_samples, 3"):
            plotter.plot_ternary_credible_regions(ax, [np.ones((10, 2))])


class TestErrorbars:
    def test_bar_changes_one_channel_and_keeps_ratio(self, plotter, ax):
        artists = plotter.plot_ternary_errorbars(ax, [0.2], [0.5], [0.3], R_err=[0.05], capsize=0)
        ends = _tlr(artists[0])
        np.testing.assert_allclose(sorted(ends[:, 0]), [0.15, 0.25], atol=1e-9)
        np.testing.assert_allclose(ends[:, 1] / ends[:, 2], 0.5 / 0.3, atol=1e-9)

    def test_bar_clipped_at_edge(self, plotter, ax):
        artists = plotter.plot_ternary_errorbars(ax, [0.02], [0.5], [0.48], R_err=[0.1], capsize=0)
        assert _tlr(artists[0])[:, 0].min() == pytest.approx(0.0, abs=1e-12)

    def test_asymmetric_and_percent(self, plotter, ax):
        artists = plotter.plot_ternary_errorbars(
            ax, [20], [50], [30], G_err=[[5], [10]], capsize=0, show_points=False
        )
        np.testing.assert_allclose(sorted(_tlr(artists[0])[:, 1]), [0.45, 0.60], atol=1e-9)

    def test_caps_and_points_drawn(self, plotter, ax):
        artists = plotter.plot_ternary_errorbars(
            ax, [0.2, 0.3], [0.5, 0.4], [0.3, 0.3], [0.05] * 2, [0.05] * 2, [0.05] * 2
        )
        assert len(artists) == 2 * 3 * 2 + 1  # (bar + caps) x 3 channels x 2 points + scatter


def _tlr(line):
    """(t, l, r) coordinates of an mpltern Line2D (whose data are stored as x, y)."""
    return line.axes.transProjection.inverted().transform(np.column_stack(line.get_data()))
