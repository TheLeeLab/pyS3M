"""Tests for BasePlotter.add_colorbar: under constrained layout (the publication style) the colour bar
gets its own space, so an image's bar never overlaps the next panel."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyS3M.PlottingBase import PublicationPlotter


def test_image_colorbars_do_not_overlap_next_panel():
    plotter = PublicationPlotter(dark_background=False)
    fig, axs = plotter.two_column_plot(ncols=3)
    cbars = []
    for ax in axs.flat:
        _, im = plotter.image_plot(ax, np.random.default_rng(0).random((14, 14)), colorbar=True,
                                   colorbar_label="photoelectrons")
        cbars.append(ax.child_axes[-1])  # the bar is an inset of the image axes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for cax, next_ax in zip(cbars[:-1], list(axs.flat)[1:]):
        # the bar's full extent, ticks and label included, ends left of the next image
        assert cax.get_tightbbox(renderer).x1 <= next_ax.get_window_extent(renderer).x0
    for ax, cax in zip(axs.flat, cbars):
        # the bar is exactly as tall as the (equal-aspect) image, not its whole grid cell
        a, c = ax.get_window_extent(renderer), cax.get_window_extent(renderer)
        assert abs(c.height - a.height) < 0.5 and abs(c.y0 - a.y0) < 0.5
    plt.close(fig)


def test_colorbar_without_layout_engine_still_works():
    plotter = PublicationPlotter(dark_background=False)
    fig, ax = plt.subplots(layout="none")
    im = ax.imshow(np.zeros((4, 4)))
    cbar = plotter.add_colorbar(im, ax, label="photoelectrons")
    assert cbar.ax.get_ylabel() == "photoelectrons"
    plt.close(fig)
