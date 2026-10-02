"""
PlottingBase.py

Base plotting utilities and common patterns for pyS3M.
Consolidates common functionality from PlottingFunctions.py and DriftPlotting.py.

:authors: jsb92
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.animation import FuncAnimation, PillowWriter
from pyS3M.Constants import DriftConstants
import logging
logger = logging.getLogger(__name__)

# Import mpltern to register ternary projection with matplotlib
# This allows using projection="ternary" in add_subplot()
try:
    import mpltern
    MPLTERN_AVAILABLE = True
except ImportError:
    MPLTERN_AVAILABLE = False
    warnings.warn(
        "mpltern not available. Ternary plots will not work. "
        "Install with: pip install mpltern",
        ImportWarning
    )

def _safe_tight_layout(fig) -> None:
    """Call fig.tight_layout() suppressing the harmless layout-changed warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The figure layout has changed to tight",
            category=UserWarning,
        )
        try:
            fig.tight_layout()
        except Exception:
            pass


def plot_bayer_pattern(
    ax,
    pattern: str = "BGGR",
    size: int = 8,
    vline: float = None,
    hline: float = None,
    marker_pos: tuple = None,
    marker_color: str = "white",
    marker_size: float = 8,
    fontsize: float = 9,
    line_color: str = "white",
    dark_background: bool = True,
):
    """Draw a Bayer pattern grid on the given axes.

    By default renders white labels/grid on a black background (``dark_background=True``).
    Pass ``dark_background=False`` for black labels/grid on a white background, which
    suits publication figures with a white page background.

    Args:
        ax: Matplotlib axes to draw on.
        pattern: 4-character Bayer pattern string (row-major, top-down).
            ``pattern[0]`` = top-left cell, ``pattern[1]`` = top-right,
            ``pattern[2]`` = bottom-left, ``pattern[3]`` = bottom-right.
            Supported values: ``"BGGR"`` (default), ``"GBRG"``, ``"RGGB"``, ``"GRBG"``.
        size: Number of pixels shown along each axis (grid is size × size).
        vline: x-coordinate for a dashed vertical reference line.
        hline: y-coordinate for a dashed horizontal reference line.
        marker_pos: ``(x, y)`` in pixel coordinates — draws a scatter marker
            indicating the molecule's current position in the pattern.
        marker_color: Marker colour string.
        marker_size: Marker size (matplotlib ``s`` parameter).
        fontsize: Font size for the B / G / R labels.
        line_color: Colour for the vline/hline reference lines.
            Defaults to ``"white"`` when ``dark_background=True`` and ``"black"``
            when ``dark_background=False`` (unless overridden explicitly).
        dark_background: If ``True`` (default), black background with white grid/labels.
            If ``False``, white background with black grid/labels.

    Returns:
        The modified axes object.
    """
    tile = {
        (0, 0): pattern[0],
        (0, 1): pattern[1],
        (1, 0): pattern[2],
        (1, 1): pattern[3],
    }

    bg_color = "black" if dark_background else "white"
    fg_color = "white" if dark_background else "black"
    _line_color = line_color if line_color != "white" or dark_background else "black"

    ax.set_facecolor(bg_color)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)  # inverted — row 0 at top (image convention)
    ax.set_aspect("equal")

    # Grid lines drawn as bounded line segments with clip_on=False so that lines
    # at the axis boundaries (i=0 and i=size) are not half-clipped by the viewport.
    for i in range(size + 1):
        ax.plot([i, i], [0, size], color=fg_color, linewidth=0.5, zorder=1, clip_on=False)
        ax.plot([0, size], [i, i], color=fg_color, linewidth=0.5, zorder=1, clip_on=False)

    # Letter labels centred in each pixel cell
    for row in range(size):
        for col in range(size):
            ax.text(
                col + 0.5, row + 0.5,
                tile[(row % 2, col % 2)],
                color=fg_color,
                ha="center", va="center",
                fontsize=fontsize,
                fontfamily="sans-serif",
                fontweight="bold",
                zorder=2,
            )

    if vline is not None:
        ax.axvline(vline, color=_line_color, linestyle="--", linewidth=1, zorder=3)
    if hline is not None:
        ax.axhline(hline, color=_line_color, linestyle="--", linewidth=1, zorder=3)

    if marker_pos is not None:
        ax.scatter(marker_pos[0], marker_pos[1],
                   color=marker_color, s=marker_size, zorder=5)

    ax.axis("off")
    return ax


class PublicationConstants:
    """Constants for publication-quality plots following journal standards.

    Based on common journal requirements for scientific publications:
    - One-column max width: 3.33 inches (240 pt)
    - Two-column max width: 6.69 inches (17 cm)
    - Maximum depth: 8.25 inches (21.1 cm)

    Font hierarchy (from PlottingFunctions):
    - Tick labels: 7 pt
    - Axis labels (panel titles): 8 pt
    - Legends and annotations: 6 pt

    This ensures figures meet publication standards by default while allowing
    explicit overrides for presentations, posters, or other special cases.
    """

    # Figure dimensions (inches)
    ONE_COLUMN_WIDTH = 3.33
    TWO_COLUMN_WIDTH = 6.69
    MAX_HEIGHT = 8.25

    # Standard mode (for publications)
    STANDARD_FONT_SIZE = 7  # Base font size
    STANDARD_TICK_LABELSIZE = 7  # Tick labels (CORRECTED from 8)
    STANDARD_AXIS_LABELSIZE = 8  # Axis titles/panel labels
    STANDARD_LEGEND_FONTSIZE = 6  # Legends and annotations
    STANDARD_LINE_WIDTH = 0.5  # CORRECTED from 1.0
    STANDARD_TICK_LENGTH = 2.0  # 4 * line_width

    # Poster mode (for presentations)
    POSTER_FONT_SIZE = 12
    POSTER_TICK_LABELSIZE = 12
    POSTER_AXIS_LABELSIZE = 15
    POSTER_LEGEND_FONTSIZE = 10
    POSTER_LINE_WIDTH = 1.0
    POSTER_TICK_LENGTH = 4.0

    # Default panel sizing
    DEFAULT_PANEL_HEIGHT_RATIO = 3.5  # Height per panel in one-column plots
    DEFAULT_TWO_COLUMN_ROW_HEIGHT = 3.0  # Height per row in two-column plots


@dataclass
class PlottingConfig:
    """Configuration class for consistent plotting styles across pyS3M.

    This class now properly implements publication standards from PlottingFunctions.
    Default values follow journal requirements for single-molecule microscopy papers.

    Font hierarchy:
    - Base font (general text): 7pt
    - Axis labels (x/y axis titles): 8pt
    - Tick labels (numbers on axes): 7pt
    - Legends and annotations: 6pt

    For poster mode, all fonts are scaled appropriately (12pt base, 15pt axis labels, etc.)
    """

    # Display properties
    DEFAULT_DPI: int = 100  # Screen display DPI for notebooks
    DEFAULT_SAVE_DPI: int = 600  # High DPI for publication quality saving
    DEFAULT_FIGSIZE: Tuple[float, float] = (3.33, 3.5)  # One-column width figure

    # Color schemes
    DEFAULT_COLORMAP: str = "gist_gray"
    DEFAULT_SCATTER_COLOR: str = "blue"
    DEFAULT_GRID_COLOR: str = "gray"
    DEFAULT_GRID_ALPHA: float = 0.3

    # Image display percentiles
    DEFAULT_VMIN_PERCENTILE: float = 1.0
    DEFAULT_VMAX_PERCENTILE: float = 99.0

    # Publication standards (set in __post_init__ based on poster mode)
    font_size: int = PublicationConstants.STANDARD_FONT_SIZE
    tick_labelsize: int = PublicationConstants.STANDARD_TICK_LABELSIZE
    axis_labelsize: int = PublicationConstants.STANDARD_AXIS_LABELSIZE
    legend_fontsize: int = PublicationConstants.STANDARD_LEGEND_FONTSIZE
    line_width: float = PublicationConstants.STANDARD_LINE_WIDTH
    tick_length: float = PublicationConstants.STANDARD_TICK_LENGTH

    # Marker properties
    DEFAULT_MARKER_SIZE: float = 1.0

    # Colorbar properties
    DEFAULT_COLORBAR_WIDTH: str = "5%"
    DEFAULT_COLORBAR_PAD: float = 0.05

    # Scale bar properties
    DEFAULT_SCALEBAR_COLOR: str = "white"
    DEFAULT_SCALEBAR_FONTSIZE: int = 6

    # Mode flags
    poster_mode: bool = False
    dark_background: bool = False

    def __post_init__(self):
        """Set up matplotlib parameters based on configuration."""
        # Use poster values if in poster mode
        if self.poster_mode:
            self.font_size = PublicationConstants.POSTER_FONT_SIZE
            self.tick_labelsize = PublicationConstants.POSTER_TICK_LABELSIZE
            self.axis_labelsize = PublicationConstants.POSTER_AXIS_LABELSIZE
            self.legend_fontsize = PublicationConstants.POSTER_LEGEND_FONTSIZE
            self.line_width = PublicationConstants.POSTER_LINE_WIDTH
            self.tick_length = PublicationConstants.POSTER_TICK_LENGTH

        # Typeface: Helvetica first, with visually/metrically-compatible fallbacks.
        # svg.fonttype="none" (below) keeps text as real text in exported SVGs, so
        # this whole list is embedded as CSS font-family -- a system with true
        # Helvetica (e.g. many Mac/Adobe setups) renders it correctly; systems
        # without it (e.g. this one) fall through to Arial, then Nimbus Sans (URW's
        # metric-compatible Helvetica clone, and what actually renders here), then
        # Liberation Sans/DejaVu Sans, with no matplotlib "font not found" warning
        # since Nimbus Sans is genuinely installed and matplotlib-visible here.
        _sans_stack = ["Helvetica", "Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"]

        # mathtext (anything in $...$, e.g. \mathrm{\mu_R} for non-italic Greek
        # letters/subscripts) does NOT gracefully cascade through a fallback list
        # the way font.sans-serif does -- its 'custom' fontset needs ONE concrete,
        # already-resolved font name per style, or it silently falls back to
        # DejaVu Sans (mismatching the plain-text font). So resolve _sans_stack to
        # whichever member actually exists on this system once, up front, and point
        # mathtext at that same concrete font -- keeping math and plain text
        # visually consistent (and matching true Helvetica too, on a system that
        # has it, since findfont would resolve to "Helvetica" there instead).
        _resolved_sans_path = font_manager.findfont(
            font_manager.FontProperties(family=_sans_stack)
        )
        _resolved_sans_name = font_manager.FontProperties(fname=_resolved_sans_path).get_name()

        # Configure matplotlib globally with proper font hierarchy
        matplotlib.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": _sans_stack,
                # Non-italic mathtext (\mathrm{...}) in the same resolved font as
                # plain text, so e.g. r'$\mathrm{\mu_R}$' renders as an upright mu
                # with a real (lowered) subscript R, not italic, matching Helvetica/
                # its fallback -- see _resolved_sans_name note above.
                "mathtext.fontset": "custom",
                "mathtext.rm": _resolved_sans_name,
                "mathtext.it": f"{_resolved_sans_name}:italic",
                "mathtext.bf": f"{_resolved_sans_name}:bold",
                "mathtext.sf": _resolved_sans_name,
                # Font sizes (proper hierarchy)
                "font.size": self.font_size,  # Base font (7pt standard, 12pt poster)
                "axes.labelsize": self.axis_labelsize,  # Axis labels (8pt standard, 15pt poster)
                "axes.titlesize": self.axis_labelsize,  # Panel titles (8pt standard, 15pt poster)
                "xtick.labelsize": self.tick_labelsize,  # Tick labels (7pt standard, 12pt poster)
                "ytick.labelsize": self.tick_labelsize,  # Tick labels (7pt standard, 12pt poster)
                "legend.fontsize": self.legend_fontsize,  # Legends (6pt standard, 10pt poster)
                # Line widths
                "axes.linewidth": self.line_width,  # Axis spines (0.5pt standard, 1.0pt poster)
                "xtick.major.width": self.line_width,  # Tick marks
                "ytick.major.width": self.line_width,
                "xtick.major.size": self.tick_length,  # Tick length
                "ytick.major.size": self.tick_length,
                # Padding
                "xtick.major.pad": 1.2,
                "ytick.major.pad": 1.2,
                # Layout
                "figure.constrained_layout.use": True,  # Auto-adjust spacing
                # Font embedding for publications
                "svg.fonttype": "none",  # Save text as text (not paths) in SVG
                "pdf.fonttype": 42,  # Embed fonts as TrueType (editable) in PDF
                "ps.fonttype": 42,  # Same for PostScript
            }
        )

        # Dark background adjustments
        if self.dark_background:
            self.DEFAULT_GRID_COLOR = "white"
            self.DEFAULT_SCALEBAR_COLOR = "white"
            plt.style.use("dark_background")


class BasePlotter(ABC):
    """Base class for all plotting functionality in pyS3M.

    This class provides common plotting utilities and patterns that are shared
    across different plotting modules in the codebase. It follows the DRY principle
    by consolidating repeated plotting patterns.
    """

    def __init__(self, config: Optional[PlottingConfig] = None):
        """Initialize the base plotter.

        Args:
            config: Plotting configuration. If None, uses default configuration.
        """
        self.config = config or PlottingConfig()
        self._setup_matplotlib()

    def _setup_matplotlib(self):
        """Setup matplotlib with consistent configuration."""
        # Configure matplotlib backend if needed
        if matplotlib.get_backend() == "Agg":
            warnings.warn("Using Agg backend - plots will not display interactively")

    def create_figure(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        facecolor: str = "white",
        edgecolor: str = "black",
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Create a standardised figure with single axis.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for figure resolution
            facecolor: Figure face color
            edgecolor: Figure edge color

        Returns:
            Tuple of (figure, axis)
        """
        figsize = figsize or self.config.DEFAULT_FIGSIZE
        dpi = dpi or self.config.DEFAULT_DPI

        fig, ax = plt.subplots(
            figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor
        )
        return fig, ax

    def one_column_plot(
        self,
        npanels: int = 1,
        ratios: Optional[List[float]] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
    ) -> Tuple[matplotlib.figure.Figure, Union[matplotlib.axes.Axes, np.ndarray]]:
        """Create a one-column width publication-quality figure.

        Defaults to 3.33" width, 3.5" per panel height at 100 DPI for display (600 DPI when saved).

        Args:
            npanels: Number of vertical panels
            ratios: Height ratios for panels. If None, all panels equal height.
            height: Total figure height in inches. If None, uses standard (3.5" per panel).
            width: Figure width in inches. If None, uses one-column standard (3.33").

        Returns:
            Tuple of (figure, axes). axes is single Axes if npanels=1, else 1D array.
        """
        if ratios is None:
            ratios = [1] * npanels

        if len(ratios) != npanels:
            raise ValueError(f"Number of ratios ({len(ratios)}) must match npanels ({npanels})")

        # Calculate dimensions with publication standards
        if width is not None:
            xsize = width
            if width > PublicationConstants.ONE_COLUMN_WIDTH:
                warnings.warn(
                    f"Width {width:.2f}\" exceeds one-column standard "
                    f"({PublicationConstants.ONE_COLUMN_WIDTH:.2f}\")"
                )
        else:
            xsize = PublicationConstants.ONE_COLUMN_WIDTH  # Default: 3.33"

        if height is not None:
            ysize = height
            if height > PublicationConstants.MAX_HEIGHT:
                warnings.warn(
                    f"Height {height:.2f}\" exceeds maximum "
                    f"({PublicationConstants.MAX_HEIGHT}\")",
                    UserWarning,
                )
        else:
            ysize = min(
                PublicationConstants.DEFAULT_PANEL_HEIGHT_RATIO * npanels,
                PublicationConstants.MAX_HEIGHT
            )

        fig, axs = plt.subplots(
            nrows=npanels, ncols=1,
            figsize=(xsize, ysize),
            height_ratios=ratios,
            frameon=False,
            squeeze=False,
            dpi=self.config.DEFAULT_DPI,  # 100 DPI for display
        )

        # Configure tick parameters
        for ax in axs.flat:
            ax.xaxis.set_tick_params(width=self.config.line_width, length=self.config.tick_length)
            ax.yaxis.set_tick_params(width=self.config.line_width, length=self.config.tick_length)

        # Return appropriately squeezed axes
        if npanels == 1:
            return fig, axs[0, 0]
        else:
            return fig, axs[:, 0]

    def two_column_plot(
        self,
        nrows: int = 1,
        ncols: int = 1,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        big: bool = False,
        subplot_kw: Optional[Dict[str, Any]] = None,
    ) -> Tuple[matplotlib.figure.Figure, Union[matplotlib.axes.Axes, np.ndarray]]:
        """Create a two-column width publication-quality figure.

        Defaults to 6.69" width, 3.0" per row height at 100 DPI for display (600 DPI when saved).

        Args:
            nrows: Number of rows
            ncols: Number of columns
            height_ratios: Relative heights of rows. If None, all equal.
            width_ratios: Relative widths of columns. If None, all equal.
            width: Total figure width in inches. If None, uses two-column standard (6.69").
            height: Total figure height in inches. If None, uses standard (3.0" per row).
            big: If True, allows larger sizes for presentations (5" per dimension).
            subplot_kw: Forwarded to ``plt.subplots()``, e.g. ``{'projection': 'ternary'}``
                for a grid of mpltern ternary axes. If None (default), plain Cartesian axes.

        Returns:
            Tuple of (figure, axes). axes shape depends on nrows/ncols:
                - nrows=1, ncols=1: single Axes
                - nrows=1: 1D array (columns)
                - ncols=1: 1D array (rows)
                - else: 2D array
        """
        if height_ratios is None:
            height_ratios = [1] * nrows
        if width_ratios is None:
            width_ratios = [1] * ncols

        # Calculate dimensions
        if width is not None:
            xsize = width
            if width > PublicationConstants.TWO_COLUMN_WIDTH and not big:
                warnings.warn(
                    f"Width {width:.2f}\" exceeds two-column standard "
                    f"({PublicationConstants.TWO_COLUMN_WIDTH:.2f}\")"
                )
        else:
            if big:
                xsize = 5.0 * ncols
            else:
                xsize = PublicationConstants.TWO_COLUMN_WIDTH  # Default: 6.69"

        if height is not None:
            ysize = height
            if height > PublicationConstants.MAX_HEIGHT:
                warnings.warn(
                    f"Height {height:.2f}\" exceeds maximum "
                    f"({PublicationConstants.MAX_HEIGHT}\")",
                    UserWarning,
                )
        else:
            if big:
                ysize = min(5.0 * nrows, PublicationConstants.MAX_HEIGHT)
            else:
                ysize = min(
                    PublicationConstants.DEFAULT_TWO_COLUMN_ROW_HEIGHT * nrows,
                    PublicationConstants.MAX_HEIGHT
                )

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(xsize, ysize),
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            frameon=False,
            squeeze=False,
            dpi=self.config.DEFAULT_DPI,
            subplot_kw=subplot_kw or {},
        )

        # Configure tick parameters
        for ax in axs.flat:
            ax.xaxis.set_tick_params(width=self.config.line_width, length=self.config.tick_length)
            ax.yaxis.set_tick_params(width=self.config.line_width, length=self.config.tick_length)

        # Return appropriately squeezed axes
        if nrows == 1 and ncols == 1:
            return fig, axs[0, 0]
        elif nrows == 1:
            return fig, axs[0, :]
        elif ncols == 1:
            return fig, axs[:, 0]
        else:
            return fig, axs

    def setup_axis(
        self,
        ax: matplotlib.axes.Axes,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        grid: bool = True,
        grid_alpha: Optional[float] = None,
        equal_aspect: bool = False,
        spine_style: Optional[str] = None,
    ) -> None:
        """Configure axis with standard settings.

        Args:
            ax: Matplotlib axis to configure
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Axis title
            grid: Whether to show grid
            grid_alpha: Grid transparency (uses default if None)
            equal_aspect: Whether to set equal aspect ratio
            spine_style: Style for axis spines ('box', 'left-bottom', 'none')
        """
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        if grid:
            alpha = grid_alpha or self.config.DEFAULT_GRID_ALPHA
            ax.grid(
                True,
                alpha=alpha,
                color=self.config.DEFAULT_GRID_COLOR,
                linestyle="--",
                linewidth=0.5,
            )

        if equal_aspect:
            ax.set_aspect("equal")

        # Configure spines
        if spine_style == "none":
            for spine in ax.spines.values():
                spine.set_visible(False)
        elif spine_style == "left-bottom":
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    def create_image_plot(
        self,
        ax: matplotlib.axes.Axes,
        data: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = None,
        origin: str = "lower",
        interpolation = None,
    ) -> matplotlib.image.AxesImage:
        """Create standardised image plot.

        Args:
            ax: Axis to plot on
            data: 2D array to display
            vmin: Minimum value for colormap (auto-calculated if None)
            vmax: Maximum value for colormap (auto-calculated if None)
            cmap: Colormap name (uses default if None)
            origin: Image origin ('lower' or 'upper')
            interpolation: Interpolation method (None uses matplotlib's default behavior,
                which avoids rendering artifacts. Can be set to 'none', 'nearest', etc. if needed.)

        Returns:
            AxesImage object for further customisation
        """
        cmap = cmap or self.config.DEFAULT_COLORMAP

        # Auto-calculate vmin/vmax using percentiles if not provided
        if vmin is None:
            vmin = np.percentile(data.ravel(), self.config.DEFAULT_VMIN_PERCENTILE)
        if vmax is None:
            vmax = np.percentile(data.ravel(), self.config.DEFAULT_VMAX_PERCENTILE)

        im = ax.imshow(
            data,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            origin=origin,
            interpolation=interpolation,
        )

        return im

    def add_colorbar(
        self,
        im: matplotlib.image.AxesImage,
        ax: matplotlib.axes.Axes,
        label: str = "",
        location: str = "right",
        size: Optional[str] = None,
        pad: Optional[float] = None,
    ) -> matplotlib.colorbar.Colorbar:
        """Add colorbar to image plot.

        Args:
            im: Image object to create colorbar for
            ax: Axis containing the image
            label: Colorbar label
            location: Colorbar location ('right', 'left', 'top', 'bottom')
            size: Colorbar size as percentage (e.g., '5%')
            pad: Padding between axis and colorbar

        Returns:
            Colorbar object
        """
        size = size or self.config.DEFAULT_COLORBAR_WIDTH
        pad = pad or self.config.DEFAULT_COLORBAR_PAD

        divider = make_axes_locatable(ax)
        cax = divider.append_axes(location, size=size, pad=pad)

        cbar = plt.colorbar(im, cax=cax)
        if label:
            cbar.set_label(label)

        return cbar

    def add_scalebar(
        self,
        ax: matplotlib.axes.Axes,
        pixelsize: float,
        length_nm: float,
        location: str = "lower right",
        color: str = None,
        fontsize: Optional[int] = None,
        label: Optional[str] = None,
    ) -> AnchoredSizeBar:
        """Add scale bar to plot.

        Args:
            ax: Axis to add scale bar to
            pixelsize: Size of one pixel in nanometres
            length_nm: Length of scale bar in nanometres
            location: Scale bar location
            color: Scale bar color
            fontsize: Font size for scale bar text
            label: Custom label (auto-generated if None)

        Returns:
            AnchoredSizeBar object
        """
        color = color or self.config.DEFAULT_SCALEBAR_COLOR
        fontsize = fontsize or self.config.DEFAULT_SCALEBAR_FONTSIZE

        # Convert nanometres to pixels
        length_pixels = length_nm / pixelsize

        # Generate label if not provided
        if label is None:
            if length_nm >= 1000:
                label = f"{length_nm/1000:.1f} μm"
            else:
                label = f"{length_nm:.0f} nm"

        scalebar = AnchoredSizeBar(
            ax.transData,
            length_pixels,
            label,
            loc=location,
            pad=0.5,
            color=color,
            frameon=False,
            size_vertical=length_pixels / 20,
            fontproperties={"size": fontsize},
        )

        ax.add_artist(scalebar)
        return scalebar

    # Convenience plotting methods for common plot types
    def line_plot(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str = "x axis",
        ylabel: str = "y axis",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        color: str = "k",
        linewidth: float = 1.0,
        linestyle: str = "-",
        label: str = "",
        alpha: float = 1.0,
        grid: bool = True,
    ) -> matplotlib.axes.Axes:
        """Create a line plot with consistent styling.

        Args:
            ax: Axes object to plot on
            x: X data
            y: Y data
            xlabel: X axis label
            ylabel: Y axis label
            xlim: X axis limits (min, max)
            ylim: Y axis limits (min, max)
            color: Line color
            linewidth: Line width
            linestyle: Line style ('-', '--', '-.', ':')
            label: Line label for legend
            alpha: Line transparency
            grid: Whether to show grid

        Returns:
            Modified axes object
        """
        ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle,
                label=label, alpha=alpha)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        self.setup_axis(ax, xlabel=xlabel, ylabel=ylabel, grid=grid)

        if label:
            ax.legend()

        return ax

    def line_plot_with_error(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray,
        xlabel: str = "x axis",
        ylabel: str = "y axis",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        color: str = "k",
        linewidth: float = 1.0,
        label: str = "",
        alpha: float = 0.3,
        grid: bool = True,
    ) -> matplotlib.axes.Axes:
        """Create a line plot with error bars/shading.

        Args:
            ax: Axes object to plot on
            x: X data
            y: Y data
            yerr: Y error values
            xlabel: X axis label
            ylabel: Y axis label
            xlim: X axis limits (min, max)
            ylim: Y axis limits (min, max)
            color: Line and error color
            linewidth: Line width
            label: Line label for legend
            alpha: Error shading transparency
            grid: Whether to show grid

        Returns:
            Modified axes object
        """
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        self.setup_axis(ax, xlabel=xlabel, ylabel=ylabel, grid=grid)

        if label:
            ax.legend()

        return ax

    def scatter_plot(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str = "x axis",
        ylabel: str = "y axis",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        color: str = "k",
        edgecolor: str = "k",
        facecolor: str = "white",
        size: float = 20,
        marker: str = "o",
        label: str = "",
        alpha: float = 1.0,
        linewidth: float = 0.75,
        rasterized: bool = False,
        grid: bool = True,
    ) -> matplotlib.axes.Axes:
        """Create a scatter plot with consistent styling.

        Args:
            ax: Axes object to plot on
            x: X data
            y: Y data
            xlabel: X axis label
            ylabel: Y axis label
            xlim: X axis limits (min, max)
            ylim: Y axis limits (min, max)
            color: Overall marker color (overridden by facecolor/edgecolor)
            edgecolor: Marker edge color
            facecolor: Marker face color
            size: Marker size
            marker: Marker style
            label: Scatter label for legend
            alpha: Marker transparency
            linewidth: Edge line width
            rasterized: Whether to rasterize markers (recommended for >1000 points)
            grid: Whether to show grid

        Returns:
            Modified axes object
        """
        ax.scatter(
            x, y,
            s=size,
            c=color if facecolor == "white" and edgecolor == "k" else None,
            edgecolors=edgecolor,
            facecolors=facecolor if facecolor != "white" or edgecolor != "k" else None,
            marker=marker,
            label=label,
            alpha=alpha,
            linewidths=linewidth,
            rasterized=rasterized,
        )

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        self.setup_axis(ax, xlabel=xlabel, ylabel=ylabel, grid=grid)

        if label:
            ax.legend()

        return ax

    def histogram_plot(
        self,
        ax: matplotlib.axes.Axes,
        data: np.ndarray,
        bins: Union[int, np.ndarray] = 50,
        xlabel: str = "Value",
        ylabel: str = "Counts",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        color: str = "blue",
        edgecolor = None,
        alpha: float = 0.7,
        density: bool = False,
        label: str = "",
        grid: bool = True,
    ) -> matplotlib.axes.Axes:
        """Create a histogram with consistent styling.

        Args:
            ax: Axes object to plot on
            data: Data to histogram
            bins: Number of bins or bin edges
            xlabel: X axis label
            ylabel: Y axis label
            xlim: X axis limits (min, max)
            ylim: Y axis limits (min, max)
            color: Bar color
            edgecolor: Bar edge color
            alpha: Bar transparency
            density: Whether to normalize to probability density
            label: Histogram label for legend
            grid: Whether to show grid

        Returns:
            Modified axes object
        """
        ax.hist(
            data,
            bins=bins,
            color=color,
            edgecolor=edgecolor,
            alpha=alpha,
            density=density,
            label=label,
        )

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        self.setup_axis(ax, xlabel=xlabel, ylabel=ylabel, grid=grid)

        if label:
            ax.legend()

        return ax

    def image_plot(
        self,
        ax: matplotlib.axes.Axes,
        image: np.ndarray,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        cmap: str = "gray",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = False,
        colorbar_label: str = "",
        aspect: str = "equal",
        interpolation: Optional[str] = None,
        origin: str = "lower",
        scalebar: bool = False,
        pixelsize: float = DriftConstants.XIMEA_PIXEL_SIZE_NM,
        scalebarsize: float = 10000.0,
        scalebarlabel: str = "10 μm",
        scalebar_color: str = "white",
        show_axes: bool = False,
    ) -> Tuple[matplotlib.axes.Axes, matplotlib.image.AxesImage]:
        """Create an image plot with consistent styling.

        For microscopy images, axes are OFF by default (show_axes=False).
        Scalebars are used instead for scale indication.

        Args:
            ax: Axes object to plot on
            image: 2D image data
            xlabel: X axis label (only used if show_axes=True)
            ylabel: Y axis label (only used if show_axes=True)
            title: Plot title
            cmap: Colormap name
            vmin: Minimum value for colormap (auto if None)
            vmax: Maximum value for colormap (auto if None)
            colorbar: Whether to add colorbar (default False for microscopy)
            colorbar_label: Label for colorbar
            aspect: Aspect ratio ('equal', 'auto', or float)
            interpolation: Interpolation method
            origin: Image origin ('lower' or 'upper')
            scalebar: Whether to add scale bar
            pixelsize: Pixel size in nm (for scalebar)
            scalebarsize: Scale bar size in nm
            scalebarlabel: Scale bar label (e.g., "10 μm")
            scalebar_color: Scale bar color
            show_axes: Whether to show axes (default False for microscopy images)

        Returns:
            Tuple of (modified axes, image object)
        """
        # Auto-calculate vmin/vmax using percentiles if not provided
        if vmin is None:
            vmin = np.percentile(image, 1)
        if vmax is None:
            vmax = np.percentile(image, 99)

        im = ax.imshow(
            image,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
            interpolation=interpolation,
            origin=origin,
        )

        # For microscopy images, axes are typically off
        if not show_axes:
            ax.axis("off")
        else:
            self.setup_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=False)

        if title and not show_axes:
            # Add title even when axes are off
            ax.set_title(title, fontsize=self.config.axis_labelsize)

        if colorbar:
            self.add_colorbar(im, ax, label=colorbar_label)

        if scalebar:
            self.add_scalebar(
                ax,
                pixelsize=pixelsize,
                length_nm=scalebarsize,
                location="lower right",
                color=scalebar_color,
                label=scalebarlabel,
            )

        return ax, im

    def colour_image_plot(
        self,
        axs,
        data: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        c_min: float = 0.0,
        c_max: float = 1.0,
        cmap: str = "jet",
        cbar: str = "on",
        cbarlabel: str = "",
        label: str = "",
        labelcolor: str = "white",
        pixelsize: float = DriftConstants.XIMEA_PIXEL_SIZE_NM,
        sbar: str = "on",
        scalebarsize: float = 10000.0,
        scalebarlabel: str = "10 μm",
    ) -> matplotlib.axes.Axes:
        """Plot a pre-rendered RGB image with a colorbar reflecting the colour parameter.

        Designed for images produced by ``render.render()`` with
        ``blur_method="gaussian_colour"``, where hue encodes a scalar
        colour parameter (e.g. A_R, wavelength) and saturation encodes
        localisation density. The colorbar is constructed from a
        ``ScalarMappable`` with the same colormap and normalisation used
        during rendering so that it accurately reflects the colour mapping.

        Args:
            axs: Axes object to plot on.
            data: RGB image array of shape ``(H, W, 3)``.
            vmin: Minimum display value. If None, uses 1st percentile.
            vmax: Maximum display value. If None, uses 99th percentile.
            c_min: Minimum value of the colour parameter used in rendering.
            c_max: Maximum value of the colour parameter used in rendering.
            cmap: Matplotlib colormap name used in rendering.
            cbar: Whether to show colorbar (``"on"`` or ``"off"``).
            cbarlabel: Colorbar label (e.g. ``"A_R"``).
            label: Text label to draw on the image.
            labelcolor: Colour for the text label.
            pixelsize: Pixel size in nm.
            sbar: Whether to show scale bar (``"on"`` or ``"off"``).
            scalebarsize: Scale bar size in nm.
            scalebarlabel: Scale bar label text.

        Returns:
            Modified axes object.
        """
        if vmin is None:
            vmin = np.percentile(data, 1)
        if vmax is None:
            vmax = np.percentile(data, 99)

        axs.imshow(data, origin="lower", vmin=vmin, vmax=vmax)
        axs.axis("off")

        if label:
            axs.text(
                0.05, 0.95, label,
                transform=axs.transAxes,
                fontsize=self.config.axis_labelsize,
                color=labelcolor,
                verticalalignment="top",
            )

        if cbar == "on":
            norm = matplotlib.colors.Normalize(vmin=c_min, vmax=c_max)
            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            self.add_colorbar(sm, axs, label=cbarlabel)

        if sbar == "on":
            self.add_scalebar(
                axs,
                pixelsize=pixelsize,
                length_nm=scalebarsize,
                location="lower right",
                color="white",
                label=scalebarlabel,
            )

        return axs

    def save_or_show(
        self,
        fig: matplotlib.figure.Figure,
        save_path: Optional[str] = None,
        show: bool = True,
        dpi: Optional[int] = None,
        bbox_inches: str = "tight",
        facecolor: str = "white",
        edgecolor: str = "none",
    ) -> None:
        """Save figure to file and/or display it.

        Args:
            fig: Figure to save/show
            save_path: Path to save figure (no saving if None)
            show: Whether to display the figure
            dpi: Resolution for saving (uses default if None)
            bbox_inches: Bounding box for saved figure
            facecolor: Face color for saved figure
            edgecolor: Edge color for saved figure
        """
        if save_path:
            dpi = dpi or self.config.DEFAULT_SAVE_DPI
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches=bbox_inches,
                facecolor=facecolor,
                edgecolor=edgecolor,
            )
            logger.info("Plot saved to: %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)


class ImagePlotMixin:
    """Mixin providing enhanced image plotting capabilities."""

    @staticmethod
    def _create_dark_to_color_cmap(color_name: str) -> LinearSegmentedColormap:
        """Create colormap from black to specified color for dark background overlays.

        Args:
            color_name: Color name, matplotlib-compatible color string, or the name of
                an existing matplotlib colormap. Single-colour names produce a black→colour
                ramp; colormap names (e.g. 'hot', 'inferno', 'viridis') are returned as-is.
                Recommended bright colors for dark backgrounds:
                - 'cyan' - Excellent visibility
                - 'yellow' - Excellent visibility
                - 'orange' - Good red alternative
                - 'hot' - Black→red→yellow→white; good warm alternative
                - 'pink' - Bright red/magenta alternative
                - 'coral', 'salmon', 'tomato' - Various red/orange shades
                - 'lime' - Brighter green
                - 'hotpink' - Very bright pink
                Less visible on dark: 'red', 'blue', 'magenta' (use brighter alternatives)

        Returns:
            LinearSegmentedColormap: Colormap ranging from black (0,0,0) to target color.

        Example:
            >>> cmap = ImagePlotMixin._create_dark_to_color_cmap('cyan')
            >>> plt.imshow(data, cmap=cmap)
        """
        # Define common color mappings for microscopy
        # Brighter colors work better on dark backgrounds
        color_dict = {
            'cyan': (0, 1, 1),           # Bright - works great on dark
            'yellow': (1, 1, 0),         # Bright - works great on dark
            'magenta': (1, 0, 1),        # Medium brightness
            'green': (0, 1, 0),          # Bright
            'red': (1, 0, 0),            # Darker - can get lost
            'blue': (0, 0, 1),           # Darker - can get lost
            'orange': (1, 0.65, 0),      # Medium-bright - good alternative to red
            'lime': (0.75, 1, 0),        # Brighter than green
            'pink': (1, 0.4, 0.7),       # Brighter than magenta - good alternative to red
            'hotpink': (1, 0.41, 0.71),  # Similar to pink but standard name
            'deeppink': (1, 0.08, 0.58), # Darker pink
            'coral': (1, 0.5, 0.31),     # Lighter red/orange - good alternative
            'salmon': (1, 0.55, 0.41),   # Light orange/pink
            'tomato': (1, 0.39, 0.28),   # Bright red-orange
        }

        if color_name in color_dict:
            rgb = color_dict[color_name]
        else:
            # Try to parse as a matplotlib single colour first
            try:
                rgb = matplotlib.colors.to_rgb(color_name)
            except ValueError:
                # Not a colour — treat as an existing matplotlib colormap name
                # (e.g. 'hot', 'inferno', 'viridis') and return it directly
                return matplotlib.colormaps.get_cmap(color_name)

        # Create colormap: black (0,0,0) -> target color
        colors = [(0, 0, 0), rgb]
        cmap = LinearSegmentedColormap.from_list(
            f'black_to_{color_name}', colors, N=256
        )

        return cmap

    def multichannel_overlay_plot(
        self,
        axs,
        images: List[np.ndarray],
        cmaps: Optional[List[str]] = None,
        alphas: Optional[List[float]] = None,
        vmins: Optional[List[float]] = None,
        vmaxs: Optional[List[float]] = None,
        brightness_boost: Optional[List[float]] = None,
        pixelsize: float = 5.0,
        sbar: str = "on",
        scalebarsize: float = 1000,
        scalebarlabel: str = "1 μm",
        cbar: str = "off",
        cbarlabels: Optional[List[str]] = None,
        background_color: str = "black",
    ) -> matplotlib.axes.Axes:
        """Create multichannel overlay plot of rendered super-resolution images.

        Overlays multiple grayscale images with different colormaps using additive RGB
        blending on a dark background, matching the behaviour of fluorescence microscopy
        software (Fiji/ImageJ merge channels). Each channel's RGB contribution is summed
        pixel-wise and clipped to [0, 1], so isolated signals appear at full colormap
        brightness and overlapping signals saturate toward white.

        Args:
            axs: Axes object to plot on.
            images: List of 2D numpy arrays (rendered images, one per channel).
            cmaps: List of colormap names for each channel. Defaults to ['cyan', 'yellow']
                for 2 channels. Recommended bright colors for dark backgrounds:
                'cyan', 'yellow', 'orange', 'pink', 'coral', 'salmon', 'lime', 'hotpink'.
                Full matplotlib colormap names (e.g. 'hot') are also accepted.
                Avoid: 'red', 'blue', 'magenta' (too dark, use brighter alternatives).
            alphas: Deprecated. Has no effect and will be removed in a future version.
                Use brightness_boost to scale individual channels.
            vmins: List of minimum intensity values for each channel. If None, uses
                1st percentile for each image.
            vmaxs: List of maximum intensity values for each channel. If None, uses
                99th percentile for each image.
            brightness_boost: List of multiplicative brightness factors for each channel.
                Values > 1.0 increase brightness (useful for dim filamentous structures),
                values < 1.0 decrease brightness, 1.0 = no change. Default is None (all 1.0).
                Example: [1.0, 2.5] boosts channel 2 by 2.5x to match brighter channel 1.
            pixelsize: Pixel size in nanometers for scale bar calculation.
            sbar: Whether to show scale bar ('on' or 'off').
            scalebarsize: Scale bar size in nanometers.
            scalebarlabel: Scale bar label text.
            cbar: Whether to show colorbars ('on' or 'off'). Default is 'off'.
            cbarlabels: List of colorbar labels for each channel (only used if cbar='on').
            background_color: Background color ('black' or 'white'). Default is 'black'.

        Returns:
            Modified axes object.

        Example:
            >>> # Render two channels
            >>> _, img1 = render(locs_ch1, info, oversampling=20, blur_method='gaussian')
            >>> _, img2 = render(locs_ch2, info, oversampling=20, blur_method='gaussian')
            >>>
            >>> # Create overlay plot
            >>> plotter = PublicationPlotter()
            >>> fig, ax = plotter.create_figure(figsize=(10, 10))
            >>> plotter.multichannel_overlay_plot(
            ...     ax, [img1, img2],
            ...     cmaps=['cyan', 'yellow'],
            ...     pixelsize=5.0,
            ...     scalebarsize=1000,
            ...     scalebarlabel='1 μm'
            ... )
            >>>
            >>> # Boost brightness of dim filamentous channel
            >>> plotter.multichannel_overlay_plot(
            ...     ax, [img_globular, img_filaments],
            ...     cmaps=['cyan', 'red'],
            ...     brightness_boost=[1.0, 2.5],  # Boost filaments 2.5x
            ...     pixelsize=5.0
            ... )

        Performance Notes:
            - Images should be pre-rendered using render.py functions
            - All images must have identical dimensions
            - Uses percentile-based intensity scaling by default
        """
        n_channels = len(images)

        # Validate inputs
        if n_channels < 2:
            raise ValueError("Need at least 2 images for multichannel overlay")

        # Check all images have same shape
        ref_shape = images[0].shape
        for i, img in enumerate(images[1:], 1):
            if img.shape != ref_shape:
                raise ValueError(
                    f"Image {i} shape {img.shape} doesn't match image 0 shape {ref_shape}"
                )

        # Set default colormaps (bright colors that work well on dark backgrounds)
        if cmaps is None:
            default_cmaps = ['cyan', 'yellow', 'pink', 'lime', 'orange', 'hotpink']
            cmaps = default_cmaps[:n_channels]
        elif len(cmaps) != n_channels:
            raise ValueError(f"Expected {n_channels} colormaps, got {len(cmaps)}")

        # alphas is deprecated — warn if caller passed anything
        if alphas is not None:
            warnings.warn(
                "The 'alphas' parameter is deprecated and has no effect. "
                "Use 'brightness_boost' to scale individual channels.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Set default brightness boost
        if brightness_boost is None:
            brightness_boost = [1.0] * n_channels
        elif len(brightness_boost) != n_channels:
            raise ValueError(f"Expected {n_channels} brightness_boost values, got {len(brightness_boost)}")

        # Validate brightness boost values
        for boost in brightness_boost:
            if boost <= 0:
                raise ValueError(f"Brightness boost must be > 0, got {boost}")

        # Set default vmin/vmax using percentiles
        if vmins is None:
            vmins = [np.percentile(img.ravel(), 1.0) for img in images]
        elif len(vmins) != n_channels:
            raise ValueError(f"Expected {n_channels} vmin values, got {len(vmins)}")

        if vmaxs is None:
            vmaxs = [np.percentile(img.ravel(), 99.0) for img in images]
        elif len(vmaxs) != n_channels:
            raise ValueError(f"Expected {n_channels} vmax values, got {len(vmaxs)}")

        # Set up axes with dark background
        axs.set_facecolor(background_color)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_aspect('equal')

        # Build additive RGB composite and hidden per-channel artists for colorbars
        composite = np.zeros((*images[0].shape, 3), dtype=np.float32)
        image_artists = []
        for image, cmap_name, vmin, vmax, boost in zip(
            images, cmaps, vmins, vmaxs, brightness_boost
        ):
            cmap = self._create_dark_to_color_cmap(cmap_name)

            img_norm = np.clip((image - vmin) / (vmax - vmin), 0, 1)
            img_norm = np.clip(img_norm * boost, 0, 1)

            # Accumulate RGB contribution additively (ignore alpha channel of cmap output)
            composite += cmap(img_norm)[:, :, :3]

            # Hidden imshow used only for colorbar attachment
            im = axs.imshow(img_norm, cmap=cmap, origin='lower', visible=False)
            image_artists.append(im)

        # Display the additive composite at full opacity
        axs.imshow(
            np.clip(composite, 0, 1),
            origin='lower',
            interpolation='bilinear',
        )

        # Add colorbars if requested
        if cbar == "on":
            divider = make_axes_locatable(axs)

            # Position colorbars side by side on the right
            for i, im in enumerate(image_artists):
                # Calculate padding (first colorbar at 0.05, subsequent ones offset)
                pad = 0.05 + i * 0.10

                cax = divider.append_axes("right", size="2%", pad=pad)
                colorbar = plt.colorbar(im, cax=cax)

                # Set label if provided
                if cbarlabels and i < len(cbarlabels):
                    label_color = 'white' if background_color == 'black' else 'black'
                    colorbar.set_label(cbarlabels[i], color=label_color)

                # Style ticks for dark background
                if background_color == 'black':
                    colorbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(colorbar.ax.axes, 'yticklabels'), color='white')

        # Add scale bar if requested
        if sbar == "on":
            scalebar_color = 'white' if background_color == 'black' else 'black'

            # Use add_scalebar method from BasePlotter
            self.add_scalebar(
                axs,
                pixelsize=pixelsize,
                length_nm=scalebarsize,
                location='lower right',
                color=scalebar_color,
                label=scalebarlabel,
            )

        return axs

    def make_animated_gif(
        self,
        image: np.ndarray,
        filename: str,
        vmin: float = 0,
        vmax: float = 150,
        pixelsize: float = DriftConstants.XIMEA_PIXEL_SIZE_NM,
        scalebarsize: float = 300,
        scalebarlabel: str = "300 nm",
        label: str = "",
        fontsz: int = 6,
        cbarlabel: str = "# of photoelectrons",
        cbar: bool = False,
        width: float = 3,
        height: float = 3,
        fps: int = 25,
        dpi: int = 400,
    ) -> None:
        """Create animated GIF from image sequence.

        Supports both grayscale and RGB image stacks. For grayscale images,
        a colormap is applied. For RGB images, the colors are displayed directly.

        Args:
            image: Image stack with shape (n_frames, H, W) for grayscale
                   or (n_frames, H, W, 3) for RGB.
            filename: Output filename (should end in .gif).
            vmin: Minimum display value (grayscale only).
            vmax: Maximum display value (grayscale only).
            pixelsize: Pixel size in nm.
            scalebarsize: Scale bar size in nm.
            scalebarlabel: Scale bar label.
            label: Image label (displayed in top-left corner).
            fontsz: Font size for label.
            cbarlabel: Colorbar label (grayscale only).
            cbar: Whether to show colorbar (grayscale only).
            width: Figure width in inches.
            height: Figure height in inches.
            fps: Frames per second for the GIF.
            dpi: Resolution of the output GIF.
        """
        n_frames = image.shape[0]

        # Detect if image is RGB (shape: n_frames, H, W, 3)
        is_rgb = image.ndim == 4 and image.shape[-1] == 3

        fig, ax = plt.subplots(figsize=(width, height), layout=None)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if cbar and not is_rgb:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

        def animate(i):
            ax.clear()

            if is_rgb:
                # RGB image - display directly
                # Ensure values are in valid range for display
                frame = image[i, :, :, :]
                if frame.dtype != np.uint8:
                    # Normalize to 0-1 range if not already uint8
                    frame = np.clip(frame, vmin, vmax)
                    frame = (frame - vmin) / (vmax - vmin)
                im = ax.imshow(frame)
            else:
                # Grayscale image - apply colormap
                im = ax.imshow(image[i, :, :], vmin=vmin, vmax=vmax, cmap="gist_gray")

                if cbar:
                    fig.colorbar(im, orientation="vertical", cax=cax)
                    cax.set_ylabel(cbarlabel, rotation=270, labelpad=8, fontsize=7)

            # Scale bar
            pixvals = scalebarsize / pixelsize
            scalebar = AnchoredSizeBar(
                ax.transData,
                pixvals,
                scalebarlabel,
                "lower right",
                pad=0.5,
                color="white",
                frameon=False,
                size_vertical=(1 / width),
            )
            ax.add_artist(scalebar)

            # Label
            if label:
                xy_coord = int(image.shape[1] * 0.05)
                ax.annotate(
                    label,
                    xy=(xy_coord, xy_coord),
                    xytext=(xy_coord, xy_coord),
                    xycoords="data",
                    color="white",
                    fontsize=fontsz + 1,
                )
            ax.axis("off")
            return [im]

        interval = 1000 / fps  # Convert fps to interval in ms
        ani = FuncAnimation(
            fig, animate, interval=interval, blit=True, repeat=True, frames=n_frames
        )
        ani.save(
            filename,
            dpi=dpi,
            writer=PillowWriter(fps=fps),
            savefig_kwargs={"transparent": True},
        )
        plt.close(fig)

    def make_animated_gif_multipanel(
        self,
        fig,
        axs: np.ndarray,
        plot_types: np.ndarray,
        xpositions: np.ndarray,
        ypositions: np.ndarray,
        images_for_figures: np.ndarray,
        n_pixels: int = 8,
        n_frames: int = None,
        filename: str = "output.gif",
        fps: int = 24,
        pattern: str = "BGGR",
        vmin: float = 0.1,
        vmax: float = 99.9,
        pixelsize: float = DriftConstants.XIMEA_PIXEL_SIZE_NM,
        scalebarsize: float = 300,
        scalebarlabel: str = "300 nm",
        marker_color = "white",
        marker_size: float = 150,
        bayer_fontsize: float = 20,
        dpi: int = 400,
    ) -> None:
        """Create a multipanel animated GIF with Bayer pattern and camera image panels.

        Produces a looping GIF showing a molecule moving through a Bayer pattern grid
        alongside the corresponding camera image for each dye channel.  Designed to
        work on dark/transparent backgrounds.

        Args:
            fig: Matplotlib figure containing the axes.
            axs: 2-D array of axes with shape ``(nrows, ncols)``.
            plot_types: String array matching ``axs`` shape.  Each entry is one of:
                ``"pattern"`` — draw Bayer grid + moving molecule marker;
                ``"image"`` — show camera image for the corresponding dye;
                ``"off"`` — hide this axes.
            xpositions: Molecule x position per frame in **pixel** coordinates,
                shape ``(n_frames,)``.
            ypositions: Molecule y position per frame in **pixel** coordinates,
                shape ``(n_frames,)``.
            images_for_figures: Camera image stack, shape
                ``(n_dyes, n_frames, H, W)``.  Column index in ``plot_types``
                maps directly to dye index (``j`` → ``images_for_figures[j]``).
            n_pixels: Size of the Bayer pattern grid shown (grid spans 0 to
                ``n_pixels``).
            n_frames: Number of frames to animate.  Defaults to
                ``images_for_figures.shape[1]``.
            filename: Output ``.gif`` path.
            fps: Frames per second.
            pattern: 4-character Bayer pattern string (row-major, top-down).
                Defaults to ``"BGGR"`` matching ``MaskFunctions.get_masks()``.
            vmin: Lower percentile clip for image display.
            vmax: Upper percentile clip for image display.
            pixelsize: Camera pixel size in nm (used for scale bars).
            scalebarsize: Image-panel scale bar length in nm.
            scalebarlabel: Image-panel scale bar label.
            marker_color: Colour of the molecule position marker.  Either a single
                colour string (applied to all pattern panels) or a list of colours,
                one per column (e.g. ``["cornflowerblue", "orange", "red"]``).
            marker_size: Size of the molecule position marker (matplotlib ``s``).
            bayer_fontsize: Font size for B / G / R labels in the pattern panel.
            dpi: Output GIF resolution.
        """
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        from matplotlib.font_manager import FontProperties

        if n_frames is None:
            n_frames = images_for_figures.shape[1]

        fp = FontProperties()
        fp.set_size(8)

        # Per-dye percentile clip (computed once across all frames)
        img_vmin = np.percentile(images_for_figures, vmin)
        img_vmax = np.percentile(images_for_figures, vmax)

        # ── per-frame render: clear and redraw every panel each frame ───────
        # This guarantees exactly one scatter point per pattern panel per frame.
        # y-positions are flipped (n_pixels - y) to match the image display
        # convention (origin='lower' in imshow has y=0 at bottom, while
        # plot_bayer_pattern uses set_ylim(size, 0) with y=0 at top).

        def animate(k):
            for i in range(axs.shape[0]):
                for j in range(axs.shape[1]):
                    ptype = plot_types[i, j]

                    if ptype == "pattern":
                        col_color = (
                            marker_color[j % len(marker_color)]
                            if isinstance(marker_color, (list, tuple, np.ndarray))
                            else marker_color
                        )
                        axs[i, j].clear()
                        plot_bayer_pattern(
                            axs[i, j],
                            pattern=pattern,
                            size=n_pixels,
                            fontsize=bayer_fontsize,
                        )
                        axs[i, j].scatter(
                            xpositions[k],
                            n_pixels - ypositions[k],
                            edgecolor=None,
                            facecolor=col_color,
                            s=marker_size,
                            zorder=np.inf,
                        )
                        pixvals = 100 / pixelsize
                        axs[i, j].add_artist(AnchoredSizeBar(
                            axs[i, j].transData,
                            pixvals, "100 nm", "lower center",
                            pad=0.5, color="white", frameon=False,
                            size_vertical=0.3 / n_pixels,
                            fontproperties=fp,
                        ))

                    elif ptype == "image":
                        axs[i, j].clear()
                        axs[i, j].imshow(
                            images_for_figures[j, k, :, :],
                            cmap="gist_gray",
                            vmin=img_vmin,
                            vmax=img_vmax,
                        )
                        pixvals = scalebarsize / pixelsize
                        axs[i, j].add_artist(AnchoredSizeBar(
                            axs[i, j].transData,
                            pixvals, scalebarlabel, "lower center",
                            pad=0.5, color="white", frameon=False,
                            size_vertical=0.5 / images_for_figures.shape[2],
                            fontproperties=fp,
                        ))
                        axs[i, j].axis("off")

                    else:
                        axs[i, j].axis("off")

            return []

        # Render frames manually and save with PIL disposal=2.
        # PillowWriter does not expose the GIF disposal parameter; without
        # disposal=2 ("restore to background before next frame") the player
        # composites each new frame onto the previous one, leaving ghost
        # scatter dots in transparent regions of earlier frames.
        import io
        from PIL import Image

        duration_ms = int(round(1000 / fps))
        pil_frames = []
        for k in range(n_frames):
            animate(k)
            buf = io.BytesIO()
            fig.savefig(buf, format="rgba", dpi=dpi)
            buf.seek(0)
            w = int(round(fig.get_figwidth() * dpi))
            h = int(round(fig.get_figheight() * dpi))
            data = np.frombuffer(buf.read(), dtype=np.uint8).reshape(h, w, 4)
            pil_frames.append(Image.fromarray(data, "RGBA"))

        plt.close(fig)

        # Convert RGBA → RGB (composite transparent regions over black) then
        # quantise to a single global palette so colours are consistent across
        # frames.  Images produced by quantize() carry all the internal PIL
        # metadata required for multi-frame GIF writing; images built with
        # Image.fromarray() do not, which causes only the first frame to land
        # in the output file.
        def _to_rgb(rgba_img):
            bg = Image.new("RGB", rgba_img.size, (0, 0, 0))
            bg.paste(rgba_img.convert("RGB"), mask=rgba_img.split()[3])
            return bg

        rgb_frames = [_to_rgb(f) for f in pil_frames]

        # Build a global 256-colour palette from the first frame.
        first_p = rgb_frames[0].quantize(colors=256)
        gif_frames = [first_p] + [
            f.quantize(colors=256, palette=first_p)
            for f in rgb_frames[1:]
        ]

        gif_frames[0].save(
            filename,
            save_all=True,
            append_images=gif_frames[1:],
            loop=0,
            duration=duration_ms,
            disposal=2,
            optimize=False,
        )


class TernaryPlotMixin:
    """Mixin for creating ternary (3-component) plots.

    Provides methods for plotting RGB color data on ternary diagrams
    using the mpltern library. Handles both scatter and density plots.
    """

    def create_ternary_plot(
        self,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        colors: Optional[np.ndarray] = None,
        marker_size: float = 10,
        marker_alpha: float = 1.0,
        edge_width: float = 0,
        title: Optional[str] = None,
        tlabel: str = 'pixel 1 /%',
        llabel: str = 'pixel 3 /%',
        rlabel: str = 'pixel 2 /%',
        show_grid: bool = True,
        maj_loc: float = 0.2,
        min_loc: float = 0.1,
        figsize: Tuple[float, float] = (6, 5),
        rasterized: bool = False,
        black_background: bool = True,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Create a standalone ternary scatter plot for RGB data.

        Creates a single-panel ternary plot with a clean layout and, by default,
        a black background inside the triangle.  RGB values are the t/l/r
        coordinates in that order (top, left, right).

        Args:
            R: T (top) values — typically the red channel fraction.
            G: L (left) values — typically the green channel fraction.
            B: R (right) values — typically the blue channel fraction.
            colors: Optional RGBA array of per-point colors.  If None the
                default matplotlib color cycle is used.
            marker_size: Scatter marker size (default: 10).
            marker_alpha: Marker opacity (default: 1.0).
            edge_width: Marker edge linewidth; 0 = no edge (default: 0).
            title: Optional plot title.
            tlabel: Label for the top (T) axis (default: 'pixel 1 /%').
            llabel: Label for the left (L) axis (default: 'pixel 3 /%').
            rlabel: Label for the right (R) axis (default: 'pixel 2 /%').
            show_grid: Whether to show dashed grid lines (default: True).
            maj_loc: Major tick / grid interval (default: 0.2).
            min_loc: Minor tick interval (default: 0.1).
            figsize: Figure size (width, height) in inches (default: (6, 5)).
            rasterized: Rasterize scatter points for smaller SVG (default: False).
            black_background: Fill triangle interior with black (default: True).
            **kwargs: Extra keyword arguments forwarded to ``ax.scatter()``.

        Returns:
            Tuple of (fig, ax) where ax is a mpltern TernaryAxes.

        Example:
            >>> fig, ax = plotter.create_ternary_plot(
            ...     A_R * 100, A_G * 100, A_B * 100,
            ...     colors=point_colors,
            ...     marker_size=10,
            ... )
            >>> fig.savefig('ternary.svg', format='svg', dpi=600)
        """
        try:
            import mpltern  # noqa: F401 — registers projection
        except ImportError:
            raise ImportError(
                "mpltern is required for ternary plots. Install with: pip install mpltern"
            )

        R = np.asarray(R)
        G = np.asarray(G)
        B = np.asarray(B)
        if len(R) != len(G) or len(R) != len(B):
            raise ValueError("R, G, B arrays must have the same length")

        # Create a clean figure with only the ternary axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='ternary')

        if black_background:
            ax.set_facecolor('black')

        # Axis labels
        ax.set_tlabel(tlabel)
        ax.set_llabel(llabel)
        ax.set_rlabel(rlabel)

        # Tick locators
        from matplotlib.ticker import MultipleLocator
        for taxis in (ax.taxis, ax.laxis, ax.raxis):
            taxis.set_major_locator(MultipleLocator(maj_loc))
            taxis.set_minor_locator(MultipleLocator(min_loc))

        # Grid
        if show_grid:
            ax.grid(lw=0.5, alpha=0.25, ls='--', which='both', axis='both')

        # Scatter
        scatter_kw = dict(
            s=marker_size,
            alpha=marker_alpha,
            edgecolors='None',
            lw=edge_width,
            marker='o',
            rasterized=rasterized,
        )
        scatter_kw.update(kwargs)
        if colors is not None:
            ax.scatter(R, G, B, c=colors, **scatter_kw)
        else:
            ax.scatter(R, G, B, **scatter_kw)

        if title:
            ax.set_title(title, pad=20)

        return fig, ax

    def create_ternary_density(
        self,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        gridsize: int = 50,
        cmap: str = 'viridis',
        show_colorbar: bool = True,
        title: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        show_grid: bool = True,
        grid_spacing: float = 0.1,
        figsize: Tuple[float, float] = (7, 5),
        log_scale: bool = False,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Create a ternary density plot (hexbin) for RGB data.

        This method creates a single-panel ternary plot showing the density
        distribution of R, G, B values using hexagonal binning.

        Args:
            R: Red channel values (normalized, 0-1)
            G: Green channel values (normalized, 0-1)
            B: Blue channel values (normalized, 0-1)
            gridsize: Number of hexagons in x direction (default: 50)
            cmap: Colormap name (default: 'viridis')
            show_colorbar: Whether to show colorbar (default: True)
            title: Plot title (optional)
            labels: Dictionary with keys 'R', 'G', 'B' for axis labels (optional)
            show_grid: Whether to show grid lines (default: True)
            grid_spacing: Spacing between grid lines (default: 0.1)
            figsize: Figure size as (width, height) (default: (7, 5))
            log_scale: Use logarithmic color scale (default: False)
            **kwargs: Additional arguments passed to ax.hexbin()

        Returns:
            Tuple of (fig, ax) where ax is a ternary axis

        Example:
            >>> plotter = PublicationPlotter()
            >>> fig, ax = plotter.create_ternary_density(
            ...     R_norm, G_norm, B_norm,
            ...     gridsize=100,
            ...     cmap='hot',
            ...     title='Color Density Distribution'
            ... )

        Notes:
            - Requires mpltern: `pip install mpltern`
            - RGB values should be normalized (sum to 1 for each point)
            - Use larger gridsize for smoother density visualization
        """
        try:
            import mpltern
        except ImportError:
            raise ImportError(
                "mpltern is required for ternary plots. Install with: pip install mpltern"
            )

        # Validate inputs
        if len(R) != len(G) or len(R) != len(B):
            raise ValueError("R, G, B arrays must have the same length")

        # Convert to numpy arrays if needed
        R = np.asarray(R)
        G = np.asarray(G)
        B = np.asarray(B)

        # Check for and handle normalization
        totals = R + G + B
        if not np.allclose(totals, 1.0, atol=1e-6):
            # Normalize
            R = R / totals
            G = G / totals
            B = B / totals
            logger.warning("RGB values were not normalized. Automatically normalized to sum=1")

        # Create figure with ternary projection
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='ternary')

        # Set up axis labels with colors
        default_labels = {'R': 'Red', 'G': 'Green', 'B': 'Blue'}
        if labels is not None:
            default_labels.update(labels)

        # Note: scatter(R, G, B) means t=R (top), l=G (left), r=B (right)
        ax.set_tlabel(default_labels['R'], color='darkred', fontsize=12)
        ax.set_llabel(default_labels['G'], color='darkgreen', fontsize=12)
        ax.set_rlabel(default_labels['B'], color='darkblue', fontsize=12)

        # Color the tick marks and tick labels
        ax.taxis.set_tick_params(colors='darkred', which='both', length=5, width=1.5)
        ax.laxis.set_tick_params(colors='darkgreen', which='both', length=5, width=1.5)
        ax.raxis.set_tick_params(colors='darkblue', which='both', length=5, width=1.5)

        # Color the axis lines (spines)
        # Note: In ternary plots, each axis runs along the OPPOSITE side:
        # - taxis (top vertex, R) runs along bottom edge = 'tside'
        # - laxis (left vertex, G) runs along right edge = 'rside'
        # - raxis (right vertex, B) runs along left edge = 'lside'
        ax.spines['lside'].set_color('darkred')      # Left edge = R axis
        ax.spines['rside'].set_color('darkgreen')    # Right edge = G axis
        ax.spines['tside'].set_color('darkblue')     # Bottom edge = B axis
        ax.spines['lside'].set_linewidth(1.5)
        ax.spines['rside'].set_linewidth(1.5)
        ax.spines['tside'].set_linewidth(1.5)

        # Set up grid with colored gridlines
        if show_grid:
            from matplotlib.ticker import MultipleLocator
            for axis in [ax.taxis, ax.laxis, ax.raxis]:
                axis.set_major_locator(MultipleLocator(grid_spacing))

            # Color the gridlines to match the axes
            ax.grid(True, which='major', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.taxis.grid(color='darkred', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.laxis.grid(color='darkgreen', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.raxis.grid(color='darkblue', alpha=0.3, linestyle='--', linewidth=0.5)

        # Create hexbin density plot
        # Note: mpltern uses (t, l, r) ordering where t=top, l=left, r=right
        # For RGB: scatter(R, G, B) means t=R (top), l=G (left), r=B (right)
        if log_scale:
            bins = 'log'
        else:
            bins = None

        hexbin = ax.hexbin(
            R, G, B,
            gridsize=gridsize,
            cmap=cmap,
            bins=bins,
            edgecolors='none',
            rasterized=True,
            **kwargs
        )

        # Add colorbar
        if show_colorbar:
            cbar = plt.colorbar(hexbin, ax=ax, pad=0.05)
            cbar.set_label('Count' if not log_scale else 'Count (log scale)', rotation=270, labelpad=20)

        # Set title
        if title:
            ax.set_title(title, pad=20)

        # Adjust layout
        _safe_tight_layout(fig)

        return fig, ax

    def plot_ternary_kde_contours(
        self,
        ax,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        color: str = 'blue',
        label: Optional[str] = None,
        levels: Union[List[float], str] = [0.5, 0.9, 0.99],
        bandwidth: Union[float, str] = 'scott',
        linewidths: Union[float, List[float]] = 2.0,
        linestyles: Union[str, List[str]] = 'solid',
        alpha: float = 0.8,
        grid_resolution: int = 100,
        **kwargs
    ) -> None:
        """Plot KDE contour lines on an existing ternary axis.

        This function calculates a 2D kernel density estimate in (R, G) space
        and plots confidence contours on a ternary diagram. This provides a
        cleaner visualization of dye separability compared to scatter plots with
        alpha transparency, avoiding the visual "overlap deception" issue.

        Args:
            ax: mpltern TernaryAxes object to plot on
            R: Red channel values (normalized, 0-1)
            G: Green channel values (normalized, 0-1)
            B: Blue channel values (normalized, 0-1) - used for validation only
            color: Color for contour lines (default: 'blue')
            label: Label for legend (optional)
            levels: Contour levels as:
                   - List of floats: [0.5, 0.9, 0.99] for confidence levels
                   - 'auto': Automatically choose N levels
                   - int: Number of levels to auto-generate
            bandwidth: KDE bandwidth selection:
                      - 'scott': Scott's rule (default, recommended)
                      - 'silverman': Silverman's rule
                      - float: Manual bandwidth value
            linewidths: Line width(s) for contours. Can be:
                       - Single float: Same width for all levels
                       - List of floats: Width per level (inner to outer)
            linestyles: Line style(s) for contours ('solid', 'dashed', 'dotted')
            alpha: Transparency for contour lines (0-1, default: 0.8)
            grid_resolution: Resolution of KDE evaluation grid (default: 100)
            **kwargs: Additional arguments passed to ax.tricontour()

        Returns:
            None (modifies ax in place)

        Example:
            >>> import matplotlib.pyplot as plt
            >>> import mpltern
            >>> from PlottingBase import PublicationPlotter
            >>>
            >>> # Create ternary figure
            >>> fig = plt.figure(figsize=(8, 6))
            >>> ax = fig.add_subplot(projection='ternary')
            >>>
            >>> # Plot KDE contours for multiple dyes
            >>> plotter = PublicationPlotter()
            >>> plotter.plot_ternary_kde_contours(
            ...     ax, R_dye1, G_dye1, B_dye1,
            ...     color='red', label='ATTO 655',
            ...     levels=[0.5, 0.9, 0.99],
            ...     linewidths=[1, 2, 3]
            ... )
            >>> plotter.plot_ternary_kde_contours(
            ...     ax, R_dye2, G_dye2, B_dye2,
            ...     color='blue', label='JF646',
            ...     levels=[0.5, 0.9, 0.99],
            ...     linewidths=[1, 2, 3]
            ... )
            >>> ax.legend()
            >>> plt.show()

        Notes:
            - Requires scipy for KDE calculation
            - Requires mpltern for ternary plotting
            - KDE is computed in 2D (R, G) space since B = 1 - R - G
            - Confidence levels [0.5, 0.9, 0.99] correspond to approximately
              [1.18σ, 4.60σ, 9.21σ] for 2D Gaussian distributions
            - Use varying linewidths to emphasize core vs tail of distribution
            - This method avoids the "alpha transparency deception" where
              overlapping scatter points visually exaggerate overlap

        See Also:
            create_ternary_plot: Basic ternary scatter plot
            create_ternary_density: Hexbin density plot
        """
        from scipy.stats import gaussian_kde

        # Validate inputs
        if len(R) != len(G) or len(R) != len(B):
            raise ValueError("R, G, B arrays must have the same length")

        # Convert to numpy arrays
        R = np.asarray(R, dtype=np.float64)
        G = np.asarray(G, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        # Remove any NaN or inf values
        valid_mask = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
        if not np.all(valid_mask):
            n_invalid = (~valid_mask).sum()
            logger.warning("Removed %s invalid values from KDE calculation", n_invalid)
            R = R[valid_mask]
            G = G[valid_mask]
            B = B[valid_mask]

        if len(R) < 10:
            logger.warning("Only %d valid points for KDE. Skipping contour plot.", len(R))
            return

        # Check normalization (should sum to 1)
        totals = R + G + B
        if not np.allclose(totals, 1.0, atol=1e-3):
            logger.warning("RGB values not normalized (sum=%.3f). Normalizing...", np.mean(totals))
            R = R / totals
            G = G / totals
            B = B / totals

        # Compute KDE in 2D (R, G) space
        # Note: B = 1 - R - G is redundant, so we only need 2D
        data = np.vstack([R, G])

        try:
            if bandwidth == 'scott':
                kde = gaussian_kde(data, bw_method='scott')
            elif bandwidth == 'silverman':
                kde = gaussian_kde(data, bw_method='silverman')
            elif isinstance(bandwidth, (int, float)):
                kde = gaussian_kde(data, bw_method=float(bandwidth))
            else:
                raise ValueError(f"Invalid bandwidth: {bandwidth}")
        except Exception as e:
            logger.error("Error creating KDE: %s. Data shape: %s, R range: [%.3f, %.3f], G range: [%.3f, %.3f]",
                         e, data.shape, R.min(), R.max(), G.min(), G.max())
            return

        # Create evaluation grid in (R, G) space
        # Grid covers valid ternary space: R + G <= 1, R >= 0, G >= 0
        r_grid = np.linspace(0, 1, grid_resolution)
        g_grid = np.linspace(0, 1, grid_resolution)
        R_grid, G_grid = np.meshgrid(r_grid, g_grid)

        # Mask invalid points (where R + G > 1)
        valid_ternary = (R_grid + G_grid) <= 1.0
        B_grid = 1.0 - R_grid - G_grid

        # Evaluate KDE only on valid ternary points
        valid_indices = valid_ternary.ravel()
        grid_points_all = np.vstack([R_grid.ravel(), G_grid.ravel()])
        grid_points_valid = grid_points_all[:, valid_indices]

        try:
            kde_values_valid = kde(grid_points_valid)
        except Exception as e:
            logger.error("Error evaluating KDE: %s", e)
            return

        # Create full KDE array with zeros for invalid points
        kde_values_full = np.zeros(R_grid.size)
        kde_values_full[valid_indices] = kde_values_valid
        kde_values = kde_values_full.reshape(R_grid.shape)

        # Determine contour levels
        if isinstance(levels, str) and levels == 'auto':
            # Auto-select levels based on NON-ZERO KDE values
            # This is critical for tight clusters where most grid points have zero density
            valid_kde = kde_values[valid_ternary]
            nonzero_kde = valid_kde[valid_kde > 1e-10]  # Exclude numerical zeros

            if len(nonzero_kde) < 10:
                logger.warning("Only %d non-zero KDE values. Skipping contour plot.", len(nonzero_kde))
                return

            # Use logarithmic spacing for better contour distribution
            min_log = np.log10(np.percentile(nonzero_kde, 1))
            max_log = np.log10(np.percentile(nonzero_kde, 99))
            levels_to_plot = np.logspace(min_log, max_log, 5)
        elif isinstance(levels, int):
            # Generate N evenly-spaced levels
            valid_kde = kde_values[valid_ternary]
            levels_to_plot = np.linspace(valid_kde.min(), valid_kde.max(), levels)
        else:
            # Convert confidence levels to density levels
            # For 2D Gaussian: P(inside ellipse) = 1 - exp(-r²/2)
            # Solving: r² = -2*ln(1-P)
            # Density at radius r: exp(-r²/2) / (2π σ²)
            # But we use empirical approach: percentiles of KDE values
            valid_kde = kde_values[valid_ternary]
            levels_array = np.asarray(levels)

            # Map confidence levels to KDE density thresholds
            # Use only non-zero KDE values (many grid points are outside data region)
            nonzero_kde = valid_kde[valid_kde > 0]

            if len(nonzero_kde) < 10:
                logger.warning("Only %d non-zero KDE values. Using all valid values.", len(nonzero_kde))
                nonzero_kde = valid_kde

            # Sort KDE values (highest to lowest)
            sorted_kde = np.sort(nonzero_kde)[::-1]
            n_points = len(sorted_kde)

            levels_to_plot = []
            for conf in levels_array:
                # For confidence level conf, we want the threshold where
                # conf% of the probability mass is above it
                # For a 2D Gaussian: 50% → ~0.39 of peak, 90% → ~0.105 of peak, 99% → ~0.018 of peak
                # Use this as approximation
                if conf == 0.5:
                    # 50% contour: roughly 0.4 of maximum
                    level_val = sorted_kde[0] * 0.4
                elif conf == 0.9:
                    # 90% contour: roughly 0.1 of maximum
                    level_val = sorted_kde[0] * 0.1
                elif conf == 0.99:
                    # 99% contour: roughly 0.02 of maximum
                    level_val = sorted_kde[0] * 0.02
                else:
                    # General case: use cumulative probability mass
                    # Calculate cumulative sum of sorted densities (high to low)
                    cumsum = np.cumsum(sorted_kde)
                    cumsum_normalized = cumsum / cumsum[-1]

                    # Find threshold where cumulative probability = confidence level
                    idx = np.searchsorted(cumsum_normalized, conf)
                    if idx >= len(sorted_kde):
                        idx = len(sorted_kde) - 1
                    level_val = sorted_kde[idx]

                levels_to_plot.append(level_val)
            levels_to_plot = np.array(levels_to_plot)

        # Ensure levels are sorted (lowest to highest) for proper contour plotting
        levels_to_plot = np.sort(levels_to_plot)

        # Handle linewidths and linestyles
        if isinstance(linewidths, (int, float)):
            linewidths_list = [linewidths] * len(levels_to_plot)
        else:
            linewidths_list = list(linewidths)
            if len(linewidths_list) < len(levels_to_plot):
                # Repeat last value
                linewidths_list += [linewidths_list[-1]] * (len(levels_to_plot) - len(linewidths_list))

        if isinstance(linestyles, str):
            linestyles_list = [linestyles] * len(levels_to_plot)
        else:
            linestyles_list = list(linestyles)
            if len(linestyles_list) < len(levels_to_plot):
                linestyles_list += [linestyles_list[-1]] * (len(levels_to_plot) - len(linestyles_list))

        # Plot contours on ternary axis
        # mpltern tricontour expects (t, l, r) = (R, G, B) and values
        try:
            contour = ax.tricontour(
                R_grid.ravel(),
                G_grid.ravel(),
                B_grid.ravel(),
                kde_values.ravel(),
                levels=levels_to_plot,
                colors=[color] * len(levels_to_plot),
                linewidths=linewidths_list,
                linestyles=linestyles_list,
                alpha=alpha,
                **kwargs
            )

            # Add label for legend (only to first contour level)
            if label is not None:
                # Create a dummy line for legend
                from matplotlib.lines import Line2D
                legend_line = Line2D([0], [0], color=color, linewidth=linewidths_list[0],
                                    linestyle=linestyles_list[0], alpha=alpha, label=label)
                # Store it for potential legend creation
                if not hasattr(ax, '_kde_legend_handles'):
                    ax._kde_legend_handles = []
                ax._kde_legend_handles.append(legend_line)

        except Exception as e:
            logger.error("Error plotting contours: %s. Levels: %s, KDE range: [%.6f, %.6f]",
                         e, levels_to_plot, np.nanmin(kde_values), np.nanmax(kde_values))
            return

    def plot_ternary_kde(
        self,
        ax,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        bandwidth: Union[float, str] = 'scott',
        grid_resolution: int = 100,
        cmap: str = 'viridis',
        n_levels: int = 20,
        show_colorbar: bool = True,
        colorbar_label: str = 'Density',
        **kwargs
    ):
        """Plot filled KDE density on an existing ternary axis.

        This method adds a smooth kernel density estimate plot to an existing ternary axis.
        Use this when you want a smoother, more publication-ready density visualization
        compared to hexbin.

        Args:
            ax: Existing ternary axis (must have projection='ternary')
            R: Red channel values (normalized, 0-1)
            G: Green channel values (normalized, 0-1)
            B: Blue channel values (normalized, 0-1)
            bandwidth: KDE bandwidth selection:
                      - 'scott': Scott's rule (default, recommended)
                      - 'silverman': Silverman's rule
                      - float: Manual bandwidth value
            grid_resolution: Resolution of KDE evaluation grid (default: 100)
            cmap: Colormap name (default: 'viridis')
            n_levels: Number of contour levels (default: 20)
            show_colorbar: Whether to show colorbar (default: True)
            colorbar_label: Label for colorbar (default: 'Density')
            **kwargs: Additional arguments passed to ax.tricontourf()

        Returns:
            contourf: The TriContourSet object from tricontourf

        Example:
            >>> import matplotlib.pyplot as plt
            >>> import mpltern
            >>> fig = plt.figure(figsize=(12, 3))
            >>> ax1 = fig.add_subplot(1, 3, 1)  # Regular plot
            >>> ax2 = fig.add_subplot(1, 3, 2)  # Regular plot
            >>> ax3 = fig.add_subplot(1, 3, 3, projection='ternary')  # Ternary plot
            >>>
            >>> # Add KDE density to the ternary axis
            >>> plotter = PublicationPlotter()
            >>> cf = plotter.plot_ternary_kde(
            ...     ax3, R_data, G_data, B_data,
            ...     bandwidth='scott',
            ...     cmap='hot'
            ... )

        Notes:
            - Requires scipy for KDE calculation
            - Requires mpltern: `pip install mpltern`
            - RGB values should be normalized (sum to 1 for each point)
            - If not normalized, the function will normalize them automatically
            - The axis must already exist with projection='ternary'
            - KDE is computed in 2D (R, G) space since B = 1 - R - G
        """
        from scipy.stats import gaussian_kde

        # Validate inputs
        if len(R) != len(G) or len(R) != len(B):
            raise ValueError("R, G, B arrays must have the same length")

        # Convert to numpy arrays
        R = np.asarray(R, dtype=np.float64)
        G = np.asarray(G, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        # Remove any NaN or inf values
        valid_mask = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
        if not np.all(valid_mask):
            n_invalid = (~valid_mask).sum()
            logger.warning("Removed %s invalid values from KDE calculation", n_invalid)
            R = R[valid_mask]
            G = G[valid_mask]
            B = B[valid_mask]

        if len(R) < 10:
            logger.warning("Only %d valid points for KDE. Returning None.", len(R))
            return None

        # Check normalization (should sum to 1)
        totals = R + G + B
        if not np.allclose(totals, 1.0, atol=1e-6):
            # Normalize
            R = R / totals
            G = G / totals
            B = B / totals

        # Compute KDE in 2D (R, G) space
        data = np.vstack([R, G])

        try:
            if bandwidth == 'scott':
                kde = gaussian_kde(data, bw_method='scott')
            elif bandwidth == 'silverman':
                kde = gaussian_kde(data, bw_method='silverman')
            elif isinstance(bandwidth, (int, float)):
                kde = gaussian_kde(data, bw_method=float(bandwidth))
            else:
                raise ValueError(f"Invalid bandwidth: {bandwidth}")
        except Exception as e:
            logger.error("Error creating KDE: %s", e)
            return None

        # Create evaluation grid in (R, G) space
        r_grid = np.linspace(0, 1, grid_resolution)
        g_grid = np.linspace(0, 1, grid_resolution)
        R_grid, G_grid = np.meshgrid(r_grid, g_grid)

        # Mask invalid points (where R + G > 1)
        valid_ternary = (R_grid + G_grid) <= 1.0
        B_grid = 1.0 - R_grid - G_grid

        # Evaluate KDE only on valid ternary points
        valid_indices = valid_ternary.ravel()
        grid_points_all = np.vstack([R_grid.ravel(), G_grid.ravel()])
        grid_points_valid = grid_points_all[:, valid_indices]

        try:
            kde_values_valid = kde(grid_points_valid)
        except Exception as e:
            logger.error("Error evaluating KDE: %s", e)
            return None

        # Create full KDE array with zeros for invalid points
        kde_values_full = np.zeros(R_grid.size)
        kde_values_full[valid_indices] = kde_values_valid
        kde_values = kde_values_full.reshape(R_grid.shape)

        # Plot filled contours on ternary axis
        # mpltern tricontourf expects (t, l, r) = (R, G, B) and values
        try:
            contourf = ax.tricontourf(
                R_grid.ravel(),
                G_grid.ravel(),
                B_grid.ravel(),
                kde_values.ravel(),
                levels=n_levels,
                cmap=cmap,
                **kwargs
            )
        except Exception as e:
            logger.error("Error plotting filled contours: %s", e)
            return None

        # Label the axes with colors
        ax.set_tlabel('R', color='darkred', fontsize=12)
        ax.set_llabel('G', color='darkgreen', fontsize=12)
        ax.set_rlabel('B', color='darkblue', fontsize=12)

        # Color the tick parameters
        ax.taxis.set_tick_params(colors='darkred', which='both', length=5, width=1.5)
        ax.laxis.set_tick_params(colors='darkgreen', which='both', length=5, width=1.5)
        ax.raxis.set_tick_params(colors='darkblue', which='both', length=5, width=1.5)

        # Add colorbar if requested
        if show_colorbar:
            cbar = plt.colorbar(contourf, ax=ax, pad=0.1, fraction=0.05)
            cbar.set_label(colorbar_label, rotation=270, labelpad=20)

        return contourf

    def plot_ternary_scatter(
        self,
        ax,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        color: Union[str, np.ndarray] = 'black',
        size: float = 20,
        marker: str = 'o',
        alpha: float = 0.6,
        edgecolor: str = 'none',
        linewidth: float = 0,
        label: str = '',
        rasterized: bool = False,
        **kwargs
    ):
        """Plot scatter points on an existing ternary axis.

        This method adds scatter points to an existing ternary axis.
        Use this when you want to add individual data points to a multi-panel figure
        with ternary plots.

        Args:
            ax: Existing ternary axis (must have projection='ternary')
            R: Red channel values (normalized, 0-1)
            G: Green channel values (normalized, 0-1)
            B: Blue channel values (normalized, 0-1)
            color: Point color - can be:
                  - String color name (e.g., 'red', 'blue')
                  - RGB/RGBA tuple
                  - Array of colors (one per point)
                  (default: 'black')
            size: Point size in points^2 (default: 20)
            marker: Marker style (default: 'o')
                   Options: 'o', 's', '^', 'v', '<', '>', 'D', 'p', '*', etc.
            alpha: Point transparency, 0-1 (default: 0.6)
            edgecolor: Edge color for markers (default: 'none')
            linewidth: Width of marker edges (default: 0)
            label: Label for legend (default: '')
            rasterized: Whether to rasterize scatter points for smaller file size (default: False)
            **kwargs: Additional arguments passed to ax.scatter()

        Returns:
            scatter: The PathCollection object from scatter (can be used for legend, etc.)

        Example:
            >>> import matplotlib.pyplot as plt
            >>> import mpltern
            >>> from PlottingBase import PublicationPlotter
            >>>
            >>> # Create figure with ternary subplot
            >>> fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={'projection': 'ternary'})
            >>>
            >>> # Add scatter points
            >>> plotter = PublicationPlotter()
            >>> scatter = plotter.plot_ternary_scatter(
            ...     ax, R_data, G_data, B_data,
            ...     color='red',
            ...     size=30,
            ...     alpha=0.7,
            ...     label='Data points'
            ... )
            >>> ax.legend()

            Multi-panel example:
            >>> import matplotlib.pyplot as plt
            >>> import mpltern
            >>> from PlottingBase import PublicationPlotter
            >>>
            >>> fig = plt.figure(figsize=(12, 4))
            >>> ax1 = fig.add_subplot(1, 3, 1)  # Regular plot
            >>> ax2 = fig.add_subplot(1, 3, 2, projection='ternary')  # Ternary plot
            >>> ax3 = fig.add_subplot(1, 3, 3)  # Regular plot
            >>>
            >>> plotter = PublicationPlotter()
            >>> # Add scatter to the ternary axis
            >>> scatter = plotter.plot_ternary_scatter(
            ...     ax2, R_data, G_data, B_data,
            ...     color='blue',
            ...     size=15,
            ...     alpha=0.5
            ... )

        Notes:
            - Requires mpltern: `pip install mpltern`
            - RGB values should be normalized (sum to 1 for each point)
            - If not normalized, the function will normalize them automatically
            - The axis must already exist with projection='ternary'
            - For large datasets (>10k points), consider using rasterized=True for smaller file sizes
            - Colors can be specified per-point using an array matching the data length
        """
        # Validate inputs
        if len(R) != len(G) or len(R) != len(B):
            raise ValueError("R, G, B arrays must have the same length")

        # Convert to numpy arrays if needed
        R = np.asarray(R)
        G = np.asarray(G)
        B = np.asarray(B)

        # Remove NaN/Inf values
        valid_mask = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
        if not np.all(valid_mask):
            logger.warning("Removing %s invalid points (NaN/Inf)", np.sum(~valid_mask))
            R = R[valid_mask]
            G = G[valid_mask]
            B = B[valid_mask]
            # Also filter color array if it's an array
            if isinstance(color, np.ndarray) and len(color) == len(valid_mask):
                color = color[valid_mask]

        if len(R) == 0:
            logger.warning("No valid points to plot")
            return None

        # Check for and handle normalization
        totals = R + G + B
        if not np.allclose(totals, 1.0, atol=1e-6):
            # Normalize
            R = R / totals
            G = G / totals
            B = B / totals

        # Create scatter plot
        # mpltern uses (t, l, r) ordering where t=top, l=left, r=right
        # For RGB: t=R (top), l=G (left), r=B (right)
        scatter = ax.scatter(
            R, G, B,
            c=color,
            s=size,
            marker=marker,
            alpha=alpha,
            edgecolors=edgecolor,
            linewidths=linewidth,
            label=label,
            rasterized=rasterized,
            **kwargs
        )

        # Label the axes with colors
        ax.set_tlabel('R', color='darkred', fontsize=12)
        ax.set_llabel('G', color='darkgreen', fontsize=12)
        ax.set_rlabel('B', color='darkblue', fontsize=12)

        # Color the tick parameters
        ax.taxis.set_tick_params(colors='darkred', which='both', length=5, width=1.5)
        ax.laxis.set_tick_params(colors='darkgreen', which='both', length=5, width=1.5)
        ax.raxis.set_tick_params(colors='darkblue', which='both', length=5, width=1.5)

        return scatter

    @staticmethod
    def _ilr(comp: np.ndarray) -> np.ndarray:
        """Isometric log-ratio transform of 3-part compositions, (n, 3) -> (n, 2)."""
        logc = np.log(comp)
        return np.column_stack([
            (logc[:, 0] - logc[:, 1]) / np.sqrt(2),
            (logc[:, 0] + logc[:, 1] - 2 * logc[:, 2]) / np.sqrt(6),
        ])

    @staticmethod
    def _ilr_inv(z: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`_ilr`, (n, 2) -> (n, 3) compositions summing to 1."""
        logc = np.column_stack([
            z[:, 0] / np.sqrt(2) + z[:, 1] / np.sqrt(6),
            -z[:, 0] / np.sqrt(2) + z[:, 1] / np.sqrt(6),
            -2 * z[:, 1] / np.sqrt(6),
        ])
        c = np.exp(logc - logc.max(axis=1, keepdims=True))
        return c / c.sum(axis=1, keepdims=True)

    def plot_ternary_credible_regions(
        self,
        ax,
        samples: Union[np.ndarray, List[np.ndarray]],
        levels: Union[float, Tuple[float, ...]] = (0.68, 0.95),
        colors: Union[str, List[Any], None] = None,
        alpha: float = 0.25,
        edgecolor: Optional[str] = None,
        linewidth: float = 0.8,
        grid_size: int = 120,
        max_samples: int = 2000,
        eps: float = 1e-6,
        seed: int = 0,
        **kwargs,
    ) -> List[Any]:
        """Shade highest-posterior-density credible regions on an existing ternary axis.

        For each point, the posterior samples are mapped to isometric log-ratio (ILR)
        coordinates, a Gaussian KDE is fitted there, and the region enclosing ``level``
        of the posterior mass (density above the ``1 - level`` quantile of the density
        at the samples) is contoured and mapped back to the simplex. Working in ILR
        space keeps regions inside the triangle and lets them curve and skew near the
        edges, as real posteriors do.

        Args:
            ax: Existing ternary axis (projection='ternary').
            samples: Posterior samples in (R, G, B) order -- i.e. (t, l, r), as in the
                other ternary methods. One point: array (n_samples, 3). Several points:
                array (n_points, n_samples, 3) or a list of (n_samples_i, 3) arrays.
                Rows are normalised to sum to 1, so fractions or percentages both work.
            levels: Credible level(s) in (0, 1), e.g. 0.95 or (0.68, 0.95). Each level
                is filled with ``alpha``, so nested levels shade darker towards the mode.
            colors: One colour for all points, or one per point (default: matplotlib
                colour cycle).
            alpha: Fill opacity per level (default: 0.25).
            edgecolor: Outline colour; None (default) uses the fill colour.
            linewidth: Outline width (default: 0.8); 0 for no outline.
            grid_size: KDE evaluation grid per side, in ILR space (default: 120).
            max_samples: Randomly subsample to at most this many samples per point,
                for speed (default: 2000).
            eps: Floor applied to fractions before the log-ratio (default: 1e-6).
            seed: Seed for the subsampling (default: 0).
            **kwargs: Passed to ``ax.fill`` (e.g. ``zorder``, ``label``).

        Returns:
            List of the filled polygons added to ``ax``.

        Example:
            >>> fig, ax = plotter.create_ternary_plot(R, G, B, black_background=False)
            >>> plotter.plot_ternary_credible_regions(ax, posterior_samples, levels=0.95)
        """
        import contourpy
        from scipy.stats import gaussian_kde

        if isinstance(samples, np.ndarray) and samples.ndim == 2:
            samples = [samples]
        levels = np.atleast_1d(levels)
        if np.any((levels <= 0) | (levels >= 1)):
            raise ValueError(f"levels must be in (0, 1); got {levels}")
        if colors is None or matplotlib.colors.is_color_like(colors):
            colors = [colors] * len(samples)
        if len(colors) != len(samples):
            raise ValueError("colors must be one colour or one per point")

        rng = np.random.default_rng(seed)
        patches = []
        for point_samples, color in zip(samples, colors):
            comp = np.asarray(point_samples, dtype=float)
            if comp.ndim != 2 or comp.shape[1] != 3:
                raise ValueError(f"each point's samples must be (n_samples, 3); got {comp.shape}")
            if len(comp) > max_samples:
                comp = comp[rng.choice(len(comp), max_samples, replace=False)]
            comp = np.clip(comp / comp.sum(axis=1, keepdims=True), eps, None)
            z = self._ilr(comp)

            kde = gaussian_kde(z.T)
            density_at_samples = kde(z.T)
            pad = 3 * np.sqrt(np.diag(kde.covariance))
            gx = np.linspace(z[:, 0].min() - pad[0], z[:, 0].max() + pad[0], grid_size)
            gy = np.linspace(z[:, 1].min() - pad[1], z[:, 1].max() + pad[1], grid_size)
            GX, GY = np.meshgrid(gx, gy)
            density = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
            contours = contourpy.contour_generator(GX, GY, density)

            if color is None:
                color = ax._get_lines.get_next_color()
            for level in levels:
                threshold = np.quantile(density_at_samples, 1 - level)
                polygons, offsets = contours.filled(threshold, np.inf)
                for poly, off in zip(polygons, offsets):
                    outer = self._ilr_inv(poly[off[0]:off[1]])  # outer ring; holes ignored
                    patches += ax.fill(
                        outer[:, 0], outer[:, 1], outer[:, 2],
                        facecolor=color, alpha=alpha,
                        edgecolor=edgecolor if edgecolor is not None else color,
                        linewidth=linewidth, **kwargs,
                    )
        return patches

    def plot_ternary_errorbars(
        self,
        ax,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        R_err: Optional[np.ndarray] = None,
        G_err: Optional[np.ndarray] = None,
        B_err: Optional[np.ndarray] = None,
        color: Union[str, Any] = "black",
        linewidth: float = 0.8,
        capsize: float = 3.0,
        alpha: float = 1.0,
        show_points: bool = True,
        marker_size: float = 12,
        **kwargs,
    ) -> List[Any]:
        """Draw per-channel error bars on an existing ternary axis.

        The error bar for a channel runs along the line from the point towards that
        channel's vertex: the channel changes by its error while the other two keep
        their ratio (and the composition still sums to 1). Bars are clipped at the
        triangle edges. Errors are in the same units as R, G, B (fractions or %).

        Args:
            ax: Existing ternary axis (projection='ternary').
            R, G, B: Point coordinates in (t, l, r) order, shape (n,).
            R_err, G_err, B_err: Errors per channel: (n,) symmetric, or (2, n) as
                (lower, upper). None skips that channel.
            color: Bar and point colour (default: 'black').
            linewidth: Bar line width (default: 0.8).
            capsize: Cap length in points; 0 for no caps (default: 3).
            alpha: Opacity (default: 1.0).
            show_points: Also draw the points (default: True).
            marker_size: Point size in points^2 (default: 12).
            **kwargs: Passed to ``ax.plot`` for the bars.

        Returns:
            List of the artists added to ``ax``.

        Example:
            >>> plotter.plot_ternary_errorbars(
            ...     ax, df.A_R, df.A_G, df.A_B, df.A_R_err, df.A_G_err, df.A_B_err)
        """
        from matplotlib.markers import MarkerStyle
        from matplotlib.transforms import Affine2D

        comp = np.column_stack([np.asarray(R, float), np.asarray(G, float), np.asarray(B, float)])
        totals = comp.sum(axis=1, keepdims=True)
        comp = comp / totals
        to_xy = ax.transProjection.transform

        artists = []
        for c, err in enumerate((R_err, G_err, B_err)):
            if err is None:
                continue
            err = np.asarray(err, float)
            lower, upper = (err[0], err[1]) if err.ndim == 2 else (err, err)
            a = comp[:, c]
            span = np.where(a < 1, 1 - a, np.nan)
            vertex = np.eye(3)[c]
            # Moving t along p -> vertex changes channel c by t * (1 - a)
            t_lo = np.maximum(-lower / totals[:, 0] / span, -a / span)
            t_hi = np.minimum(upper / totals[:, 0] / span, 1.0)
            for i in np.flatnonzero(np.isfinite(span)):
                ends = comp[i] + np.outer([t_lo[i], t_hi[i]], vertex - comp[i])
                artists += ax.plot(ends[:, 0], ends[:, 1], ends[:, 2], color=color,
                                   linewidth=linewidth, alpha=alpha, **kwargs)
                if capsize > 0:
                    # '|' is vertical, so rotating by the bar angle makes it perpendicular
                    dx, dy = to_xy(vertex[None])[0] - to_xy(comp[i][None])[0]
                    cap = MarkerStyle("|", transform=Affine2D().rotate(np.arctan2(dy, dx)))
                    artists += ax.plot(ends[:, 0], ends[:, 1], ends[:, 2], linestyle="none",
                                       marker=cap, markersize=capsize, color=color,
                                       markeredgewidth=linewidth, alpha=alpha)
        if show_points:
            artists.append(ax.scatter(comp[:, 0], comp[:, 1], comp[:, 2], s=marker_size,
                                      color=color, alpha=alpha, zorder=3))
        return artists


class DatashaderMixin:
    """Mixin for handling large datasets with datashader when available.

    This mixin automatically switches between matplotlib and datashader rendering
    based on dataset size, with user-tunable thresholds for optimal performance.
    """

    def __init__(self, *args, datashader_threshold: int = 1000, **kwargs):
        """Initialize DatashaderMixin.

        Args:
            datashader_threshold: Number of points above which to use datashader.
                Default is 1000. Set to None to disable auto-switching.
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)

        self.datashader_threshold = datashader_threshold
        self._datashader_warned = False  # Track if we've shown warning

        # Try to import datashader components
        try:
            import datashader as ds
            import datashader.transfer_functions as tf
            import pandas as pd

            self.ds = ds
            self.tf = tf
            self.pd = pd
            self.datashader_available = True
        except ImportError:
            self.datashader_available = False
            if datashader_threshold is not None and not self._datashader_warned:
                warnings.warn(
                    "Datashader not available. Install with 'pip install datashader' "
                    "for better performance with large datasets (>1k points). "
                    "Falling back to matplotlib which may be slow for >10k points."
                )
                self._datashader_warned = True

    def plot_large_scatter(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        c: Optional[np.ndarray] = None,
        threshold: Optional[int] = None,
        canvas_size: Optional[Tuple[int, int]] = None,
        cmap: str = "viridis",
        downsample: bool = False,
        downsample_factor: int = 10,
        **scatter_kwargs,
    ) -> Union[matplotlib.image.AxesImage, matplotlib.collections.PathCollection]:
        """Plot scatter data, using datashader for large datasets.

        This method automatically selects the best rendering method based on
        dataset size. For datasets larger than the threshold, it uses datashader
        for fast rendering. For smaller datasets, it uses matplotlib for better
        interactivity.

        Args:
            ax: Axis to plot on
            x: X coordinates
            y: Y coordinates
            c: Color values (optional, for matplotlib or datashader aggregation)
            threshold: Use datashader if more than this many points.
                If None, uses self.datashader_threshold. Set to None to force matplotlib.
            canvas_size: Canvas size for datashader rendering (width, height).
                If None, uses figure DPI and size.
            cmap: Colormap for datashader or matplotlib
            downsample: If True and using matplotlib, downsample large datasets
            downsample_factor: Keep every Nth point when downsampling
            **scatter_kwargs: Arguments for matplotlib scatter (s, alpha, marker, etc.)

        Returns:
            Either AxesImage (datashader) or PathCollection (matplotlib)

        Examples:
            >>> # Small dataset - uses matplotlib
            >>> plotter.plot_large_scatter(ax, x[:500], y[:500])

            >>> # Large dataset - auto-switches to datashader
            >>> plotter.plot_large_scatter(ax, x, y)  # 50k points

            >>> # Force matplotlib with downsampling
            >>> plotter.plot_large_scatter(ax, x, y, threshold=None,
            ...                           downsample=True, downsample_factor=10)
        """
        # Determine threshold
        if threshold is None:
            threshold = (
                self.datashader_threshold if self.datashader_threshold else float("inf")
            )

        n_points = len(x)
        use_datashader = (
            n_points > threshold and self.datashader_available and threshold is not None
        )

        if use_datashader:
            # Use datashader for large datasets
            if canvas_size is None:
                # Auto-determine canvas size from figure
                bbox = ax.get_window_extent().transformed(
                    ax.figure.dpi_scale_trans.inverted()
                )
                canvas_size = (
                    int(bbox.width * ax.figure.dpi),
                    int(bbox.height * ax.figure.dpi),
                )

            return self._plot_with_datashader(ax, x, y, c, canvas_size, cmap)
        else:
            # Use matplotlib
            if downsample and n_points > threshold:
                # Downsample for preview
                indices = np.arange(0, n_points, downsample_factor)
                x_down = x[indices]
                y_down = y[indices]
                c_down = c[indices] if c is not None else None

                scatter = ax.scatter(
                    x_down, y_down, c=c_down, cmap=cmap, **scatter_kwargs
                )
                ax.set_title(
                    ax.get_title() + f" (showing {len(indices)}/{n_points} points)"
                )
                return scatter
            else:
                return ax.scatter(x, y, c=c, cmap=cmap, **scatter_kwargs)

    def _plot_with_datashader(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        c: Optional[np.ndarray],
        canvas_size: Tuple[int, int],
        cmap: str,
    ) -> matplotlib.image.AxesImage:
        """Create datashader plot.

        Args:
            ax: Axis to plot on
            x: X coordinates
            y: Y coordinates
            c: Color values (optional, used for aggregation)
            canvas_size: Canvas size (width, height)
            cmap: Colormap name

        Returns:
            AxesImage object
        """
        # Create DataFrame
        if c is not None:
            df = self.pd.DataFrame({"x": x, "y": y, "c": c})
            agg_column = "c"
            agg_func = "mean"  # Average color values
        else:
            df = self.pd.DataFrame({"x": x, "y": y})
            agg_column = None
            agg_func = "count"  # Count points per pixel

        # Create canvas with bounds
        x_range = (float(x.min()), float(x.max()))
        y_range = (float(y.min()), float(y.max()))

        canvas = self.ds.Canvas(
            plot_width=canvas_size[0],
            plot_height=canvas_size[1],
            x_range=x_range,
            y_range=y_range,
        )

        # Aggregate points
        if agg_column:
            agg = canvas.points(df, "x", "y", agg=self.ds.mean(agg_column))
        else:
            agg = canvas.points(df, "x", "y")

        # Create image with proper colormap
        try:
            # Try using colormap from colorcet if available
            img = self.tf.shade(agg, cmap=cmap, how="linear")
        except (ValueError, KeyError):
            # Fall back to default if colormap not found
            img = self.tf.shade(agg, how="linear")

        # Convert to numpy array and display
        img_array = img.to_numpy()
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

        im = ax.imshow(
            img_array,
            extent=extent,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        # Add note about rendering method
        n_points = len(x)
        ax.text(
            0.02,
            0.98,
            f"Datashader: {n_points:,} points",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            alpha=0.7,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        return im

    def plot_multi_dataset_scatter(
        self,
        ax: matplotlib.axes.Axes,
        datasets: List[Dict[str, np.ndarray]],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        threshold: Optional[int] = None,
        canvas_size: Optional[Tuple[int, int]] = None,
        alpha: float = 0.6,
        sizes: Optional[Union[float, List[float]]] = None,
        **scatter_kwargs,
    ) -> List[Union[matplotlib.image.AxesImage, matplotlib.collections.PathCollection]]:
        """Plot multiple datasets efficiently, auto-selecting best rendering method.

        For large multi-dataset plots, this method intelligently chooses between:
        - Individual datashader layers (best for very large datasets)
        - Matplotlib scatter with downsampling (good for moderate datasets)
        - Standard matplotlib scatter (best for small datasets)

        Args:
            ax: Axis to plot on
            datasets: List of dicts with 'x' and 'y' keys
            labels: Labels for each dataset (for legend)
            colors: Colors for each dataset
            threshold: Total points threshold for datashader
            canvas_size: Canvas size for datashader
            alpha: Transparency for matplotlib rendering
            sizes: Marker sizes (single or per-dataset)
            **scatter_kwargs: Additional matplotlib scatter kwargs

        Returns:
            List of plot objects (one per dataset)

        Example:
            >>> datasets = [
            ...     {'x': locs1['xc'], 'y': locs1['yc']},
            ...     {'x': locs2['xc'], 'y': locs2['yc']}
            ... ]
            >>> plotter.plot_multi_dataset_scatter(ax, datasets,
            ...     labels=['Fiducial 1', 'Fiducial 2'])
        """
        if threshold is None:
            threshold = (
                self.datashader_threshold if self.datashader_threshold else float("inf")
            )

        total_points = sum(len(d["x"]) for d in datasets)
        use_datashader = (
            total_points > threshold
            and self.datashader_available
            and threshold is not None
        )

        # Prepare colors and sizes
        if colors is None:
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]

        if isinstance(sizes, (int, float)):
            sizes = [sizes] * len(datasets)
        elif sizes is None:
            sizes = [self.config.DEFAULT_MARKER_SIZE] * len(datasets)

        plots = []

        if use_datashader and len(datasets) > 1:
            # Combine datasets with category labels for datashader
            all_x = np.concatenate([d["x"] for d in datasets])
            all_y = np.concatenate([d["y"] for d in datasets])
            categories = np.concatenate(
                [np.full(len(d["x"]), i, dtype=int) for i, d in enumerate(datasets)]
            )

            # Convert to categorical for datashader
            df = self.pd.DataFrame({"x": all_x, "y": all_y, "category": categories})
            df["category"] = df["category"].astype("category")

            if canvas_size is None:
                bbox = ax.get_window_extent().transformed(
                    ax.figure.dpi_scale_trans.inverted()
                )
                canvas_size = (
                    int(bbox.width * ax.figure.dpi),
                    int(bbox.height * ax.figure.dpi),
                )

            x_range = (float(all_x.min()), float(all_x.max()))
            y_range = (float(all_y.min()), float(all_y.max()))

            canvas = self.ds.Canvas(
                plot_width=canvas_size[0],
                plot_height=canvas_size[1],
                x_range=x_range,
                y_range=y_range,
            )

            # Aggregate by category
            agg = canvas.points(
                df, "x", "y", agg=self.ds.by("category", self.ds.count())
            )

            # Create color mapping
            color_key = {i: colors[i % len(colors)] for i in range(len(datasets))}
            img = self.tf.shade(
                agg, color_key=color_key, how="linear", alpha=int(alpha * 255)
            )

            img_array = img.to_numpy()
            extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

            im = ax.imshow(
                img_array,
                extent=extent,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            plots.append(im)

            # Add legend manually for datashader
            if labels:
                from matplotlib.patches import Patch

                legend_elements = [
                    Patch(facecolor=colors[i % len(colors)], label=labels[i])
                    for i in range(len(datasets))
                ]
                ax.legend(handles=legend_elements)

            # Add rendering note
            ax.text(
                0.02,
                0.98,
                f"Datashader: {total_points:,} points",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                alpha=0.7,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )

        else:
            # Use matplotlib for each dataset
            for i, dataset in enumerate(datasets):
                x, y = dataset["x"], dataset["y"]
                color = colors[i % len(colors)]
                size = sizes[i % len(sizes)]
                label = labels[i] if labels else None

                # Check if individual dataset is too large
                if len(x) > threshold and threshold is not None:
                    # Downsample this dataset
                    downsample_factor = max(1, len(x) // threshold)
                    indices = np.arange(0, len(x), downsample_factor)
                    x_down = x[indices]
                    y_down = y[indices]

                    scatter = ax.scatter(
                        x_down,
                        y_down,
                        c=color,
                        s=size,
                        alpha=alpha,
                        label=label,
                        **scatter_kwargs,
                    )

                    if i == 0:  # Only show note once
                        ax.text(
                            0.02,
                            0.98,
                            f"Downsampled: {len(indices):,}/{total_points:,} points",
                            transform=ax.transAxes,
                            fontsize=8,
                            verticalalignment="top",
                            alpha=0.7,
                            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
                        )
                else:
                    scatter = ax.scatter(
                        x,
                        y,
                        c=color,
                        s=size,
                        alpha=alpha,
                        label=label,
                        **scatter_kwargs,
                    )

                plots.append(scatter)

            if labels:
                ax.legend()

        return plots

    def create_preview_plot(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        preview_points: int = 5000,
        method: str = "random",
        **scatter_kwargs,
    ) -> matplotlib.collections.PathCollection:
        """Create fast preview plot by intelligently downsampling large datasets.

        Args:
            ax: Axis to plot on
            x: X coordinates
            y: Y coordinates
            preview_points: Target number of points for preview
            method: Downsampling method ('random', 'uniform', 'density')
                - 'random': Random sampling
                - 'uniform': Evenly spaced sampling
                - 'density': Density-aware sampling (keeps more points in sparse regions)
            **scatter_kwargs: Additional scatter plot arguments

        Returns:
            PathCollection from scatter plot

        Example:
            >>> # Quick preview of 100k points showing 5k representative points
            >>> plotter.create_preview_plot(ax, x, y, preview_points=5000, method='density')
        """
        n_points = len(x)

        if n_points <= preview_points:
            # No downsampling needed
            return ax.scatter(x, y, **scatter_kwargs)

        # Select indices based on method
        if method == "random":
            indices = np.random.choice(n_points, preview_points, replace=False)
        elif method == "uniform":
            step = n_points // preview_points
            indices = np.arange(0, n_points, step)[:preview_points]
        elif method == "density":
            # Density-aware sampling: use 2D histogram to identify sparse/dense regions
            # Keep more points from sparse regions for better coverage
            hist, x_edges, y_edges = np.histogram2d(x, y, bins=50)

            # Assign each point to a bin
            x_bin_idx = np.digitize(x, x_edges) - 1
            y_bin_idx = np.digitize(y, y_edges) - 1

            # Clip to valid range
            x_bin_idx = np.clip(x_bin_idx, 0, hist.shape[0] - 1)
            y_bin_idx = np.clip(y_bin_idx, 0, hist.shape[1] - 1)

            # Calculate sampling probability (inversely proportional to density)
            densities = hist[x_bin_idx, y_bin_idx]
            # Avoid division by zero
            probs = 1.0 / (densities + 1)
            probs /= probs.sum()

            indices = np.random.choice(n_points, preview_points, replace=False, p=probs)
        else:
            raise ValueError(f"Unknown downsampling method: {method}")

        # Plot downsampled data
        scatter = ax.scatter(x[indices], y[indices], **scatter_kwargs)

        # Add note about preview
        ax.text(
            0.02,
            0.98,
            f"Preview: {len(indices):,}/{n_points:,} points ({method})",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            alpha=0.7,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        return scatter


class PublicationPlotter(TernaryPlotMixin, BasePlotter, ImagePlotMixin):
    """Publication-quality plotter with enhanced image and ternary plot capabilities.

    This class provides high-quality plotting functionality suitable for
    scientific publications, with consistent styling and professional appearance.
    Includes support for ternary (3-component) plots via the TernaryPlotMixin.
    """

    def __init__(self, poster: bool = False, dark_background: bool = False):
        """Initialize publication plotter.

        Args:
            poster: Whether to use poster-style formatting (12pt fonts vs 7pt, 1.0pt lines vs 0.5pt)
            dark_background: Whether to use dark background theme
        """
        # Create config with proper poster and dark background flags
        # Must set flags BEFORE __post_init__ runs (which happens at creation time for dataclass)
        config = PlottingConfig(poster_mode=poster, dark_background=dark_background)

        super().__init__(config)

        # Store mode for helper methods
        self.poster = poster
        self.dark_background = dark_background

    # ------------------------------------------------------------------
    # Legacy-compatible methods migrated from PlottingFunctions.Plotter
    # ------------------------------------------------------------------

    def _get_plot_font_size(self, plot_type: str = "standard") -> int:
        """Return plot element font size for the current mode.

        Args:
            plot_type: ``"standard"`` (8 pt) or ``"scatter"`` (7 pt).

        Returns:
            Font size in points.
        """
        if self.poster:
            return 15
        return 7 if plot_type == "scatter" else 8

    def _setup_colorbar(self, im, axs, cbarlabel: str, location: str = "right") -> None:
        """Add a styled colorbar using BasePlotter.add_colorbar.

        Args:
            im: Image / mappable object.
            axs: Parent axes.
            cbarlabel: Label for the colorbar.
            location: ``"left"``, ``"right"``, ``"top"``, or ``"bottom"``.
        """
        cbar = self.add_colorbar(im, axs, label=cbarlabel, location=location)
        font_size = self._get_plot_font_size()
        cbar.ax.tick_params(labelsize=font_size - 1, pad=0.1, width=0.5, length=2)

    def _setup_scalebar(
        self,
        axs,
        pixelsize: float,
        scalebarsize: float,
        scalebarlabel: str,
        labelcolor: str,
        location: str = "lower right",
    ) -> None:
        """Add a scale bar using BasePlotter.add_scalebar.

        Args:
            axs: Axes to annotate.
            pixelsize: Pixel size in nm.
            scalebarsize: Scale bar length in nm.
            scalebarlabel: Text label for the scale bar.
            labelcolor: Color for the scale bar and label.
            location: Legend location string.
        """
        self.add_scalebar(
            axs,
            pixelsize=pixelsize,
            length_nm=scalebarsize,
            location=location,
            color=labelcolor,
            label=scalebarlabel,
        )

    def image_scatter_plot(
        self,
        axs,
        data: np.ndarray,
        xdata: np.ndarray,
        ydata: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "gist_gray",
        cbar: str = "on",
        cbarlabel: str = "photons",
        label: str = "",
        labelcolor: str = "white",
        pixelsize: float = DriftConstants.XIMEA_PIXEL_SIZE_NM,
        scalebarsize: float = 10000,
        scalebarlabel: str = "10 μm",
        alpha: float = 1,
        scatteralpha: float = 1,
        scattercolor: str = "red",
        facecolor: str = "None",
        marker: str = "o",
        s: float = 20,
        lws: float = 0.75,
    ):
        """Create an image plot with scatter overlay.

        Args:
            axs: Axes object.
            data: Image data.
            xdata: Scatter X coordinates.
            ydata: Scatter Y coordinates.
            vmin: Minimum display value (default: 1st percentile).
            vmax: Maximum display value (default: 99th percentile).
            cmap: Colormap.
            cbar: ``"on"`` to show colorbar.
            cbarlabel: Colorbar label.
            label: Text annotation on the image.
            labelcolor: Annotation color.
            pixelsize: Pixel size in nm (for scale bar).
            scalebarsize: Scale bar length in nm.
            scalebarlabel: Scale bar text label.
            alpha: Image transparency.
            scatteralpha: Scatter point transparency.
            scattercolor: Scatter edge color.
            facecolor: Scatter face color.
            marker: Scatter marker style.
            s: Scatter marker size.
            lws: Scatter line width.

        Returns:
            Modified axes object.
        """
        font_size = self._get_plot_font_size()

        if vmin is None:
            vmin = np.percentile(data.ravel(), 1)
        if vmax is None:
            vmax = np.percentile(data.ravel(), 99)

        im = axs.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, origin="lower")

        if cbar == "on":
            self._setup_colorbar(im, axs, cbarlabel, "left")

        axs.set_xticks([])
        axs.set_yticks([])

        self._setup_scalebar(axs, pixelsize, scalebarsize, scalebarlabel, labelcolor)

        axs.annotate(
            label,
            xy=(5, 5),
            xytext=(20, 60),
            xycoords="data",
            color=labelcolor,
            fontsize=font_size - 1,
        )

        axs.scatter(
            xdata,
            ydata,
            lw=lws,
            edgecolor=scattercolor,
            s=s,
            marker=marker,
            facecolors=facecolor,
            alpha=scatteralpha,
        )
        return axs

    def line_error_plot(
        self,
        axs,
        x: np.ndarray,
        y: np.ndarray,
        yerror: np.ndarray,
        xlim: Optional[Tuple] = None,
        ylim: Optional[Tuple] = None,
        color: str = "k",
        lw: float = 0.75,
        label: str = "",
        xaxislabel: str = "x axis",
        yaxislabel: str = "y axis",
        ls: str = "-",
        alpha: float = 1.0,
    ):
        """Create a line plot with error bands.

        Thin wrapper around :meth:`BasePlotter.line_plot_with_error` using the
        legacy parameter names accepted by notebooks.

        Args:
            axs: Axes object.
            x: X data.
            y: Y data.
            yerror: Y error data.
            xlim: X axis limits (min, max).
            ylim: Y axis limits (min, max).
            color: Line color.
            lw: Line width.
            label: Line label.
            xaxislabel: X axis label.
            yaxislabel: Y axis label.
            ls: Line style (currently unused — passed for API compat).
            alpha: Error band transparency.

        Returns:
            Modified axes object.
        """
        xlim_t = tuple(xlim) if xlim is not None else None
        ylim_t = tuple(ylim) if ylim is not None else None
        return self.line_plot_with_error(
            axs, x, y, yerror,
            xlabel=xaxislabel,
            ylabel=yaxislabel,
            xlim=xlim_t,
            ylim=ylim_t,
            color=color,
            linewidth=lw,
            label=label,
            alpha=alpha,
        )

    def make_animated_gif_image(
        self,
        image: np.ndarray,
        n_frames: int,
        filename: str,
        vmin: float = 0,
        vmax: float = 150,
        pixelsize: float = DriftConstants.XIMEA_PIXEL_SIZE_NM,
        scalebarsize: float = 300,
        scalebarlabel: str = "300 nm",
        label: str = "",
        fontsz: int = 6,
        cbarlabel: str = "# of photoelectrons",
        cbar: bool = False,
        width: float = 3,
        height: float = 3,
    ) -> None:
        """Create animated GIF from a grayscale image sequence.

        Thin wrapper around :meth:`make_animated_gif` kept for backwards
        compatibility with notebooks that call ``make_animated_gif_image``.

        Args:
            image: 3-D image stack (n_frames × H × W).
            n_frames: Ignored — frame count is taken from ``image.shape[0]``.
            filename: Output GIF path.
            vmin: Minimum display value.
            vmax: Maximum display value.
            pixelsize: Pixel size in nm.
            scalebarsize: Scale bar length in nm.
            scalebarlabel: Scale bar text label.
            label: Text annotation on each frame.
            fontsz: Font size for annotations.
            cbarlabel: Colorbar label.
            cbar: Whether to show colorbar.
            width: Figure width in inches.
            height: Figure height in inches.
        """
        self.make_animated_gif(
            image=image,
            filename=filename,
            vmin=vmin,
            vmax=vmax,
            pixelsize=pixelsize,
            scalebarsize=scalebarsize,
            scalebarlabel=scalebarlabel,
            label=label,
            fontsz=fontsz,
            cbarlabel=cbarlabel,
            cbar=cbar,
            width=width,
            height=height,
        )

    def _setup_ternary_axis(
        self,
        ax,
        maj_loc: float,
        min_loc: float,
        maxt: float,
        maxl: float,
        maxr: float,
        trianglesize: float,
        black_background: bool = True,
    ) -> None:
        """Configure a ternary plot axis with ticks, limits, and labels.

        Args:
            ax: Ternary axes object (mpltern).
            maj_loc: Major tick interval.
            min_loc: Minor tick interval.
            maxt: Maximum t value.
            maxl: Maximum l value.
            maxr: Maximum r value.
            trianglesize: Side length of the displayed triangle region.
            black_background: Fill triangle interior with black (default: True).
        """
        if black_background:
            ax.set_facecolor('black')

        for axis in [ax.taxis, ax.laxis, ax.raxis]:
            axis.set_major_locator(MultipleLocator(maj_loc))
            axis.set_minor_locator(MultipleLocator(min_loc))

        ax.set_ternary_lim(
            maxt - trianglesize, maxt,
            maxl - trianglesize, maxl,
            maxr - trianglesize, maxr,
        )

        ax.set_tlabel(r"pixel 1 QE")
        ax.set_llabel(r"pixel 3 QE")
        ax.set_rlabel(r"pixel 2 QE")

        ax.grid(lw=0.5, alpha=0.25, ls="--", which="both", axis="both", color="white")

    def ternary_scatter_plot(
        self,
        fig,
        axs,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        colours: np.ndarray,
        xlevel: int = 1,
        ylevel: int = 2,
        location_pos: int = 2,
        maj_loc: float = 0.2,
        min_loc: float = 0.1,
        maxt: float = 1,
        maxl: float = 1,
        maxr: float = 1,
        trianglesize: float = 1,
        s: float = 25,
        lws: float = 0.5,
        black_background: bool = True,
    ) -> Tuple[Any, Any]:
        """Create a ternary scatter plot using the legacy subplot-replacement API.

        All existing axes are cleared from *fig* before the ternary axis is
        added, so no rectangular background panels remain visible.

        Args:
            fig: Figure object.
            axs: Axes array — all axes are removed and replaced by the ternary axis.
            R: T (top) scatter values.
            G: L (left) scatter values.
            B: R (right) scatter values.
            colours: RGBA edge colors for each scatter point.
            xlevel: Row count for :func:`~matplotlib.figure.Figure.add_subplot`.
            ylevel: Column count for :func:`~matplotlib.figure.Figure.add_subplot`.
            location_pos: Subplot position index.
            maj_loc: Major tick interval.
            min_loc: Minor tick interval.
            maxt: Maximum t value.
            maxl: Maximum l value.
            maxr: Maximum r value.
            trianglesize: Side length of the displayed triangle region.
            s: Scatter marker size.
            lws: Scatter marker edge width.
            black_background: Fill triangle interior with black (default: True).

        Returns:
            Tuple of (figure, axes).
        """
        # Remove every pre-existing rectangular axis so none remain in the background
        for a in fig.axes:
            a.remove()

        try:
            import mpltern  # noqa: F401 — registers projection
            ax = fig.add_subplot(xlevel, ylevel, location_pos, projection="ternary")
        except ImportError:
            raise ImportError(
                "mpltern is required for ternary plots. Install with: pip install mpltern"
            )

        self._setup_ternary_axis(ax, maj_loc, min_loc, maxt, maxl, maxr, trianglesize,
                                 black_background=black_background)

        ax.scatter(
            R, G, B,
            s=s,
            facecolors="None",
            edgecolors=colours,
            lw=lws,
            marker="o",
        )
        return fig, axs

    def ternary_contour_plot(
        self,
        fig,
        axs,
        t: np.ndarray,
        l: np.ndarray,
        r: np.ndarray,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        maj_loc: float = 0.2,
        min_loc: float = 0.1,
        gridsize: int = 100,
        bins: Optional[int] = None,
        cmap: str = "gist_gray",
        maxt: float = 1,
        maxl: float = 1,
        maxr: float = 1,
        trianglesize: float = 1,
        ecolour: str = "red",
        s: float = 25,
        lws: float = 0.5,
        black_background: bool = True,
    ) -> Tuple[Any, Any]:
        """Create a ternary contour (hexbin) plot with scatter overlay.

        ``axs[1]`` is replaced by the ternary axis; ``axs[0]`` is preserved so
        that callers can continue to use it for an accompanying histogram or other
        panel.

        Args:
            fig: Figure object.
            axs: Axes array — ``axs[1]`` is removed and replaced by the ternary axis.
            t: T (top) hexbin values.
            l: L (left) hexbin values.
            r: R (right) hexbin values.
            R: T scatter values.
            G: L scatter values.
            B: R scatter values.
            maj_loc: Major tick interval.
            min_loc: Minor tick interval.
            gridsize: Hexbin grid size.
            bins: Hexbin bins argument.
            cmap: Colormap for hexbin.
            maxt: Maximum t value.
            maxl: Maximum l value.
            maxr: Maximum r value.
            trianglesize: Side length of the displayed triangle region.
            ecolour: Edge color for scatter overlay points.
            s: Scatter marker size.
            lws: Scatter marker edge width.
            black_background: Fill triangle interior with black (default: True).

        Returns:
            Tuple of (figure, axes).
        """
        axs[1].remove()

        try:
            import mpltern  # noqa: F401 — registers projection
            ax = fig.add_subplot(2, 1, 2, projection="ternary")
        except ImportError:
            raise ImportError(
                "mpltern is required for ternary plots. Install with: pip install mpltern"
            )

        self._setup_ternary_axis(ax, maj_loc, min_loc, maxt, maxl, maxr, trianglesize,
                                 black_background=black_background)

        ax.hexbin(
            t, l, r,
            gridsize=gridsize,
            edgecolors="none",
            bins=bins,
            cmap=cmap,
            rasterized=True,
        )

        ax.scatter(
            R, G, B,
            s=s,
            facecolors="None",
            edgecolors=ecolour,
            lw=lws,
            marker="o",
        )
        return fig, axs


class AnalysisPlotter(TernaryPlotMixin, DatashaderMixin, ImagePlotMixin, BasePlotter):
    """Analysis-focused plotter with large dataset handling and ternary plots.

    This class is optimised for interactive data analysis and exploration,
    with support for large datasets and quick visualisation. Automatically
    switches to datashader for datasets >1k points (configurable).
    Includes support for ternary (3-component) plots via the TernaryPlotMixin.

    Note: MRO is TernaryPlotMixin -> DatashaderMixin -> ImagePlotMixin -> BasePlotter
    to ensure proper initialization order.
    """

    def __init__(self, datashader_threshold: int = 1000):
        """Initialize analysis plotter with defaults optimised for exploration.

        Args:
            datashader_threshold: Number of points above which to use datashader
                for faster rendering. Default is 1000. Set to None to disable
                auto-switching and always use matplotlib.
        """
        config = PlottingConfig()
        config.DEFAULT_FIGSIZE = (10, 6)  # Larger for exploration/analysis
        # No need to override DEFAULT_DPI - already 100 for display

        # Initialize with proper MRO
        super().__init__(config=config, datashader_threshold=datashader_threshold)
