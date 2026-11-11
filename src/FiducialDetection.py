"""
Fiducial Detection Module

Contains functions for detecting and processing fiducial markers for drift correction.
Extracted from DriftCorrectionFunctions.py for better code organisation.

This module handles:
- High-density region detection from images
- Puncta selection from detected regions
- Fiducial detection with temporal chunking
- Fiducial clustering and validation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import warnings
import sys
from pathlib import Path
import gc

sys.path.append(str(Path(__file__).parent))

from pyS3M.ImportManager import get_module, is_available
from pyS3M.PlottingBase import AnalysisPlotter, PublicationPlotter
from pyS3M.Constants import DriftConstants, AnalysisConfig
import logging
logger = logging.getLogger(__name__)


# Import scipy components
try:
    from scipy import ndimage
    from scipy.optimize import curve_fit
except ImportError:
    warnings.warn(
        "scipy not available - some fiducial detection features may not work."
    )
    ndimage = None
    curve_fit = None

# Get modules through import manager
plt = get_module("matplotlib.pyplot")

# Lazy load local modules to avoid circular imports
postprocess = None


def _ensure_postprocess():
    """Lazy load postprocess module."""
    global postprocess
    if postprocess is None:
        postprocess = get_module("postprocess")
    return postprocess


class FiducialDetector:
    """Class containing fiducial detection and selection functionality."""

    def __init__(self, drift_correction_instance=None, config: AnalysisConfig = None):
        """
        Initialise with reference to main drift correction instance if needed.

        Args:
            drift_correction_instance: Reference to main DriftCorrectionFunctions instance
            config: AnalysisConfig controlling display, save, and callback behaviour.
        """
        self.drift_correction = drift_correction_instance
        self.config = config if config is not None else AnalysisConfig()

    def detect_high_density_regions_from_image(
        self,
        smoothed_image: np.ndarray,
        histogram_bins: int = 256,
        threshold_percentile: float = 99.0,
        pixelsize: float = 100.0,
        output_figure_path: Optional[str] = None,
        title: str = "High-Density Region Detection",
        create_plot: bool = True,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, float, Dict[str, Any]]:
        """Detect high-density regions from a smoothed image using histogram analysis.

        This function takes a pre-smoothed/rendered image and identifies high-density
        regions based on histogram analysis. It provides clear visualization of the
        detection process and outputs region coordinates for downstream processing.

        Args:
            smoothed_image: Pre-smoothed 2D image array (e.g., from render functions)
            histogram_bins: Number of bins for histogram analysis
            threshold_percentile: Percentile threshold for region detection (0-100)
            pixelsize: Pixel size in nm for scale bar visualization
            output_figure_path: Optional path to save the detection figure
            title: Title for the detection plot
            create_plot: Whether to create visualization plots

        Returns:
            Tuple containing:
            - List of (y, x) coordinates of detected high-density region centres
            - Binary mask of detected regions
            - Threshold value used for detection
            - Metadata dictionary with detection statistics
        """
        # Calculate histogram and threshold
        image_flat = smoothed_image.ravel()
        image_flat = image_flat[image_flat > 0]  # Exclude zero values

        if len(image_flat) == 0:
            raise ValueError("Image contains no non-zero values")

        hist, bin_edges = np.histogram(image_flat, bins=histogram_bins)
        threshold = np.percentile(image_flat, threshold_percentile)

        # Create binary mask of high-density regions
        binary_mask = smoothed_image > threshold

        # Find connected components / regions
        if ndimage is None:
            raise RuntimeError("scipy.ndimage required for region detection")

        labeled_regions, n_regions = ndimage.label(binary_mask)

        # Calculate region centres and properties
        region_centres = []
        region_stats = []

        for region_id in range(1, n_regions + 1):
            region_mask = labeled_regions == region_id
            region_coords = np.where(region_mask)

            if len(region_coords[0]) > 0:
                # Calculate centre of mass
                centre_y = np.mean(region_coords[0])
                centre_x = np.mean(region_coords[1])
                region_centres.append((int(centre_y), int(centre_x)))

                # Calculate region statistics
                region_area = np.sum(region_mask)
                region_intensity = np.sum(smoothed_image[region_mask])
                region_max_intensity = np.max(smoothed_image[region_mask])

                region_stats.append(
                    {
                        "centre": (centre_y, centre_x),
                        "area_pixels": region_area,
                        "total_intensity": region_intensity,
                        "max_intensity": region_max_intensity,
                        "mean_intensity": (
                            region_intensity / region_area if region_area > 0 else 0
                        ),
                    }
                )

        # Create visualization if requested
        if create_plot:
            self._plot_density_detection_results(
                smoothed_image,
                binary_mask,
                region_centres,
                hist,
                bin_edges,
                threshold,
                pixelsize,
                output_figure_path,
                title,
            )

        # Prepare metadata
        metadata = {
            "n_regions_detected": n_regions,
            "threshold_value": threshold,
            "threshold_percentile": threshold_percentile,
            "histogram_bins": histogram_bins,
            "image_shape": smoothed_image.shape,
            "image_max": np.max(smoothed_image),
            "image_mean": np.mean(smoothed_image[smoothed_image > 0]),
            "region_statistics": region_stats,
            "total_region_area": np.sum(binary_mask),
            "region_area_fraction": np.sum(binary_mask) / binary_mask.size,
        }

        return region_centres, binary_mask, threshold, metadata

    def select_puncta_from_regions(
        self,
        locs: np.recarray,
        region_centres: List[Tuple[int, int]],
        binary_mask: np.ndarray,
        pixelsize: float = 100.0,
        selection_box_size_nm: float = 600.0,
        min_localisations_per_region: int = 10,
        output_figure_path: Optional[str] = None,
        title: str = "Puncta Selection from Regions",
        create_plot: bool = True,
        plot_individual_regions: bool = True,
        use_datashader_threshold: int = 1000,
        memory_optimise: bool = True,
    ) -> Tuple[List[np.recarray], Dict[str, Any]]:
        """Select puncta (localisations) from detected high-density regions.

        This function takes the output from detect_high_density_regions_from_image
        and selects localisations within rectangular boxes around each detected region centre
        to create potential fiducial candidates. Uses the optimized postprocess.picked_locs
        function with Rectangle shape.

        Args:
            locs: Localisation data with xc, yc, frame fields
            region_centres: List of (y, x) coordinates from density detection
            binary_mask: Binary mask from density detection
            pixelsize: Pixel size in nm for coordinate conversion
            selection_box_size_nm: Size of square selection box around each region centre (nm)
            min_localisations_per_region: Minimum number of localisations required for valid region
            output_figure_path: Optional path to save selection visualisation
            title: Title for visualisation plots
            create_plot: Whether to create visualisation plots
            plot_individual_regions: Whether to plot individual region details
            use_datashader_threshold: Use datashader for scatter plots with more than this many points
            memory_optimise: Whether to use memory optimisation

        Returns:
            Tuple containing:
            - List of localisation arrays, one per valid region
            - Metadata dictionary with selection statistics
        """
        # Ensure postprocess is loaded
        pp = _ensure_postprocess()
        if pp is None:
            raise RuntimeError(
                "postprocess module not available - cannot use picked_locs function"
            )

        # Handle empty region centres
        if not region_centres:
            metadata = {
                "n_regions_input": 0,
                "n_regions_selected": 0,
                "selection_criteria": {
                    "min_localisations": min_localisations_per_region,
                    "selection_box_size_nm": selection_box_size_nm,
                    "selection_box_size_pixels": 0.0,
                },
                "rejection_reasons": {"too_few_localisations": 0, "accepted": 0},
                "region_statistics": [],
            }
            return [], metadata

        # Convert box size from nm to pixels
        box_size_pixels = selection_box_size_nm / pixelsize
        half_box = box_size_pixels / 2.0

        # Create horizontal line picks for Rectangle shape
        picks = []
        for centre_y, centre_x in region_centres:
            picks.append(
                ((centre_x - half_box, centre_y), (centre_x + half_box, centre_y))
            )

        # Use postprocess.picked_locs with parallelization if we have 8+ picks
        width = max(locs.xc.max() + 10, 100)
        height = max(locs.yc.max() + 10, 100)

        picked_locs_arrays = pp.picked_locs(
            locs=locs,
            width=width,
            height=height,
            picks=picks,
            pick_shape="Rectangle",
            pick_size=box_size_pixels,
            add_group=False,
            callback="console",
            parallel=len(picks) >= 8,
        )

        # Memory cleanup
        if memory_optimise:
            del picks
            gc.collect()

        # Filter results based on minimum localisation count
        selected_puncta = []
        region_stats = []

        if picked_locs_arrays is None:
            picked_locs_arrays = []

        rejected_count = 0
        for region_id, (region_locs, (centre_y, centre_x)) in enumerate(
            zip(picked_locs_arrays, region_centres)
        ):
            n_locs = len(region_locs)

            if n_locs >= min_localisations_per_region:
                selected_puncta.append(region_locs)

                region_stat = {
                    "region_id": region_id,
                    "centre_y": centre_y,
                    "centre_x": centre_x,
                    "n_localisations": n_locs,
                    "mean_x": np.mean(region_locs.xc),
                    "mean_y": np.mean(region_locs.yc),
                    "std_x": np.std(region_locs.xc),
                    "std_y": np.std(region_locs.yc),
                    "frame_range": [
                        int(region_locs.frame.min()),
                        int(region_locs.frame.max()),
                    ],
                    "frame_span": int(
                        region_locs.frame.max() - region_locs.frame.min() + 1
                    ),
                    "selection_box_size_nm": selection_box_size_nm,
                    "selection_box_size_pixels": box_size_pixels,
                    "box_boundaries": {
                        "x_min": centre_x - half_box,
                        "x_max": centre_x + half_box,
                        "y_min": centre_y - half_box,
                        "y_max": centre_y + half_box,
                    },
                }

                if hasattr(region_locs, "photoelectrons"):
                    region_stat["mean_photoelectrons"] = np.mean(region_locs.photoelectrons)
                    region_stat["std_photoelectrons"] = np.std(region_locs.photoelectrons)

                region_stats.append(region_stat)
            else:
                rejected_count += 1
                if memory_optimise:
                    del region_locs

            if memory_optimise and region_id % 100 == 0 and region_id > 0:
                gc.collect()
                _n = len(picked_locs_arrays)
                msg = (f"Processed {region_id + 1}/{_n} regions "
                       f"({len(selected_puncta)} accepted, {rejected_count} rejected)")
                logger.info(msg)
                if self.config.logging_callback:
                    self.config.logging_callback(msg)
                if self.config.progress_callback:
                    self.config.progress_callback((region_id + 1) / _n, msg)

        # Final memory cleanup
        if memory_optimise:
            del picked_locs_arrays
            gc.collect()
            msg = "Memory optimisation: Freed intermediate arrays after region processing"
            logger.info(msg)
            if self.config.logging_callback:
                self.config.logging_callback(msg)

        # Create visualization if requested
        if create_plot:
            self._plot_puncta_selection_results(
                locs,
                selected_puncta,
                region_centres,
                binary_mask,
                region_stats,
                box_size_pixels,
                pixelsize,
                output_figure_path,
                title,
                plot_individual_regions,
                use_datashader_threshold,
            )

            if memory_optimise and plt is not None:
                plt.close("all")
                gc.collect()

        # Prepare metadata
        total_locs_selected = sum(len(puncta) for puncta in selected_puncta)

        metadata = {
            "n_regions_input": len(region_centres),
            "n_regions_selected": len(selected_puncta),
            "n_regions_rejected": rejected_count,
            "selection_rate": (
                len(selected_puncta) / len(region_centres) if region_centres else 0
            ),
            "selection_criteria": {
                "min_localisations": min_localisations_per_region,
                "selection_box_size_nm": selection_box_size_nm,
                "selection_box_size_pixels": box_size_pixels,
            },
            "region_statistics": region_stats,
            "total_selected_localisations": total_locs_selected,
            "memory_optimized": memory_optimise,
            "rejection_reasons": {
                "too_few_localisations": rejected_count,
                "accepted": len(selected_puncta),
            },
        }

        return selected_puncta, metadata

    def remove_puncta_locs(
        self,
        locs: np.recarray,
        selected_puncta: List[np.recarray],
    ) -> np.recarray:
        """Remove fiducial / puncta localisations from a localisation array.

        Takes the ``selected_puncta`` list returned by
        :meth:`select_puncta_from_regions` and removes every localisation that
        belongs to any of those regions from ``locs``, returning the remainder.

        Matching is performed by exact equality on ``(frame, xc, yc)`` — this
        is safe because the localisations in ``selected_puncta`` are direct
        copies of rows in ``locs`` with no arithmetic transformation.

        Args:
            locs: Full localisation recarray (all molecules).
            selected_puncta: List of recarrays as returned by
                ``select_puncta_from_regions``.  May be empty.

        Returns:
            Localisation recarray with all puncta rows removed.
        """
        if not selected_puncta:
            return locs

        all_puncta = np.concatenate(selected_puncta)

        # Build a set of (frame, xc, yc) keys from puncta for O(1) lookup.
        # Use ['field'] indexing: works for both recarrays and plain structured arrays.
        puncta_keys = set(
            zip(
                all_puncta['frame'].tolist(),
                all_puncta['xc'].tolist(),
                all_puncta['yc'].tolist(),
            )
        )

        keep = np.array(
            [
                (int(f), float(x), float(y)) not in puncta_keys
                for f, x, y in zip(locs['frame'], locs['xc'], locs['yc'])
            ],
            dtype=bool,
        )

        n_removed = int((~keep).sum())
        msg = (f"remove_puncta_locs: removed {n_removed:,} localisations "
               f"({n_removed / max(len(locs), 1) * 100:.1f}% of {len(locs):,} total)")
        logger.info(msg)
        if self.config.logging_callback:
            self.config.logging_callback(msg)

        return locs[keep]

    def identify_real_fiducials_with_clustering(
        self,
        selected_puncta: List[np.recarray],
        retention_percentage: float = 0.9,
        min_samples_factor: float = 0.7,
        frame_count: int = 100000,
        pixelsize: float = DriftConstants.XIMEA_PIXEL_SIZE_NM,
        output_figure_path: Optional[str] = None,
        title: str = "Fiducial Gaussian Fitting Analysis",
        create_plot: bool = True,
    ) -> Tuple[List[np.recarray], Dict[str, Any]]:
        """Identify real fiducials using single Gaussian distribution fitting.

        This function applies single Gaussian mixture fitting to identify real fiducial markers
        by fitting each region to a single 2D Gaussian and keeping a specified percentage
        of points based on their distance from the Gaussian centre.

        Args:
            selected_puncta: List of puncta arrays from region selection
            retention_percentage: Fraction of points to retain (0-1)
            min_samples_factor: Minimum fraction of points required for valid fit
            frame_count: Total number of frames for normalization
            pixelsize: Pixel size in nm
            output_figure_path: Optional path to save plots
            title: Title for plots
            create_plot: Whether to create visualisation plots

        Returns:
            Tuple of (validated_fiducials, validation_metadata)
        """
        validated_fiducials = []
        validation_metadata = []
        _n_puncta = len(selected_puncta)

        for i, puncta in enumerate(selected_puncta):
            if self.config.progress_callback and _n_puncta > 0:
                self.config.progress_callback(i / _n_puncta, f"Validating region {i}/{_n_puncta}")
            if len(puncta) == 0:
                continue

            # Fit 2D Gaussian to the localizations
            x = puncta.xc
            y = puncta.yc

            try:
                # Calculate centre and spread
                x_mean, y_mean = np.mean(x), np.mean(y)
                x_std, y_std = np.std(x), np.std(y)

                # Calculate distances from centre
                distances = np.sqrt((x - x_mean) ** 2 + (y - y_mean) ** 2)

                # Keep only the closest retention_percentage of points
                n_keep = int(len(distances) * retention_percentage)
                if n_keep < min_samples_factor * len(distances):
                    continue

                closest_indices = np.argsort(distances)[:n_keep]
                validated_puncta = puncta[closest_indices]

                validated_fiducials.append(validated_puncta)

                metadata = {
                    "region_id": i,
                    "n_input": len(puncta),
                    "n_output": len(validated_puncta),
                    "retention_rate": len(validated_puncta) / len(puncta),
                    "centre_x": x_mean,
                    "centre_y": y_mean,
                    "std_x": x_std,
                    "std_y": y_std,
                }
                validation_metadata.append(metadata)

            except Exception as e:
                warnings.warn(f"Failed to validate region {i}: {e}")
                continue

        if create_plot:
            self._plot_clustering_results(
                selected_puncta,
                validated_fiducials,
                validation_metadata,
                output_figure_path,
                title,
            )

        return validated_fiducials, validation_metadata

    def _plot_density_detection_results(
        self,
        smoothed_image,
        binary_mask,
        region_centres,
        hist,
        bin_edges,
        threshold,
        pixelsize,
        output_figure_path,
        title,
    ):
        """Create visualization of density detection results."""
        if not is_available("matplotlib.pyplot"):
            return

        try:
            DriftPlotter(config=self.config).create_separate_plots(
                smoothed_image,
                binary_mask,
                region_centres,
                hist,
                bin_edges,
                threshold,
                pixelsize,
                output_figure_path,
                title,
            )
        except Exception as e:
            warnings.warn(f"Failed to create density detection plots: {e}")

    def _plot_puncta_selection_results(
        self,
        locs,
        selected_puncta,
        region_centres,
        binary_mask,
        region_stats,
        box_size_pixels,
        pixelsize,
        output_figure_path,
        title,
        plot_individual_regions,
        use_datashader_threshold,
    ):
        """Create visualization of puncta selection results."""
        if not is_available("matplotlib.pyplot"):
            return

        try:
            DriftPlotter(config=self.config).plot_puncta_selection_results(
                locs,
                selected_puncta,
                region_centres,
                binary_mask,
                region_stats,
                box_size_pixels,
                pixelsize,
                output_figure_path,
                title,
                plot_individual_regions,
                use_datashader_threshold,
            )
        except Exception as e:
            warnings.warn(f"Failed to create puncta selection plots: {e}")

    def _plot_clustering_results(
        self,
        selected_puncta,
        validated_fiducials,
        validation_metadata,
        output_figure_path,
        title,
    ):
        """Create visualization of clustering results."""
        if not is_available("matplotlib.pyplot"):
            return

        try:
            DriftPlotter(config=self.config).plot_clustering_results(
                selected_puncta,
                validated_fiducials,
                validation_metadata,
                output_figure_path,
                title,
            )
        except Exception as e:
            warnings.warn(f"Failed to create clustering plots: {e}")


class DriftPlotter(AnalysisPlotter):
    """Specialised drift correction / fiducial plotting utilities.

    Inherits from AnalysisPlotter for consistent styling and large-dataset support.
    Moved from DriftPlotting.py (2026-04-10).
    """

    def __init__(self, config: AnalysisConfig = None):
        super().__init__()
        self.io_config = config if config is not None else AnalysisConfig()

    def plot_fiducial_detection_steps(
        self,
        image: np.ndarray,
        hist: tuple,
        threshold: float,
        all_picks: list,
        valid_picks: list,
        result,
        info: List[dict],
        save_path: Optional[str] = None,
    ) -> None:
        """Create step-by-step visualization of fiducial detection process."""
        try:
            from typing import TYPE_CHECKING
            fig, axes = self.two_column_plot(
                ncols=2,
                nrows=3,
                width_ratios=[1.0, 1.0],
                height_ratios=[1.0, 1.0, 0.8],
                big=True,
            )

            pixelsize_nm = info[0]["Pixelsize"] * 1000 if info else 100.0

            ax1 = axes[0, 0]
            self.image_plot(
                ax1,
                image,
                pixelsize=pixelsize_nm / 1000,
                vmax=np.percentile(image, 99),
                vmin=0,
                title=f"Rendered Image ({len(all_picks)} candidates)",
                colorbar=True,
            )

            if all_picks:
                picks_array = np.array(all_picks)
                ax1.scatter(
                    picks_array[:, 1],
                    picks_array[:, 0],
                    c="red", s=20, alpha=0.7, marker="x",
                    label=f"All candidates ({len(all_picks)})",
                )
                ax1.legend()

            ax2 = axes[0, 1]
            if hist:
                hist_values, bin_edges = hist
                bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax2.bar(bin_centres, hist_values,
                        width=bin_centres[1] - bin_centres[0],
                        alpha=0.7, color="skyblue", edgecolor="black")
                ax2.axvline(threshold, color="red", linestyle="--", linewidth=2,
                            label=f"Threshold: {threshold:.1f}")
                self.setup_axis(ax2, xlabel="Intensity", ylabel="Frequency",
                                title="Intensity Histogram Analysis", grid=True)
                ax2.legend()

            ax3 = axes[1, 0]
            self.image_plot(
                ax3,
                image,
                pixelsize=pixelsize_nm / 1000,
                vmax=np.percentile(image, 99),
                vmin=0,
                title=f"Valid Fiducials ({len(valid_picks)})",
                colorbar=True,
            )

            if valid_picks:
                valid_array = np.array(valid_picks)
                ax3.scatter(
                    valid_array[:, 1], valid_array[:, 0],
                    c="lime", s=30, alpha=0.8, marker="o",
                    edgecolors="black", linewidth=0.5,
                    label=f"Valid fiducials ({len(valid_picks)})",
                )
                ax3.legend()

            ax4 = axes[1, 1]
            if result.locs_with_groups is not None and len(result.locs_with_groups) > 0:
                try:
                    unique_groups = np.unique(result.locs_with_groups.group)
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
                    for i, group_id in enumerate(unique_groups):
                        group_locs = result.locs_with_groups[
                            result.locs_with_groups.group == group_id
                        ]
                        ax4.scatter(group_locs.xc, group_locs.yc,
                                    c=[colors[i]], s=8, alpha=0.7,
                                    label=f"Fid {group_id} ({len(group_locs)})")
                    self.setup_axis(ax4, xlabel="X (pixels)", ylabel="Y (pixels)",
                                    title=f"Grouped Localisations ({result.n_fiducials} fiducials)",
                                    grid=True, equal_aspect=True)
                    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                except Exception as e:
                    ax4.text(0.5, 0.5, f"Error plotting groups: {e}",
                             transform=ax4.transAxes, ha="center", va="center")

            # axes[2, :] is a row of 2 separate Axes (two_column_plot has no
            # gridspec-merge support to span them as one) -- hide both, write
            # the summary text on the first.
            for _ax in axes[2, :]:
                _ax.axis("off")
            ax5 = axes[2, 0]
            summary_text = "Detection Summary:\n"
            summary_text += f"• Total candidates found: {len(all_picks)}\n"
            summary_text += f"• Valid fiducials: {len(valid_picks)}\n"
            summary_text += f"• Final fiducial groups: {result.n_fiducials}\n"
            if result.metadata:
                summary_text += f"• Total localisations: {result.metadata.get('total_localisations', 'N/A')}\n"
                summary_text += f"• Threshold used: {result.metadata.get('threshold_used', 'N/A'):.1f}\n"
            ax5.text(0.05, 0.9, summary_text, transform=ax5.transAxes, fontsize=12,
                     verticalalignment="top",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))

            self.save_or_show(fig, save_path=save_path, show=self.io_config.display, dpi=self.io_config.dpi)

        except Exception as e:
            logger.warning(f"⚠️ Failed to create fiducial detection steps plot: {e}")
            import traceback
            traceback.print_exc()

    def plot_fiducial_detection_results(
        self,
        result,
        info: List[dict],
        save_path: Optional[str] = None,
    ) -> None:
        """Create a plot of fiducial detection results."""
        try:
            from pyS3M.CoordinateProcessing import CoordinateProcessor

            meta = CoordinateProcessor.extract_metadata(info)
            pixelsize = meta.get("pixelsize", 130.0)

            fig, (ax1, ax2) = self.two_column_plot(nrows=1, ncols=2)

            fiducial_x = [pick[0] for pick in result.picks]
            fiducial_y = [pick[1] for pick in result.picks]

            # image_scatter_plot is only on PublicationPlotter — use a dedicated instance
            pub = PublicationPlotter(poster=False, dark_background=False)
            pub.image_scatter_plot(
                ax1,
                data=result.detection_image,
                xdata=np.array(fiducial_x),
                ydata=np.array(fiducial_y),
                cmap="hot",
                cbar="on",
                cbarlabel="Intensity",
                label=f"Detected Fiducials ({result.n_fiducials})",
                labelcolor="cyan",
                pixelsize=pixelsize,
                scalebarsize=1000,
                scalebarlabel="1 μm",
                scattercolor="cyan",
                s=100,
                scatteralpha=0.8,
            )
            ax1.set_title("Fiducial Detection Results")

            fiducial_locs = result.locs_with_groups[result.locs_with_groups.group >= 0]
            if len(fiducial_locs) > 0:
                unique_groups = np.unique(fiducial_locs.group)
                color_list = ["red", "blue", "green", "orange", "purple",
                               "brown", "pink", "gray", "olive", "cyan"]
                colors = (color_list * (len(unique_groups) // len(color_list) + 1))[
                    : len(unique_groups)
                ]
                for i, group_id in enumerate(unique_groups):
                    group_locs = fiducial_locs[fiducial_locs.group == group_id]
                    ax2.scatter(
                        group_locs.xc * 1000,
                        group_locs.yc * 1000,
                        s=2, alpha=0.6, c=[colors[i]],
                        label=f"Fiducial {group_id+1} ({len(group_locs)} locs)",
                        rasterized=True,
                    )

            self.setup_axis(ax2, xlabel="X (nm)", ylabel="Y (nm)",
                            title="Fiducial Localisations by Group",
                            grid=True, equal_aspect=True)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            summary_text = (
                f"Detection Summary:\n"
                f"• Threshold: {result.detection_params['threshold_percentile']:.1f}%\n"
                f"• Box size: {result.detection_params['box_size_nm']:.0f} nm\n"
                f"• Min frames: {result.detection_params['min_frames_fraction']:.1%}\n"
                f"• Candidates found: {result.metadata['total_candidates']}\n"
                f"• Valid fiducials: {result.n_fiducials}"
            )
            fig.text(0.02, 0.98, summary_text, transform=fig.transFigure,
                     verticalalignment="top", fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

            self.save_or_show(fig, save_path=save_path, show=self.io_config.display, dpi=self.io_config.dpi)

        except Exception as e:
            logger.warning(f"⚠️ Failed to create fiducial detection plot: {e}")

    def plot_region_data_with_datashader(self, ax, data_list, color_list, title):
        """Plot region data using optimised rendering based on dataset size."""
        if len(data_list) == 1:
            data = data_list[0]
            self.plot_large_scatter(
                ax, data["xc"], data["yc"],
                threshold=10000,
                cmap=color_list[0] if color_list else "blue",
                s=2, alpha=0.7, rasterized=True,
            )
        else:
            datasets = [{"x": data["xc"], "y": data["yc"]} for data in data_list]
            labels = [f"Group {i+1}" for i in range(len(data_list))]
            self.plot_multi_dataset_scatter(
                ax, datasets, labels=labels,
                colors=color_list if color_list else None,
                threshold=10000, alpha=0.6, sizes=2.0, rasterized=True,
            )
        self.setup_axis(ax, xlabel="X (pixels)", ylabel="Y (pixels)",
                        title=title, grid=True, equal_aspect=True)

    def plot_clustering_overlay(self, ax, all_x, all_y, types, title):
        """Plot clustering overlay showing original vs validated points."""
        original_mask = np.array(types) == "original"
        validated_mask = np.array(types) == "validated"

        datasets, labels, colors, sizes = [], [], [], []
        if np.any(original_mask):
            datasets.append({"x": all_x[original_mask], "y": all_y[original_mask]})
            labels.append("Original"); colors.append("lightgray"); sizes.append(1.0)
        if np.any(validated_mask):
            datasets.append({"x": all_x[validated_mask], "y": all_y[validated_mask]})
            labels.append("Validated"); colors.append("red"); sizes.append(3.0)

        if len(all_x) > 10000:
            self.plot_multi_dataset_scatter(ax, datasets, labels=labels,
                colors=colors, sizes=sizes, threshold=10000, alpha=0.6, rasterized=True)
        else:
            for dataset, label, color, size in zip(datasets, labels, colors, sizes):
                alpha = 0.3 if label == "Original" else 0.9
                ax.scatter(dataset["x"], dataset["y"], c=color, s=size,
                           alpha=alpha, label=label, rasterized=True)
            ax.legend()

        self.setup_axis(ax, xlabel="X (pixels)", ylabel="Y (pixels)",
                        title=title, grid=True, equal_aspect=True)

    def plot_puncta_selection_results(
        self,
        all_locs: np.recarray,
        selected_puncta: List[np.recarray],
        region_centres: List[Tuple[int, int]],
        binary_mask: np.ndarray,
        region_stats: List[Dict[str, Any]],
        box_size_pixels: float,
        pixelsize: float,
        output_figure_path: Optional[str],
        title: str,
        plot_individual_regions: bool = True,
        use_datashader_threshold: int = 10000,
    ) -> None:
        """Create visualization of puncta selection results with optimized rendering."""
        try:
            fig, axes = self.two_column_plot(nrows=2, ncols=2)
            axes = axes.flatten()

            ax = axes[0]
            n_locs = len(all_locs)
            if n_locs > use_datashader_threshold:
                self.create_preview_plot(ax, all_locs.xc, all_locs.yc,
                                         preview_points=use_datashader_threshold,
                                         method="density", s=1, alpha=0.5, c="blue",
                                         rasterized=True)
            else:
                ax.scatter(all_locs.xc, all_locs.yc, s=1, alpha=0.5, c="blue",
                           label="All localisations", rasterized=True)

            if region_centres:
                centres = np.array(region_centres)
                ax.scatter(centres[:, 1], centres[:, 0], s=100, c="red", marker="x",
                           linewidth=2, label=f"Region centres ({len(region_centres)})")
            self.setup_axis(ax, xlabel="X (pixels)", ylabel="Y (pixels)",
                            title=f"All Localisations ({n_locs:,} points)",
                            grid=True, equal_aspect=True)
            ax.legend()

            ax = axes[1]
            if selected_puncta:
                datasets = [{"x": p.xc, "y": p.yc} for p in selected_puncta if len(p) > 0]
                labels = [f"Region {i} ({len(p)})"
                          for i, p in enumerate(selected_puncta) if len(p) > 0]
                if datasets:
                    self.plot_multi_dataset_scatter(
                        ax, datasets, labels=labels if len(datasets) <= 10 else None,
                        threshold=use_datashader_threshold, alpha=0.6, sizes=2.0,
                        rasterized=True)
            self.setup_axis(ax, xlabel="X (pixels)", ylabel="Y (pixels)",
                            title=f"Selected Puncta ({len(selected_puncta)} regions)",
                            grid=True, equal_aspect=True)

            ax = axes[2]
            ax.imshow(binary_mask, cmap="gray", origin="lower")
            if region_centres:
                centres = np.array(region_centres)
                ax.scatter(centres[:, 1], centres[:, 0], s=50, c="red", marker="o", alpha=0.7)
            self.setup_axis(ax, xlabel="X (pixels)", ylabel="Y (pixels)",
                            title="Density Mask", grid=False)

            ax = axes[3]
            if region_stats:
                stats_text = [
                    f"PUNCTA SELECTION SUMMARY", f"",
                    f"Total localisations: {n_locs:,}",
                    f"Detected regions: {len(region_centres)}",
                    f"Selected regions: {len(selected_puncta)}",
                    f"Box size: {box_size_pixels:.1f} px ({box_size_pixels * pixelsize:.1f} nm)",
                    f"Pixel size: {pixelsize:.2f} nm",
                ]
                if selected_puncta:
                    stats_text.extend([
                        f"",
                        f"Average locs per region: {np.mean([len(p) for p in selected_puncta]):.1f}",
                        f"Min/Max locs: {min(len(p) for p in selected_puncta)}/{max(len(p) for p in selected_puncta)}",
                    ])
                ax.text(0.05, 0.95, "\n".join(stats_text), transform=ax.transAxes,
                        fontsize=11, verticalalignment="top", fontfamily="monospace",
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

            plt.suptitle(title, fontsize=14, fontweight="bold")
            self.save_or_show(fig, save_path=output_figure_path, show=self.io_config.display, dpi=self.io_config.dpi)

        except Exception as e:
            logger.warning(f"⚠️ Failed to create puncta selection plot: {e}")
            import traceback
            traceback.print_exc()

    def plot_individual_clustering_details(
        self,
        selected_puncta: List[np.recarray],
        validated_fiducials: List[np.recarray],
        clustering_metadata: List[Dict[str, Any]],
        base_path: str,
        title: str,
    ) -> None:
        """Create detailed plots for individual clustering results."""
        try:
            n_validated = len(validated_fiducials)
            if n_validated == 0:
                logger.warning("⚠️ No validated fiducials to plot")
                return

            cols = min(3, n_validated)
            rows = (n_validated + cols - 1) // cols
            fig, axes = self.two_column_plot(nrows=rows, ncols=cols, big=True)
            if n_validated == 1:
                axes = [axes]
            elif rows > 1:
                axes = axes.flatten()

            for i, (selected, validated, metadata) in enumerate(
                zip(selected_puncta[:n_validated], validated_fiducials[:n_validated],
                    clustering_metadata[:n_validated])
            ):
                ax = axes[i]
                if len(selected) > 0:
                    ax.scatter(selected.xc, selected.yc, c="lightblue", s=20,
                               alpha=0.6, label="Selected")
                if len(validated) > 0:
                    ax.scatter(validated.xc, validated.yc, c="red", s=30,
                               alpha=0.8, label="Validated")
                ax.set_title(f"Region {i}: {len(validated)} validated")
                ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
                ax.legend(); ax.grid(True, alpha=0.3)
                ax.set_aspect("equal", adjustable="box")

            for i in range(n_validated, len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(f"{title} - Individual Clustering Details", fontsize=14)
            filename = f"{base_path}_details.{self.io_config.figure_format}"
            self.save_or_show(fig, save_path=filename, show=self.io_config.display, dpi=self.io_config.dpi)
            logger.info(f"Saved individual clustering details: {filename}")

        except Exception as e:
            logger.warning(f"⚠️ Error creating individual clustering details: {e}")

    def plot_clustering_results(
        self,
        selected_puncta: List[np.recarray],
        validated_fiducials: List[np.recarray],
        clustering_metadata: List[Dict[str, Any]],
        output_figure_path: Optional[str],
        title: str,
    ) -> None:
        """Create visualization of clustering results."""
        try:
            n_regions = len(selected_puncta)
            n_validated = len(validated_fiducials)
            if n_regions == 0:
                logger.warning("⚠️ No puncta regions to plot")
                return

            fig, axes = self.two_column_plot(nrows=2, ncols=2)
            axes = axes.flatten()

            ax = axes[0]
            for i, puncta in enumerate(selected_puncta):
                if len(puncta) > 0:
                    ax.scatter(puncta.xc, puncta.yc, c=[plt.cm.tab10(i % 10)],
                               s=20, alpha=0.6, label=f"Region {i}")
            ax.set_title(f"Original Puncta ({n_regions} regions)")
            ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)"); ax.grid(True, alpha=0.3)
            if n_regions <= 10:
                ax.legend()

            ax = axes[1]
            for i, validated in enumerate(validated_fiducials):
                if len(validated) > 0:
                    ax.scatter(validated.xc, validated.yc, c=[plt.cm.tab10(i % 10)],
                               s=30, alpha=0.8, label=f"Validated {i}")
            ax.set_title(f"Validated Fiducials ({n_validated} regions)")
            ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)"); ax.grid(True, alpha=0.3)
            if n_validated <= 10:
                ax.legend()

            ax = axes[2]
            retention_rates = []
            region_sizes = []
            for selected, validated in zip(selected_puncta, validated_fiducials):
                if len(selected) > 0:
                    retention_rates.append(len(validated) / len(selected) * 100)
                    region_sizes.append(len(selected))
                else:
                    retention_rates.append(0)
                    region_sizes.append(0)

            if retention_rates:
                bars = ax.bar(range(len(retention_rates)), retention_rates, alpha=0.7)
                ax.set_xlabel("Region ID"); ax.set_ylabel("Retention Rate (%)")
                ax.set_title("Clustering Retention Rates"); ax.grid(True, alpha=0.3)
                for bar, rate in zip(bars, retention_rates):
                    if rate > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 1, f"{rate:.1f}%",
                                ha="center", va="bottom", fontsize=8)

            ax = axes[3]
            total_selected = sum(len(p) for p in selected_puncta)
            total_validated = sum(len(v) for v in validated_fiducials)
            overall_retention = (total_validated / total_selected * 100
                                  if total_selected > 0 else 0)
            stats_text = [
                f"Regions analyzed: {n_regions}",
                f"Total puncta: {total_selected}",
                f"Validated fiducials: {total_validated}",
                f"Overall retention: {overall_retention:.1f}%",
                f"Avg per region: {total_validated/n_regions:.1f}" if n_regions > 0 else "Avg per region: 0",
            ]
            ax.text(0.1, 0.9, "\n".join(stats_text), transform=ax.transAxes, fontsize=12,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
            ax.set_title("Summary Statistics")

            plt.suptitle(f"{title} - Clustering Results", fontsize=14)

            if output_figure_path:
                base_path = (output_figure_path.rsplit(".", 1)[0]
                             if "." in output_figure_path else output_figure_path)
                _save = f"{base_path}_clustering_results.{self.io_config.figure_format}"
            else:
                _save = None
            self.save_or_show(fig, save_path=_save, show=self.io_config.display, dpi=self.io_config.dpi)
            if _save:
                logger.info(f"Clustering results saved to: {_save}")

        except Exception as e:
            logger.warning(f"⚠️ Error creating clustering results plot: {e}")

    def plot_clustering_summary_only(
        self,
        selected_puncta: List[np.recarray],
        validated_fiducials: List[np.recarray],
        clustering_metadata: List[Dict[str, Any]],
        output_figure_path: Optional[str],
        title: str,
    ) -> None:
        """Create summary visualization only."""
        try:
            n_regions = len(selected_puncta)
            n_validated = len(validated_fiducials)
            if n_regions == 0:
                logger.warning("⚠️ No data to plot in summary")
                return

            fig, axes = self.two_column_plot(nrows=2, ncols=2)
            axes = axes.flatten()

            n_locs = [len(s) for s in selected_puncta]
            n_validated_per_region = [len(v) for v in validated_fiducials]
            retention_rates = [(len(v) / len(s) * 100) if len(s) > 0 else 0
                               for s, v in zip(selected_puncta, validated_fiducials)]

            ax = axes[0]
            bars = ax.bar(range(len(retention_rates)), retention_rates,
                          alpha=0.7, color="skyblue")
            ax.set_xlabel("Region ID"); ax.set_ylabel("Retention Rate (%)")
            ax.set_title("Validation Retention by Region"); ax.grid(True, alpha=0.3)
            if retention_rates:
                mean_retention = np.mean(retention_rates)
                ax.axhline(mean_retention, color="red", linestyle="--",
                           label=f"Mean: {mean_retention:.1f}%")
                ax.legend()

            ax = axes[1]
            if retention_rates:
                ax.hist(retention_rates, bins=10, alpha=0.7,
                        color="lightgreen", edgecolor="black")
                ax.set_xlabel("Retention Rate (%)")
                ax.set_ylabel("Number of Regions")
                ax.set_title("Quality Distribution"); ax.grid(True, alpha=0.3)

            ax = axes[2]
            if n_locs and retention_rates:
                colors_scatter = ["red" if r < 50 else "orange" if r < 75 else "green"
                                  for r in retention_rates]
                ax.scatter(n_locs, retention_rates, c=colors_scatter, alpha=0.7, s=50)
                ax.set_xlabel("Initial Puncta Count"); ax.set_ylabel("Retention Rate (%)")
                ax.set_title("Size vs Quality"); ax.grid(True, alpha=0.3)
                if len(n_locs) > 3:
                    z = np.polyfit(n_locs, retention_rates, 1)
                    p = np.poly1d(z)
                    ax.plot(sorted(n_locs), p(sorted(n_locs)), "r--", alpha=0.8, linewidth=1)

            ax = axes[3]
            total_input = sum(n_locs)
            total_output = sum(n_validated_per_region)
            overall_retention = (total_output / total_input * 100) if total_input > 0 else 0
            stats_text = [
                f"CLUSTERING SUMMARY", f"",
                f"Regions processed: {n_regions}",
                f"Total input puncta: {total_input}",
                f"Total validated fiducials: {total_output}",
                f"Overall retention rate: {overall_retention:.1f}%", f"",
                (f"Mean retention: {np.mean(retention_rates):.1f}% ± {np.std(retention_rates):.1f}%"
                 if retention_rates else "Mean retention: 0%"),
                (f"Best region: {max(retention_rates):.1f}%" if retention_rates else "Best region: 0%"),
                (f"Worst region: {min(retention_rates):.1f}%" if retention_rates else "Worst region: 0%"),
                f"Regions >50% retention: {sum(1 for r in retention_rates if r > 50)}/{n_regions}",
            ]
            ax.text(0.05, 0.95, "\n".join(stats_text), transform=ax.transAxes,
                    fontsize=11, verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

            plt.suptitle(f"{title} - Clustering Summary", fontsize=14, fontweight="bold")

            if output_figure_path:
                base_path = (output_figure_path.rsplit(".", 1)[0]
                             if "." in output_figure_path else output_figure_path)
                _save = f"{base_path}_clustering_summary.{self.io_config.figure_format}"
            else:
                _save = None
            self.save_or_show(fig, save_path=_save, show=self.io_config.display, dpi=self.io_config.dpi)
            if _save:
                logger.info(f"Clustering summary saved to: {_save}")

        except Exception as e:
            logger.warning(f"⚠️ Error creating clustering summary plot: {e}")

    def create_separate_plots(
        self,
        smoothed_image: np.ndarray,
        binary_mask: np.ndarray,
        region_centres: List[Tuple[int, int]],
        hist: np.ndarray,
        bin_edges: np.ndarray,
        threshold: float,
        pixelsize: float,
        output_figure_path: Optional[str],
        title: str,
    ) -> None:
        """Create 4-panel density detection visualization."""
        try:
            fig, axes = self.two_column_plot(nrows=2, ncols=2)
            axes = axes.flatten()

            axes[0].imshow(smoothed_image, cmap="hot", origin="lower")
            self.setup_axis(axes[0], xlabel="X (pixels)", ylabel="Y (pixels)",
                            title="Smoothed Image", grid=False)

            axes[1].imshow(binary_mask, cmap="gray", origin="lower")
            self.setup_axis(axes[1], xlabel="X (pixels)", ylabel="Y (pixels)",
                            title="Binary Mask", grid=False)

            if len(hist) > 0 and len(bin_edges) > 0:
                bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
                axes[2].plot(bin_centres, hist)
                axes[2].axvline(threshold, color="red", linestyle="--",
                                label=f"Threshold: {threshold:.2f}")
                axes[2].legend(fontsize=7)
            self.setup_axis(axes[2], xlabel="Intensity", ylabel="Count",
                            title="Intensity Histogram", grid=True)

            axes[3].imshow(smoothed_image, cmap="hot", origin="lower", alpha=0.7)
            for i, (cy, cx) in enumerate(region_centres):
                axes[3].plot(cx, cy, "wo", markersize=8,
                             markeredgecolor="red", markeredgewidth=2)
                axes[3].text(cx, cy, str(i), color="white",
                             ha="center", va="center", fontsize=8)
            self.setup_axis(axes[3], xlabel="X (pixels)", ylabel="Y (pixels)",
                            title=f"{title} – {len(region_centres)} regions", grid=False)

            if output_figure_path:
                base_path = (output_figure_path.rsplit(".", 1)[0]
                             if "." in output_figure_path else output_figure_path)
                filename = f"{base_path}_density_detection.{self.io_config.figure_format}"
                self.save_or_show(fig, save_path=filename, show=self.io_config.display, dpi=self.io_config.dpi)
                logger.info(f"Density detection plot saved to: {filename}")
            else:
                self.save_or_show(fig, save_path=None, show=self.io_config.display, dpi=self.io_config.dpi)

        except Exception as e:
            logger.warning(f"⚠️ Error creating density detection plots: {e}")
            import traceback
            traceback.print_exc()
