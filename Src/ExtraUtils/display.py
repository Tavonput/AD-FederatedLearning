from typing import Tuple, List, Union, Optional, Dict, Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

import numpy as np


ScalarList = Union[np.ndarray, List[float], List[int]]


@dataclass
class PlotConfig:
    """A configuration class for common plot settings."""
    title:        str                           = "title"
    xlabel:       str                           = "xlabel"
    ylabel:       str                           = "ylabel"
    ylim:         Optional[Tuple[float, float]] = None
    figsize:      Tuple[int, int]               = (10, 6)
    grid:         Tuple[bool, bool]             = (True, True)
    tight_layout: bool                          = True


    def apply(self, fig: Figure, ax: Axes) -> None:
        fig.set_size_inches(self.figsize)
        if self.tight_layout:
            fig.tight_layout()

        ax.set_title(self.title, fontweight="bold")
        ax.set_xlabel(self.xlabel, fontweight="bold")
        ax.set_ylabel(self.ylabel, fontweight="bold")

        ax.set_axisbelow(True)

        if self.grid[0]:
            ax.grid(True, axis="x", linestyle="-")
        if self.grid[1]:
            ax.grid(True, axis="y", linestyle="-")

        if self.ylim is not None:
            ax.set_ylim(self.ylim)


def plot_histogram(
    data:      Dict[str, ScalarList],
    conifg:    PlotConfig,
    bins:      Union[int, str] = "auto",
    alpha:     float           = 0.7,
    kde:       bool            = False,
    element:   str             = "bars",
    show:      bool            = True,
    save_path: Optional[str]   = None,
) -> Tuple[Figure, Axes]:
    """Create a histogram.

    Args:
        data: Input data {name: list}.
        config: PlotConfig object.
        bins: Number of bins or binning strategy (see sns.histplot).
        alpha: Transparency of bars.
        kde: Whether to overlay a kernel density estimate.
        element: Visual representation.
        show: Whether to show the plot.
        save_path: Path to save the plot (optional).

    Returns:
        Tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots()
    conifg.apply(fig, ax)

    sns.histplot(
        data=data,
        bins=bins,
        edgecolor="white",
        alpha=alpha,
        kde=kde,
        element=element,  # type: ignore
        ax=ax,
    )

    save_and_show(save_path, show)
    return fig, ax


def plot_line(
    x:          List[ScalarList],
    y:          List[ScalarList],
    config:     PlotConfig,
    errors:     Optional[List[ScalarList]]       = None,
    linestyles: Optional[List[str]]              = None,
    markers:    Optional[List[Optional[str]]]    = None,
    labels:     Optional[List[Optional[str]]]    = None,
    show:       bool                             = True,
    save_path:  Optional[str]                    = None,
    callback:   Optional[Callable[[Axes], None]] = None
) -> Tuple[Figure, Axes]:
    """Create a line plot with multiple lines.

    Args:
        x: List of X-axis data for each line.
        y: List of Y-axis data for each line.
        config: PlotConfig object with common plot settings.
        errors: Error values to match y.
        linestyles: List of line styles for each line (default: solid line for all).
        markers: List of marker styles for each line (default: None for all).
        labels: List of labels for each line (default: None for no labels).
        show: Whether to show the plot.
        save_path: Path to save the plot (optional).

    Returns:
        Tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots()
    config.apply(fig, ax)

    n_lines = len(x)
    if linestyles is None:
        linestyles = ["-" for _ in range(n_lines)]
    if markers is None:
        markers = [None for _ in range(n_lines)]
    if labels is None:
        labels = [None for _ in range(n_lines)]

    for i in range(n_lines):
        sns.lineplot(
            x=x[i],
            y=y[i],
            linestyle=linestyles[i],
            marker=markers[i],
            label=labels[i],
            ax=ax,
        )

        if errors is not None:
            y_np = np.asarray(y[i])
            error_np = np.asarray(errors[i])

            plt.fill_between(x[i], y_np - error_np, y_np + error_np, alpha=0.3)

    if any(label is not None for label in labels):
        ax.legend()

    if callback is not None:
        callback(ax)

    save_and_show(save_path, show)
    return fig, ax


def plot_scatter(
    x:         ScalarList,
    y:         ScalarList,
    config:    PlotConfig,
    alpha:     float         = 0.6,
    marker:    str           = 'o',
    size:      float         = 50,
    label:     Optional[str] = None,
    show:      bool          = True,
    save_path: Optional[str] = None,
) -> tuple[Figure, Axes]:
    """Create a scatter plot.

    Args:
        x: X-axis data (numpy array or list).
        y: Y-axis data (numpy array or list).
        config: PlotConfig object with common plot settings.
        alpha: Transparency of points.
        marker: Marker style.
        size: Size of scatter points.
        label: Label for the scatter points (for legend).
        show: Whether to show the plot.
        save_path: Path to save the plot (optional).

    Returns:
        Tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots()
    config.apply(fig, ax)

    sns.scatterplot(
        x=x,
        y=y,
        # color=color,
        alpha=alpha,
        marker=marker,
        # s=size,
        label=label,
        ax=ax
    )

    # This is for the client delay map. Should be removed in general
    sns.move_legend(ax, loc="lower right")

    save_and_show(save_path, show)
    return fig, ax


def set_sns_colors(palette: Optional[str] = None) -> None:
    """Set the sns color palette."""
    sns.color_palette(palette)


def set_sns_theme(*args, **kwargs) -> None:
    """Set the sns theme."""
    sns.set_theme(*args, **kwargs)


def save_and_show(save_path: Optional[str], show: bool) -> None:
    """Maybe save and/or show a plot."""
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        show_plot()


def show_plot() -> None:
    """Wrapper around plt.show()."""
    plt.show()


def set_global_rcparams() -> None:
    """Set the global plt.rcParams for a professional look."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["lines.markersize"] = 6
