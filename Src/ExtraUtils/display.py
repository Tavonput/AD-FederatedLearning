from typing import Tuple, List, Union, Optional
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
    title:        str               = "title"
    xlabel:       str               = "xlabel"
    ylabel:       str               = "ylabel"
    figsize:      Tuple[int, int]   = (10, 6)
    grid:         Tuple[bool, bool] = (True, True)
    tight_layout: bool              = True


    def apply(self, fig: Figure, ax: Axes) -> None:
        fig.set_size_inches(self.figsize)
        if self.tight_layout:
            fig.tight_layout()

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        ax.set_axisbelow(True)

        if self.grid[0]:
            ax.grid(True, axis="x", linestyle="-")
        if self.grid[1]:
            ax.grid(True, axis="y", linestyle="-")


def plot_histogram(
    data:      ScalarList,
    conifg:    PlotConfig,
    bins:      Union[int, str] = "auto",
    alpha:     float           = 0.7,
    kde:       bool            = False,
    show:      bool            = True,
    save_path: Optional[str]   = None,
) -> Tuple[Figure, Axes]:
    """Create a histogram.

    Args:
        data: Input data (np.ndarray or list).
        config: PlotConfig object.
        bins: Number of bins or binning strategy (see sns.histplot).
        alpha: Transparency of bars.
        kde: Whether to overlay a kernel density estimate.
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
        ax=ax,
    )

    save_and_show(save_path, show)
    return fig, ax


def plot_line(
    x:         ScalarList,
    y:         ScalarList,
    config:    PlotConfig,
    linestyle: str           = "-",
    marker:    Optional[str] = None,
    show:      bool          = True,
    save_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Create a line plot.

    Args:
        x: X-axis data.
        y: Y-axis data.
        config: PlotConfig object with common plot settings.
        linestyle: Line style.
        marker: Marker style.
        show: Whether to show the plot.
        save_path: Path to save the plot (optional).

    Returns:
        Tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots()
    config.apply(fig, ax)

    sns.lineplot(
        x=x,
        y=y,
        linestyle=linestyle,
        marker=marker,
        ax=ax,
    )

    save_and_show(save_path, show)
    return fig, ax


def plot_scatter(
    x:         ScalarList,
    y:         ScalarList,
    config:    PlotConfig,
    color:     str           = 'dodgerblue',
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
        color: Color of scatter points.
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
        color=color,
        alpha=alpha,
        marker=marker,
        s=size,
        label=label,
        ax=ax
    )

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

