from typing import Tuple, List, Union, Optional
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

import numpy as np


ListFloat = Union[np.ndarray, List[float]]


@dataclass
class PlotConfig:
    """A configuration class for common plot settings."""
    title: str         = "title"
    xlabel: str        = "xlabel"
    ylabel: str        = "ylabel"
    figsize: Tuple     = (10, 6)
    style: str         = "seaborn"
    grid: bool         = True
    tight_layout: bool = True


    def apply(self, fig: Figure, ax: Axes) -> None:
        fig.set_size_inches(self.figsize)
        if self.tight_layout:
            fig.tight_layout()

        plt.style.use(self.style)

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(self.grid, linestype="--", alpha=0.7)


def plot_histogram(
    data:      ListFloat,
    conifg:    PlotConfig,
    show:      bool          = True,
    save_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Create a histogram.

    Args:
        data: Input data (np.ndarray or list).
        config: PlotConfig object.
        show: Whether to show the plot.
        save_path: Path to save the plot (optional).

    Returns:
        Tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = plt.subplots()
    conifg.apply(fig, ax)

    sns.histplot(data=data)

    save_and_show(save_path, show)
    return fig, ax


def save_and_show(save_path: Optional[str], show: bool) -> None:
    """Maybe save and/or show a plot."""
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        show_plot()


def show_plot() -> None:
    """Wrapper around plt.show()."""
    plt.show()

