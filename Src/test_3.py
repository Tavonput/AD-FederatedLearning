from ADFL.sampling import sample_half_normal
from ExtraUtils.display import *

set_sns_theme()

compute = sample_half_normal(100, sigma=10, shift=0, seed=1, reverse=False)
network = sample_half_normal(100, sigma=8, shift=10, seed=5, reverse=True)

config = PlotConfig(
    title="Delay Simulation",
    xlabel="Delay",
    ylabel="Count",
)
plot_histogram(
    {"Compute": compute, "Connectivity": network},
    config,
    element="step",
    show=False,
    save_path="./hist.png"
)

config = PlotConfig(
    title="Delay Simulation",
    xlabel="Compute",
    ylabel="Connectivity",
)
plot_scatter(
    compute,
    network,
    config,
    marker="o",
    show=False,
    save_path="./scatter.png",
)

