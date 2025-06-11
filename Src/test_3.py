from ADFL.sampling import sample_half_normal

from ExtraUtils.display import *


set_sns_theme(style="white")


delays = sample_half_normal(8, 1, seed=0)

config = PlotConfig(
    title="Client Delay",
    xlabel="Client ID",
    ylabel="Delay",
    grid=(True, True)
)

y = sorted(delays, key=lambda x: x, reverse=True)
x = list(range(len(y)))
plot_line(
    x=x,
    y=y,
    config=config,
    marker="o",
    show=False,
    save_path="./output.png"
)

plot_histogram(
    data=delays,
    conifg=config,
    bins="auto",
    show=False,
    save_path="./output.png"
)
