import pypsa
import pandas as pd
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from plot import get_price_duration

if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("plot_price_durations")

    plt.style.use(["bmh", snakemake.input.matplotlibrc])

    scenarios = snakemake.config["price_duration_curve"]["scenarios"]

    ns = [pypsa.Network(fn) for fn in snakemake.input.networks]

    df = pd.DataFrame({
        n.meta["run"]: get_price_duration(n) for n in ns
    })

    fig, ax = plt.subplots(figsize=(4,4))

    df.plot(
        ax=ax,
        ylim=(0, 2000),
        xlim=(0, 100),
        ylabel="Electricity Price [€/MWh]",
        xlabel="Fraction of Time [%]",
        linewidth=1,
        grid=True,
    )

    ax.grid()

    plt.savefig(snakemake.output.price_durations)

    fig, ax = plt.subplots(figsize=(4,4))

    df.plot(
        ax=ax,
        ylim=(100, 10000),
        xlim=(0, 10),
        logy=True,
        ylabel="Electricity Price [€/MWh]",
        xlabel="Fraction of Time [%]",
        linewidth=1,
        grid=True,
    )

    plt.savefig(snakemake.output.price_durations_log)