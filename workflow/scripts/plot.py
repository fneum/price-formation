import pypsa
import pandas as pd
import matplotlib.pyplot as plt

from helpers import set_scenario_config

def plot_price_duration(n):
    df = (
        n.buses_t.marginal_price["electricity"]
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )

    df.plot(
        ylim=(df.min(), df.max()),
        xlim=(0, len(df)),
        ylabel="Price [€/MWh]",
        xlabel="Snapshots [-]",
    )

    plt.savefig(snakemake.output.price_duration)


def plot_mu_energy_balance(n):
    df = n.stores_t.mu_energy_balance

    df.plot(
        ylabel="Shadow Price [€/MWh]",
        xlabel="Snapshots",
        ylim=(0, df.max().max())
    )

    plt.savefig(snakemake.output.mu_energy_balance)


def plot_energy_balance(n):
    eb = (
        n.statistics.energy_balance(aggregate_time=False)
        .xs("electricity", level="bus_carrier")
        .groupby("carrier")
        .sum()
        .T
    )

    crt = (
        n.statistics.curtailment(aggregate_time=False)
        .groupby("carrier")
        .sum()
        .drop(0, axis=1)
        .drop("load", axis=0, errors="ignore")
        .T
    )
    crt.columns += " curtailed"
    crt.index = pd.DatetimeIndex(crt.index)

    eb = pd.concat([eb, crt], axis=1)

    preferred_order = pd.Index(snakemake.config["preferred_order"])
    order = preferred_order.intersection(eb.columns).append(
        eb.columns.difference(preferred_order)
    )
    eb = eb[order]

    eb.plot.area(
        figsize=(20, 3),
        linewidth=0,
        ylabel="Electricity Balance [MW]",
        color=eb.columns.map(n.carriers.color),
    )

    plt.legend(bbox_to_anchor=(1, 1))

    plt.savefig(snakemake.output.energy_balance)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("plot")

    set_scenario_config(
        snakemake.config,
        snakemake.input.scenarios,
        snakemake.wildcards.run,
    )

    plt.style.use(["bmh", snakemake.input.matplotlibrc])

    n = pypsa.Network(snakemake.input.network)

    colors = snakemake.config["colors"]
    n.mremove("Carrier", n.carriers.index)
    n.madd("Carrier", colors.keys(), color=colors.values())

    plot_price_duration(n)

    plot_mu_energy_balance(n)

    plot_energy_balance(n)
