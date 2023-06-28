import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import set_scenario_config

def plot_price_duration(n):
    df = (
        n.buses_t.marginal_price["electricity"]
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )
    df.index = np.arange(0, 100, 100 / len(df.index))

    fig, ax = plt.subplots()

    df.plot(
        ax=ax,
        ylim=(0, df.max() * 1.1),
        xlim=(0, 100),
        ylabel="Electricity Price [€/MWh]",
        xlabel="Fraction of Time [%]",
    )

    plt.savefig(snakemake.output.price_duration)


def plot_price_time_series(n):
    df = n.buses_t.marginal_price["electricity"]

    fig, ax = plt.subplots()

    df.plot(
        ax=ax,
        ylabel="Electricity Price [€/MWh]",
        xlabel="Snapshots",
        ylim=(0, df.max() * 1.1)
    )

    plt.savefig(snakemake.output.price_time_series)


def plot_mu_energy_balance(n):
    df = n.stores_t.mu_energy_balance

    fig, ax = plt.subplots()

    df.plot(
        ax=ax,
        ylabel="Shadow Price [€/MWh]",
        xlabel="Snapshots",
        ylim=(0, df.max().max() * 1.1)
    )

    plt.savefig(snakemake.output.mu_energy_balance)


def plot_hydrogen_bidding(n):
    
    mcp = n.buses_t.marginal_price["hydrogen"]
    
    electrolyser_bid = mcp * n.links.at["hydrogen electrolyser", "efficiency"]

    fuel_cell_bid = mcp / n.links.at["hydrogen fuel cell", "efficiency"]

    fig, ax = plt.subplots()

    electrolyser_bid.plot(
        ax=ax,
        ylabel="Bid [€/MWh]",
        xlabel="Snapshots",
        label="hydrogen electrolyser",
    )

    fuel_cell_bid.plot(
        ax=ax,
        ylabel="Bid [€/MWh]",
        xlabel="Snapshots",
        ylim=(0, fuel_cell_bid.max() * 1.1),
        label="hydrogen fuel cell",
    )

    plt.savefig(snakemake.output.hydrogen_bidding)


def plot_battery_bidding(n):
    
    mcp = n.buses_t.marginal_price["battery"]
    
    charger_bid = mcp * n.links.at["battery charger", "efficiency"]

    discharger_bid = mcp / n.links.at["battery discharger", "efficiency"]

    fig, ax = plt.subplots()

    charger_bid.plot(
        ax=ax,
        ylabel="Bid [€/MWh]",
        xlabel="Snapshots",
        label="battery charger",
    )

    discharger_bid.plot(
        ax=ax,
        ylabel="Bid [€/MWh]",
        xlabel="Snapshots",
        ylim=(0, discharger_bid.max() * 1.1),
        label="battery discharger",
    )

    plt.savefig(snakemake.output.battery_bidding)

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

    fig, ax = plt.subplots()

    eb.plot.area(
        ax=ax,
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

    plot_price_time_series(n)

    plot_mu_energy_balance(n)

    plot_hydrogen_bidding(n)
    
    plot_battery_bidding(n)

    plot_energy_balance(n)
