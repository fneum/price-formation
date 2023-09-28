from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from helpers import set_scenario_config
from pypsa.descriptors import get_switchable_as_dense as as_dense


def set_xticks(ax, index):
    major_tick_positions = [index.get_loc(date) for date in index if date == datetime(date.year, 1, 1, 0)]
    formatter = "%b\n%Y" if len(major_tick_positions) <= 8 else "'%y"
    major_labels = [date.strftime(formatter) for date in index if date == datetime(date.year, 1, 1, 0)]
    ax.set_xticks(major_tick_positions)
    ax.set_xticklabels(major_labels)

    if len(major_tick_positions) <= 3:
        months = range(1, 13) if len(major_tick_positions) <= 2 else [1, 4, 7, 10]
        minor_tick_positions = [index.get_loc(date) for date in index if date.month in months and date.day == 1 and date.hour == 0]
        minor_labels = [date.strftime("%b") for date in index if  date.month in months and date.day == 1 and date.hour == 0]
        ax.set_xticks(minor_tick_positions, minor=True)
        ax.set_xticklabels(minor_labels, minor=True)

        ax.tick_params(axis="x", which="minor", labelcolor="grey")


def get_hydrogen_bids(n, sns=None):
    mcp = n.buses_t.marginal_price["hydrogen"]
    if sns:
        mcp = mcp[sns]
    el_bid = mcp * n.links.at["hydrogen electrolyser", "efficiency"]
    fc_bid = mcp / n.links.at["hydrogen fuel cell", "efficiency"]
    if not sns:
        el_bid.name = "electrolyser bids"
        fc_bid.name = "fuel cell bids"
    return el_bid, fc_bid


def get_battery_bids(n, sns=None):
    mcp = n.buses_t.marginal_price["battery"]
    if sns:
        mcp = mcp[sns]
    charger_bid = mcp * n.links.at["battery charger", "efficiency"]
    discharger_bid = mcp / n.links.at["battery discharger", "efficiency"]
    if not sns:
        charger_bid.name = "battery charger bids"
        discharger_bid.name = "battery discharger bids"
    return charger_bid, discharger_bid


def get_price_duration(n, bus="electricity"):
    df = (
        n.buses_t.marginal_price[bus]
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )
    df.index = np.arange(0, 100, 100 / len(df.index))
    return df


def plot_price_duration(n):
    df = get_price_duration(n)

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

    df.reset_index(drop=True, inplace=True)

    fig, ax = plt.subplots()

    df.plot(
        ax=ax,
        ylabel="Electricity Price [€/MWh]",
        xlabel="Snapshots",
        ylim=(0, df.max() * 1.1),
    )

    ax.set_xlim(df.index[0], df.index[-1])

    set_xticks(ax, n.snapshots)

    plt.savefig(snakemake.output.price_time_series)


def plot_mu_energy_balance(n):
    df = n.stores_t.mu_energy_balance

    if df.empty:
        fig, ax = plt.subplots()
        plt.savefig(snakemake.output.mu_energy_balance)
        return

    df.reset_index(drop=True, inplace=True)

    fig, ax = plt.subplots()

    df.plot(
        ax=ax,
        ylabel="Shadow Price [€/MWh]",
        xlabel="Snapshots",
        ylim=(0, df.max().max() * 1.1),
    )

    set_xticks(ax, n.snapshots)

    plt.xlim(df.index[0], df.index[-1])

    plt.savefig(snakemake.output.mu_energy_balance)


def plot_hydrogen_bidding(n):
    if not "hydrogen" in n.buses.index:
        fig, ax = plt.subplots()
        plt.savefig(snakemake.output.hydrogen_bidding)
        return

    electrolyser_bid, fuel_cell_bid = get_hydrogen_bids(n)

    electrolyser_bid.reset_index(drop=True, inplace=True)
    fuel_cell_bid.reset_index(drop=True, inplace=True)

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

    set_xticks(ax, n.snapshots)
    
    plt.xlim(electrolyser_bid.index[0], electrolyser_bid.index[-1])

    plt.savefig(snakemake.output.hydrogen_bidding)


def plot_battery_bidding(n):
    if not "battery" in n.buses.index:
        fig, ax = plt.subplots()
        plt.savefig(snakemake.output.battery_bidding)
        return

    charger_bid, discharger_bid = get_battery_bids(n)

    charger_bid.reset_index(drop=True, inplace=True)
    discharger_bid.reset_index(drop=True, inplace=True)

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

    set_xticks(ax, n.snapshots)

    plt.xlim(charger_bid.index[0], charger_bid.index[-1])

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
        .drop(["load", "load-shedding"], axis=0, errors="ignore")
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

    eb.reset_index(drop=True, inplace=True)

    fig, ax = plt.subplots()

    eb.plot.area(
        ax=ax,
        figsize=(20, 3),
        linewidth=0,
        ylabel="Electricity Balance [MW]",
        color=eb.columns.map(n.carriers.color),
    )

    set_xticks(ax, n.snapshots)

    plt.legend(bbox_to_anchor=(1, 1))

    plt.xlim(eb.index[0], eb.index[-1])

    plt.savefig(snakemake.output.energy_balance)


def plot_supply_demand_curve(n, sns):
    res = 0.01

    ylim_max = snakemake.config["supply_demand_curve"]["ylim_max"]

    if "load" in n.generators.index:
        mc = n.generators.at["load", "marginal_cost"]
        mc2 = n.generators.at["load", "marginal_cost_quadratic"]
        p_nom = n.generators.at["load", "p_nom"]

        d = np.arange(0, p_nom, res)

        if mc2 > 0:
            load_bid = list(mc - 2 * mc2 * d)
        else:
            load_bid = [mc for _ in d]

    elif "load" in n.loads.index:
        p_set = as_dense(n, "Load", "p_set").loc[sns, "load"]
        x = np.arange(0, p_set, res)
        load_bid = [ylim_max * 1.1 for _ in x]

    vre = (as_dense(n, "Generator", "p_max_pu").loc[sns] * n.generators.p_nom_opt).sum()
    x = np.arange(0, vre, res)
    vre_bid = [0 for _ in x]

    supply = vre_bid + [ylim_max]
    demand = load_bid + [0]

    if snakemake.config["hydrogen"]:
        electrolyser_bid, fuel_cell_bid = get_hydrogen_bids(n, sns)

        electrolyser_p_nom = n.links.at["hydrogen electrolyser", "p_nom_opt"]
        fuel_cell_p_nom = (
            n.links.at["hydrogen fuel cell", "p_nom_opt"]
            * n.links.at["hydrogen fuel cell", "efficiency"]
        )

        x = np.arange(0, electrolyser_p_nom, res)
        el_bid = [electrolyser_bid for _ in x]
        x = np.arange(0, fuel_cell_p_nom, res)
        fc_bid = [fuel_cell_bid for _ in x]

        supply += fc_bid
        demand += el_bid

    if snakemake.config["battery"]:
        charger_bid, discharger_bid = get_battery_bids(n, sns)

        charger_p_nom = n.links.at["battery charger", "p_nom_opt"]
        discharger_p_nom = (
            n.links.at["battery discharger", "p_nom_opt"]
            * n.links.at["battery discharger", "efficiency"]
        )

        x = np.arange(0, charger_p_nom, res)
        ch_bid = [charger_bid for _ in x]
        x = np.arange(0, discharger_p_nom, res)
        dch_bid = [discharger_bid for _ in x]

        supply += dch_bid
        demand += ch_bid

    supply = np.sort(supply)
    demand = -np.sort(-np.array(demand))

    x_supply = res * np.arange(0, len(supply))
    x_demand = res * np.arange(0, len(demand))

    mcp = n.buses_t.marginal_price.at[sns, "electricity"]

    ac_balance = (
        n.statistics.energy_balance(aggregate_time=False)
        .loc[:, sns]
        .xs("electricity", level="bus_carrier")
    )
    mcv = ac_balance[ac_balance > 0].sum()

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.axhline(mcp, linestyle="--", color="gray", linewidth=1, label="market clearing")
    ax.axvline(mcv, linestyle="--", color="gray", linewidth=1)

    plt.plot(x_supply, supply, label="supply curve", drawstyle="steps-post")
    plt.plot(x_demand, demand, label="demand curve", drawstyle="steps-post")

    ax.grid()

    plt.ylabel("Bids [€/MWh]")
    plt.xlabel("Volumes [MWh]")

    xlim_max = max(np.append(len(supply), len(demand))) * res
    plt.ylim(-0.03 * ylim_max, 1.03 * ylim_max)
    plt.xlim(0, 1.03 * xlim_max)

    cc = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax.text(xlim_max * 1.04, 0, "_ VRE", color=cc[0])
    ax.text(xlim_max * 1.26, np.mean(load_bid), "_ demand", color=cc[1])

    if snakemake.config["hydrogen"]:
        ax.text(xlim_max * 1.04, fuel_cell_bid, "_ fuel cell", color=cc[0])
        ax.text(xlim_max * 1.26, electrolyser_bid, "_ electrolyser", color=cc[1])

    if snakemake.config["battery"]:
        ax.text(xlim_max * 1.04, charger_bid, "_ battery", color=cc[0])
        ax.text(xlim_max * 1.26, discharger_bid, "_ battery", color=cc[1])

    plt.legend(bbox_to_anchor=(1.53, 0.9))

    plt.title(sns, fontsize="medium")

    plt.savefig(snakemake.output[sns])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("plot", lt="country+DE-number_years+2-inelastic+true")
        # snakemake = mock_snakemake("plot_myopic_dispatch", lt="inelastic+true", st="horizon+96")

    set_scenario_config(
        snakemake.config,
        snakemake.wildcards,
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

    for sns in snakemake.config["supply_demand_curve"]["snapshots"]:
        plot_supply_demand_curve(n, sns)
