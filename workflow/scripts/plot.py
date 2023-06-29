import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pypsa.descriptors import get_switchable_as_dense as as_dense

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
        ylim=(0, df.max() * 1.1),
    )

    plt.savefig(snakemake.output.price_time_series)


def plot_mu_energy_balance(n):
    df = n.stores_t.mu_energy_balance

    if df.empty:
        fig, ax = plt.subplots()
        plt.savefig(snakemake.output.mu_energy_balance)
        return

    fig, ax = plt.subplots()

    df.plot(
        ax=ax,
        ylabel="Shadow Price [€/MWh]",
        xlabel="Snapshots",
        ylim=(0, df.max().max() * 1.1),
    )

    plt.savefig(snakemake.output.mu_energy_balance)


def plot_hydrogen_bidding(n):
    if not "hydrogen" in n.buses.index:
        fig, ax = plt.subplots()
        plt.savefig(snakemake.output.hydrogen_bidding)
        return

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
    if not "battery" in n.buses.index:
        fig, ax = plt.subplots()
        plt.savefig(snakemake.output.battery_bidding)
        return

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
        mcp_h2 = n.buses_t.marginal_price.at[sns, "hydrogen"]
        electrolyser_bid = mcp_h2 * n.links.at["hydrogen electrolyser", "efficiency"]
        fuel_cell_bid = mcp_h2 / n.links.at["hydrogen fuel cell", "efficiency"]

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
        mcp_ba = n.buses_t.marginal_price.at[sns, "battery"]
        charger_bid = mcp_ba * n.links.at["battery charger", "efficiency"]
        discharger_bid = mcp_ba / n.links.at["battery discharger", "efficiency"]

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

        snakemake = mock_snakemake("plot", run="zero-cost-storage-inelastic")

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

    for sns in snakemake.config["supply_demand_curve"]["snapshots"]:
        plot_supply_demand_curve(n, sns)
