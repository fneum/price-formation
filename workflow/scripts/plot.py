import calendar
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pypsa
from helpers import set_scenario_config
from pypsa.descriptors import get_switchable_as_dense as as_dense

NDIGITS = 3


def set_xticks(ax, index):
    major_tick_positions = [
        index.get_loc(date) for date in index if date == datetime(date.year, 1, 1, 0)
    ]
    formatter = "%b\n%Y" if len(major_tick_positions) <= 8 else "'%y"
    major_labels = [
        date.strftime(formatter)
        for date in index
        if date == datetime(date.year, 1, 1, 0)
    ]
    ax.set_xticks(major_tick_positions)
    ax.set_xticklabels(major_labels)

    if len(major_tick_positions) <= 3:
        months = range(1, 13) if len(major_tick_positions) <= 2 else [1, 4, 7, 10]
        minor_tick_positions = [
            index.get_loc(date)
            for date in index
            if date.month in months and date.day == 1 and date.hour == 0
        ]
        minor_labels = [
            date.strftime("%b")
            for date in index
            if date.month in months and date.day == 1 and date.hour == 0
        ]
        ax.set_xticks(minor_tick_positions, minor=True)
        ax.set_xticklabels(minor_labels, minor=True)

        ax.tick_params(axis="x", which="minor", labelcolor="grey")


def get_energy_balance(n):
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

    return pd.concat([eb, crt], axis=1)


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


def get_cost_recovery(n, segments="pricebands"):
    # remove artificial bids from myopic dispatch optimisation
    carriers = n.stores.index.intersection(["hydrogen storage", "battery storage"])
    n.stores.loc[carriers, "marginal_cost"] = 0.0
    n.stores_t.marginal_cost.loc[:, carriers] = 0.0

    def merge_battery(s):
        if "charger" in s:
            return "battery dis-/charging"
        return s

    comps = {"Generator", "Link", "Store"}
    revenue = (
        n.statistics.revenue(comps=comps, aggregate_time=False)
        .droplevel(0)
        .drop("load", errors="ignore")
        .groupby(merge_battery)
        .sum()
        .div(1e6)
    )

    if segments == "months":
        revenue = revenue.T.groupby(revenue.columns.month).sum().T
        revenue.columns = revenue.columns.map(calendar.month_abbr.__getitem__)
    elif segments == "pricebands":
        lmps = n.buses_t.marginal_price["electricity"]
        bins = [0, 10, 50, 100, 200, 300, 400, 500, 1000, 2000, 5000]
        bins = [v for v in bins if v < max(lmps)] + [max(lmps) + 1]
        revenue = (
            revenue.T.groupby(pd.cut(lmps, bins=bins, precision=0), observed=False)
            .sum()
            .T
        )

    capex = n.statistics.capex()
    opex = n.statistics.opex()
    index = opex.index.union(capex.index)
    opex = opex.reindex(index).fillna(0.0)
    capex = capex.reindex(index).fillna(0.0)
    costs = (
        (capex + opex)
        .droplevel(0)
        .drop("load", errors="ignore")
        .groupby(merge_battery)
        .sum()
        .div(1e6)
    )

    revenue.loc["total"] = revenue.sum()
    costs["total"] = costs.sum()

    return revenue.div(costs, axis=0) * 100


def get_price_duration(n, bus="electricity"):
    df = (
        n.buses_t.marginal_price[bus]
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )
    df.index = np.arange(0, 100, 100 / len(df.index))
    return df


def get_storage_value_duration(n, name="hydrogen storage"):
    df = (
        n.stores_t.mu_energy_balance[name]
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )
    df.index = np.arange(0, 100, 100 / len(df.index))
    return df


def get_load_duration(n):

    df = (
        n.statistics.energy_balance(aggregate_time=False)
        .xs("load", axis=0, level="carrier")
        .sum()
        .mul(-1)
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


def plot_load_duration(n):

    df = get_load_duration(n)

    fig, ax = plt.subplots()

    df.plot(
        ax=ax,
        ylim=(0, df.max() * 1.1),
        xlim=(0, 100),
        ylabel="Electricity Demand [MW]",
        xlabel="Fraction of Time [%]",
    )

    plt.savefig(snakemake.output.load_duration)


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


def plot_state_of_charge(n):

    soc = n.stores_t.e / n.stores.e_nom_opt * 100

    soc.reset_index(drop=True, inplace=True)

    colors = n.carriers.color.to_dict()
    colors.update({"battery storage": "lightgrey"})

    fig, ax = plt.subplots()

    soc.plot(
        ax=ax,
        color=soc.columns.map(colors),
        ylabel="State of Charge [%]",
        xlabel="",
        ylim=(0, 100),
    )

    plt.legend(title="", bbox_to_anchor=(0.2, 1.02), ncol=2)

    set_xticks(ax, n.snapshots)

    plt.xlim(soc.index[0], soc.index[-1])

    plt.savefig(snakemake.output.soc)


def plot_hydrogen_bidding(n):
    if "hydrogen" not in n.buses.index:
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
    if "battery" not in n.buses.index:
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

    if len(n.snapshots) > 8784 * 5:
        fig, ax = plt.subplots()
        plt.savefig(snakemake.output.energy_balance)
        return

    eb = get_energy_balance(n)

    preferred_order = pd.Index(snakemake.config["preferred_order"])
    order = preferred_order.intersection(eb.columns).append(
        eb.columns.difference(preferred_order)
    )
    eb = eb[order]

    eb.reset_index(drop=True, inplace=True)

    # protect against numerical inaccuracies (e.g. small negative values in positive column)
    eb = eb.apply(
        lambda c: c.clip(lower=0) if (c >= 0).mean() > 0.6 else c.clip(upper=0)
    )

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

    if "load-shedding" in n.generators.index:
        mc = n.generators.at["load-shedding", "marginal_cost"]
        mc2 = n.generators.at["load-shedding", "marginal_cost_quadratic"]
        p_nom = n.generators.at["load-shedding", "p_nom"]

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


def plot_cost_recovery(n, segments="pricebands"):

    crf = get_cost_recovery(n, segments)

    preferred_order = pd.Index(
        [
            "hydrogen storage",
            "battery storage",
            "hydrogen electrolyser",
            "hydrogen fuel cell",
            "battery dis-/charging",
            "solar",
            "wind",
            "total",
        ]
    )

    order = preferred_order.intersection(crf.index).append(
        crf.index.difference(preferred_order)
    )
    crf = crf.loc[order]

    fig, ax = plt.subplots(figsize=(8, 3))

    crf_sum = crf.sum(axis=1)

    # to separate total
    ax.axhline(6.5, linestyle="--", color="gray", linewidth=0.5)
    ax.axhline(1.5, linestyle="--", color="gray", linewidth=0.5)

    ax.scatter(
        crf_sum,
        crf_sum.index,
        linewidth=0,
        marker=".",
        color="k",
        label="cost recovery",
        zorder=2,
    )
    crf.plot.barh(ax=ax, stacked=True, cmap="viridis")
    plt.grid(axis="x")
    plt.ylabel("")
    plt.xlabel("cost recovery [%]")

    for name, value in enumerate(crf_sum):
        ax.annotate(
            f"{value:.1f}%",
            (value, name),
            va="center",
            textcoords="offset points",
            xytext=(5, 0),
        )

    plt.legend(bbox_to_anchor=(1.05, 1), title="price band [€/MWh]", reverse=True)

    plt.savefig(snakemake.output.cost_recovery)


def get_metrics(n):

    energy_balance = n.statistics.energy_balance().groupby("carrier").sum() / 1e6  # TWh

    weightings = n.snapshot_weightings.generators

    metrics = pd.Series()
    # for opex, do not consider load shedding generators and heuristic storage bids
    metrics["opex"] = (
        n.statistics.opex(comps={"Generator", "Link"})
        .drop("load", level=1, errors="ignore")
        .sum()
        / 1e9
    )
    metrics["capex"] = n.statistics.capex().sum() / 1e9
    metrics["system-costs"] = metrics["capex"] + metrics["opex"]
    metrics["energy-served"] = -energy_balance["load"]
    metrics["average-load-served"] = -energy_balance["load"] * 1e6 / weightings.sum()
    metrics["primary-energy"] = energy_balance.filter(
        regex="solar|wind|dispatchable"
    ).sum()
    metrics["wind-share"] = energy_balance["wind"] / metrics["primary-energy"] * 100
    metrics["solar-share"] = energy_balance["solar"] / metrics["primary-energy"] * 100
    metrics["dispatchable-share"] = (
        energy_balance.filter(like="dispatchable").sum() / metrics["primary-energy"]
    )

    market_values = n.statistics.market_value()
    metrics["wind-lcoe"] = market_values.loc["Generator", "wind"]
    metrics["solar-lcoe"] = market_values.loc["Generator", "solar"]

    metrics["wind-cf"] = n.generators_t.p_max_pu["wind"].mean() * 100
    metrics["solar-cf"] = n.generators_t.p_max_pu["solar"].mean() * 100

    metrics["hydrogen-consumed"] = n.links_t.p0["hydrogen fuel cell"] @ weightings / 1e6

    if "load" in n.generators.index:
        metrics["peak-load-shedding"] = (
            n.generators.at["load", "p_nom"] + n.generators_t.p["load"].max()
        )
    else:
        metrics["peak-load-shedding"] = (
            n.generators_t.p.filter(like="load-shedding").sum(axis=1).max()
        )

    # capacities
    capacities = (
        n.statistics.optimal_capacity()
        .groupby("carrier")
        .sum()
        .drop("load", errors="ignore")
    )
    capacities["hydrogen fuel cell"] *= n.links.at[
        "hydrogen fuel cell", "efficiency"
    ]  # MWe
    capacities["hydrogen storage"] /= 1e3  # GWh
    if "battery discharger" in capacities.index:
        capacities["battery discharger"] *= n.links.at[
            "battery discharger", "efficiency"
        ]  # MWe
    if "battery storage" in capacities.index:
        capacities["battery storage"] /= 1e3  # GWh
    capacities.index = ["p_nom_opt-" + i.replace(" ", "-") for i in capacities.index]
    metrics = pd.concat([metrics, capacities])

    # time-weighted average prices
    metrics["average-electricity-price"] = n.buses_t.marginal_price[
        "electricity"
    ].mean()
    metrics["average-hydrogen-price"] = n.buses_t.marginal_price["hydrogen"].mean()
    metrics["std-electricity-price"] = n.buses_t.marginal_price["electricity"].std()
    metrics["std-hydrogen-price"] = n.buses_t.marginal_price["hydrogen"].std()

    # marginal storage values (including heuristic bids)
    msv = n.stores_t.mu_energy_balance + as_dense(n, "Store", "marginal_cost")
    metrics["average-hydrogen-msv"] = msv["hydrogen storage"].mean()
    metrics["std-hydrogen-msv"] = msv["hydrogen storage"].std()

    battery_msv = msv.get("battery storage", pd.Series([pd.NA]))
    metrics["average-battery-msv"] = battery_msv.mean()
    metrics["std-battery-msv"] = battery_msv.std()

    curtailment_twh = n.statistics.curtailment().filter(regex="solar|wind").sum() / 1e6
    metrics["curtailment"] = (
        curtailment_twh
        / (curtailment_twh + energy_balance.filter(regex="solar|wind").sum())
        * 100
    )

    # multiple if statements since mixing of voll and elastic is possible
    U = 0.0
    if case := n.meta["elastic_pwl"]:
        pwl = n.meta["elastic_pwl_params"][case]
        for i, (a, b, load) in enumerate(
            zip(pwl["intercept"], pwl["slope"], pwl["nominal"])
        ):
            Q = n.generators_t.p[f"load-shedding-segment-{i}"]
            constant = (a * load - b * load / 2) * weightings.sum()
            intersection = a - b * load
            load_shedding_cost = intersection * Q + b / 2 * Q**2
            U += constant - load_shedding_cost @ weightings
    if n.meta["elastic"]:
        a = n.meta["elastic_intercept"]
        b = n.meta["elastic_intercept"] / n.meta["load"]
        constant = a**2 / (2 * b) * weightings.sum()
        Q2 = n.generators_t.p["load-shedding"] ** 2
        U += constant - b / 2 * Q2 @ weightings
    if n.meta["voll"]:
        if "load-shedding" in n.generators.index:
            Q = n.meta["load"] - n.generators_t.p["load-shedding"]
        else:
            Q = -n.generators_t.p["load"]
        U += n.meta["voll_price"] * Q @ weightings
    if n.meta["inelastic"]:
        U = pd.NA

    metrics["utility"] = U / 1e9

    metrics["welfare"] = metrics["utility"] - metrics["system-costs"]

    metrics["average-costs"] = (
        metrics["system-costs"] * 1e3 / metrics["energy-served"]
    )  # €/MWh

    crf = get_cost_recovery(n).sum(axis=1).rename(index=lambda x: "cost-recovery " + x)
    metrics = pd.concat([metrics, crf])

    metrics.round(NDIGITS).to_csv(snakemake.output.metrics)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("plot", lt="country+DE-elastic_pwl+default")

    set_scenario_config(
        snakemake.config,
        snakemake.wildcards,
    )

    plt.style.use(["bmh", snakemake.input.matplotlibrc])
    sns.set_palette(snakemake.config["color_palette"])

    n = pypsa.Network(snakemake.input.network)

    colors = snakemake.config["colors"]
    n.mremove("Carrier", n.carriers.index)
    n.madd("Carrier", colors.keys(), color=colors.values())

    get_metrics(n)

    plot_cost_recovery(n, "pricebands")

    plot_price_duration(n)

    plot_load_duration(n)

    plot_price_time_series(n)

    plot_mu_energy_balance(n)

    plot_state_of_charge(n)

    plot_hydrogen_bidding(n)

    plot_battery_bidding(n)

    plot_energy_balance(n)

    for sns in snakemake.config["supply_demand_curve"]["snapshots"]:
        plot_supply_demand_curve(n, sns)
