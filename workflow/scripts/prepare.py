import logging

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from helpers import set_scenario_config

logger = logging.getLogger(__name__)

def annuity(r, n):
    if r == 0:
        return 1 / n
    else:
        return r / (1 - 1 / (1 + r) ** n)


def load_technology_data(fn, defaults, years=1):
    df = pd.read_csv(fn, index_col=[0, 1])

    df.loc[df.unit.str.contains("/kW"), "value"] *= 1e3
    df.unit = df.unit.str.replace("/kW", "/MW")

    df = df.value.unstack()[defaults.keys()].fillna(defaults)

    annuity_factor = df.apply(lambda x: annuity(x["discount rate"], x["lifetime"]) * years, axis=1)

    df["capital_cost"] = (annuity_factor + df["FOM"] / 100) * df["investment"]

    return df


def load_time_series(fn, country, snapshots):
    snapshots = pd.date_range("1950-01-01", "2021-01-01", inclusive="left", freq="H")

    ds = xr.open_dataset(fn)

    ds = ds.assign_coords(NUTS=ds["NUTS_keys"], time=snapshots)

    s = ds.timeseries_data.sel(NUTS=country, time=snapshots).to_pandas()

    return s


def add_load(n, config):
    assert (
        sum([config["voll"], config["elastic"], config["inelastic"]]) == 1
    ) or config["voll_share"], "Must choose exactly one of 'voll', 'elastic', 'inelastic' if elasticities are not mixed."

    if config["voll"]:
        logger.info("Adding demand with VOLL.")
        n.add(
            "Generator",
            "load",
            bus="electricity",
            carrier="load",
            marginal_cost=config["voll_price"],
            p_max_pu=0,
            p_min_pu=-1,
            p_nom=config["load"],
        )
    if config["elastic"]:
        logger.info("Adding elastic demand.")
        n.add(
            "Generator",
            "load-shedding",
            bus="electricity",
            carrier="load",
            marginal_cost_quadratic=config["elastic_intercept"] / ( 2 * config["load"]),
            p_nom=config["load"],
        )
        n.add("Load", "load", bus="electricity", carrier="load", p_set=config["load"])
    if config["inelastic"]:
        logger.info("Adding inelastic demand.")
        n.add("Load", "load", bus="electricity", carrier="load", p_set=config["load"])

    voll_share = config["voll_share"]
    if voll_share:
        assert sum([config["voll"], config["elastic"]]) == 2, "Need both 'voll' and 'elastic' to mix elasticities."
        logger.info("Mixing VOLL and elastic demand.")
        sel = n.generators.query("marginal_cost_quadratic == 0. & carrier == 'load'").index
        n.generators.loc[sel, "p_nom"] *= voll_share
        sel = n.generators.query("marginal_cost_quadratic != 0. & carrier == 'load'").index
        n.generators.loc[sel, "p_nom"] *= (1 - voll_share)
        n.generators.loc[sel, "marginal_cost_quadratic"] /= (1 - voll_share)
        n.loads.at["load", "p_set"] *= (1 - voll_share)

def add_solar(n, config, tech_data, p_max_pu):
    if not config["solar"]:
        return

    n.add(
        "Generator",
        "solar",
        bus="electricity",
        carrier="solar",
        p_max_pu=p_max_pu,
        p_nom_extendable=True,
        marginal_cost=0.1,  # Small cost to prefer curtailment to destroying energy in storage, solar curtails before wind
        capital_cost=tech_data.at["solar", "capital_cost"],
    )

def add_wind(n, config, tech_data, p_max_pu):
    if not config["wind"]:
        return

    n.add(
        "Generator",
        "wind",
        bus="electricity",
        carrier="wind",
        p_max_pu=p_max_pu,
        p_nom_extendable=True,
        marginal_cost=0.2,  # Small cost to prefer curtailment to destroying energy in storage, solar curtails before wind
        capital_cost=tech_data.at["onwind", "capital_cost"],
    )


def add_battery(n, config, tech_data):
    if not config["battery"]:
        return

    n.add("Bus", "battery", carrier="battery")

    n.add(
        "Store",
        "battery storage",
        bus="battery",
        carrier="battery storage",
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=tech_data.at["battery storage", "capital_cost"],
    )

    n.add(
        "Link",
        "battery charger",
        bus0="electricity",
        bus1="battery",
        carrier="battery charger",
        efficiency=tech_data.at["battery inverter", "efficiency"],
        p_nom_extendable=True,
        capital_cost=tech_data.at["battery inverter", "capital_cost"],
    )

    n.add(
        "Link",
        "battery discharger",
        bus0="battery",
        bus1="electricity",
        carrier="battery discharger",
        p_nom_extendable=True,
        efficiency=tech_data.at["battery inverter", "efficiency"],
    )


def add_hydrogen(n, config, tech_data):
    if not config["hydrogen"]:
        return

    n.add("Bus", "hydrogen", carrier="hydrogen")

    n.add(
        "Link",
        "hydrogen electrolyser",
        bus0="electricity",
        bus1="hydrogen",
        carrier="hydrogen electrolyser",
        p_nom_extendable=True,
        efficiency=tech_data.at["electrolysis", "efficiency"],
        capital_cost=tech_data.at["electrolysis", "capital_cost"],
    )

    n.add(
        "Store",
        "hydrogen storage",
        bus="hydrogen",
        carrier="hydrogen storage",
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=tech_data.at["hydrogen storage underground", "capital_cost"],
    )

    n.add(
        "Link",
        "hydrogen fuel cell",
        bus0="hydrogen",
        bus1="electricity",
        carrier="hydrogen fuel cell",
        p_nom_extendable=True,
        efficiency=tech_data.at["fuel cell", "efficiency"],
        capital_cost=tech_data.at["fuel cell", "capital_cost"]
        * tech_data.at["fuel cell", "efficiency"],
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("prepare", lt='number_years+1-elastic+true-elastic_intercept+200')

    set_scenario_config(
        snakemake.config,
        snakemake.wildcards,
    )

    country = snakemake.config["country"]

    n = pypsa.Network()

    n.snapshots = pd.date_range(**snakemake.config["snapshots"])

    n.snapshot_weightings.loc[:, :] = float(snakemake.config["snapshots"]["freq"][:-1])

    years = snakemake.config["number_years"]
    if not years:
        freq = snakemake.config["snapshots"]["freq"]
        years = (n.snapshots[-1] - n.snapshots[0] + pd.Timedelta(freq)) / np.timedelta64(1,'Y')

    tech_data = load_technology_data(
        snakemake.input.tech_data,
        snakemake.config["technology_data"]["fill_values"],
        years,
    )

    solar_cf = load_time_series(snakemake.input.solar_cf, country, n.snapshots)
    onwind_cf = load_time_series(snakemake.input.onwind_cf, country, n.snapshots)

    n.add("Bus", "electricity", carrier="electricity")

    colors = snakemake.config["colors"]
    n.madd("Carrier", colors.keys(), color=colors.values())

    add_load(n, snakemake.config)
    add_solar(n, snakemake.config, tech_data, solar_cf)
    add_wind(n, snakemake.config, tech_data, onwind_cf)
    add_battery(n, snakemake.config, tech_data)
    add_hydrogen(n, snakemake.config, tech_data)

    if snakemake.config["zero_cost_storage"]:
        n.stores.loc[:, "capital_cost"] = 0

    n.meta = snakemake.config

    export_kwargs = snakemake.config["export_to_netcdf"]
    n.export_to_netcdf(snakemake.output.network, **export_kwargs)
