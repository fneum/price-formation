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


def load_technology_data(fn, defaults, overrides=False, years=1):
    df = pd.read_csv(fn, index_col=[0, 1])

    df.loc[df.unit.str.contains("/kW"), "value"] *= 1e3
    df.unit = df.unit.str.replace("/kW", "/MW")

    df = df.value.unstack()[defaults.keys()].fillna(defaults)

    df.loc["OCGT", "fuel"] = df.loc["gas", "fuel"]

    if overrides:
        for tech, tech_overrides in overrides.items():
            for attr, value in tech_overrides.items():
                # overrides are in kW, convert to MW
                df.loc[tech, attr] = value * 1e3

    annuity_factor = df.apply(
        lambda x: annuity(x["discount rate"], x["lifetime"]) * years, axis=1
    )

    df["capital_cost"] = (annuity_factor + df["FOM"] / 100) * df["investment"]

    df["marginal_cost"] = df["VOM"] + df["fuel"] / df["efficiency"]

    return df


def load_time_series(fn, country, snapshots, clip_p_max_pu=1e-2):
    snapshots = pd.date_range("1950-01-01", "2021-01-01", inclusive="left", freq="h")

    ds = xr.open_dataset(fn)

    ds = ds.assign_coords(NUTS=ds["NUTS_keys"], time=snapshots)

    s = ds.timeseries_data.sel(NUTS=country, time=snapshots).to_pandas()

    s.where(s > clip_p_max_pu, other=0.0, inplace=True)

    return s


def add_load(n, config):
    elastic_pwl = True if config["elastic_pwl"] else False
    number_options = sum(
        [config["voll"], config["elastic"], config["inelastic"], elastic_pwl]
    )
    assert (number_options == 1) or config[
        "voll_share"
    ], "Must choose exactly one of 'voll', 'elastic', 'elastic_pwl', 'inelastic' if elasticities are not mixed."
    if config["voll"]:
        logger.info("Adding demand with VOLL.")
        n.add(
            "Generator",
            "load-shedding",
            bus="electricity",
            carrier="load",
            marginal_cost=config["voll_price"],
            p_nom=config["load"],
        )
        n.add("Load", "load", bus="electricity", carrier="load", p_set=config["load"])
    if config["elastic"]:
        logger.info("Adding elastic demand.")
        n.add(
            "Generator",
            "load-shedding",
            bus="electricity",
            carrier="load",
            marginal_cost_quadratic=config["elastic_intercept"] / (2 * config["load"]),
            p_nom=config["load"],
        )
        n.add("Load", "load", bus="electricity", carrier="load", p_set=config["load"])
    if param_set := config["elastic_pwl"]:
        logger.info(f"Adding piecewise linear elastic demand with set '{param_set}'.")
        pwl = config["elastic_pwl_params"][param_set]
        assert (
            len(pwl["intercept"]) == len(pwl["slope"]) == len(pwl["nominal"])
        ), "Piecewise linear demand must have same number of points for intercept, slope, and nominal."
        for i, (intercept, slope, nominal) in enumerate(
            zip(pwl["intercept"], pwl["slope"], pwl["nominal"])
        ):
            n.add(
                "Generator",
                f"load-shedding-segment-{i}",
                bus="electricity",
                carrier="load",
                marginal_cost=intercept - slope * nominal,
                marginal_cost_quadratic=slope / 2,
                p_nom=nominal,
            )
        n.add(
            "Load",
            "load",
            bus="electricity",
            carrier="load",
            p_set=sum(pwl["nominal"]),
        )
    if config["inelastic"]:
        logger.info("Adding inelastic demand.")
        n.add("Load", "load", bus="electricity", carrier="load", p_set=config["load"])

    voll_share = config["voll_share"]
    if voll_share:
        assert (
            sum([config["voll"], config["elastic"]]) == 2
        ), "Need both 'voll' and 'elastic' to mix elasticities."
        logger.info("Mixing VOLL and elastic demand.")
        sel = n.generators.query(
            "marginal_cost_quadratic == 0. & carrier == 'load'"
        ).index
        n.generators.loc[sel, "p_nom"] *= voll_share
        sel = n.generators.query(
            "marginal_cost_quadratic != 0. & carrier == 'load'"
        ).index
        n.generators.loc[sel, "p_nom"] *= 1 - voll_share
        n.generators.loc[sel, "marginal_cost_quadratic"] /= 1 - voll_share
        n.loads.at["load", "p_set"] *= 1 - voll_share


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


def add_dispatchable(n, config, tech_data):
    if not config["dispatchable"]:
        return

    n.add(
        "Generator",
        "dispatchable",
        bus="electricity",
        carrier="dispatchable",
        p_nom=config["dispatchable"],
        efficiency=tech_data.at["OCGT", "efficiency"],
        marginal_cost=tech_data.at["OCGT", "marginal_cost"],
        capital_cost=tech_data.at["OCGT", "capital_cost"],
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
        p_nom_min=config["reserve"],
        efficiency=tech_data.at["fuel cell", "efficiency"],
        capital_cost=tech_data.at["fuel cell", "capital_cost"]
        * tech_data.at["fuel cell", "efficiency"],
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake
        from pathlib import Path

        snakemake = mock_snakemake(
            "prepare",
            lt="number_years+1-elastic+true-elastic_intercept+200",
            configfiles=[Path("../config/config.yaml")],
        )

    set_scenario_config(
        snakemake.config,
        snakemake.wildcards,
    )

    country = snakemake.config["country"]

    n = pypsa.Network()

    n.snapshots = pd.date_range(**snakemake.config["snapshots"])

    n.snapshot_weightings.loc[:, :] = float(snakemake.config["snapshots"]["freq"][:-1])

    if snakemake.config["fixed_year"]:
        years = 1
    else:
        years = snakemake.config["number_years"]

    if not years:
        freq = snakemake.config["snapshots"]["freq"]
        years = (
            n.snapshots[-1] - n.snapshots[0] + pd.Timedelta(freq)
        ) / np.timedelta64(365, "D")

    tech_data = load_technology_data(
        snakemake.input.tech_data,
        snakemake.config["technology_data"]["fill_values"],
        snakemake.config["technology_data"]["overrides"],
        years,
    )

    clip_p_max_pu = snakemake.config["clip_p_max_pu"]
    solar_cf = load_time_series(
        snakemake.input.solar_cf, country, n.snapshots, clip_p_max_pu
    )
    onwind_cf = load_time_series(
        snakemake.input.onwind_cf, country, n.snapshots, clip_p_max_pu
    )

    n.add("Bus", "electricity", carrier="electricity")

    colors = snakemake.config["colors"]
    n.madd("Carrier", colors.keys(), color=colors.values())

    add_load(n, snakemake.config)
    add_solar(n, snakemake.config, tech_data, solar_cf)
    add_wind(n, snakemake.config, tech_data, onwind_cf)
    add_dispatchable(n, snakemake.config, tech_data)
    add_battery(n, snakemake.config, tech_data)
    add_hydrogen(n, snakemake.config, tech_data)

    if snakemake.config["zero_cost_storage"]:
        n.stores.loc[:, "capital_cost"] = 0

    n.meta = snakemake.config

    export_kwargs = snakemake.config["export_to_netcdf"]
    n.export_to_netcdf(snakemake.output.network, **export_kwargs)
