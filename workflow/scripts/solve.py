import pypsa
import numpy as np
import pandas as pd
import xarray as xr

from helpers import set_scenario_config

import logging
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
    ), "Must choose exactly one of 'voll', 'elastic', 'inelastic'"

    if config["voll"]:
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
    elif config["elastic"]:
        #create inverse demand curve where elastic_intercept is price p where demand d
        #vanishes and load is demand d for zero p
        #inverse demand curve: p(d) = intercept - intercept/load*d
        #utility: U(d) = intercept*d - intercept/(2*load)*d^2
        #since demand is negative generator, take care with signs!
        n.add(
            "Generator",
            "load",
            bus="electricity",
            carrier="load",
            marginal_cost=config["elastic_intercept"],
            marginal_cost_quadratic=config["elastic_intercept"] / ( 2 * config["load"]),
            p_max_pu=0,
            p_min_pu=-1,
            p_nom=config["load"],
        )
    elif config["inelastic"]:
        n.add("Load", "load", bus="electricity", carrier="load", p_set=config["load"])


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


def add_battery_constraints(n, sns):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def solve_network(n, config):
    profile = config["solver"]["options"]

    status, condition = n.optimize(
        solver_name=config["solver"]["name"],
        solver_options=config["solver_options"][profile],
        extra_functionality=add_battery_constraints
    )

    if status != "ok":
        logger.info(f"Solving status '{status}' with condition '{condition}'")
    if "infeasible" in condition:
        raise RuntimeError("Solving status 'infeasible'")

if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("solve", run='run-inelastic')

    set_scenario_config(
        snakemake.config,
        snakemake.input.scenarios,
        snakemake.wildcards.run,
    )

    country = snakemake.config["country"]
    
    n = pypsa.Network()

    n.snapshots = pd.date_range(**snakemake.config["snapshots"])

    n.snapshot_weightings.loc[:, :] = float(snakemake.config["snapshots"]["freq"][:-1])

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

    solve_network(n, snakemake.config)

    n.meta = snakemake.config

    n.export_to_netcdf(snakemake.output.network)

    energy_balance = n.statistics.energy_balance(aggregate_time=False, aggregate_bus=False).T
    energy_balance.to_csv(snakemake.output.energy_balance)

    n.statistics().to_csv(snakemake.output.statistics)
