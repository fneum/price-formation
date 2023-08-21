import pypsa

from helpers import set_scenario_config

import logging
logger = logging.getLogger(__name__)

import random
random.seed(123)


def set_snapshots(n, number_years=False, random_years=False, exclude_years=[]):

    if not number_years:
        logger.info("No subset of years selected. Keep all snapshots.")
        return

    all_years = set(n.snapshots.year.unique())
    allowed_years = list(all_years.difference(exclude_years))

    assert len(allowed_years) > number_years, "Fewer allowed years than selected years"

    if random_years:
        random.shuffle(allowed_years)
    years = allowed_years[-number_years:]

    logger.info("Clipping snapshot to years: %s", years)
    n.snapshots = n.snapshots[n.snapshots.year.isin(years)]


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

        snakemake = mock_snakemake("solve", run='inelastic')

    set_scenario_config(
        snakemake.config,
        snakemake.input.scenarios,
        snakemake.wildcards.run,
    )
    
    n = pypsa.Network(snakemake.input.network)

    kwargs = snakemake.config["planning"]
    set_snapshots(n, **kwargs)

    solve_network(n, snakemake.config)

    n.meta = snakemake.config

    n.meta["run"] = snakemake.wildcards.run

    export_kwargs = snakemake.config["export_to_netcdf"]
    n.export_to_netcdf(snakemake.output.network, **export_kwargs)

    energy_balance = n.statistics.energy_balance(aggregate_time=False, aggregate_bus=False).T
    energy_balance.to_csv(snakemake.output.energy_balance)

    n.statistics().to_csv(snakemake.output.statistics)
