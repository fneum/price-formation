import logging

import pypsa
from helpers import set_scenario_config

logger = logging.getLogger(__name__)

import random

random.seed(123)

from numpy.random import randint


def set_snapshots(
    n, number_years=False, random_years=False, fixed_year=False, exclude_years=[]
):

    if fixed_year:
        logger.info("Fixed year %s specified. Clipping snapshots.", fixed_year)
        n.snapshots = n.snapshots[n.snapshots.year.isin([fixed_year])]
        return

    if not number_years:
        logger.info("No subset of years selected. Keep all snapshots.")
        return

    all_years = set(n.snapshots.year.unique())
    allowed_years = list(all_years.difference(exclude_years))

    assert len(allowed_years) >= number_years, "Fewer allowed years than selected years"

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


def add_cross_elastic_terms(n, sns):
    """
    Add cross-elasticity terms to the objective function.
    """
    scenario = n.meta["cross_elasticity"]
    if not scenario:
        return
    extent = n.meta["cross_elasticity_params"][scenario]["extent"]
    divisor = n.meta["cross_elasticity_params"][scenario]["divisor"]
    myopic_and_cyclic = n.meta.get("myopic_and_cyclic", False)

    logger.info(
        f"Adding cross-elasticity terms ({scenario}: {extent}, {divisor}) to the objective function."
    )

    load = n.generators.query("carrier == 'load'")
    b = load.marginal_cost_quadratic * 2  # marginal_cost_quadratic = b / 2
    gamma = b / divisor

    p_t = n.model["Generator-p"].loc[:, load.index]
    d_gamma_half = 0.5 * load.p_nom * gamma
    for k in range(-extent, extent + 1):
        if k == 0:
            continue
        p_k = p_t.roll(snapshot=k)
        snapshots = p_t.indexes["snapshot"]
        if not myopic_and_cyclic:
            snapshots = snapshots[:k] if k < 0 else snapshots[k:]
        n.model.objective += (
            (p_t * d_gamma_half + p_k * d_gamma_half - p_t * p_k * 0.5 * gamma)
            .sel(snapshot=snapshots)
            .sum()
        )


def add_extra_functionality(n, sns):
    """
    Add extra functionality to the network before solving.
    """
    add_battery_constraints(n, sns)
    add_cross_elastic_terms(n, sns)


def solve_network(n, config, attempt=1):

    solver_name = config["solver"]["name"]
    profile = config["solver"]["options"]
    solver_options = config["solver_options"][profile]

    if solver_name == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)
        solver_options["threads"] = config["solver"]["threads"]

    if attempt > 1 and solver_name == "gurobi":
        numeric_profile = "gurobi-numeric"
        logger.info(f"Retry with {numeric_profile} solver settings.")
        solver_options = config["solver_options"][numeric_profile]
        solver_options["threads"] = config["solver"]["threads"]
        solver_options["NumericFocus"] = min(2, max(attempt - 1, 1))
        solver_options["Seed"] = randint(1, 999)

    status, condition = n.optimize(
        solver_name=solver_name,
        solver_options=solver_options,
        assign_all_duals=True,
        extra_functionality=add_extra_functionality,
    )

    if status != "ok":
        logger.info(f"Solving status '{status}' with condition '{condition}'")
    if condition in ["infeasible", "suboptimal", "unbounded", "error"]:
        raise RuntimeError(f"Solving status '{condition}'")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("solve", lt="inelastic+true")

    set_scenario_config(
        snakemake.config,
        snakemake.wildcards,
    )

    n = pypsa.Network(snakemake.input.network)

    set_snapshots(
        n,
        snakemake.config["number_years"],
        snakemake.config["random_years"],
        snakemake.config["fixed_year"],
    )

    solve_network(n, snakemake.config, snakemake.resources.attempt)

    n.meta = snakemake.config

    export_kwargs = snakemake.config["export_to_netcdf"]
    n.export_to_netcdf(snakemake.output.network, **export_kwargs)

    n.statistics().to_csv(snakemake.output.statistics)
