import contextlib
import logging

import pypsa
from helpers import set_scenario_config
from pypsa.descriptors import nominal_attrs
from solve import set_snapshots

logger = logging.getLogger(__name__)


def fix_optimal_capacities_from_other(n, other):
    for c, attr in nominal_attrs.items():
        ext_i = n.get_extendable_i(c)
        n.df(c).loc[ext_i, attr] = other.df(c).loc[ext_i, attr + "_opt"]
        n.df(c)[attr + "_extendable"] = False


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_myopic_dispatch", lt="inelastic+true", st="horizon+100"
        )

    set_scenario_config(
        snakemake.config,
        snakemake.wildcards,
    )

    solver_name = snakemake.config["myopic_solver"]["name"]
    profile = snakemake.config["myopic_solver"]["options"]
    solver_options = snakemake.config["solver_options"][profile]

    n = pypsa.Network(snakemake.input.prepared_network)
    n_solved = pypsa.Network(snakemake.input.solved_network)

    fix_optimal_capacities_from_other(n, n_solved)

    hydrogen_bid = snakemake.config["myopic"]["hydrogen_bid"]
    if "hydrogen storage" in n.stores.index:
        if hydrogen_bid == "series":
            # this only works if long-term and short-term model share same snapshots
            n.stores_t.marginal_cost["hydrogen storage"] = (
                n_solved.buses_t.marginal_price["hydrogen"]
            )
        elif hydrogen_bid == "mean":
            n.stores.at["hydrogen storage", "marginal_cost"] = (
                n_solved.buses_t.marginal_price["hydrogen"].mean()
            )
        elif isinstance(hydrogen_bid, (float, int)):
            n.stores.at["hydrogen storage", "marginal_cost"] = hydrogen_bid

    battery_bid = snakemake.config["myopic"]["battery_bid"]
    if "battery storage" in n.stores.index:
        if battery_bid == "series":
            # this only works if long-term and short-term model share same snapshots
            n.stores_t.marginal_cost["battery storage"] = (
                n_solved.buses_t.marginal_price["battery"]
            )
        elif battery_bid == "mean":
            n.stores.at["battery storage", "marginal_cost"] = (
                n_solved.buses_t.marginal_price["battery"].mean()
            )
        elif isinstance(battery_bid, (float, int)):
            n.stores.at["battery storage", "marginal_cost"] = battery_bid

    n.stores.e_cyclic = snakemake.config["myopic"]["cyclic"]

    perturbation = snakemake.config["myopic"]["perturbation"]
    if perturbation != 1:
        logger.info("Applying capacity perturbation of factor %s", perturbation)
    for c in n.iterate_components({"Generator", "Link", "Store"}):
        attr = "e_nom" if c.name == "Store" else "p_nom"
        if isinstance(perturbation, dict):
            for carrier, perturbation in perturbation.items():
                c.df.loc[c.df.carrier == carrier, attr] *= perturbation
        elif isinstance(perturbation, (float, int)):
            c.df.loc[c.df.carrier != "load", attr] *= perturbation
        else:
            raise ValueError(f"Unknown perturbation type {type(perturbation)}")

    exclude_years = list(n_solved.snapshots.year.unique())
    number_years = snakemake.config["myopic"]["number_years"]
    random_years = snakemake.config["myopic"]["random_years"]
    set_snapshots(
        n,
        number_years=number_years,
        random_years=random_years,
        exclude_years=exclude_years,
    )

    if solver_name == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)
        solver_options["threads"] = snakemake.config["myopic_solver"]["threads"]

    if snakemake.config["myopic"]["perfect_foresight"]:
        n.optimize(
            solver_name=solver_name,
            solver_options=solver_options,
            assign_all_duals=True,
        )
    else:
        with contextlib.ExitStack() as stack:

            if solver_name == "gurobi":
                import gurobipy

                env = stack.enter_context(gurobipy.Env())
            else:
                env = None

            n.optimize.optimize_with_rolling_horizon(
                solver_name=solver_name,
                solver_options=solver_options,
                assign_all_duals=True,
                horizon=snakemake.config["myopic"]["horizon"],
                overlap=snakemake.config["myopic"]["overlap"],
                env=env,
            )

    export_kwargs = snakemake.config["export_to_netcdf"]
    n.export_to_netcdf(snakemake.output.network, **export_kwargs)

    n.statistics().to_csv(snakemake.output.statistics)
