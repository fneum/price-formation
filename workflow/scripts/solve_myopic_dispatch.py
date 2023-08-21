import pypsa

from pypsa.descriptors import nominal_attrs
from helpers import set_scenario_config
from solve import set_snapshots

import logging
logger = logging.getLogger(__name__)


def fix_optimal_capacities_from_other(n, other):
    for c, attr in nominal_attrs.items():
        ext_i = n.get_extendable_i(c)
        n.df(c).loc[ext_i, attr] = other.df(c).loc[ext_i, attr + "_opt"]
        n.df(c)[attr + "_extendable"] = False


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("solve_myopic_dispatch", run='inelastic')

    set_scenario_config(
        snakemake.config,
        snakemake.input.scenarios,
        snakemake.wildcards.run,
    )

    solver_name = snakemake.config["solver"]["name"]
    profile = snakemake.config["solver"]["options"]
    solver_options = snakemake.config["solver_options"][profile]

    n = pypsa.Network(snakemake.input.prepared_network)
    n_solved = pypsa.Network(snakemake.input.solved_network)

    fix_optimal_capacities_from_other(n, n_solved)

    hydrogen_bid = snakemake.config["myopic"]["hydrogen_bid"]
    if hydrogen_bid == "series":
        n.stores_t.marginal_cost["hydrogen storage"] = n.buses_t.marginal_price["hydrogen"]
    elif hydrogen_bid == "mean":
        n.stores.at["hydrogen storage", "marginal_cost"] = n.buses_t.marginal_price["hydrogen"].mean()
    elif isinstance(hydrogen_bid, (float, int)):
        n.stores.at["hydrogen storage", "marginal_cost"] = hydrogen_bid

    battery_bid = snakemake.config["myopic"]["battery_bid"]
    if battery_bid == "series":
        n.stores_t.marginal_cost["battery storage"] = n.buses_t.marginal_price["battery"]
    elif battery_bid == "mean":
        n.stores.at["battery storage", "marginal_cost"] = n.buses_t.marginal_price["battery"].mean()
    elif isinstance(battery_bid, (float, int)):
        n.stores.at["battery storage", "marginal_cost"] = battery_bid

    n.stores.e_cyclic = snakemake.config["myopic"]["cyclic"]

    for c in n.iterate_components({"Generator", "Link", "Store"}):
        attr = "e_nom" if c.name == "Store" else "p_nom"
        for carrier, perturbation in snakemake.config["myopic"]["perturbation"].items():
            c.df.loc[c.df.carrier == carrier, attr] *= perturbation

    exclude_years = list(n_solved.snapshots.year.unique())
    number_years = snakemake.config["myopic"]["number_years"]
    random_years = snakemake.config["myopic"]["random_years"]
    set_snapshots(
        n,
        number_years=number_years,
        random_years=random_years,
        exclude_years=exclude_years,
    )

    if snakemake.config["myopic"]["perfect_foresight"]:
        status, condition = n.optimize(
            solver_name=solver_name,
            solver_options=solver_options,
        )
    else:
        status, condition = n.optimize.optimize_with_rolling_horizon(
            solver_name=solver_name,
            solver_options=solver_options,
        )

    export_kwargs = snakemake.config["export_to_netcdf"]
    n.export_to_netcdf(snakemake.output.network, **export_kwargs)

    energy_balance = n.statistics.energy_balance(aggregate_time=False, aggregate_bus=False).T
    energy_balance.to_csv(snakemake.output.energy_balance)

    n.statistics().to_csv(snakemake.output.statistics)
