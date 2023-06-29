import pypsa
import numpy as np
import pandas as pd
import xarray as xr

from helpers import set_scenario_config

from solve import add_battery_constraints, solve_network

import logging
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("solve_operations", run='run-inelastic')

    set_scenario_config(
        snakemake.config,
        snakemake.input.scenarios,
        snakemake.wildcards.run,
    )

    n = pypsa.Network(snakemake.input.network)

    n.optimize.fix_optimal_capacities()

    solve_network(n, snakemake.config)

    n.export_to_netcdf(snakemake.output.network)

    energy_balance = n.statistics.energy_balance(aggregate_time=False, aggregate_bus=False).T
    energy_balance.to_csv(snakemake.output.energy_balance)

    n.statistics().to_csv(snakemake.output.statistics)
