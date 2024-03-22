import pandas as pd
from helpers import set_scenario_config

if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("solve")

    set_scenario_config(
        snakemake.config,
        snakemake.wildcards,
    )

    config = snakemake.config["technology_data"]
    version = config["version"]
    year = config["year"]

    df = pd.read_csv(
        f"https://raw.githubusercontent.com/PyPSA/technology-data/{version}/outputs/costs_{year}.csv",
        index_col=0,
    )

    df.to_csv(snakemake.output.tech_data)
