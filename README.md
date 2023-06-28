# Price formation in 100% VRE systems

## Data Sources

The solar and wind time series for 1950-2020 are taken from from [Bloomfield and Brayshaw (2021)](https://doi.org/10.17864/1947.000321).

## Installation

```sh
mamba create -f workflow/envs/environment.yaml
```

## Run

From root of repository:

```sh
snakemake -call
```