# Price formation without fuel costs: the interaction of elastic demand with storage bidding

**Link to preprint:** https://arxiv.org/TBC *(in preparation)*

**Link to data:** https://zenodo.org/TBC *(in preparation)*

## Abstract

Studies looking at electricity market design for very high shares of wind and
solar often conclude that the energy-only market will break down. Without fuel
costs, it is said that there is nothing to set prices. Symptoms of breakdown
include long phases of zero prices, scarcity prices too high to be politically
acceptable, prices that collapse under small perturbations of capacities from
the long-term equilibrium, cost recovery that is impossible due to low market
values, high variability of revenue between different weather years, and
difficulty operating long-term storage with limited foresight. We argue that all
these problems are an artefact of modeling with perfectly inelastic demand. Even
with a small amount of short-term elasticity representing today's flexible
demand (-5\%), the problems are significantly reduced. The combined interaction
of demand willingness to pay and storage opportunity costs is enough to produce
stable pricing. Considering a simplified model with just wind, solar, batteries,
and hydrogen-based storage, the price duration curve is significantly smoothed
with a piecewise linear demand curve. This removes high price peaks, reduces the
fraction of zero-price hours from 90\% to around 30\%, and guarantees more price
stability for perturbations of capacity and different weather years. Fuels
derived from green hydrogen take over the role of fossil fuels as the backup of
final resort. Furthermore, we show that with elastic demand, the long-term model
exactly reproduces the prices of the short-term model with the same capacities.
We can then use insights from the long-term model to derive simple bidding
strategies for storage so that we can also run the short-term model with limited
operational foresight. We demonstrate this short-term operation in a model
trained on 35 years of weather data and tested on another 35 years of unseen
data. We conclude that the energy-only market can still play a key role in
coordinating dispatch and investment in the future.

## Data Sources

The solar and wind time series for 1950-2020 are taken from from [Bloomfield and
Brayshaw (2021)](https://doi.org/10.17864/1947.000321).

The techno-economic assumptions about costs and efficiencies are taken from
[`technology-data`
(v0.8.1)](https://github.com/PyPSA/technology-data/tree/v0.8.1), which largely
come from the [Danish Energy
Agency](https://ens.dk/en/our-services/technology-catalogues).

The assumptions about demand elasticity are taken from [Hirth et al.
(2024)](https://doi.org/10.1016/j.eneco.2024.107652) and [Arnold
(2023)](https://www.ewi.uni-koeln.de/en/publications/on-the-functional-form-of-short-term-electricity-demand-response-insights-from-high-price-years-in-germany-2/).

## Installation

Use [`conda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment manager:

```sh
conda update conda
conda env create -f workflow/envs/environment.fixed.yaml
conda activate price-formation
```

The main dependencies are:

- pypsa (v0.27.1)
- linopy (v0.3.8)
- snakemake (v8.5)
- gurobi (v11.0.2)

## Run

From root of repository:

```sh
snakemake -call --use-conda --conda-frontend conda
```

Or with specific scenario configuration file:

```sh
snakemake -call --use-conda --conda-frontend conda --configfile config/config.foo.yaml
```

## Cluster

On HPC cluster, run:

```sh
snakemake -call --profile slurm --use-conda --conda-frontend conda
```

## Compress Results

Use `tar` to run (excluding report directory):

```sh
tar -cJf price-formation-results.tar.xz \
    config data figures results resources workflow \
    .gitignore .pre-commit-config.yaml .syncignore-receive \
    .syncignore-receive CITATION.cff LICENSE matplotlibrc README.md
```

## License

The code in this repository is MIT licensed, see [`./LICENSE`](`./LICENSE`).
