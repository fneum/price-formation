import pypsa
import pandas as pd
import matplotlib.pyplot as plt

def plot_price_duration(n):

    fig, ax = plt.subplots()

    df = n.buses_t.marginal_price["electricity"].sort_values(ascending=False).reset_index(drop=True)

    df.plot(
        ax=ax,
        ylim=(df.min(), df.max()),
        xlim=(0, len(df)),
        ylabel="Price [€/MWh]",
        xlabel="Snapshots [-]",
    )

    plt.savefig(snakemake.output.price_duration)

def plot_mu_energy_balance(n):

    fig, ax = plt.subplots()

    df = n.stores_t.mu_energy_balance

    df.plot(
        ax=ax,
        ylabel="Shadow Price [€/MWh]",
        xlabel="Snapshots",
    )

    plt.savefig(snakemake.output.mu_energy_balance)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("plot")

    plt.style.use(['bmh', snakemake.input.matplotlibrc])

    n = pypsa.Network(snakemake.input.network)

    plot_price_duration(n)

    plot_mu_energy_balance(n)