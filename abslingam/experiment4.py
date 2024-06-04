"""
Contains the code to replicate
the LINGAM experiments.
"""

import os
import time

import fire
import pandas as pd
import ray
from abs_lingam import evaluate_lingam

from causabs.dataset import generate_datasets


def plot(
    results_dir: str = "results/",
    flavor: str = "small",
    plot_dir: str = "plots/",
    store: bool = True,
    show: bool = False,
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    results_fname = os.path.join(results_dir, f"experiment4_{flavor}.csv")
    results = pd.read_csv(results_fname)

    print(f"== Experiment 4 ({flavor}) ==")

    # filter results
    print(f"Total Entries: {len(results)}")
    if flavor == "small":
        results = results[results["dset/abs_nodes"] == 5]
        results = results[results["dset/min_block_size"] == 5]
        results = results[results["dset/max_block_size"] == 10]
    elif flavor == "medium":
        results = results[results["dset/abs_nodes"] == 10]
        results = results[results["dset/min_block_size"] == 5]
        results = results[results["dset/max_block_size"] == 10]
    elif flavor == "large":
        results = results[results["dset/abs_nodes"] == 10]
        results = results[results["dset/min_block_size"] == 10]
        results = results[results["dset/max_block_size"] == 15]
    else:
        raise ValueError(f"Unknown flavor: {flavor}")
    print(f"Flavor Entries: {len(results)}")

    avg_cnc_nodes = results["dset/cnc_nodes"].mean()
    std_cnc_nodes = results["dset/cnc_nodes"].std() * 1.96
    print(
        "Concrete Nodes:",
        round(avg_cnc_nodes, 2),
        "±",
        round(std_cnc_nodes, 2),
    )

    results["Relevance Ratio"] = results["dset/relevant_ratio"].apply(
        lambda x: str(round(1 - x, 2))
    )

    def add_bar(y):
        min_y = results[y].min()
        max_y = 1.0  # results[y].max()
        plt.axvline(avg_cnc_nodes, color="black", alpha=0.5, linestyle="--")
        plt.fill_between(
            [avg_cnc_nodes - std_cnc_nodes, avg_cnc_nodes + std_cnc_nodes],
            min_y,
            max_y,
            color="b",
            alpha=0.05,
        )

    for y, y_label in [
        ("eval/abstract_roc_auc", r"ROCAUC $\mathcal{H}$"),
        ("eval/abstract_precision", r"Precision $\mathcal{H}$"),
        ("eval/abstract_recall", r"Recall $\mathcal{H}$"),
        ("eval/abstract_f1", r"F1 $\mathcal{H}$"),
        ("eval/tau_roc_auc", r"ROCAUC $\mathbf{T}$"),
        ("eval/tau_precision", r"Precision $\mathbf{T}$"),
        ("eval/tau_recall", r"Recall $\mathbf{T}$"),
        ("eval/tau_f1", r"F1 $\mathbf{T}$"),
        ("eval/concrete_roc_auc", r"ROCAUC $\mathcal{L}$"),
        ("eval/concrete_precision", r"Precision $\mathcal{L}$"),
        ("eval/concrete_recall", r"Recall $\mathcal{L}$"),
        ("eval/concrete_f1", r"F1 $\mathcal{L}$"),
        ("eval/pk_precision", "Prior Knowledge Precision"),
        ("eval/pk_recall", "Prior Knowledge Recall"),
        ("eval/time", "Time (s)"),
    ]:
        sns.lineplot(
            x="dset/paired_samples",
            y=y,
            hue="Relevance Ratio",
            data=results,
        )
        # rename axis
        plt.xlabel(r"Paired Samples $|\mathcal{D}_P|$")
        plt.ylabel(y_label)
        # rename legend
        plt.legend(title="Relevance Ratio", loc="lower right")
        if y == "eval/concrete_roc_auc":
            plt.ylim(0.5, 1)
        elif "concrete" in y:
            plt.ylim(0, 1)
        add_bar(y)
        # store plots/exp1_rocauc.pdf
        metric = y.split("/")[1]
        if store:
            plt.savefig(os.path.join(plot_dir, f"exp4_{metric}_{flavor}.pdf"))
            plt.savefig(os.path.join(plot_dir, f"exp4_{metric}_{flavor}.pgf"))
        if show:
            plt.show()
        plt.clf()


def run(
    seed: int = 13121110,
    num_cpus: int = 2,
    num_runs: int = 5,
    flavor: str = "small",
    data_dir: str = "data/",
    results_dir: str = "results/",
    verbose: bool = False,
):
    """run"""

    # initialize ray
    assert num_cpus > 1
    ray.init(num_cpus=num_cpus, num_gpus=0)

    datetime = time.strftime("%Y-%m-%d-%H-%M-%S")
    dset_params = {
        "abs_nodes": 5,
        "abs_edges": 8,
        "abs_type": "ER",
        "min_block_size": 5,
        "max_block_size": 10,
        "alpha": 1e3,
        "relevant_ratio": 0.5,
        "internal": True,
        "n_samples": 20000,
        "noise_term": "exponential",
        "noise_abs": 0.0,
    }

    assert flavor in ["small", "medium", "large"], f"Unknown flavour {flavor}."

    if flavor == "small":
        shift = 10
    elif flavor == "medium":
        dset_params["abs_nodes"] = 10
        dset_params["abs_edges"] = 20
        shift = 20
    elif flavor == "large":
        dset_params["abs_nodes"] = 10
        dset_params["abs_edges"] = 20
        dset_params["min_block_size"] = 10
        dset_params["max_block_size"] = 15
        shift = 20

    futures = []
    for relevant_ratio in [0.0, 0.2, 0.5, 0.8, 1.0]:
        dset_params["relevant_ratio"] = relevant_ratio
        generate_datasets(dset_params, data_dir, num_runs, force=False)
        n_concrete = dset_params["n_samples"] - 2000
        max_concrete_nodes = (
            dset_params["abs_nodes"] * dset_params["max_block_size"]
        )
        max_paired_samples = int(max_concrete_nodes * 2)
        for run in range(num_runs):
            for n_paired in list(range(shift, max_paired_samples, shift)) + [
                max_paired_samples
            ]:
                # Test the Abs-LiNGAM method
                futures.append(
                    evaluate_lingam.remote(
                        dset_params.copy(),
                        data_dir,
                        method="Abs-LiNGAM",
                        n_paired=n_paired,
                        n_concrete=n_concrete,
                        run=run,
                        shuffle_features=True,
                        normalize=True,
                        seed=seed,
                        verbose=verbose,
                    )
                )

    # get records and add experiment info
    print(f"Launched {len(futures)} total jobs.")
    records = ray.get(futures)
    print(f"Finished {len(records)} total jobs.")
    for record in records:
        record["experiment/seed"] = seed
        record["experiment/datetime"] = datetime

    # build and append dataframe
    df = pd.DataFrame.from_records(records)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_fname = os.path.join(results_dir, f"experiment4_{flavor}.csv")
    df.to_csv(results_fname, index=False)


if __name__ == "__main__":
    fire.Fire()
