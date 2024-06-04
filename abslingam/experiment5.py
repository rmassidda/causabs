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

    results_fname = os.path.join(results_dir, f"experiment5_{flavor}.csv")
    results = pd.read_csv(results_fname)

    if flavor == "small":
        dset = "d5_e8_tER_m5_M10"
    elif flavor == "medium":
        dset = "d10_e20_tER_m5_M10"
    elif flavor == "large":
        dset = "d10_e20_tER_m10_M15"
    dset += "_h1000.0_p0.5_iTrue_n20000_Nexponential_I0_jFalse_vNone_aTrue"

    print(f"== Experiment 5 ({flavor}) ==")

    # filter results
    print(f"Total Entries: {len(results)}")
    print(f"Flavor Entries: {len(results)}")

    # print(len(results))
    avg_cnc_nodes = results["dset/cnc_nodes"].mean()
    std_cnc_nodes = results["dset/cnc_nodes"].std() * 1.96
    print(
        "Concrete Nodes:",
        round(avg_cnc_nodes, 2),
        "±",
        round(std_cnc_nodes, 2),
    )

    # discretize tau_threshold (print in scientific notation 0.01 -> 1e-2)
    results["params/tau_threshold"] = results["params/tau_threshold"].apply(
        lambda x: f"{x:.0e}"
    )

    def add_bar():
        plt.axvline(avg_cnc_nodes, color="black", alpha=0.5, linestyle="--")

    if show:
        labels = [
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
            ("dset/paired_rank", r"Rank $\mathcal{D}_{|\mathbf{X}|}$"),
            ("eval/time", "Time (s)"),
        ]
    else:
        labels = [
            ("eval/concrete_roc_auc", r"ROCAUC $\mathcal{L}$"),
            ("eval/pk_precision", "Prior Knowledge Precision"),
            ("eval/pk_recall", "Prior Knowledge Recall"),
            ("eval/time", "Time (s)"),
        ]

    for y, y_label in labels:
        plt.figure(figsize=(6, 3.5))
        if y == "dset/paired_rank":
            hue = None
        else:
            hue = "params/tau_threshold"

        sns.lineplot(
            x="dset/paired_samples",
            y=y,
            hue=hue,
            style=hue,
            data=results,
        )
        # rename axis
        plt.xlabel(r"Paired Samples $|\mathcal{D}_P|$")
        plt.ylabel(y_label)
        # log y
        # plt.yscale("log")
        # rename legend
        if hue is not None:
            plt.legend(title=r"$\tau$ Threshold", loc="lower right")
        add_bar()
        # store plots/exp5_rocauc.pdf
        metric = y.split("/")[1]
        if store:
            plt.savefig(os.path.join(plot_dir, f"exp5_{metric}_{flavor}.pdf"))
            plt.savefig(os.path.join(plot_dir, f"exp5_{metric}_{flavor}.pgf"))
        if show:
            plt.show()
        plt.clf()

    if show:
        return results


def run(
    seed: int = 1011329608,
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
        "n_samples": 50000,
        "noise_term": "exponential",
        "noise_abs": 0.0,
    }

    assert flavor in ["small", "medium", "large"], f"Unknown flavour {flavor}."

    if flavor == "small":
        shift = 20
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

    generate_datasets(dset_params, data_dir, num_runs, force=False)
    futures = []
    n_concrete = dset_params["n_samples"] - 2000
    max_concrete_nodes = (
        dset_params["abs_nodes"] * dset_params["max_block_size"]
    )
    max_paired_samples = int(max_concrete_nodes * 3)
    # max_paired_samples = dset_params["n_samples"] - n_concrete
    for run in range(num_runs):
        for tau_threshold in [1e-3, 1e-2, 5e-2, 1e-1]:
            for n_paired in list(range(shift, max_paired_samples, shift)) + [
                max_paired_samples
            ]:
                # Test the Abs-LiNGAM method
                futures.append(
                    evaluate_lingam.remote(
                        dset_params,
                        data_dir,
                        method="Abs-LiNGAM",
                        n_paired=n_paired,
                        n_concrete=n_concrete,
                        run=run,
                        shuffle_features=True,
                        normalize=True,
                        seed=seed,
                        tau_threshold=tau_threshold,
                        skip_concrete=True,
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
    results_fname = os.path.join(results_dir, f"experiment5_{flavor}.csv")
    df.to_csv(results_fname, index=False)


if __name__ == "__main__":
    fire.Fire()
