"""
Contains the code to replicate
the LINGAM experiments.
"""

import os
import time

import fire
import pandas as pd
import ray
from abs_lingam import evaluate_lingam, preprocess_results

from causabs.dataset import generate_datasets


def plot(
    results_dir: str = "results/",
    flavor: str = "small",
    plot_dir: str = "plots/",
    store: bool = True,
    show: bool = False,
    main_body: bool = False,
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    results_fname = os.path.join(results_dir, f"experiment3_{flavor}.csv")
    results = pd.read_csv(results_fname)

    print(f"== Experiment 3 ({flavor}) ==")

    # filter results
    print(f"Total Entries: {len(results)}")
    results["Abs Threshold"] = results["params/abs_threshold"]
    results["params/abs_threshold"] = 0.0
    results = preprocess_results(results)
    print(f"Flavor Entries: {len(results)}")

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
        if main_body and y not in [
            "eval/concrete_roc_auc",
            "eval/time",
        ]:
            continue

        if show:
            plt.figure(figsize=(6, 6))
        else:
            plt.figure(figsize=(6, 3.5))

        if main_body:
            results = results[
                results["Method"].isin(
                    [
                        "DirectLiNGAM",
                        "Abs-LiNGAM (Bootstrap=0)",
                        "Abs-LiNGAM (Bootstrap=5)",
                    ]
                )
            ]
            # figure size
            plt.figure(figsize=(5, 4))
            hue_order = [
                "DirectLiNGAM",
                "Abs-LiNGAM (Bootstrap=5)",
                "Abs-LiNGAM (Bootstrap=0)",
            ]
            # figure size
            plt.figure(figsize=(5, 4))
            sns.set(context="paper", font_scale=1.3, style="white")
        else:
            hue_order = None

        sns.lineplot(
            x="dset/cnc_nodes",
            y=y,
            hue="Method",
            style="Method",
            data=results,
            hue_order=hue_order,
        )
        # plt.yscale("log")
        # rename axis
        plt.xlabel(r"Concrete Nodes $|\mathbf{X}|$")
        plt.ylabel(y_label)

        if y in ["eval/time"]:
            plt.legend(loc="upper left")
        elif y == "eval/pk_recall":
            plt.legend(loc="upper right")
        else:
            plt.legend(loc="lower left")

        # if not main_body:
        #     if y == "eval/concrete_roc_auc":
        #         plt.ylim(0.5, 1)
        #     elif "concrete" in y:
        #         plt.ylim(0, 1)
        # else:
        if main_body:
            # remove legend
            plt.legend().remove()

        # store plots/exp2_rocauc.pdf
        metric = y.split("/")[1]
        plt.tight_layout()
        if store:
            plt.savefig(os.path.join(plot_dir, f"exp3_{metric}_{flavor}.pdf"))
            plt.savefig(os.path.join(plot_dir, f"exp3_{metric}_{flavor}.pgf"))
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

    range_readout = [(1, 1), (2, 2), (5, 5), (10, 10)]

    if flavor == "small":
        dset_params["abs_nodes"] = 5
        dset_params["abs_edges"] = 8
    elif flavor == "medium":
        dset_params["abs_nodes"] = 10
        dset_params["abs_edges"] = 20
    elif flavor == "large":
        dset_params["abs_nodes"] = 20
        dset_params["abs_edges"] = 40
    else:
        raise ValueError(f"Unknown flavour {flavor}.")

    futures = []
    for min_block_size, max_block_size in range_readout:
        dset_params["min_block_size"] = min_block_size
        dset_params["max_block_size"] = max_block_size
        generate_datasets(dset_params, data_dir, num_runs, force=False)
        n_concrete = 15000
        n_paired = dset_params["abs_nodes"] * dset_params["max_block_size"] * 2
        for run in range(num_runs):
            # Test the DirectLiNGAM method
            futures.append(
                evaluate_lingam.remote(
                    dset_params.copy(),
                    data_dir,
                    method="DirectLiNGAM",
                    n_paired=0,
                    n_concrete=n_concrete,
                    run=run,
                    shuffle_features=True,
                    normalize=True,
                    seed=seed,
                    verbose=verbose,
                )
            )
            # The the Abs-LiNGAM-GT method
            futures.append(
                evaluate_lingam.remote(
                    dset_params.copy(),
                    data_dir,
                    method="Abs-LiNGAM-GT",
                    n_paired=0,
                    n_concrete=n_concrete,
                    run=run,
                    shuffle_features=True,
                    normalize=True,
                    seed=seed,
                    verbose=verbose,
                )
            )
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
            for bootstrap_samples in [1, 2, 5, 10]:
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
                        abs_threshold=0.1,
                        bootstrap_samples=bootstrap_samples,
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
    results_fname = os.path.join(results_dir, f"experiment3_{flavor}.csv")
    df.to_csv(results_fname, index=False)


if __name__ == "__main__":
    fire.Fire()
