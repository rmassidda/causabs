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
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    results_fname = os.path.join(results_dir, f"experiment1_{flavor}.csv")
    results = pd.read_csv(results_fname)

    print(f"== Experiment 1 ({flavor}) ==")

    # filter results
    print(f"Total Entries: {len(results)}")
    results = preprocess_results(results)
    print(f"Flavor Entries: {len(results)}")

    def replicate_per_paired_samples(target: str):
        # get all rows where "params/method" = DirectLiNGAM
        rows = results[results["params/method"] == target]
        # convert to list of dictionary
        rows = rows.to_dict("records")
        # unique paired samples
        paired_samples = results["dset/paired_samples"].unique()
        # remove 0
        paired_samples = paired_samples[1:]
        new_rows = []
        for row in rows:
            for paired_sample in paired_samples:
                new_row = row.copy()
                new_row["dset/paired_samples"] = paired_sample
                new_rows.append(new_row)
        return pd.concat([results, pd.DataFrame(new_rows)], ignore_index=True)

    # print(len(results))
    results = replicate_per_paired_samples("DirectLiNGAM")
    # print(len(results))
    results = replicate_per_paired_samples("Abs-LiNGAM-GT")
    # print(len(results))
    avg_cnc_nodes = results["dset/cnc_nodes"].mean()
    std_cnc_nodes = results["dset/cnc_nodes"].std() * 1.96
    print(
        "Concrete Nodes:",
        round(avg_cnc_nodes, 2),
        "±",
        round(std_cnc_nodes, 2),
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
        # figure size
        plt.figure(figsize=(6, 3.5))
        hue_order = None

        sns.lineplot(
            x="dset/paired_samples",
            y=y,
            hue="Method",
            style="Method",
            data=results,
            hue_order=hue_order,
        )
        # rename axis
        plt.xlabel(r"Paired Samples $|\mathcal{D}_J|$")
        plt.ylabel(y_label)
        # log y
        # plt.yscale("log")
        # rename legend
        if y in ["eval/time", "eval/pk_recall"]:
            plt.legend(loc="upper right")
        else:
            plt.legend(loc="lower right")

        # # remove legend
        # plt.legend().remove()
        # # add title
        # plt.title(r"$|\mathcal{D}_{\mathcal{L}}| = 20000$")

        add_bar()
        # store plots/exp1_rocauc.pdf
        metric = y.split("/")[1]
        # tight layout
        plt.tight_layout()
        if store:
            plt.savefig(os.path.join(plot_dir, f"exp1_{metric}_{flavor}.pgf"))
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
        pass
    elif flavor == "medium":
        dset_params["abs_nodes"] = 10
        dset_params["abs_edges"] = 20
    elif flavor == "large":
        dset_params["abs_nodes"] = 10
        dset_params["abs_edges"] = 20
        dset_params["min_block_size"] = 10
        dset_params["max_block_size"] = 15

    generate_datasets(dset_params, data_dir, num_runs, force=False)
    futures = []
    n_concrete = 20000
    max_paired_samples = (
        dset_params["abs_nodes"] * dset_params["max_block_size"] * 3
    )
    shift = int(max_paired_samples / 10)
    paired_range = list(range(shift, max_paired_samples, shift))
    if max_paired_samples not in paired_range:
        paired_range.append(max_paired_samples)
    for run in range(num_runs):
        # Test the Abs-LiNGAM-GT method
        futures.append(
            evaluate_lingam.remote(
                dset_params,
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
        futures.append(
            evaluate_lingam.remote(
                dset_params,
                data_dir,
                method="DirectLiNGAM",
                n_paired=0,
                n_concrete=n_concrete,
                run=run,
                shuffle_features=True,
                normalize=True,
                seed=seed,
                verbose=verbose,
                tau_threshold=0.0,
                abs_threshold=0.0,
            )
        )
        for n_paired in paired_range:
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
                    verbose=verbose,
                )
            )

            for bootstrap_samples in [1, 2, 5, 10]:
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
                        verbose=verbose,
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
    results_fname = os.path.join(results_dir, f"experiment1_{flavor}.csv")
    df.to_csv(results_fname, index=False)


if __name__ == "__main__":
    fire.Fire()
