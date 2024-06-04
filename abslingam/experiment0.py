"""
Experiment 0:
Evaluate the performance of Abs-LiNGAM for
different levels of noise in the observed
paired abstract observations against
different kinds of retrieval methods.
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
    pgf: bool = False,
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    results_fname = os.path.join(results_dir, f"experiment0_{flavor}.csv")
    results = pd.read_csv(results_fname)

    print(f"== Experiment 10 ({flavor}) ==")

    # filter results
    print(f"Total Entries: {len(results)}")
    results = preprocess_results(results)
    print(f"Flavor Entries: {len(results)}")

    # Sort by "params/{method, style, tau, bootstrap}"
    results = results.sort_values(
        by=[
            "params/method",
            "params/style",
            "params/tau_threshold",
            "params/bootstrap_samples",
        ]
    )

    # rename Noisy in "Top-1"
    results["params/style"] = results["params/style"].apply(
        lambda x: x.replace("Noisy", "Top-1")
    )

    # filter when threshold 1e-1
    results = results.where(results["params/tau_threshold"] == 1e-1)

    # discretize tau_threshold (print in scientific notation 0.01 -> 1e-2)
    results["params/tau_threshold"] = results["params/tau_threshold"].apply(
        lambda x: f"{x:.0e}"
    )

    def format_method(entry):
        if entry["params/method"] != "Abs-LiNGAM":
            return entry["params/method"]

        # tau_threshold = f"Tau={entry['params/tau_threshold']}"
        style = entry["params/style"]
        if style != "Perfect":
            return f"Abs-LiNGAM ({style})"
        else:
            return f"Abs-LiNGAM"

        # if entry["params/bootstrap_samples"] == 0:
        #     return f"Abs-LiNGAM ({style}, {tau_threshold})"
        # else:
        #     bootstrap = f"Bootstrap={entry['params/bootstrap_samples']}"
        #     return f"Abs-LiNGAM ({style}, {tau_threshold}, {bootstrap})"

    results["Method"] = results.apply(format_method, axis=1)

    if show:
        labels = [
            ("eval/abstract_roc_auc", r"ROCAUC $\mathcal{H}$"),
            # ("eval/abstract_precision", r"Precision $\mathcal{H}$"),
            # ("eval/abstract_recall", r"Recall $\mathcal{H}$"),
            # ("eval/abstract_f1", r"F1 $\mathcal{H}$"),
            ("eval/tau_roc_auc", r"ROCAUC $\mathbf{T}$"),
            # ("eval/tau_precision", r"Precision $\mathbf{T}$"),
            # ("eval/tau_recall", r"Recall $\mathbf{T}$"),
            # ("eval/tau_f1", r"F1 $\mathbf{T}$"),
            ("eval/concrete_roc_auc", r"ROCAUC $\mathcal{L}$"),
            # ("eval/concrete_precision", r"Precision $\mathcal{L}$"),
            # ("eval/concrete_recall", r"Recall $\mathcal{L}$"),
            # ("eval/concrete_f1", r"F1 $\mathcal{L}$"),
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
        plt.figure(figsize=(6, 3.5))

        sns.lineplot(
            x="dset/noise_abs",
            y=y,
            hue="Method",
            # hue="params/tau_threshold",
            data=results,
        )

        # rename axis
        plt.xlabel(r"Abstract Noise Variance $\sigma^2$")
        plt.ylabel(y_label)

        # # get axes
        # ax = plt.gca()
        # # Shrink current axis's height by 10% on the bottom
        # box = ax.get_position()
        # ax.set_position(
        #     [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.6]
        # )

        # # Put a legend below current axis
        # ax.legend(
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, 0.5),
        #     fancybox=True,
        #     shadow=True,
        #     ncol=3,
        # )

        # store plots/exp8_rocauc.pdf
        metric = y.split("/")[1]
        if store:
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"exp0_{metric}_{flavor}.pdf"))
            if pgf:
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plot_dir, f"exp0_{metric}_{flavor}.pgf")
                )
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

    # Define the dataset size (concrete and paired samples)
    n_concrete = 20000
    max_paired_samples = (
        dset_params["abs_nodes"] * dset_params["max_block_size"] * 3
    )
    shift = int(max_paired_samples / 10)
    paired_range = list(range(shift, max_paired_samples, shift))
    if max_paired_samples not in paired_range:
        paired_range.append(max_paired_samples)

    # Prepare future jobs for each dataset and model configuration
    futures = []
    n_paired = dset_params["abs_nodes"] * dset_params["max_block_size"] * 2
    for noise_abs in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        # Generate datasets
        dset_params["noise_abs"] = noise_abs
        generate_datasets(dset_params, data_dir, num_runs, force=False)
        for n_run in range(num_runs):
            # Test the Abs-LiNGAM-GT method
            futures.append(
                evaluate_lingam.remote(
                    dset_params,
                    data_dir,
                    method="Abs-LiNGAM-GT",
                    n_paired=0,
                    n_concrete=n_concrete,
                    run=n_run,
                    shuffle_features=True,
                    normalize=True,
                    seed=seed,
                    verbose=verbose,
                )
            )
            for tau_threshold in [1e-3, 1e-2, 1e-1]:
                for style in ["Perfect", "Noisy", "Noisy-Refit"]:
                    # Test the Abs-LiNGAM method w/o bootstrap
                    futures.append(
                        evaluate_lingam.remote(
                            dset_params,
                            data_dir,
                            method="Abs-LiNGAM",
                            n_paired=n_paired,
                            n_concrete=n_concrete,
                            run=n_run,
                            shuffle_features=True,
                            normalize=True,
                            seed=seed,
                            verbose=verbose,
                            tau_threshold=tau_threshold,
                            style=style,
                        )
                    )

                    # for bootstrap_samples in [2, 5, 10]:
                    #     # Test the Abs-LiNGAM method w/ bootstrap
                    #     futures.append(
                    #         evaluate_lingam.remote(
                    #             dset_params,
                    #             data_dir,
                    #             method="Abs-LiNGAM",
                    #             n_paired=n_paired,
                    #             n_concrete=n_concrete,
                    #             run=n_run,
                    #             shuffle_features=True,
                    #             normalize=True,
                    #             seed=seed,
                    #             verbose=verbose,
                    #             bootstrap_samples=bootstrap_samples,
                    #             tau_threshold=tau_threshold,
                    #             style=style,
                    #         )
                    #     )

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
    results_fname = os.path.join(results_dir, f"experiment0_{flavor}.csv")
    df.to_csv(results_fname, index=False)


if __name__ == "__main__":
    fire.Fire()
