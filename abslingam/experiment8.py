"""
Experiment 8 studies how the reconstruction
of the abstraction function is affected
by the noise level on the observations
for different strategies.
"""

import os
import time

import fire
import numpy as np
import pandas as pd
import ray
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from causabs.abstraction import noisy_abstraction, perfect_abstraction
from causabs.dataset import (
    config_to_signature,
    generate_datasets,
    load_dataset,
)
from causabs.utils import preprocess_dataset, seed_everything


def _evaluate_abs(
    dset_params: dict,
    data_dir: str,
    method: str,
    n_paired: int,
    run_id: int,
    shuffle_features: bool,
    normalize: bool,
    seed: int,
    tau_threshold: float = 1e-2,
    verbose: bool = False,
) -> dict:
    """
    Evaluates the method performance.
    """
    seed_everything(seed)

    # load dataset
    signature = config_to_signature(dset_params)
    (
        cnc_weights,
        _,
        tau_adj,
        _,
        _,
        dset,
    ) = load_dataset(data_dir, signature, run_id)

    # preprocess the dataset
    (
        _,
        paired_samples,
        concrete_permutation,
        abstract_permutation,
    ) = preprocess_dataset(
        dset,
        n_paired,
        0,
        shuffle_features=shuffle_features,
        normalize=normalize,
    )

    if verbose:
        print("Not much to say, but I'm verbose.")

    # permute ground truth objects for evaluation
    tau_adj_gt = tau_adj[concrete_permutation, :][:, abstract_permutation]

    start_time = time.time()
    # switch over method
    if method == "Perfect":
        tau_adj_hat = perfect_abstraction(*paired_samples, tau_threshold)
    elif method == "Noisy":
        tau_adj_hat = noisy_abstraction(*paired_samples, tau_threshold, False)
    elif method == "Noisy-Refit":
        tau_adj_hat = noisy_abstraction(*paired_samples, tau_threshold, True)
    else:
        raise ValueError(f"Unknown method {method}.")
    end_time = time.time()

    # evaluation (precision, recall, roc-auc score, f1)
    def evaluate_adjacency(adj_hat, adj_gt, threshold=1e-3):
        adj_hat_mask = np.abs(adj_hat) > threshold
        adj_gt_mask = np.abs(adj_gt) > 0.0
        return {
            "precision": precision_score(
                adj_gt_mask.flatten(), adj_hat_mask.flatten()
            ),
            "recall": recall_score(
                adj_gt_mask.flatten(), adj_hat_mask.flatten()
            ),
            "roc_auc": roc_auc_score(
                adj_gt_mask.flatten(), np.abs(adj_hat).flatten()
            ),
            "f1": f1_score(adj_gt_mask.flatten(), adj_hat_mask.flatten()),
            "shd": hamming_loss(adj_gt_mask.flatten(), adj_hat_mask.flatten()),
            "nhd": hamming_loss(adj_gt_mask.flatten(), adj_hat_mask.flatten())
            / cnc_weights.shape[0],
        }

    # record of the run
    record = {}
    record.update(
        {
            f"eval/tau_{k}": v
            for k, v in evaluate_adjacency(
                tau_adj_hat, tau_adj_gt, tau_threshold
            ).items()
        }
    )
    record["eval/tau_MSE"] = np.mean((tau_adj_hat - tau_adj_gt) ** 2)
    record["eval/time"] = end_time - start_time
    record.update({f"dset/{k}": v for k, v in dset_params.items()})
    record["dset/cnc_nodes"] = cnc_weights.shape[0]
    record["dset/signature"] = signature
    record["dset/paired_samples"] = n_paired
    record["params/method"] = method
    record["params/run"] = run_id
    record["params/shuffle_features"] = shuffle_features
    record["params/normalize"] = normalize
    record["params/tau_threshold"] = tau_threshold
    record["params/seed"] = seed
    return record


@ray.remote
def evaluate_abs(*args, **kwargs):
    """Ray remote wrapper for _evaluate_abs."""
    return _evaluate_abs(*args, **kwargs)


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

    results_fname = os.path.join(results_dir, f"experiment8_{flavor}.csv")
    results = pd.read_csv(results_fname)

    print(f"== Experiment 8 ({flavor}) ==")

    # filter results
    print(f"Total Entries: {len(results)}")
    print(f"Flavor Entries: {len(results)}")

    print(len(results))

    for noise_abs in [0.0, 0.01, 0.1, 0.2, 0.5]:
        # Select the subset of results
        subframe = results[results["dset/noise_abs"] == noise_abs]
        # Group by method and tau threshold and take the mean and std
        subframe = subframe.groupby(
            ["params/method", "params/tau_threshold"]
        ).agg(
            {
                "eval/tau_shd": ["mean", "std"],
                "eval/tau_f1": ["mean", "std"],
                "eval/tau_MSE": ["mean", "std"],
                "eval/time": ["mean", "std"],
            }
        )
        # print("== Noise Abs:", noise_abs, "==")
        # print(subframe)

    # Select only threshold 1e-3 1e-2
    results = results[results["params/tau_threshold"].isin([1e-3, 1e-2, 1e-1])]

    # discretize tau_threshold (print in scientific notation 0.01 -> 1e-2)
    results["params/tau_threshold"] = results["params/tau_threshold"].apply(
        lambda x: f"{x:.0e}"
    )

    def format_method(entry):
        method = entry["params/method"]
        tau_threshold = f"Tau={entry['params/tau_threshold']}"
        return f"{method} ({tau_threshold})"

    # Create unique methods
    results["Method"] = results.apply(format_method, axis=1)
    results["Noise"] = results["dset/noise_abs"]

    labels = [
        ("eval/tau_roc_auc", r"ROCAUC $\mathbf{T}$"),
        ("eval/tau_precision", r"Precision $\mathbf{T}$"),
        ("eval/tau_recall", r"Recall $\mathbf{T}$"),
        ("eval/tau_f1", r"F1 $\mathbf{T}$"),
        ("eval/tau_shd", r"SHD $\mathbf{T}$"),
    ]

    for y, y_label in labels:
        plt.figure(figsize=(6, 3.5))
        sns.lineplot(
            x="Noise",
            y=y,
            hue="params/method",
            # hue="params/tau_threshold",
            style="params/tau_threshold",
            data=results,
        )

        # rename axis
        plt.xlabel(r"Abstract Noise Corruption")
        plt.ylabel(y_label)

        # store plots/exp8_rocauc.pdf
        metric = y.split("/")[1]
        if store:
            plt.savefig(os.path.join(plot_dir, f"exp8_{metric}_{flavor}.pdf"))
            plt.savefig(os.path.join(plot_dir, f"exp8_{metric}_{flavor}.pgf"))
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
        "n_samples": 500,
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

    futures = []
    n_paired = dset_params["abs_nodes"] * dset_params["max_block_size"] * 2
    for run_id in range(num_runs):
        for noise_abs in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            dset_params["noise_abs"] = noise_abs
            generate_datasets(dset_params, data_dir, num_runs, force=False)
            for tau_threshold in [1e-3, 1e-2, 5e-2, 1e-1, 2e-1]:
                for method in ["Perfect", "Noisy", "Noisy-Refit"]:
                    futures.append(
                        evaluate_abs.remote(
                            dset_params,
                            data_dir,
                            method,
                            n_paired,
                            run_id,
                            shuffle_features=True,
                            normalize=True,
                            seed=seed,
                            tau_threshold=tau_threshold,
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
    results_fname = os.path.join(results_dir, f"experiment8_{flavor}.csv")
    df.to_csv(results_fname, index=False)


if __name__ == "__main__":
    fire.Fire()
