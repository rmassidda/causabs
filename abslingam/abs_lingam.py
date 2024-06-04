"""
Implementation of the Abs-LiNGAM method
and its evaluation.
"""

import time
from typing import Optional, Tuple

import lingam
import numpy as np
import pandas as pd
import ray
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from causabs.abstraction import noisy_abstraction, perfect_abstraction
from causabs.dataset import config_to_signature, load_dataset
from causabs.utils import (
    compute_direct_paths,
    preprocess_dataset,
    seed_everything,
)

pp_names = {
    "params/bootstrap_samples": "Bootstrap",
}


def preprocess_results(results: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the results by appending parameters to the method."""
    # copy column "params/method" in "Method"
    results["Method"] = results["params/method"]

    # iterate over columns starting with "params/"
    params = results.filter(regex="params/").columns
    # ignore params/run
    params = [param for param in params if param != "params/run"]

    for param in params:
        # check if unique when "params/method" = "Abs-LiNGAM"
        sub_frame = results[results["params/method"] == "Abs-LiNGAM"]
        unique = sub_frame[param].unique()
        if len(unique) > 1:
            print("log: found", param, "with multiple values", unique)

            # pretty name
            try:
                param_name = pp_names[param]
            except KeyError:
                param_name = param.split("/")[-1]

            # Create string param_name=value and append to Method
            index = results["params/method"] == "Abs-LiNGAM"
            results.loc[index, "Method"] = (
                results[index]["Method"]
                + " ("
                + param_name
                + "="
                + results[index][param].astype(str)
                + ")"
            )

    return results


def false_positive_rate(y_true, y_pred):
    """
    Compute the false positive rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def full_lingam(cnc_dataset: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Runs DirectLiNGAM on the concrete dataset."""
    cnc_nodes = cnc_dataset.shape[1]
    model = lingam.DirectLiNGAM()
    try:
        model.fit(cnc_dataset)
        cnc_weights = model.adjacency_matrix_.T
    except ValueError:
        # samples < variables
        cnc_weights = np.zeros((cnc_nodes, cnc_nodes))
    return cnc_weights


def abs_lingam(
    cnc_dataset: np.ndarray,
    joint_dataset: Optional[Tuple[np.ndarray, np.ndarray]],
    tau_adj: Optional[np.ndarray],
    abs_weights: Optional[np.ndarray],
    tau_threshold: float,
    skip_concrete: bool,
    abs_threshold: float,
    max_abstract_samples: Optional[int],
    bootstrap_samples: int,
    verbose: bool = False,
    style: str = "Perfect",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs DirectLiNGAM on the concrete model by exploiting
    prior knowledge given the abstraction function.
    """
    # get number of nodes
    cnc_nodes = cnc_dataset.shape[1]

    # tau-abstraction weights
    if tau_adj is None:
        if style == "Perfect":
            tau_adj = perfect_abstraction(*joint_dataset, tau_threshold)
        elif style == "Noisy":
            tau_adj = noisy_abstraction(*joint_dataset, tau_threshold, False)
        elif style == "Noisy-Refit":
            tau_adj = noisy_abstraction(*joint_dataset, tau_threshold, True)
        else:
            raise ValueError(
                f"Unknown style {style} for Abs-LiNGAM (Perfect, Noisy, Noisy-Refit)."
            )
        tau_adj_mask = np.abs(tau_adj) > tau_threshold
    else:
        tau_adj_mask = np.abs(tau_adj) > 0.0

    # enforce mask
    tau_adj_store = np.copy(tau_adj)
    tau_adj = tau_adj * tau_adj_mask

    if verbose:
        print("Computed tau abstraction.")

    # fit abstract model
    if abs_weights is None:
        abs_dataset = cnc_dataset @ tau_adj

        # Eventually reduce the number of abstract samples
        if max_abstract_samples is not None:
            abs_dataset = abs_dataset[:max_abstract_samples]

        model = lingam.DirectLiNGAM()

        # Directly fit the model or bootstrap
        if bootstrap_samples == 0:
            model.fit(abs_dataset)
            abs_weights = model.adjacency_matrix_.T
        else:
            result = model.bootstrap(abs_dataset, n_sampling=bootstrap_samples)
            abs_weights = result.get_probabilities().T

        abs_weights_mask = np.abs(abs_weights) > abs_threshold
    else:
        abs_weights_mask = np.abs(abs_weights) > 0.0

    # enforce mask
    abs_weights_store = np.copy(abs_weights)
    abs_weights = abs_weights * abs_weights_mask

    if verbose:
        print("Fitted abstract model.")

    # count abstract nodes
    abs_nodes = abs_weights.shape[0]

    # get relevant variables
    relevant = []
    for y in range(abs_nodes):
        relevant.append(np.where(tau_adj_mask[:, y] == 1)[0])

    # compute abstract directed paths
    abs_directed = compute_direct_paths(abs_weights)

    # compute prior knowledge
    prior_knowledge = -np.ones((cnc_nodes, cnc_nodes))
    for y_a in range(abs_nodes):
        for y_b in range(abs_nodes):
            # no path y_a -> y_b
            if abs_directed[y_a, y_b] == 0.0:
                for x_a in relevant[y_a]:
                    for x_b in relevant[y_b]:
                        prior_knowledge[x_a, x_b] = 0
    prior_knowledge_store = np.copy(prior_knowledge)

    # skip concrete
    if skip_concrete:
        cnc_weights = np.zeros((cnc_nodes, cnc_nodes))
        return (
            cnc_weights,
            abs_weights_store,
            tau_adj_store,
            prior_knowledge_store,
        )

    model = lingam.DirectLiNGAM(
        prior_knowledge=prior_knowledge.T,
        apply_prior_knowledge_softly=False,
    )

    try:
        model.fit(cnc_dataset)
        cnc_weights = model.adjacency_matrix_.T
    except ValueError:
        # samples < variables
        cnc_weights = np.zeros((cnc_nodes, cnc_nodes))

    if verbose:
        print("Fitted concrete model.")

    return cnc_weights, abs_weights_store, tau_adj_store, prior_knowledge_store


def _evaluate_lingam(
    dset_params: dict,
    data_dir: str,
    method: str,
    n_paired: int,
    n_concrete: int,
    run: int,
    shuffle_features: bool,
    normalize: bool,
    seed: int,
    verbose: bool,
    tau_threshold: float = 1e-2,
    abs_threshold: float = 1e-2,
    skip_concrete: bool = False,
    max_abstract_samples: Optional[int] = None,
    bootstrap_samples: int = 0,
    style: str = "Perfect",
) -> dict:
    """
    Evaluates the method performance.
    """
    seed_everything(seed)

    # load dataset
    signature = config_to_signature(dset_params)
    (
        cnc_weights,
        abs_weights,
        tau_adj,
        _,
        _,
        dset,
    ) = load_dataset(data_dir, signature, run)

    # preprocess the dataset
    (
        concrete_samples,
        paired_samples,
        concrete_permutation,
        abstract_permutation,
    ) = preprocess_dataset(
        dset,
        n_paired,
        n_concrete,
        shuffle_features=shuffle_features,
        normalize=normalize,
    )

    # permute ground truth objects for evaluation
    tau_adj_gt = tau_adj[concrete_permutation, :][:, abstract_permutation]
    cnc_weights_gt = cnc_weights[concrete_permutation, :][
        :, concrete_permutation
    ]
    abs_weights_gt = abs_weights[abstract_permutation, :][
        :, abstract_permutation
    ]

    start_time = time.time()
    # switch over method
    if method == "DirectLiNGAM":
        # Fit the concrete model without prior knowledge
        cnc_weights_hat = full_lingam(concrete_samples, verbose=verbose)
        abs_weights_hat = abs_weights_gt
        tau_adj_hat = tau_adj_gt
        pk = None
    elif method == "Abs-LiNGAM-GT":
        # Fit the concrete model with ground truth prior knowledge
        cnc_weights_hat, abs_weights_hat, tau_adj_hat, pk = abs_lingam(
            concrete_samples,
            None,
            tau_adj=tau_adj_gt,
            abs_weights=abs_weights_gt,
            tau_threshold=tau_threshold,
            abs_threshold=abs_threshold,
            skip_concrete=skip_concrete,
            max_abstract_samples=max_abstract_samples,
            bootstrap_samples=bootstrap_samples,
            style=None,
            verbose=verbose,
        )
    elif method == "Abs-Only":
        # Fit the abstract model only from data
        abs_weights_hat = full_lingam(paired_samples[1], verbose=verbose)
        cnc_weights_hat = cnc_weights_gt
        tau_adj_hat = tau_adj_gt
        pk = None
    elif method == "Abs-LiNGAM":
        # Fit everything from data
        cnc_weights_hat, abs_weights_hat, tau_adj_hat, pk = abs_lingam(
            concrete_samples,
            paired_samples,
            tau_adj=None,
            abs_weights=None,
            tau_threshold=tau_threshold,
            abs_threshold=abs_threshold,
            skip_concrete=skip_concrete,
            max_abstract_samples=max_abstract_samples,
            bootstrap_samples=bootstrap_samples,
            style=style,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown method {method}.")
    end_time = time.time()

    # evaluation (precision, recall, roc-auc score, f1)
    def evaluate_adjacency(adj_hat, adj_gt, threshold=1e-3):
        adj_hat_mask = np.abs(adj_hat) > threshold
        adj_gt_mask = np.abs(adj_gt) > 0.0
        return {
            "fpr": false_positive_rate(
                adj_gt_mask.flatten(), adj_hat_mask.flatten()
            ),
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
        }

    # record of the run
    record = {}
    record.update(
        {
            f"eval/concrete_{k}": v
            for k, v in evaluate_adjacency(
                cnc_weights_hat, cnc_weights_gt
            ).items()
        }
    )
    record.update(
        {
            f"eval/abstract_{k}": v
            for k, v in evaluate_adjacency(
                abs_weights_hat, abs_weights_gt, abs_threshold
            ).items()
        }
    )
    record.update(
        {
            f"eval/tau_{k}": v
            for k, v in evaluate_adjacency(
                tau_adj_hat, tau_adj_gt, tau_threshold
            ).items()
        }
    )
    if pk is not None:
        # Compute concrete directed paths
        cnc_directed = compute_direct_paths(cnc_weights_gt)
        # compute precision
        record["eval/pk_precision"] = precision_score(
            (1 - cnc_directed).flatten(), (pk + 1).flatten()
        )
        record["eval/pk_recall"] = recall_score(
            (1 - cnc_directed).flatten(), (pk + 1).flatten()
        )
    record["eval/time"] = end_time - start_time
    record.update({f"dset/{k}": v for k, v in dset_params.items()})
    record["dset/cnc_nodes"] = cnc_weights.shape[0]
    record["dset/signature"] = signature
    record["dset/paired_samples"] = n_paired
    record["dset/concrete_samples"] = n_concrete
    if skip_concrete:
        # check rank of paired_samples[0] if concrete is skipped
        record["dset/paired_rank"] = (
            np.linalg.matrix_rank(paired_samples[0]) / cnc_weights.shape[0]
        )
    record["params/method"] = method
    record["params/run"] = run
    record["params/shuffle_features"] = shuffle_features
    record["params/normalize"] = normalize
    record["params/tau_threshold"] = tau_threshold
    record["params/abs_threshold"] = abs_threshold
    record["params/skip_concrete"] = skip_concrete
    record["params/max_abstract_samples"] = max_abstract_samples
    record["params/bootstrap_samples"] = bootstrap_samples
    record["params/seed"] = seed
    record["params/verbose"] = verbose
    record["params/style"] = style

    return record


@ray.remote
def evaluate_lingam(*args, **kwargs):
    """
    Ray remote function to evaluate the method.
    """
    return _evaluate_lingam(*args, **kwargs)
