"""Utilities to generate, store, and load datasets."""

import os

import numpy as np

from .sampling import (
    sample_linear_abstracted_models,
    sample_linear_realizations,
)


def linear_dataset(
    abs_nodes: int = 3,
    abs_edges: int = 3,
    abs_type: str = "ER",
    min_block_size: int = 1,
    max_block_size: int = 5,
    alpha: float = 1e4,
    relevant_ratio: float = 0.2,
    internal: bool = True,
    n_samples: int = 1000,
    noise_term: str = "gaussian",
    noise_abs: float = 0.1,
):
    """Samples a causal abstraction between random ANMs."""

    # Get the concrete and abstract models, with the ground truth abstraction
    (
        concrete,
        abstract,
        tau,
        gamma,
        pi,
    ) = sample_linear_abstracted_models(
        abs_nodes,
        abs_edges,
        abs_type,
        min_block_size,
        max_block_size,
        alpha,
        relevant_ratio,
        internal,
    )

    # Sample the dataset
    dataset = sample_linear_realizations(
        cnc_weights=concrete,
        abs_weights=abstract,
        tau=tau,
        gamma=gamma,
        n_samples=n_samples,
        noise_term=noise_term,
        noise_abs=noise_abs,
    )

    return concrete, abstract, tau, gamma, pi, dataset


PARAM_TO_ID = {
    "abs_nodes": "d",
    "abs_edges": "e",
    "abs_type": "t",
    "min_block_size": "m",
    "max_block_size": "M",
    "alpha": "h",
    "relevant_ratio": "p",
    "internal": "i",
    "n_samples": "n",
    "noise_term": "N",
    "noise_abs": "v",
}


def config_to_signature(dset_params: dict) -> str:
    """
    Given the configuration used to generate
    a random dataset, it returns the signature
    that is then used to store the dataset.
    """
    if set(dset_params.keys()) != set(PARAM_TO_ID.keys()):
        raise ValueError(
            "The configuration must contain all the parameters "
            f"{PARAM_TO_ID.keys()}."
        )
    return "_".join(
        [
            f"{PARAM_TO_ID[param]}{value}"
            for param, value in dset_params.items()
        ]
    )


def signature_to_config(signature: str) -> dict:
    """
    Given the signature of a dataset, it returns
    the configuration used to generate it.
    """
    dset_params = {}
    tokens = signature.split("_")
    for param, param_id in PARAM_TO_ID.items():
        for token in tokens:
            if token.startswith(param_id):
                dset_params[param] = token[1:]
                if param in [
                    "relevant_ratio",
                    "alpha",
                    "noise_abs",
                ]:
                    try:
                        dset_params[param] = float(dset_params[param])
                    except ValueError:
                        dset_params[param] = None
                elif param in [
                    "internal",
                ]:
                    dset_params[param] = dset_params[param] == "True"
                elif param not in ["graph_type", "noise_term"]:
                    try:
                        dset_params[param] = int(dset_params[param])
                    except ValueError:
                        dset_params[param] = None
    return dset_params


def load_dataset(data_path: str, signature: str, num: int):
    """Loads from disk"""
    # filename
    fname = f"{signature}_run{num}.npz"
    path = os.path.join(data_path, fname)

    # load dataset
    data = np.load(path)

    concrete = data["concrete"]
    abstract = data["abstract"]
    tau = data["tau"]
    gamma = data["gamma"]
    pi = data["pi"]
    dataset = (data["samples_x"], data["samples_y"])

    return concrete, abstract, tau, gamma, pi, dataset


def check_dataset(data_path: str, signature: str, num: int):
    """Checks if the dataset exists."""
    # filename
    fname = f"{signature}_run{num}.npz"
    path = os.path.join(data_path, fname)
    return os.path.exists(path)


def generate_datasets(
    dset_params: dict, data_path: str, n_repetitions: int, force: bool = False
) -> str:
    """Stores a simulated linear dataset."""
    # create data path if it does not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # generates signature
    signature = config_to_signature(dset_params)

    # iterate over number of repetitions
    for rep_num in range(n_repetitions):
        # try to load the dataset
        if check_dataset(data_path, signature, rep_num):
            print(f"Dataset {signature}_IT{rep_num} already exists.")
            if not force:
                continue

        # generate dataset
        (
            concrete,
            abstract,
            tau,
            gamma,
            pi,
            dataset,
        ) = linear_dataset(**dset_params)

        samples_x, samples_y = dataset

        # filename
        fname = f"{signature}_run{rep_num}.npz"

        # save dataset
        np.savez(
            os.path.join(data_path, fname),
            concrete=concrete,
            abstract=abstract,
            tau=tau,
            gamma=gamma,
            pi=pi,
            samples_x=samples_x,
            samples_y=samples_y,
        )

        print(f"Dataset {fname} generated.")

    return signature
