"""Utility functions"""

import random
from itertools import chain, combinations
from typing import Optional

import numpy as np


def seed_everything(seed: int):
    """Seed everything."""
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seeded everything with {seed}")


def powerset(iterable):
    """power-set"""
    s = list(iterable)
    return list(
        chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    )


def compute_direct_paths(adj_matrix: np.ndarray) -> np.ndarray:
    """Matrix of direct paths."""
    n_nodes = adj_matrix.shape[0]
    # set 1 if there is an arc, zero otherwise
    adj_matrix = np.where(np.abs(adj_matrix) > 0, 1, 0)

    mechanism = np.zeros((n_nodes, n_nodes))
    cumulator = np.eye(n_nodes)
    mechanism += cumulator
    for _ in range(n_nodes):
        cumulator = cumulator @ adj_matrix
        cumulator = np.where(cumulator > 0, 1, 0)
        mechanism += cumulator

    mechanism = np.where(mechanism > 0, 1, 0)

    return mechanism


def compute_mechanism(adj_matrix: np.ndarray) -> np.ndarray:
    """Reduced form of the linear ANM."""
    n_nodes = adj_matrix.shape[0]
    mechanism = np.zeros((n_nodes, n_nodes))
    cumulator = np.eye(n_nodes)
    mechanism += cumulator
    for _ in range(n_nodes):
        cumulator = cumulator @ adj_matrix
        mechanism += cumulator
    return mechanism


def check_cancelling_paths(adj_matrix: np.ndarray) -> bool:
    """
    In a linear SCM, the model does not have cancelling paths
    whenever the direct paths coincide with the dependencies
    of the mechanism.
    """
    return np.allclose(
        compute_direct_paths(adj_matrix),
        np.abs(compute_mechanism(adj_matrix)) > 0,
    )


def preprocess_dataset(
    dset: list,
    n_paired: int,
    n_concrete: int,
    shuffle_features: bool = True,
    normalize: bool = True,
):
    """Preprocesses the dataset by normalizing and shuffling features."""
    # get samples
    px_samples, py_samples = dset[0], dset[1]

    # shuffle features
    if shuffle_features:
        concrete_permutation = np.random.permutation(px_samples.shape[1])
        abstract_permutation = np.random.permutation(py_samples.shape[1])
    else:
        concrete_permutation = np.arange(px_samples.shape[1])
        abstract_permutation = np.arange(py_samples.shape[1])

    # permutation of concrete features
    px_samples = px_samples[:, concrete_permutation]
    py_samples = py_samples[:, abstract_permutation]

    # assert no overlap
    assert n_concrete + n_paired <= px_samples.shape[0]

    # split samples
    concrete_samples = px_samples[:n_concrete]
    paired_samples = [px_samples[-n_paired:], py_samples[-n_paired:]]

    # data normalization
    if normalize:
        concrete_samples = (
            concrete_samples - np.mean(concrete_samples, axis=0)
        ) / np.std(concrete_samples, axis=0)
        paired_samples[0] = (
            paired_samples[0] - np.mean(paired_samples[0], axis=0)
        ) / np.std(paired_samples[0], axis=0)
        paired_samples[1] = (
            paired_samples[1] - np.mean(paired_samples[1], axis=0)
        ) / np.std(paired_samples[1], axis=0)

    return (
        concrete_samples,
        paired_samples,
        concrete_permutation,
        abstract_permutation,
    )


def linear_anm(
    weights: np.ndarray,
    exogenous: np.ndarray,
    intervention: Optional[dict] = None,
) -> np.ndarray:
    """Linear Additive Noise Model (ANM)"""
    endogenous = np.zeros_like(exogenous)
    n_nodes = weights.shape[0]
    assert weights.shape == (n_nodes, n_nodes)

    # max number of steps
    for _ in range(n_nodes):
        # update endogenous
        endogenous = endogenous @ weights + exogenous

        # apply intervention
        if intervention is not None:
            for target, value in intervention.items():
                endogenous[:, target] = value

    return endogenous
