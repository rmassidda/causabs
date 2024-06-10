"""
Causal Abstraction Consistency measures
on linear Structural Causal Models.
"""

import numpy as np

from causabs.utils import block_idx, linear_anm, measure_blocks


def _int_consistency(
    cnc_model: np.array,
    abs_model: np.array,
    tau_adj: np.array,
    gamma_adj: np.array,
    interventions: list[tuple[dict, dict]],
    n_samples: int = 1000,
    noise: callable = np.random.normal,
) -> float:
    """
    Computes the interventional consistency of the
    two models given the endogenous and exogenous
    abstraction function on the provided list of
    concrete and abstract interventions.
    """
    # Number of concrete nodes
    cnc_nodes = cnc_model.shape[0]

    # Initialize error
    error = 0.0

    # Iterate over interventions
    for cnc_int, abs_int in interventions:
        # Sample the exogenous noise
        exog = noise(size=(n_samples, cnc_nodes))
        # Compute and abstract
        path_a = linear_anm(cnc_model, exog, cnc_int) @ tau_adj
        # Abstract and compute
        path_b = linear_anm(abs_model, exog @ gamma_adj, abs_int)
        # Update error (MSE)
        error += np.mean((path_a - path_b) ** 2)

    # Normalize error
    error /= len(interventions) * n_samples

    return error


def obs_consistency(
    cnc_model: np.array,
    abs_model: np.array,
    tau_adj: np.array,
    gamma_adj: np.array,
    n_samples: int = 1000,
    noise: callable = np.random.normal,
) -> float:
    """
    Computes the observational consistency of the
    two models given the endogenous and exogenous
    abstraction function.

    Parameters
    ----------
    cnc_model : np.array
        Concrete model.
    abs_model : np.array
        Abstract model.
    tau_adj : np.array
        Endogenous Abstraction function.
    gamma_adj : np.array
        Exogenous Abstraction function.
    n_samples : int
        Number of samples.
    noise : callable
        Noise function.
    """
    return _int_consistency(
        cnc_model, abs_model, tau_adj, gamma_adj, [({}, {})], n_samples, noise
    )


def int_consistency(
    cnc_model: np.array,
    abs_model: np.array,
    tau_adj: np.array,
    gamma_adj: np.array,
    style: str = "random_normal",
    max_targets: int = 1,
    n_ints: int = 100,
    n_samples: int = 1000,
    noise: callable = np.random.normal,
) -> float:
    """
    Computes the observational consistency of two
    Linear SCMs given an endogenous and exogenous
    abstraction function. The consistency is computed
    over a list of interventions that is generated
    according to the style parameter.

    Parameters
    ----------
    cnc_model : np.array
        Concrete model.
    abs_model : np.array
        Abstract model.
    tau_adj : np.array
        Endogenous Abstraction function.
    gamma_adj : np.array
        Exogenous Abstraction function.
    style : str
        Intervention style. The following styles are
        supported:
            - "random_normal": Randomly generated interventions,
            - "random_zero": Randomly generated interventions,
                setting each node to zero.
    max_targets : int
        Maximum number of abstract targets per intervention.
    n_ints : int
        Number of interventions, ignored if style is
        "all_zero".
    n_samples : int
        Number of samples.
    noise : callable
        Noise function.
    """
    # Empty list of interventions
    interventions = []

    # Get blocks
    block_size = measure_blocks(tau_adj)
    abs_nodes = abs_model.shape[0]
    blocks = block_idx(block_size)

    # Check max targets
    if max_targets > abs_nodes:
        raise ValueError(
            "Max targets cannot exceed the number of abstract nodes."
        )

    if style == "random_zero":
        # Generate random interventions
        for _ in range(n_ints):
            targets = np.random.choice(abs_nodes, max_targets, replace=False)
            abs_int = {target: 0.0 for target in targets}
            cnc_int = {}
            for target in targets:
                for cnc_target in list(range(*blocks[target])):
                    cnc_int[cnc_target] = 0.0
            interventions.append((cnc_int, abs_int))
    elif style == "random_normal":
        # Generate random interventions
        for _ in range(n_ints):
            targets = np.random.choice(abs_nodes, max_targets, replace=False)
            abs_int = {}
            cnc_int = {}
            for target in targets:
                target_val = 0.0
                for cnc_target in list(range(*blocks[target])):
                    cnc_int[cnc_target] = np.random.normal()
                    target_val += (
                        cnc_int[cnc_target] * tau_adj[cnc_target, target]
                    )
                abs_int[target] = target_val
            interventions.append((cnc_int, abs_int))
    else:
        raise NotImplementedError(f"Style {style} not implemented.")

    return _int_consistency(
        cnc_model,
        abs_model,
        tau_adj,
        gamma_adj,
        interventions,
        n_samples,
        noise,
    )
