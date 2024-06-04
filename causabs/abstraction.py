"""
Module containing several strategies
to fit an abstraction function from
data.
"""

import numpy as np


def perfect_abstraction(
    px_samples: np.array, py_samples: np.array, tau_threshold: float = 1e-2
):
    """
    Fits an abstraction function assumed to be perfect.
    """
    tau_adj = np.linalg.pinv(px_samples) @ py_samples
    tau_adj_mask = np.abs(tau_adj) > tau_threshold
    tau_adj = tau_adj * tau_adj_mask
    return tau_adj


def noisy_abstraction(
    px_samples: np.array,
    py_samples: np.array,
    tau_threshold: float = 1e-1,
    refit_coeff: bool = False,
):
    """
    Fits an abstraction function assumed to be noisy.
    """
    # Reconstruct T
    tau_adj_hat = np.linalg.pinv(px_samples) @ py_samples

    # Max entries
    tau_mask_hat = np.argmax(np.abs(tau_adj_hat), axis=1)
    # To one hot
    abs_nodes = py_samples.shape[1]
    tau_mask_hat = np.eye(abs_nodes)[tau_mask_hat]
    # Filter out small values (irrelevant variables)
    tau_mask_hat *= np.array(
        np.abs(tau_adj_hat) > tau_threshold, dtype=np.int32
    )

    # Eventually refit coefficients
    if refit_coeff:

        def get_block(y):
            """Return non-zero indices for each block."""
            return np.where(tau_mask_hat[:, y] == 1)[0]

        for y in range(tau_mask_hat.shape[1]):
            block = get_block(y)
            if len(block) > 0:
                tau_adj_hat[block, y] = (
                    np.linalg.pinv(px_samples[:, block]) @ py_samples[:, y]
                )

    # Compute the final abstraction
    tau_adj_hat = tau_mask_hat * tau_adj_hat

    return tau_adj_hat
