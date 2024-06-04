"""Sampling of models, abstractions, and datasets."""

from typing import Tuple

import igraph as ig
import numpy as np

from .utils import check_cancelling_paths, compute_mechanism, linear_anm

AdjacencyMatrix = np.ndarray


def sample_random_dag(
    num_nodes: int,
    num_edges: int,
    graph_type: str,
    ordered: bool = False,
) -> ig.Graph:
    """
    Samples a random DAG with the expected number of edges,
    originally taken from https://github.com/xunzheng/notears.
    """

    def _random_permutation(matrix: AdjacencyMatrix) -> AdjacencyMatrix:
        # np.random.permutation permutes first axis only
        p_matrix = np.random.permutation(np.eye(matrix.shape[0]))
        return p_matrix.T @ matrix @ p_matrix

    def _random_acyclic_orientation(b_und):
        return np.tril(_random_permutation(b_und), k=-1)

    def _graph_to_adjmat(graph: ig.Graph) -> AdjacencyMatrix:
        return np.array(graph.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        graph = ig.Graph.Erdos_Renyi(n=num_nodes, m=num_edges)
        binary = _graph_to_adjmat(graph)
        binary = _random_acyclic_orientation(binary)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        graph = ig.Graph.Barabasi(
            n=num_nodes, m=int(round(num_edges / num_nodes)), directed=True
        )
        binary = _graph_to_adjmat(graph)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * num_nodes)
        graph = ig.Graph.Random_Bipartite(
            top, num_nodes - top, m=num_edges, directed=True, neimode=ig.OUT
        )
        binary = _graph_to_adjmat(graph)
    else:
        raise ValueError("unknown graph type")

    # Random permutation of the node labels
    binary = _random_permutation(binary)
    graph = ig.Graph.Weighted_Adjacency(binary.tolist())

    if ordered:
        # Get topological ordering
        order = graph.topological_sorting()
        # Reorder adjacency matrix
        binary = binary[order, :][:, order]
        graph = ig.Graph.Weighted_Adjacency(binary.tolist())

    return graph


def sample_linear_anm(
    nodes: int,
    edges: int,
    graph_type: str,
    w_ranges=((-2.0, -0.5), (0.5, 2.0)),
) -> AdjacencyMatrix:
    """
    Samples ANM parameters for a DAG.
    """
    # sample a DAG binary adjacency matrix
    graph = sample_random_dag(nodes, edges, graph_type, ordered=True)
    binary = np.array(graph.get_adjacency().data)

    # sample the weights
    weight_adj = np.zeros(binary.shape)
    sign_mat = np.random.randint(len(w_ranges), size=binary.shape)
    for i, (low, high) in enumerate(w_ranges):
        unif_sample = np.random.uniform(low=low, high=high, size=binary.shape)
        weight_adj += binary * (sign_mat == i) * unif_sample

    return weight_adj


def sample_linear_concretization(
    abs_weights: AdjacencyMatrix,
    tau_adj: np.ndarray,
    block_size: list[int],
    internal: bool = True,
    alpha: float = 1e4,
    style: str = "positive",
) -> Tuple[AdjacencyMatrix, np.ndarray]:
    """
    Given the abstract adjacency matrix and the endogenous
    abstraction function tau, it returns the concrete adjacency
    matrix and the exogenous abstraction function gamma.

    Parameters:
    -----------
    abs_weights: AdjacencyMatrix
        The adjacency matrix of the abstract model.
    tau_adj: np.ndarray
        The endogenous abstraction function.
    block_size: list[int]
        The size of each block for any abstract variable.
    internal: bool
        If True, the internal connections are sampled
        as well. Otherwise, the blocks are considered
        to be internally disconnected.
    alpha: float
        Concentration parameter for the Dirichlet distribution.
    style: str
        The style of the concrete weights.
    """
    # Abstract model
    abs_nodes = abs_weights.shape[0]
    block_start = np.cumsum([0] + block_size[:-1])

    # Helper to get block-indices
    def get_block(y: int) -> slice:
        return slice(block_start[y], block_start[y] + block_size[y])

    # Exogenous abstraction
    gamma_adj = np.copy(tau_adj)

    # Handle block internal
    f_blocks = []
    cnc_nodes = tau_adj.shape[0]
    cnc_weights = np.zeros((cnc_nodes, cnc_nodes))
    for y in range(abs_nodes):
        block_y = get_block(y)
        size_y = block_size[y]
        if internal:
            # NOTE: Blocks are internally fully connected,
            #       but in principle connections could be
            #       missing without affecting the theory.

            # sample the internal connections
            cnc_weights[block_y, block_y] = np.random.normal(
                size=(size_y, size_y)
            ) / np.sqrt(size_y)
            # remove the lower-triangular part
            cnc_weights[block_y, block_y] = np.triu(
                cnc_weights[block_y, block_y], k=1
            )
            # compute internal mechanism
            f_yy = compute_mechanism(cnc_weights[block_y, block_y])
        else:
            f_yy = np.eye(size_y)
        f_blocks.append(f_yy)

        # compute gamma
        gamma_adj[block_y, y] = f_blocks[y] @ tau_adj[block_y, y]

    # Abstract mechanism
    abs_model = compute_mechanism(abs_weights)

    # Remainder matrix (cnc_nodes, cnc_nodes)
    remainder = np.zeros_like(cnc_weights)

    # Concrete weights
    for y_b in range(abs_nodes):
        # target block
        block_b = slice(block_start[y_b], block_start[y_b] + block_size[y_b])

        # target abstraction
        s_b = gamma_adj[block_b, y_b]

        # allowed concrete targets
        all_targets = list(np.where(np.abs(s_b) > 0.0)[0])

        for y_a in list(reversed(range(y_b))):
            # source block
            block_a = slice(
                block_start[y_a], block_start[y_a] + block_size[y_a]
            )

            # abstraction function
            t_a = tau_adj[block_a, y_a]

            # abstract mechanism
            g_ab = abs_model[y_a, y_b]
            m_ab = abs_weights[y_a, y_b]

            # Update the remainder
            i, j = y_a, y_b
            start = min(i, j) + 1
            end = max(i, j)
            for k in range(start, end):
                # compute k-th step
                block_i = get_block(i)
                block_j = get_block(j)
                block_k = get_block(k)
                w_ik = cnc_weights[block_i, block_k]
                w_kj = cnc_weights[block_k, block_j]
                r_kj = remainder[block_k, block_j]
                f_kk = f_blocks[k]

                # update the remainder
                remainder[block_i, block_j] += w_ik @ f_kk @ (w_kj + r_kj)

            # extract the remainder
            r_ab = remainder[block_a, block_b]

            # compute the weights
            for k in range(block_size[y_a]):
                # random choice half of the possible targets variables
                targets = np.random.choice(
                    all_targets,
                    size=max(len(all_targets) // 2, 1),
                    replace=False,
                )

                # build the assignment vector
                v = np.zeros_like(s_b)

                if style == "uniform":
                    dist = 1 / len(targets)
                elif style == "positive":
                    dist = np.random.dirichlet(np.ones(len(targets)) * alpha)
                elif style == "negative_bound":
                    # to avoid exploding weights try at least 100
                    # times to sample a vector with already sum larger than one
                    found = False
                    for _ in range(100):
                        dist = np.random.normal(size=len(targets))
                        if np.abs(np.sum(dist)) > 1.0:
                            found = True
                            break
                    if not found:
                        # if not found, assign all positive
                        print("WARNING: Could not find a valid assignment.")
                        dist = np.random.dirichlet(
                            np.ones(len(targets)) * alpha
                        )
                    dist = dist / np.sum(dist)
                elif style == "negative_unbound":
                    dist = np.random.normal(size=len(targets))
                    dist = dist / np.sum(dist)
                elif style == "negative":
                    # assign negative weights
                    if len(targets) == 1:
                        dist = np.ndarray([1.0])
                    elif len(targets) == 2:
                        dist = np.random.dirichlet(np.ones(2) * alpha)
                    else:
                        # number of targets following the abstraction sign
                        n_positives = np.random.randint(2, len(targets))
                        n_negatives = len(targets) - n_positives
                        dist_pos = np.random.dirichlet(
                            np.ones(n_positives) * alpha
                        )
                        dist_neg = np.random.dirichlet(
                            np.ones(n_negatives) * alpha
                        )
                        # the positive weights are larger
                        # difference must be equal to 1.0
                        dist_pos *= 2.0
                        dist_neg *= -1.0
                        # concatenate
                        dist = np.concatenate([dist_pos, dist_neg])
                        # shuffle
                        dist = np.random.permutation(dist)
                else:
                    raise ValueError(f"Unknown style {style}.")

                # right-inverse of s_b
                c_b = np.zeros_like(s_b)
                v[targets] = dist
                c_b[all_targets] = v[all_targets] / s_b[all_targets]

                # compute block
                cnc_weights[block_a, block_b][k, :] = m_ab * t_a[k] * c_b

            # check weights closed form
            w_ab = cnc_weights[block_a, block_b]
            s_b = s_b.reshape((len(s_b), 1))
            t_a = t_a.reshape((len(t_a), 1))
            assert np.allclose(w_ab @ s_b, g_ab * t_a - r_ab @ s_b)
            s_b = s_b.reshape((len(s_b),))

    # set connections towards ignored variables
    block_ignored = get_block(abs_nodes)
    for y in range(abs_nodes):
        block_y = get_block(y)
        # random weights from block_y to block_ignored
        cnc_weights[block_y, block_ignored] = np.random.normal(
            size=(block_size[y], block_size[-1])
        ) / np.sqrt(block_size[y])
        # randomly mask half of the weights
        mask = np.random.randint(
            2, size=cnc_weights[block_y, block_ignored].shape
        )
        # set the masked weights to zero
        cnc_weights[block_y, block_ignored] *= mask
    # set connections from block_ignored to block_ignored
    cnc_weights[block_ignored, block_ignored] = np.random.normal(
        size=(block_size[-1], block_size[-1])
    ) / np.sqrt(block_size[-1])
    # remove the lower-triangular part
    cnc_weights[block_ignored, block_ignored] = np.triu(
        cnc_weights[block_ignored, block_ignored], k=1
    )

    # consistency when FT = SG
    cnc_model = compute_mechanism(cnc_weights)
    assert np.allclose(cnc_model @ tau_adj, gamma_adj @ abs_model, atol=1e-4)

    # check cancelling paths
    if not check_cancelling_paths(cnc_weights):
        # NOTE: It would be better to raise an exception.
        print("WARNING: The concrete model is not faithful.")

    return cnc_weights, gamma_adj


def sample_linear_abstraction(
    abs_nodes: int,
    min_block_size: int,
    max_block_size: int,
    relevant_ratio: float = 0.2,
    tau_ranges: tuple = (0.5, 2.0),
    ignored_block: bool = True,
) -> np.ndarray:
    """Samples a linear abstraction function."""
    # Sample readouts
    block_size = [
        np.random.randint(min_block_size, max_block_size + 1)
        for _ in range(abs_nodes + 1)
    ]

    # Eventually remove ignored variables
    if not ignored_block:
        block_size = block_size[:-1]

    # Concrete model pointers
    concrete_nodes = sum(block_size)
    block_start = np.cumsum([0] + block_size[:-1])

    # Sample abstraction function
    tau_adj = np.zeros((concrete_nodes, abs_nodes))
    for y in range(abs_nodes):
        interval = slice(block_start[y], block_start[y] + block_size[y])
        sign_mat = np.random.randint(2, size=block_size[y])
        tau_adj[interval, y] = (
            np.random.uniform(
                low=tau_ranges[0], high=tau_ranges[1], size=block_size[y]
            )
            * (-1) ** sign_mat
        )

        # Sample irrelevant variables
        irrelevant_ratio = 1.0 - relevant_ratio
        max_irrelevant = block_size[y] - 1
        n_irrelevant = np.random.randint(
            0, int(max_irrelevant * irrelevant_ratio) + 1
        )
        irrelevant = np.random.choice(
            max_irrelevant, n_irrelevant, replace=False
        )

        # Set the abstraction to zero for irrelevant variables
        tau_adj[interval, y][irrelevant] = 0.0

    return tau_adj


def sample_linear_abstracted_models(
    abs_nodes: int,
    abs_edges: int,
    abs_type: str,
    min_block_size: int = 2,
    max_block_size: int = 5,
    alpha: float = 1e-4,
    relevant_ratio: float = 0.2,
    internal: bool = True,
    ignored: bool = True,
    style: str = "positive",
) -> Tuple[AdjacencyMatrix, AdjacencyMatrix, np.ndarray, np.ndarray, list]:
    """
    The function samples a pair of ANMs that are
    abstracted by a randomly sampled abstraction
    linear function. It returns the adjacencies
    of the abstract and concrete models, the
    endogenous abstraction function the
    exogenous abstraction function.
    """

    # Sample the abstract model
    abs_weights = sample_linear_anm(abs_nodes, abs_edges, abs_type)

    # Sample the abstraction
    tau_adj = sample_linear_abstraction(
        abs_nodes, min_block_size, max_block_size, relevant_ratio
    )
    cnc_nodes = tau_adj.shape[0]

    # Measure the blocks
    block_size = []
    for y in range(abs_nodes):
        start = sum(block_size)
        last = None
        for x in range(cnc_nodes):
            if tau_adj[x, y] != 0.0:
                last = x
        assert last is not None
        block_size.append(last + 1 - start)
    n_ignored = cnc_nodes - sum(block_size)
    block_size.append(n_ignored)

    # Sample the concretization
    cnc_weights, gamma_adj = sample_linear_concretization(
        abs_weights,
        tau_adj,
        block_size,
        internal=internal,
        alpha=alpha,
        style=style,
    )

    # eventually remove ignored variables
    if not ignored:
        tau_adj = tau_adj[:-n_ignored, :]
        gamma_adj = gamma_adj[:-n_ignored, :]
        block_size = block_size[:-1]
        cnc_weights = cnc_weights[:-n_ignored, :-n_ignored]

    return cnc_weights, abs_weights, tau_adj, gamma_adj, block_size


def sample_linear_realizations(
    cnc_weights: AdjacencyMatrix,
    abs_weights: AdjacencyMatrix,
    tau: np.ndarray,
    gamma: np.ndarray,
    n_samples: int = 1000,
    noise_term: str = "gaussian",
    noise_abs: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given two abstracted linear ANMs and their abstraction
    function, it returns a dataset of samples from the
    concrete and abstract models in the observational distribution
    and the random interventional distributions.
    """

    # get the number of nodes
    cnc_nodes = cnc_weights.shape[0]

    # Concrete exogenous distribution
    def sample_exogenous() -> np.array:
        if noise_term == "gaussian":
            samples_e = np.random.normal(size=(n_samples, cnc_nodes))
        elif noise_term == "exponential":
            samples_e = np.random.exponential(size=(n_samples, cnc_nodes))
        elif noise_term == "gumbel":
            samples_e = np.random.gumbel(size=(n_samples, cnc_nodes))
        elif noise_term == "uniform":
            samples_e = np.random.uniform(size=(n_samples, cnc_nodes))
        elif noise_term == "logistic":
            samples_e = np.random.logistic(size=(n_samples, cnc_nodes))
        else:
            raise ValueError(f"Unknown noise_term type {noise_term}")

        return samples_e

    # Sample exogenous
    samples_e = sample_exogenous()

    # Compute endogenous
    samples_x = linear_anm(cnc_weights, samples_e)

    # Abstract
    samples_y = samples_x @ tau

    # Eventually add noise
    if noise_abs > 0.0:
        samples_y += np.random.normal(scale=noise_abs, size=samples_y.shape)

    # Check consistency
    samples_y_bis = linear_anm(abs_weights, samples_e @ gamma)
    if not np.allclose(samples_y, samples_y_bis):
        norm2error = np.linalg.norm(samples_y - samples_y_bis, ord=2)
        print(f"WARNING: Consistency error {norm2error}.")
        if noise_abs > 0.0:
            print(f"...as expected, due to noise {noise_abs}.")

    return samples_x, samples_y
