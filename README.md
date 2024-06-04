# Causal Abstraction

This repository contains the code for the paper "[Learning Causal Abstractions of Linear Structural Causal Models](https://arxiv.org/abs/2406.00394)"
by [Riccardo Massidda](https://pages.di.unipi.it/massidda), [Sara Magliacane](https://saramagliacane.github.io/), and [Davide Bacciu](http://pages.di.unipi.it/bacciu/).

The code is organized in a package, `causabs`, and a set of scripts to run the experiments reported in the paper, in the `abslingam` directory.

## Causabs

The `causabs` package implements the procedure to sample pairs of low-level and abstract linear causal models. First, we suggest to create new virtual environment with Python 3.10, and then install as:

```bash
pip install .
```

The `causabs.sampling` module contains the functions to sample linearly abstracted linear structural causal models (SCMs). The main functions are `sample_linear_abstraction` and `sample_linear_concretization` that, given a known abstract SCM $\mathcal{H}$, sample a low-level SCM $\mathcal{L}$.

Given a fixed number of abstract nodes,
`sample_linear_abstraction` extracts
for each abstract node a random number
of low-level nodes, between `min_block_size` and `max_block_size`.
Then, the `relevant_ratio` parameter controls the proportion of low-level nodes
that are relevant (Definition 2) to the abstract node.
The function returns a matrix $\mathbf{T}$ with non-overlapping blocks
representing the $\mathbf{T}$-abstraction function (Definition 1).
Then, we count the number of low-level nodes in each block to measure the block size.

```python
from causabs.sampling import sample_linear_abstraction
from causabs.utils import measure_blocks

# Sample an abstract model
T = sample_linear_abstraction(abs_nodes, min_block_size, max_block_size, relevant_ratio)
blocks = measure_blocks(T)
```

The `sample_linear_concretization` implements Algorithm 1 from the paper and samples a low-level SCM $\mathcal{L}$ given the adjacency matrix $\mathbf{M}$ of an abstract SCM $\mathcal{H}$ and the $\mathbf{T}$-abstraction function. The function returns the adjacency matrix $\mathbf{W}$ of the concrete model and
the exogenous abstraction function $\gamma$, represented by a matrix $\mathbf{S}$.

```python
from causabs.sampling import sample_linear_concretization

# Sample a concrete model
W, S = sample_linear_concretization(M, T, blocks)
```

The `sample_linear_abstracted_models`
jointly samples an abstract SCM $\mathcal{H}$ and a low-level SCM $\mathcal{L}$.
It returns the adjacency matrices $\mathbf{W}$ and $\mathbf{M}$,
the endogenous abstraction function $\mathbf{T}$,
the exogenous abstraction function $\mathbf{S}$,
and the block sizes.

```python
from causabs.sampling import sample_linear_abstracted_models

W, M, T, S, blocks = sample_linear_abstracted_models(
    abs_nodes, abs_edges,
    min_block_size, max_block_size,
    relevant_ratio=relevant_ratio)
```

With a similar interface, the `linear_dataset` function
generates the SCMs and a dataset from their joint distribution.

```python
from causabs.dataset import sample_linear_abstracted_models

W, M, T, S, blocks, [L_samples, H_samples] = linear_dataset(
    abs_nodes, abs_edges,
    min_block_size, max_block_size,
    relevant_ratio=relevant_ratio,
    n_samples=n_samples)
```


## Abs-LiNGAM

We report our implentation of Abs-LiNGAM
with the experimental setup used in the paper
in the `abslingam` directory.
The code has additional dependencies
that can be installed with:

```bash
cd abslingam
pip install -r requirements.txt
```

The method described in Algorithm 2 of the paper
is implemented in the `abs_lingam` module.
Given a dataset of low-level observations,
as an array of shape `(n_cnc_samples, cnc_nodes)`,
and a dataset of joint concrete-abstract observations,
as an array of shape `(n_joint_samples, cnc_nodes + abs_nodes)`,
it returns:
- the adjacency matrix $\mathbf{W}$ of the low-level SCM,
- the adjacency matrix $\mathbf{A}$ of the abstract SCM,
- the endogenous abstraction function $\mathbf{T}$,
- the constraints $\mathbf{K}$, as a matrix `(cnc_nodes, cnc_nodes)` containing 0 if and only if there cannot be a causal link between the two low-level nodes.

```python
from abs_lingam import abs_lingam

W, A, T, K = abs_lingam(low_level_data, joint_data)
```

The same folder also contains Jupyter Notebooks to plot and visualze the results (`Experimental Analysis.ipynb`), to visualize the models in each generate dataset (`Visualization.ipynb`), and to monitor a single execution of Abs-LiNGAM (`Run Abs-LiNGAM.ipynb`). The experiments are numbered and described in the `Experimental Analysis.ipynb` notebook. Each experiment is implemented in a separate script in the `abslingam` directory that can be run from the command line.


```bash
python experiment1.py \
    --seed="42" --num_cpus="2" \
    --num_runs="5" --flavor="small" \
    --data_dir="data/" --results_dir="results/"
```