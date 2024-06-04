# Causal Abstraction

This repository contains the code for the paper "[Learning Causal Abstractions of Linear Structural Causal Models](https://arxiv.org/abs/2406.00394)"
by [Riccardo Massidda](https://pages.di.unipi.it/massidda), [Sara Magliacane](https://saramagliacane.github.io/), and [Davide Bacciu](http://pages.di.unipi.it/bacciu/).
The repository contains both a generic `causabs` Python package and the code to reproduce the experiments in the paper, in the `abslingam` directory.

## Causabs

The `causabs` package can be used standalone to generate pairs of concrete/abstract linear causal models, sample realizations from them, and fit an abstraction function from paired realizations.

We suggest to create new virtual environment with Python 3.10, and then install as a package:

```bash
pip install .
```

## Abs-LiNGAM

Each experiment can be run from the `abslingam` directory, as in

```bash
cd abslingam
pip install -r abslingam/requirements.txt
python experiment1.py \
    --seed="42" --num_cpus="2" \
    --num_runs="5" --flavor="small" \
    --data_dir="data/" --results_dir="results/"
```

The same folder also contains Jupyter Notebooks to plot and visualze the results (`Experimental Analysis.ipynb`), to visualize the models in each generate dataset (`Visualization.ipynb`), and to monitor a single execution of Abs-LiNGAM (`Run Abs-LiNGAM.ipynb`). 

