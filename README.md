# Multithreaded hyperparameter tuning

In this project, we show how you can quickly run a parallel hyperparameter tuning using the [Neu.ro](https://neu.ro/platform/) platform. As a benchmark, we solve a simple task of classifying images 
from [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

### Used technologies

* [Catalyst](https://github.com/catalyst-team/catalyst) as a pipeline runner for deep learning tasks. This new and rapidly developing library can significantly reduce the amount of boilerplate code. If you are familiar with the TensorFlow ecosystem, you can think of Catalyst as Keras for PyTorch.
* [Weights & biases](https://www.wandb.com/) as hyperparameter tuning backend and logging system.

### Steps to run
* `make setup` - Before we start doing something, we have to run the command, which prepares a Docker container with all the necessary dependencies.
* Installing `W&B` and configuring its credentials.
* `make hypertrain`- Run `N_JOBS` jobs on our platform
(number of jobs can be specified in `Makefile` or as environment variable). Additional parameters of tuning you can set in `src/wandb-sweep.yaml` file; see [W&B documentation about
sweeps](https://docs.wandb.com/library/sweeps) for more details.
* `make train` - Run a single training process with default hyperparameters.

Outcomes:
* Charts and a table with comparisons of runs with different hyperparameters in [W&B Web UI](https://app.wandb.ai/home) (see `Sweep` section on the left bar). There you can also find a button for early stop the search (or you can use `make kill-hypertrain` for this purpose).
* Training logs and checkpoints you can found in the `results` directory (see `RESULTS_DIR` in `Makefile`).
