# Multithreaded hyperparameters search

In this project, we will show how you can easily run a parallel hyperparameters 
search using the [neu.ro](https://neu.ro/platform/) platform. 
As a benchmark, we chose the simple task of classifying images 
from [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dsataset.

### Used technologies
* [Catalyst](https://github.com/catalyst-team/catalyst) as a pipeline runner for deep learning tasks.
This new and rapidly developing library can significantly reduce the amount of boilerplate code.
If you are familiar with the TensorFlow ecosystem, you can think of Catalyst as Keras for PyTorch.
* [Weights & biases](https://www.wandb.com/) as hyperparameters search backend and logging system.

### Steps to run
* `make setup` - Before we start doing something, we have to run the command, 
which prepares a Docker container with all the necessary dependencies.
* Installing `W&B` and configurating its credentials (see `Weights & Biases integration` below).
* `make hypertrain` - will run `N_JOBS` jobs on our platform
(number of jobs can be specified in `Makefile` or as environment variable).
Additional parameters of search can be set in `src/wandb-sweep.yaml` file, see [W&B documentation about
sweeps](https://docs.wandb.com/library/sweeps) for more details.
* `make train` - you can also run single training process with default hyperparameters.

Outcomes:
* Charts and table with comparisons of runs with different hyperparameters are available 
through [W&B Web UI](https://app.wandb.ai/home) (see `Sweep` section on the left bar).
Here you can also find a button for early stop the search
(or you can use `make kill-hypertrain` for this purpose).
* Training logs and checkpoints can be found in the results directory (see `RESULTS_DIR` in `Makefile`).
