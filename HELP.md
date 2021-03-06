# Neuro Project Template Reference

## Development Environment

This template runs on [Neuro Platform](https://neu.ro). 

To dive into the problem solving, you need to sign up at [Neuro Platform](https://neu.ro) website, set up your local machine according to [instructions](https://neu.ro/docs) and log into the Neuro CLI:

```shell
neuro login
```

## Directory structure

| Local directory | Description | Storage URI | Environment mounting point |
|:--------------- |:----------- |:----------- |:-------------------------- | 
| `data/` | Data | `storage:ml-recipe-hyper-search/data/` | `/ml-recipe-hyper-search/data/` | 
| `src/` | Python modules | `storage:ml-recipe-hyper-search/src/` | `/ml-recipe-hyper-search/src/` |
| `config/` | Configuration files | `storage:ml-recipe-hyper-search/config/` | `/ml-recipe-hyper-search/config/` |
| `notebooks/` | Jupyter notebooks | `storage:ml-recipe-hyper-search/notebooks/` | `/ml-recipe-hyper-search/notebooks/` |
| `results/` | Logs and results | `storage:ml-recipe-hyper-search/results/` | `/ml-recipe-hyper-search/results/` |

## Development

Follow the instructions below to set up the environment on Neuro and start a Jupyter development session.

### Setup development environment 

```shell
make setup
```

* Several files from the local project are uploaded to the platform storage (namely, `requirements.txt`,  `apt.txt`, `setup.cfg`).
* A new job is started in our [base environment](https://hub.docker.com/r/neuromation/base). 
* Pip requirements from `requirements.txt` and apt applications from `apt.txt` are installed in this environment.
* The updated environment is saved under a new project-dependent name and is used further on.

### Run Jupyter with GPU 

```shell
make jupyter
```

* The content of the `src` and `notebooks` directories is uploaded to the platform storage.
* A job with Jupyter is started, and its web interface is opened in the local web browser window.

### Kill Jupyter

```shell 
make kill-jupyter
```

* The job with Jupyter Notebooks is terminated. The notebooks are saved on the platform storage. You may run `make download-notebooks` to download them to the local `notebooks/` directory.

### Help

```shell 
make help
```

* The list of all available template commands is printed.

## Data

### Uploading to the Storage via Web UI

On local machine, run `make filebrowser` and open the job's URL on your mobile device or desktop.
Through a simple file explorer interface, you can upload test images and perform file operations.

### Uploading to the Storage via CLI

On local machine, run `make upload-data`. This command pushes local files stored in `./data`
into `storage:ml-recipe-hyper-search/data` mounted to your development environment's `/project/data`.

### Uploading data to the Job from Google Cloud Storage

Google Cloud SDK is pre-installed on all jobs produced from the Base Image.

Neuro Project Template provides a fast way to authenticate Google Cloud SDK to work with Google Service Account (see instructions on setting up your Google Project and Google Service Account and creating the secret key for this Service Account in [documentation](https://neu.ro/docs/google_cloud_storage)).

Download service account key to the local config directory `./config/` and set appropriate permissions on it:

```shell
$ SA_NAME="neuro-job"
$ gcloud iam service-accounts keys create ./config/$SA_NAME-key.json \
  --iam-account $SA_NAME@$PROJECT_ID.iam.gserviceaccount.com
$ chmod 600 ./config/$SA_NAME-key.json
```

Inform Neuro about this file:

```shell
$ export GCP_SECRET_FILE=$SA_NAME-key.json
```

Alternatively, set this value directly in `Makefile`.

Check that Neuro can access and use this file for authentication:

```shell
$ make gcloud-check-auth
Using variable: GCP_SECRET_FILE='neuro-job-key.json'
Google Cloud will be authenticated via service account key file: '/path/to/project/config/neuro-job-key.json'
```

Now, if you run a `develop`, `train`, or `jupyter` job, Neuro authenticates Google Cloud SDK via your secret file so that you can use `gsutil` or `gcloud` there:

```shell
$ make develop
...
$ make connect-develop
...
root@job-56e9b297-5034-4492-ba1a-2284b8dcd613:/# gsutil cat gs://my-neuro-bucket-42/hello.txt
Hello World
```

Also, the environment variable `GOOGLE_APPLICATION_CREDENTIALS` is set up for these jobs, so that you can access your data on Google Cloud Storage via Python API (see example in [Google Cloud Storage documentation](https://cloud.google.com/storage/docs/reference/libraries)).

### Uploading data to the Job from AWS S3

AWS CLI is pre-installed on all jobs produced from the Base Image.

Neuro Project Template provides a fast way to authenticate AWS CLI to work with AWS user account (see instructions on setting up your AWS user account credentials and creating the secret key in [documentation](https://neu.ro/docs/aws_s3)).

In the project directory, write your AWS credentials to a file `./config/aws-credentials.txt`, set appropriate permissions on it,
inform Neuro about this file by setting a specific env var, and check that Neuro can access and use this file for authentication:

```shell
$ export AWS_SECRET_FILE=aws-credentials.txt
$ chmod 600 ./config/$AWS_SECRET_FILE
$ make aws-check-auth
AWS will be authenticated via user account credentials file: '/path/to/project/config/aws-credentials.txt'
```

Now, if you run a `develop`, `train`, or `jupyter` job, Neuro authenticates AWS CLI via your secret file so that you can use `aws` there:

```shell
$ make develop
...
$ make connect-develop
...
root@job-098b8584-1003-4cb9-adfb-3606604a3855:/# aws s3 cp s3://my-neuro-bucket-42/hello.txt -
Hello World
```

## Customization

Several variables in `Makefile` are intended to be modified according to the project specifics. 
To change them, find the corresponding line in `Makefile` and update.

### Data location

`DATA_DIR_STORAGE?=$(PROJECT_PATH_STORAGE)/$(DATA_DIR)`

This project template implies that your data is stored alongside the project. If this is the case, you don't have to change this variable. However, if your data is shared between several projects on the platform, 
you need to change the following line to point to its location. For example:

`DATA_DIR_STORAGE?=storage:datasets/cifar10`

### Run development job

If you want to debug your code on GPU, you can run a sleeping job via `make develop`, then connect to its bash over SSH
via `make connect-develop` (type `exit` or `^D` to close SSH connection), see its logs via `make logs-develop`, or 
forward port 22 from the job to localhost via `make port-forward-develop` to use it for remote debugging.
Please find instructions on remote debugging via PyCharm Pro in the [documentation](https://neu.ro/docs/remote_debugging_pycharm). 

Please don't forget to kill your job via `make kill-develop` not to waste your quota!   

### Weights & Biases integration

Neuro Platform offers easy integration with [Weights & Biases](https://www.wandb.com), an experiment tracking tool for deep learning.
The instructions look similar to ones for Google Cloud integration above. 
First, you need to [register your W&B account](https://app.wandb.ai/login?signup=true). 
Then, find your API key on [W&B's settings page](https://app.wandb.ai/settings) (section "API keys"),
save it to a file in local directory `./config/`, protect by setting appropriate permissions 
and check that Neuro can access and use this file for authentication:

```shell
$ export WANDB_SECRET_FILE=wandb-token.txt
$ echo "cf23df2207d99a74fbe169e3eba035e633b65d94" > config/$WANDB_SECRET_FILE
$ chmod 600 config/$WANDB_SECRET_FILE
$ make wandb-check-auth 
Using variable: WANDB_SECRET_FILE=wandb-token.txt
Weights & Biases will be authenticated via key file: '/path/to/project/config/wandb-token.txt'
```

Now, if you run `develop`, `train`, or `jupyter` job, Neuro authenticates W&B via your API key, so that you can use `wandb` there:

```shell
$ make develop
...
$ make connect-develop
...
root@job-fe752aaf-5f76-4ba8-a477-0809632c4a59:/# wandb status
Logged in? True
...
```

So now, you can do `import wandb; api = wandb.Api()` in your Python code and use W&B.

Technically, authentication is being done as follows: 
when you start any job derived from the base environment, Neuro Platform checks if the env var `NM_WANDB_TOKEN_PATH`
is set and stores path to existing file, and then it runs the command `wandb login $(cat $NM_WANDB_TOKEN_PATH)`
before the job starts.
 
Please find instructions on using Weights & Biases in your code in [W&B documentation](https://docs.wandb.com/library/api/examples).
You can also find [W&B example projects](https://github.com/wandb/examples) or an example of Neuro Project Template-based 
[ML Recipe that uses W&B as a part of the workflow](https://neu.ro/docs/cookbook/ml-recipe-hier-attention). 

### Training machine type

`PRESET?=gpu-small`

There are several machine types supported on the platform. Run `neuro config show` to see the list.

### HTTP authentication

`HTTP_AUTH?=--http-auth`

When jobs with HTTP interface are executed (for example, with Jupyter Notebooks or TensorBoard), this interface requires a user to be authenticated on the platform. However, if you want to share the link with someone who is not registered on the platform, you may disable the authentication updating this line to `HTTP_AUTH?=--no-http-auth`.

### Training command

To tweak the training command, change the line in `Makefile`:
 
```shell
TRAIN_CMD=python -u $(CODE_DIR)/train.py --data $(DATA_DIR)
```

And then, just run `make train`. Alternatively, you can specify training command for one separate training job:

```shell
make train TRAIN_CMD='python -u $(CODE_DIR)/train.py --data $(DATA_DIR)'
```

Note that in this case, we use single quotes so that local `bash` does not resolve environment variables. You can assume that training command `TRAIN_CMD` runs in the project's root directory.

### Multiple training jobs

You can run multiple training experiments simultaneously by setting up `RUN` environment variable:

```shell
make train RUN=new-idea
```

Note, this label becomes a postfix of the job name, which may contain only alphanumeric characters and hyphen `-`, and cannot end with hyphen or be longer than 40 characters.

Please, don't forget to kill the jobs you started:
- `make kill-train` to kill the training job started via `make train`,
- `make kill-train RUN=new-idea` to kill the training job started via `make train RUN=new-idea`,
- `make kill-train-all` to kill all training jobs started in the current project,
- `make kill-jupyter` to kill the job started via `make jupyter`,
- ...
- `make kill-all` to kill all jobs started in the current project.

### Multi-threaded hyperparameter tuning

Neuro Platform supports hyperparameter tuning via [Weights & Biases](https://www.wandb.com/articles/running-hyperparameter-sweeps-to-pick-the-best-model-using-w-b).

To run hyperparameter tuning for the model, you need to define the list of hyperparameters and send the metrics to WandB after each run. Your code may look as follows:

```python
import wandb

def train() -> None:
    hyperparameter_defaults = dict(
        lr=0.1,
        optimizer='sgd',
        scheduler='const'
    )
    wandb.init(config=hyperparameter_defaults)
    # your model training code here
    metrics = {'accuracy': accuracy, 'loss': loss}
    wandb.log(metrics)

if __name__ == "__main__":
    train()
```   

This list of hyper-parameters corresponds to the default configuration we provide in `src/wandb-sweep.yaml` file. See [W&B documentation page](https://docs.wandb.com/library/sweeps) for more details. The name of the sweep file can be modified in `Makefile` or as environment variable `WANDB_SWEEP_FILE`.

You also need to put your WandB token in `config/wandb-token.txt` file.

After that, you can run `make hypertrain`, which submits `N_JOBS` (`3` by default) jobs on Neuro Platform (number of jobs can be modified in `Makefile` or as corresponding environment variable). Use `make ps-hypertrain` to list active jobs of the latest sweep. To monitor the hyperparameter tuning process, follow the link which `wandb` provides at the beginning of the process.

To terminate all jobs over all hyperparameter tuning sweeps, run `make kill-hypertrain-all`. After that, verify that the jobs were killed `make ps`, and then delete unused sweeps from the local file `.wandb_sweeps`.
