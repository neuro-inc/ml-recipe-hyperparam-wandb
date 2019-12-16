import os
import sys

import wandb
import yaml

config = yaml.load(open('sweep.yaml', 'r'), yaml.FullLoader)

with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    sys.stdout = devnull
    sweep_id = wandb.sweep(config, project='hyper')

sys.stdout = old_stdout
print(sweep_id)
