from argparse import ArgumentParser

import numpy as np
import wandb

wandb.init(project="neuro_hyper_dummy")

"""
    Dummy example of hyperparameters search config (sweep.yaml):

    program: src/dummy_train.py
    method: grid
    metric:
      name: accuracy
      goal: maximize
    parameters:
      th:
        values: [0.1, 0.5, 0.9]
"""


def dummy_optimization_task(th: float) -> None:
    # the best threshold is 0.5

    probs = np.array([0.2, 0.25, 0.1, 0.3,
                      0.7, 0.6, 0.55, 0.9]
                     )
    labels = np.array([0, 0, 0, 0,
                       1, 1, 1, 1]
                      )

    accuracy = sum(labels == (probs > th)) / len(labels)

    wandb.log({'accuracy': accuracy})

    print(f'Th: {th}, accuracy: {accuracy}.')


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--th', type=float, default=0.5)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    dummy_optimization_task(th=args.th)
