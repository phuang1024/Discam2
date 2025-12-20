import argparse
from pathlib import Path

import torch

from constants import *
from model import DiscamModel


def train(args):
    model = DiscamModel().to(DEVICE)

    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(args.epochs):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
