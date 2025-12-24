import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from constants import *
from dataset import DiscamDataset
from model import DiscamModel


def train(args):
    model = DiscamModel().to(DEVICE)

    loader_args = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 8,
    }
    train_data = DiscamDataset(args.data)
    train_loader = DataLoader(train_data, **loader_args)

    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(args.epochs):
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optim.zero_grad()
            pred = model(x)
            loss = criterion(y, pred)
            loss.backward()
            optim.step()

            print(loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
