"""
--data: Path to main data directory. Structure:

data/
    train/
        film1/
            0.jpg
            0.json
            ...
        film2/
            ...
        ...
    val/
        ...
"""

import argparse
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from constants import *
from dataset import DiscamDataset
from model import DiscamModel


def forward_dataset(data, model, criterion):
    for x, y in data:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        model.zero_grad()
        pred = model(x)
        loss = criterion(y, pred)
        yield loss


def train(args):
    model = DiscamModel().to(DEVICE)

    loader_args = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 16,
    }
    train_data = DiscamDataset(args.data / "train")
    train_loader = DataLoader(train_data, **loader_args)
    val_data = DiscamDataset(args.data / "val")
    val_loader = DataLoader(val_data, **loader_args)

    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    writer = SummaryWriter(args.output / "logs")
    global_step = 0

    for epoch in range(args.epochs):
        for loss in forward_dataset(tqdm(train_loader, desc=f"Train epoch {epoch}"), model, criterion):
            loss.backward()
            optim.step()

            writer.add_scalar("train_loss", loss.item(), global_step)

            global_step += 1

        with torch.no_grad():
            total_loss = 0
            for loss in forward_dataset(tqdm(val_loader, desc=f"Val epoch {epoch}"), model, criterion):
                total_loss += loss.item()
            total_loss /= len(val_loader)

            writer.add_scalar("val_loss", total_loss, global_step)


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
