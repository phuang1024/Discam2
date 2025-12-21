"""
Torch dataset class.

Run this file to preview data.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class DiscamDataset(Dataset):
    def __init__(self, dir: Path):
        """
        dir: Main data directory.
            Should have subdirs, which each have the image and JSON files.
        """
        self.subdirs = list(dir.iterdir())
        print("Found data dirs:", self.subdirs)

        self.total_len = 0
        self.start_inds = []

        for dir in self.subdirs:
            self.start_inds.append(self.total_len)
            self.total_len += len(list(dir.iterdir())) // 2

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        for dir_index in range(len(self.subdirs)):
            if index >= self.start_inds[dir_index]:
                break

        file_index = index - self.start_inds[dir_index]
        img_file = self.subdirs[dir_index] / f"{file_index}.jpg"
        bbox_file = self.subdirs[dir_index] / f"{file_index}.json"

        img = read_image(str(img_file)).float() / 255.0
        with open(bbox_file, "r") as f:
            bbox = json.load(f)
        bbox = bbox["bbox"]

        return img, bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    args = parser.parse_args()

    dataset = DiscamDataset(args.dir)

    while True:
        index = random.randint(0, len(dataset) - 1)
        img, bbox = dataset[index]
        img = (img * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        bbox = list(map(int, bbox))

        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow("a", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
