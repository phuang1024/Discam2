"""
Torch dataset class.

Run this file to preview data.
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image

from constants import *
from utils import apply_edge_weights, check_bbox_bounds


class VideoDataset:
    """
    Reader for a single video.
    Reads image and corresponding bbox GT by index.
    """

    def __init__(self, dir: Path):
        """
        dir: Data directory. Contains image and JSON files.
        """
        self.dir = dir
        print(dir)

        self.length = len(list(self.dir.iterdir())) // 2

    def read(self, index):
        """
        return: (image, label)
            image: tensor float [0-1] (C, H, W)
            bbox: tensor float (4,)
        """
        img_file = self.dir / f"{index}.jpg"
        bbox_file = self.dir / f"{index}.json"

        img = read_image(str(img_file)).float() / 255.0
        with open(bbox_file, "r") as f:
            bbox = json.load(f)["bbox"]
        bbox = torch.tensor(bbox, dtype=torch.float32)

        return img, bbox


class DiscamDataset(Dataset):
    """
    Loads chunks (frames of a single video separated by constant step).

    Returns X, Y pairs for the neural network.
    X is the stack of images.
    Y is the expected edge weights.

    For each chunk loaded, computes a random offset.
    The random offset is applied to the *ground truth* bbox of each frame,
    and crops the frame according to this new bbox.
    Therefore, the returned chunk will have frame crops offset from the ground truth.
    Using the random offset, the ground truth edge weights are computed:
    The edge weights that *should* be output, in order to reverse this random offset.
    """

    def __init__(self, dir: Path):
        """
        dir: Main data directory. Should have subdirs of individual videos.
        """
        self.dir = dir

        self.videos = []
        self.total_len = 0
        self.start_inds = []

        for dir in self.dir.iterdir():
            self.videos.append(VideoDataset(dir))
            self.start_inds.append(self.total_len)
            self.total_len += self.videos[-1].length

        self.resize = T.Resize(MODEL_INPUT_RES[::-1])

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        for dir_index in range(len(self.videos)):
            if index >= self.start_inds[dir_index]:
                break

        file_index = index - self.start_inds[dir_index]

        # In [-1, 1]
        rand_edge_weights = torch.rand([4]) * 2 - 1

        img = self.read_crop_image(dir_index, file_index, rand_edge_weights)
        img = self.resize(img)

        img = img.unsqueeze(0)

        # In addition to reversing the random edge weights, we want the NN to expand
        # or contract the bbox based on the magnitude of displacement.
        # If displacement is large, we want to zoom out, so more is visible, in order to
        # reposition well. The opposite if displacement is small.
        # disp: [0, 1]
        disp = math.hypot(
            rand_edge_weights[0].item() + rand_edge_weights[2].item(),
            rand_edge_weights[1].item() + rand_edge_weights[3].item(),
        ) / math.hypot(2, 2)
        gt_edge_weights = -1 * rand_edge_weights
        gt_edge_weights += disp * 0.3

        gt_edge_weights = gt_edge_weights.clamp(-1, 1)

        #print("rand", rand_edge_weights)
        #print("disp", disp)

        return img, gt_edge_weights

    def read_crop_image(self, dir_index, file_index, edge_weights):
        """
        Read image given by (dir_index, file_index).
        Apply edge weights to bbox, and crop image with new bbox.
        """
        img, bbox = self.videos[dir_index].read(file_index)

        # Max displacement is roughly equal to image dimension.
        velocity = (img.shape[1] + img.shape[2]) / 2
        bbox = apply_edge_weights(bbox, edge_weights, img.shape[2] / img.shape[1], velocity)
        bbox = check_bbox_bounds(bbox, (img.shape[2], img.shape[1]))

        bbox = list(map(int, bbox))
        crop = img[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]

        return crop


def vis_frame(dir):
    """
    Visualize single image frame and GT bbox.
    """
    dataset = VideoDataset(dir)

    while True:
        index = random.randint(0, dataset.length - 1)
        img, bbox = dataset.read(index)

        img = (img * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        bbox = list(map(int, bbox))

        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow("a", img)
        cv2.waitKey(0)


def vis_dataset(dir):
    """
    Visualize dataset crop and edge weights.
    """
    dataset = DiscamDataset(dir)

    while True:
        index = random.randint(0, len(dataset) - 1)
        # img: (N, C, H, W)
        img, edge_weights = dataset[index]
        img = img[-1, ...]

        img = (img * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        print("Edge weights:", edge_weights)
        cv2.imshow("a", img)
        cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    args = parser.parse_args()

    vis_dataset(args.dir)


if __name__ == "__main__":
    main()
