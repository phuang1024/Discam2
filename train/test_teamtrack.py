"""
Test NN on teamtrack data.

Note: This script uses the original teamtrack data, not the processed.
Because it is not possible to keep a persistent bbox across frames of the
processed data.
"""

import argparse
from pathlib import Path

import cv2
import torch

from constants import *
from model import DiscamModel
from utils import apply_edge_weights, check_bbox_bounds


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Dir containing video files.")
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    model = DiscamModel().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()

    video_files = sorted(list(args.data.iterdir()))
    bbox = [0, 0, 1920, 1080]

    for file in video_files:
        video = cv2.VideoCapture(str(file))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ret, frame = video.read()
            if not ret:
                break

            input_frame = cv2.resize(frame, MODEL_INPUT_RES)
            input_tensor = torch.from_numpy(input_frame).to(DEVICE)
            input_tensor = input_tensor.permute(2, 0, 1)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.float() / 255

            edge_weights = model(input_tensor).squeeze(0).cpu().numpy()
            bbox = apply_edge_weights(bbox, edge_weights, 16 / 9, 20)
            bbox = check_bbox_bounds(bbox, (width, height))

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("a", frame)
            if cv2.waitKey(30) == ord("q"):
                break

        video.release()


if __name__ == "__main__":
    main()
