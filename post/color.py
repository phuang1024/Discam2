"""
Color balance script.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def process_frame(frame):
    # Desaturate.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[..., 1] = frame[..., 1] * 0.7
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to input video.")
    parser.add_argument("output", type=Path, help="Path to output video.")
    args = parser.parse_args()

    in_video = cv2.VideoCapture(str(args.input))
    width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(in_video.get(cv2.CAP_PROP_FPS))
    frame_count = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    pbar = tqdm(total=frame_count)

    while True:
        ret, frame = in_video.read()
        if not ret:
            break
        pbar.update(1)

        out_frame = process_frame(frame)
        out_video.write(out_frame)

        # Visualize result.
        if True:
            vis = np.zeros((height // 2, width, 3), dtype=np.uint8)
            vis[:, :width // 2] = cv2.resize(frame, (width // 2, height // 2))
            vis[:, width // 2:] = cv2.resize(out_frame, (width // 2, height // 2))

            cv2.imshow("vis", vis)
            cv2.waitKey(10)

    pbar.close()
    in_video.release()
    out_video.release()


if __name__ == "__main__":
    main()
