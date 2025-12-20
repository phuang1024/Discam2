"""
Process and load data from the teamtrack dataset.

Run this script to process data.
Args:
    data: Path to data dir. Should have subdirs `videos` and `annotations`.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_anno_file(anno_file) -> list[list[tuple]]:
    """
    Return list of bounding boxes for each frame, with ball being the last one.
    """
    with open(anno_file, "r") as fp:
        lines = fp.read().strip().split("\n")
    # Remove key names.
    lines = lines[4:]

    # List of lists of tuples (x, y, w, h)
    bboxes = []
    for line in lines:
        values = list(map(float, line.strip().split(",")))
        values = values[1:]
        assert len(values) % 4 == 0

        bboxes.append([])
        for i in range(0, len(values), 4):
            bbox = values[i : i+4]
            bboxes[-1].append((bbox[1], bbox[2], bbox[3], bbox[0]))

    return bboxes


def compute_gt(bboxes, z_thres=1.5):
    """
    Compute ground truth bbox (i.e. where the PTZ camera should be looking at)
    given all detection bboxes for a frame,
    using a statistical approach.

    The last element in bboxes should be the bbox of the ball.

    Let std be the standard deviation of the locations of detection bboxes.
    The ground truth is a bbox around all bboxes within a constant times std distance
    to either the mean of all detections, or the ball.

    z_thres: `(value - mean) / std <= z_thres` to be considered within GT bbox.
    return: (x, y, w, h) of GT bbox.
    """
    # List of centers of each bbox.
    centers = []
    for x, y, w, h in bboxes:
        centers.append((x + w/2, y + h/2))
    centers = np.array(centers)

    # Find mean and std of each axis.
    mean_x = np.mean(centers[:, 0])
    mean_y = np.mean(centers[:, 1])
    std_x = np.std(centers[:, 0])
    std_y = np.std(centers[:, 1])

    x_min = np.max(centers[:, 0])
    x_max = 0
    y_min = np.max(centers[:, 1])
    y_max = 0

    for x, y in centers:
        z_x = (x - mean_x) / std_x
        z_y = (y - mean_y) / std_y
        if abs(z_x) <= z_thres and abs(z_y) <= z_thres:
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def vis_frame(frame, detect_boxes, gt_box):
    """
    Draw bboxes from annotations file.

    frame: Frame to use. Will make a copy and draw.
    bboxes: Bboxes for this frame; i.e. bboxes[i].
    """
    def draw_rect(frame, box, color):
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    frame = frame.copy()

    for i, box in enumerate(detect_boxes):
        color = (0, 0, 255) if i == len(detect_boxes) - 1 else (0, 255, 0)
        draw_rect(frame, box, color)

    draw_rect(frame, gt_box, (255, 0, 0))

    # Resize to around 1000px
    """
    width = 1000
    height = int(width * frame.shape[0] / frame.shape[1])
    frame = cv2.resize(frame, (width, height))
    """

    return frame


def process_clip(video_file, anno_file, output_dir):
    """
    """
    video = cv2.VideoCapture(video_file)
    bboxes = read_anno_file(anno_file)

    frame_i = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_i >= len(bboxes):
            break

        gt_box = compute_gt(bboxes[frame_i])

        if True:
            cv2.imshow("a", vis_frame(frame, bboxes[frame_i], gt_box))
            cv2.waitKey(100)

        frame_i += 1

    if frame_i != len(bboxes):
        print("Warning: Length mismatch between video and annotations.\n"
              f"video_file={video_file}, anno_file={anno_file}\n"
              f"video_len={frame_i}, anno_len={len(bboxes)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    videos_dir = args.data / "videos"
    annos_dir = args.data / "annotations"
    for file in videos_dir.iterdir():
        if file.suffix != ".mp4":
            continue
        anno_file = annos_dir / f"{file.stem}.csv"
        if not anno_file.exists():
            continue

        process_clip(file, anno_file, args.output)

        break


if __name__ == "__main__":
    main()
