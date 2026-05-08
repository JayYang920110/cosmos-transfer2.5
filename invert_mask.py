#!/usr/bin/env python3
"""Invert a binary mask video: white <-> black."""

import argparse
import cv2
import numpy as np
from pathlib import Path


def invert_mask_video(input_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(255 - frame)

    cap.release()
    out.release()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input mask video (.mp4)")
    parser.add_argument("output", help="Output inverted mask video (.mp4)")
    args = parser.parse_args()
    invert_mask_video(args.input, args.output)
