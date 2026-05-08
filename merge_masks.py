"""
Merge multiple binary mask videos into one by taking the union of all white regions.

Usage:
    python merge_masks.py mask1.mp4 mask2.mp4 [mask3.mp4 ...] -o merged.mp4
    python merge_masks.py mask1.mp4 mask2.mp4 -o merged.mp4 --threshold 128
"""

import argparse
import sys

import cv2
import numpy as np


def read_mask_video(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    return frames, fps


def merge_masks(mask_paths: list[str], output_path: str, threshold: int = 128) -> None:
    all_frames: list[list[np.ndarray]] = []
    all_fps: list[float] = []

    print("Loading masks...")
    for path in mask_paths:
        frames, fps = read_mask_video(path)
        all_frames.append(frames)
        all_fps.append(fps)
        print(f"  {path}: {len(frames)} frames, {fps:.1f} fps, size {frames[0].shape[1]}x{frames[0].shape[0]}")

    # Check frame count consistency
    frame_counts = [len(f) for f in all_frames]
    if len(set(frame_counts)) != 1:
        mismatch = ", ".join(f"{p}: {n}" for p, n in zip(mask_paths, frame_counts))
        print(f"\n[ERROR] Frame count mismatch: {mismatch}", file=sys.stderr)
        sys.exit(1)

    # Check spatial size consistency
    sizes = [(f[0].shape[1], f[0].shape[0]) for f in all_frames]
    if len(set(sizes)) != 1:
        mismatch = ", ".join(f"{p}: {s}" for p, s in zip(mask_paths, sizes))
        print(f"\n[ERROR] Spatial size mismatch: {mismatch}", file=sys.stderr)
        sys.exit(1)

    n_frames = frame_counts[0]
    H, W = all_frames[0][0].shape
    fps = all_fps[0]
    print(f"\nAll masks consistent: {n_frames} frames, {W}x{H}, {fps:.1f} fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    print(f"Merging -> {output_path}")
    for i in range(n_frames):
        merged = np.zeros((H, W), dtype=np.uint8)
        for frames in all_frames:
            binary = (frames[i] >= threshold).astype(np.uint8) * 255
            merged = cv2.bitwise_or(merged, binary)
        bgr = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
        writer.write(bgr)

    writer.release()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Merge binary mask videos (union of white regions).")
    parser.add_argument("masks", nargs="+", help="Input mask video paths (.mp4)")
    parser.add_argument("-o", "--output", required=True, help="Output merged mask video path")
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Pixel value threshold for treating a pixel as foreground (default: 128)",
    )
    args = parser.parse_args()

    if len(args.masks) < 2:
        parser.error("At least 2 mask videos are required.")

    merge_masks(args.masks, args.output, threshold=args.threshold)


if __name__ == "__main__":
    main()
