import argparse
import json
import os
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--model", required=True, help="Path to trained stump model (Ultralytics YOLO)")
    ap.add_argument("--out", default="calibration_stumps.json", help="Output JSON path")
    ap.add_argument("--min-conf", type=float, default=0.50, help="Min confidence to accept a stump detection")
    ap.add_argument("--no-preview", action="store_true", help="Disable preview window")
    return ap.parse_args()


def bbox_bottom_center(xyxy: np.ndarray) -> Tuple[int, int]:
    x1, y1, x2, y2 = map(float, xyxy)
    cx = int((x1 + x2) / 2.0)
    by = int(y2)
    return cx, by


def bbox_bottom_corners(xyxy: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x1, y1, x2, y2 = map(int, xyxy)
    return (x1, y2), (x2, y2)


def select_two_stumps(xyxy_list: np.ndarray, conf_list: np.ndarray, min_conf: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return the two best stump boxes (xyxy) filtered by min_conf, or None if <2."""
    if len(xyxy_list) == 0:
        return None
    # filter by confidence
    keep = conf_list >= min_conf
    xy = xyxy_list[keep]
    cf = conf_list[keep]
    if len(xy) < 2:
        return None
    # pick top-2 by confidence
    order = np.argsort(-cf)
    idx_top2 = order[:2]
    return xy[idx_top2[0]], xy[idx_top2[1]]


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    model = YOLO(args.model)

    out_json = None
    frame_idx = -1
    preview = not args.no_preview

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[calib] Reached end of video without finding two stumps.")
            break
        frame_idx += 1
        h, w = frame.shape[:2]

        # Run inference on this frame
        results = model(frame)
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            if preview:
                cv2.imshow("Calib (stumps)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        # If your model has multiple classes, you can filter by class here:
        # cls = res.boxes.cls.cpu().numpy().astype(int)
        # mask = (cls == 0)  # e.g., if class 0 is 'stump'
        # xyxy, conf = xyxy[mask], conf[mask]

        picked = select_two_stumps(xyxy, conf, args.min_conf)
        annotated = frame.copy()

        # Draw all detections (optional)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{conf[i]:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if picked is not None:
            b1, b2 = picked

            # Decide batsman (upper) vs bowler (lower) using bbox center-Y
            def center_y(xyxy: np.ndarray) -> float:
                return float((xyxy[1] + xyxy[3]) / 2.0)
            y1 = center_y(b1)
            y2 = center_y(b2)
            batsman_box, bowler_box = (b1, b2) if y1 < y2 else (b2, b1)

            # Bottom corners (left/right) and bottom-centers
            bat_bl, bat_br = bbox_bottom_corners(batsman_box)
            bow_bl, bow_br = bbox_bottom_corners(bowler_box)
            bat_cx, bat_by = bbox_bottom_center(batsman_box)
            bow_cx, bow_by = bbox_bottom_center(bowler_box)

            # Draw four-point ground quad (bowler BL->BR->batsman BR->BL)
            quad = np.array([bow_bl, bow_br, bat_br, bat_bl], dtype=np.int32)
            cv2.polylines(annotated, [quad], isClosed=True, color=(255, 255, 255), thickness=3)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [quad], color=(255, 255, 255))
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

            # Visualize endpoints and labels
            cv2.circle(annotated, (bat_cx, bat_by), 7, (255, 0, 0), -1)
            cv2.putText(annotated, "BATSMAN", (bat_cx + 8, bat_by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.circle(annotated, (bow_cx, bow_by), 7, (0, 0, 255), -1)
            cv2.putText(annotated, "BOWLER", (bow_cx + 8, bow_by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Build JSON with four bottom points (+ middle bases for backward compatibility)
            out_json = {
                "video": os.path.abspath(args.video),
                "frame_index": int(frame_idx),
                "image_size": [int(w), int(h)],
                "image_points": {
                    "bowler_bottom_left":  [int(bow_bl[0]), int(bow_bl[1])],
                    "bowler_bottom_right": [int(bow_br[0]), int(bow_br[1])],
                    "batsman_bottom_left":  [int(bat_bl[0]), int(bat_bl[1])],
                    "batsman_bottom_right": [int(bat_br[0]), int(bat_br[1])],
                    # Back-compat (optional): bottom centers
                    "bowler_middle_stump": [int(bow_cx), int(bow_by)],
                    "batsman_middle_stump": [int(bat_cx), int(bat_by)]
                },
            }

            # Save JSON and a preview image, then exit
            with open(args.out, "w") as f:
                json.dump(out_json, f, indent=2)
            preview_path = os.path.splitext(args.out)[0] + "_preview.jpg"
            cv2.imwrite(preview_path, annotated)
            print(f"[calib] Saved: {args.out}\n[calib] Preview: {preview_path}")
            break

        if preview:
            cv2.imshow("Calib (stumps)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    if out_json is None:
        print("[calib] No two-stump frame detected; nothing saved.")


class StumpDetector:
    """Wrapper class for programmatic stump detection"""

    def __init__(self, model_path: str, min_conf: float = 0.50):
        """
        Initialize stump detector

        Args:
            model_path: Path to YOLO model file
            min_conf: Minimum confidence threshold for detections
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(model_path)
        self.min_conf = min_conf

    def detect(self, video_path: str, output_json_path: str = None) -> dict:
        """
        Detect stumps in video and return calibration data

        Args:
            video_path: Path to input video
            output_json_path: Optional path to save JSON output

        Returns:
            Dictionary with stump calibration data
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        out_json = None
        frame_idx = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            h, w = frame.shape[:2]

            # Run inference on this frame
            results = self.model(frame)
            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue

            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()

            picked = select_two_stumps(xyxy, conf, self.min_conf)

            if picked is not None:
                b1, b2 = picked

                # Decide batsman (upper) vs bowler (lower) using bbox center-Y
                def center_y(xyxy: np.ndarray) -> float:
                    return float((xyxy[1] + xyxy[3]) / 2.0)
                y1 = center_y(b1)
                y2 = center_y(b2)
                batsman_box, bowler_box = (b1, b2) if y1 < y2 else (b2, b1)

                # Bottom corners (left/right) and bottom-centers
                bat_bl, bat_br = bbox_bottom_corners(batsman_box)
                bow_bl, bow_br = bbox_bottom_corners(bowler_box)
                bat_cx, bat_by = bbox_bottom_center(batsman_box)
                bow_cx, bow_by = bbox_bottom_center(bowler_box)

                # Build JSON with four bottom points
                out_json = {
                    "video": os.path.abspath(video_path),
                    "frame_index": int(frame_idx),
                    "image_size": [int(w), int(h)],
                    "image_points": {
                        "bowler_bottom_left":  [int(bow_bl[0]), int(bow_bl[1])],
                        "bowler_bottom_right": [int(bow_br[0]), int(bow_br[1])],
                        "batsman_bottom_left":  [int(bat_bl[0]), int(bat_bl[1])],
                        "batsman_bottom_right": [int(bat_br[0]), int(bat_br[1])],
                        "bowler_middle_stump": [int(bow_cx), int(bow_by)],
                        "batsman_middle_stump": [int(bat_cx), int(bat_by)]
                    },
                }

                # Save JSON if path provided
                if output_json_path:
                    with open(output_json_path, "w") as f:
                        json.dump(out_json, f, indent=2)

                break

        cap.release()

        if out_json is None:
            raise RuntimeError("No two-stump frame detected in video")

        return out_json


if __name__ == "__main__":
    main()
