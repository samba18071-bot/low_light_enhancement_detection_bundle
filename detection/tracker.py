"""
Real-Time Multi-Object Video Tracker using IoU Association and State Smoothing.
Maintains continuous object identity and bounding box stability across low-light video frames.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def box_iou(b1: List[int], b2: List[int]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes [x, y, w, h].
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


class VideoTracker:
    """
    Lightweight, high-throughput tracker designed for low-light video streams.
    Prevents detection dropouts and box jitter by associating detections frame-to-frame
    and smoothing trajectories using Exponential Moving Average (EMA).
    """

    def __init__(
        self,
        max_missed: int = 4,
        iou_threshold: float = 0.25,
        smooth_alpha: float = 0.70,
    ):
        """
        Args:
            max_missed: Maximum consecutive missed frames before terminating a track.
            iou_threshold: Minimum IoU overlap to associate a detection with an existing track.
            smooth_alpha: EMA smoothing factor for bounding box coordinates (0.0 to 1.0).
        """
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.smooth_alpha = smooth_alpha
        self.tracks: List[Dict[str, Any]] = []
        self.next_id: int = 1

    def reset(self):
        """Resets all active tracks."""
        self.tracks = []
        self.next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates tracks with detections from the current frame.

        Args:
            detections: List of detection dicts with 'bbox' [x, y, w, h], 'confidence',
                        'class_id', and 'class_name'.

        Returns:
            List of active track dicts including 'track_id', smoothed 'bbox', and 'confidence'.
        """
        updated_tracks: List[Dict[str, Any]] = []
        unmatched_dets = list(range(len(detections)))

        # 1. Match existing tracks to new detections
        for track in self.tracks:
            best_iou = 0.0
            best_det_idx = -1

            for d_idx in unmatched_dets:
                det = detections[d_idx]
                # Prefer matching the same class, but allow close matches
                iou = box_iou(track["bbox"], det["bbox"])
                if det.get("class_id") == track.get("class_id"):
                    iou += 0.05

                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx

            if best_iou >= self.iou_threshold and best_det_idx >= 0:
                det = detections[best_det_idx]
                # Smooth bbox coordinates using EMA
                smoothed_box = [
                    int(round(self.smooth_alpha * det["bbox"][k] + (1.0 - self.smooth_alpha) * track["bbox"][k]))
                    for k in range(4)
                ]
                bx, by, bw, bh = smoothed_box
                track["bbox"] = smoothed_box
                track["box_xyxy"] = [bx, by, bx + bw, by + bh]
                track["confidence"] = float(self.smooth_alpha * det["confidence"] + (1.0 - self.smooth_alpha) * track["confidence"])
                track["class_id"] = det["class_id"]
                track["class_name"] = det["class_name"]
                track["missed"] = 0
                track["age"] = track.get("age", 0) + 1
                updated_tracks.append(track)
                unmatched_dets.remove(best_det_idx)
            else:
                track["missed"] = track.get("missed", 0) + 1
                if track["missed"] <= self.max_missed:
                    # Decay confidence slightly during occlusion/missed frames
                    track["confidence"] *= 0.95
                    track["age"] = track.get("age", 0) + 1
                    updated_tracks.append(track)

        # 2. Initialize new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_track = {
                "track_id": self.next_id,
                "bbox": list(det["bbox"]),
                "box_xyxy": list(det.get("box_xyxy", [
                    det["bbox"][0], det["bbox"][1],
                    det["bbox"][0] + det["bbox"][2],
                    det["bbox"][1] + det["bbox"][3]
                ])),
                "confidence": float(det["confidence"]),
                "class_id": int(det["class_id"]),
                "class_name": str(det["class_name"]),
                "missed": 0,
                "age": 1,
            }
            self.next_id += 1
            updated_tracks.append(new_track)

        self.tracks = updated_tracks

        # Return deep copy of active tracks
        return [
            {
                "track_id": t["track_id"],
                "bbox": list(t["bbox"]),
                "box_xyxy": list(t["box_xyxy"]),
                "confidence": float(t["confidence"]),
                "class_id": t["class_id"],
                "class_name": t["class_name"],
                "missed": t["missed"],
            }
            for t in self.tracks
        ]
