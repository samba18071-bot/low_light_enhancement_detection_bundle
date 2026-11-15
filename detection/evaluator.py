"""
Detection Evaluator & Low-Light Benchmark Utility.
Quantitatively and visually compares object detection performance on raw low-light inputs
versus enhanced inputs to demonstrate enhancement efficacy.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class DetectionEvaluator:
    """
    Evaluates and compares object detection metrics across original low-light
    and enhanced imagery.
    """

    @staticmethod
    def compute_metrics(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes summary statistics for a set of detections.
        """
        total_count = len(detections)
        if total_count == 0:
            return {
                "total_count": 0,
                "mean_confidence": 0.0,
                "class_counts": {},
            }

        confs = [d["confidence"] for d in detections]
        class_counts: Dict[str, int] = {}
        for d in detections:
            cname = d["class_name"]
            class_counts[cname] = class_counts.get(cname, 0) + 1

        return {
            "total_count": total_count,
            "mean_confidence": float(np.mean(confs)),
            "class_counts": class_counts,
        }

    @classmethod
    def compare_detections(
        cls,
        raw_detections: List[Dict[str, Any]],
        enhanced_detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compares detection performance between raw and enhanced versions.
        """
        raw_metrics = cls.compute_metrics(raw_detections)
        enh_metrics = cls.compute_metrics(enhanced_detections)

        count_gain = enh_metrics["total_count"] - raw_metrics["total_count"]
        conf_gain = enh_metrics["mean_confidence"] - raw_metrics["mean_confidence"]

        return {
            "raw": raw_metrics,
            "enhanced": enh_metrics,
            "count_gain": count_gain,
            "confidence_gain": conf_gain,
            "recall_improved": count_gain > 0,
        }

    @staticmethod
    def create_comparison_canvas(
        raw_annotated: np.ndarray,
        enh_annotated: np.ndarray,
        metrics: Dict[str, Any],
        title_raw: str = "Raw Low-Light Input",
        title_enh: str = "Enhanced Output"
    ) -> np.ndarray:
        """
        Builds a high-visibility side-by-side comparison canvas with detection stats banner.
        """
        h, w = raw_annotated.shape[:2]
        enh_resized = cv2.resize(enh_annotated, (w, h))

        # Horizontal stack
        combined = np.hstack([raw_annotated, enh_resized])

        # Add top banner for metrics
        banner_h = 70
        banner = np.zeros((banner_h, combined.shape[1], 3), dtype=np.uint8)
        banner[:] = (25, 25, 28)

        # Raw info
        raw_count = metrics["raw"]["total_count"]
        raw_conf = metrics["raw"]["mean_confidence"]
        text_raw = f"{title_raw} | Detections: {raw_count} (Avg Conf: {raw_conf:.1%})"
        cv2.putText(banner, text_raw, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 220), 2, cv2.LINE_AA)

        # Enhanced info
        enh_count = metrics["enhanced"]["total_count"]
        enh_conf = metrics["enhanced"]["mean_confidence"]
        gain_text = f"(+{metrics['count_gain']} new)" if metrics['count_gain'] > 0 else ""
        text_enh = f"{title_enh} | Detections: {enh_count} {gain_text} (Avg Conf: {enh_conf:.1%})"
        cv2.putText(banner, text_enh, (w + 20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 240, 140), 2, cv2.LINE_AA)

        # Stack banner on top
        result = np.vstack([banner, combined])
        return result

