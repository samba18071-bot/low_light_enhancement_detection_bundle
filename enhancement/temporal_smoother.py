"""
Temporal Smoother and Scene Cut Detector for Video Enhancement.
Provides Exponential Moving Average (EMA) parameter smoothing across frames
and resets parameters when a scene cut is detected via color/luminance histogram correlation.
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np


class TemporalSmoother:
    """
    Maintains temporal consistency across consecutive video frames to prevent
    brightness flicker and noise pumping, while handling scene transitions cleanly.
    """

    def __init__(self, alpha: float = 0.88, scene_cut_threshold: float = 0.55):
        """
        Args:
            alpha: EMA smoothing coefficient (0.0 to 1.0). Higher = smoother changes.
            scene_cut_threshold: Histogram correlation threshold below which a scene cut is triggered.
        """
        self.alpha = float(np.clip(alpha, 0.0, 0.99))
        self.scene_cut_threshold = scene_cut_threshold
        self.previous_params: Optional[Dict[str, float]] = None
        self.previous_hist: Optional[np.ndarray] = None
        self.scene_cut_count: int = 0
        self.frame_index: int = 0

    def compute_histogram(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Computes a normalized 32-bin grayscale histogram for rapid similarity checking."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def is_scene_cut(self, current_hist: np.ndarray) -> bool:
        """Checks if the correlation between previous and current frame histograms falls below threshold."""
        if self.previous_hist is None:
            return False
        similarity = cv2.compareHist(self.previous_hist, current_hist, cv2.HISTCMP_CORREL)
        return bool(similarity < self.scene_cut_threshold)

    def update(self, current_params: Dict[str, Any], frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Updates internal state with new frame parameters and returns temporally smoothed parameters.
        Accepts arguments in either (params, frame) or (frame, params) order.
        """
        # Gracefully handle swapped argument order
        if isinstance(current_params, np.ndarray) and isinstance(frame_bgr, dict):
            current_params, frame_bgr = frame_bgr, current_params

        self.frame_index += 1
        current_hist = self.compute_histogram(frame_bgr)
        cut_detected = self.is_scene_cut(current_hist)

        if cut_detected:
            self.scene_cut_count += 1
            # Reset temporal smoothing on scene cut so new scene isn't distorted
            self.previous_params = None

        if self.previous_params is None:
            smoothed = current_params.copy()
        else:
            smoothed = current_params.copy()
            smooth_keys = ("gamma", "denoise_h", "clahe_clip", "clahe_mix", "sharpen", "sat_boost")
            for key in smooth_keys:
                if key in current_params and key in self.previous_params:
                    smoothed[key] = float(
                        self.alpha * self.previous_params[key] + (1.0 - self.alpha) * current_params[key]
                    )

        self.previous_params = smoothed.copy()
        self.previous_hist = current_hist
        smoothed["scene_cut"] = cut_detected
        return smoothed

    def reset(self):
        """Resets the temporal smoother state."""
        self.previous_params = None
        self.previous_hist = None
        self.scene_cut_count = 0
        self.frame_index = 0

