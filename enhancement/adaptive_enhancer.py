"""
Adaptive Low-Light Enhancer
Executes the continuous content-adaptive enhancement pipeline:
  1. Adaptive Parameter Selection & Noise Evaluation
  2. Edge-Preserving Denoising (Bilateral or FastNLMeans)
  3. LAB Luminance Gamma Curve Correction via 256-entry precomputed LUT
  4. Controlled Contrast Limited Adaptive Histogram Equalization (CLAHE)
  5. Unsharp Masking for detail preservation without ringing
  6. Chroma and Saturation Vibrance Balancing
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np

from .noise_estimator import select_adaptive_parameters


class AdaptiveEnhancer:
    """
    OpenCV-based continuous content-adaptive low-light enhancement engine.
    """

    def __init__(self, denoise_method: str = "bilateral"):
        """
        Args:
            denoise_method: 'bilateral' (real-time, fast) or 'fastnlmeans' (high quality).
        """
        self.denoise_method = denoise_method.lower()

    def _denoise(self, frame_bgr: np.ndarray, h: float) -> np.ndarray:
        """Applies edge-preserving denoising based on configured method."""
        if self.denoise_method == "fastnlmeans":
            # Fast Non-Local Means colored denoising (high quality)
            h_val = float(np.clip(h, 2.0, 10.0))
            return cv2.fastNlMeansDenoisingColored(
                frame_bgr, None,
                h=h_val, hColor=h_val,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            # Bilateral filter (orders of magnitude faster, ideal for video processing)
            sigma_color = float(np.clip(h * 6.0, 15.0, 65.0))
            sigma_space = float(np.clip(h * 1.5, 3.0, 15.0))
            return cv2.bilateralFilter(frame_bgr, d=5, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    def enhance(self, frame_bgr: np.ndarray, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Enhances a single BGR frame using content-adaptive OpenCV operations.

        Args:
            frame_bgr: Input BGR uint8 image (H, W, 3).
            params: Optional precomputed/smoothed parameters. If None, derived automatically.

        Returns:
            np.ndarray: Enhanced BGR uint8 image (H, W, 3).
        """
        if params is None:
            params = select_adaptive_parameters(frame_bgr)

        # 1. Edge-preserving denoising / color space conversion
        h = float(params.get("denoise_h", 4.0))
        if self.denoise_method == "fastnlmeans":
            denoised = self._denoise(frame_bgr, h)
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        else:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 2. Luminance Gamma Correction via LUT
        gamma = float(params.get("gamma", 0.7))
        # Build 256-element lookup table for fast execution
        lut = np.array([
            np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255)
            for i in range(256)
        ], dtype=np.uint8)
        bright_l = cv2.LUT(l_channel, lut)

        # Apply edge-preserving bilateral filter on illuminated luminance channel to preserve vehicle edges
        if self.denoise_method != "fastnlmeans":
            sigma_color = float(np.clip(h * 5.0, 15.0, 50.0))
            sigma_space = float(np.clip(h * 1.5, 3.0, 10.0))
            bright_l = cv2.bilateralFilter(bright_l, d=5, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        # 4. Controlled CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe_clip = float(params.get("clahe_clip", 1.4))
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        clahe_l = clahe.apply(bright_l)

        # Alpha-blend local contrast enhancement with global gamma output
        clahe_mix = float(params.get("clahe_mix", 0.3))
        enhanced_l = cv2.addWeighted(bright_l, 1.0 - clahe_mix, clahe_l, clahe_mix, 0)

        # 5. Gentle Unsharp Masking on luminance
        sharpen = float(params.get("sharpen", 0.03))
        if sharpen > 0.005:
            blurred_l = cv2.GaussianBlur(enhanced_l, (0, 0), sigmaX=1.0)
            enhanced_l = cv2.addWeighted(
                enhanced_l, 1.0 + sharpen,
                blurred_l, -sharpen, 0
            )

        enhanced_l = np.clip(enhanced_l, 0, 255).astype(np.uint8)

        # 6. Reconstruct LAB and convert back to BGR
        enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        result_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 7. Adaptive Saturation Balancing (recovering muted night colors)
        sat_boost = float(params.get("sat_boost", 1.0))
        if sat_boost > 1.01:
            hsv = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_boost, 0, 255)
            result_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return result_bgr


# Convenience functional interface matching original notebook signature
def enhance_frame(frame_bgr: np.ndarray, params: Optional[Dict[str, Any]] = None, denoise_method: str = "bilateral") -> np.ndarray:
    """Convenience wrapper for single frame enhancement."""
    enhancer = AdaptiveEnhancer(denoise_method=denoise_method)
    return enhancer.enhance(frame_bgr, params)

