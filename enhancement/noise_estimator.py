"""
Adaptive Low-Light Noise and Brightness Estimator
Analyzes luminance statistics, lower/upper percentiles, and high-pass residual MAD noise.
"""

from typing import Dict, Any
import cv2
import numpy as np


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Estimates Gaussian noise standard deviation (sigma) using high-pass residual
    Median Absolute Deviation (MAD).

    Args:
        gray: Grayscale 8-bit image (H, W).

    Returns:
        float: Estimated noise sigma.
    """
    gray_f = gray.astype(np.float32)
    # High-pass filter via subtracting Gaussian blur
    smooth = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=1.0, sigmaY=1.0)
    residual = gray_f - smooth
    med = np.median(residual)
    mad = np.median(np.abs(residual - med))
    # Scale MAD to approximate normal distribution standard deviation (MAD / 0.6745)
    sigma = float(mad / 0.6745)
    return sigma


def analyze_luminance_and_noise(frame_bgr: np.ndarray) -> Dict[str, float]:
    """
    Computes luminance percentiles in LAB space and estimates sensor noise.

    Args:
        frame_bgr: Input BGR image (H, W, 3).

    Returns:
        Dict containing statistical metrics and normalized indicator scores.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    y = lab[:, :, 0]  # Luminance channel [0, 255]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    p10 = float(np.percentile(y, 10))
    p50 = float(np.percentile(y, 50))
    p90 = float(np.percentile(y, 90))
    mean_y = float(np.mean(y))
    noise_sigma = estimate_noise_sigma(gray)

    # Continuous darkness metric: 0.0 (well lit) to 1.0 (extremely dark)
    darkness = float(np.clip((110.0 - p50) / 105.0, 0.0, 1.0))

    # Continuous noise indicator: 0.0 (clean) to 1.0 (heavy noise)
    noise_level = float(np.clip((noise_sigma - 2.0) / 14.0, 0.0, 1.0))

    # Highlight guard: prevents washing out bright spotlights/streetlamps
    highlight_guard = float(np.clip((245.0 - p90) / 100.0, 0.0, 1.0))

    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "mean_y": mean_y,
        "noise_sigma": noise_sigma,
        "darkness": darkness,
        "noise_level": noise_level,
        "highlight_guard": highlight_guard,
    }


def select_adaptive_parameters(frame_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Derives optimal continuous enhancement parameters based on frame metrics.

    Calculates:
      - gamma: Luminance stretching exponent (lower = brighter midtones)
      - denoise_h: Denoising strength (scaled with noise level)
      - clahe_clip: Local contrast limit
      - clahe_mix: Blend ratio for CLAHE enhancement
      - sharpen: Unsharp masking strength
      - sat_boost: Color vibrancy recovery factor for low-light desaturation
    """
    stats = analyze_luminance_and_noise(frame_bgr)
    darkness = stats["darkness"]
    noise_level = stats["noise_level"]
    highlight_guard = stats["highlight_guard"]

    # Continuous gamma: darkness dictates lift, guarded against bright highlights
    gamma = 1.0 - 0.52 * darkness * (0.65 + 0.35 * highlight_guard)
    gamma = float(np.clip(gamma, 0.45, 1.0))

    # Denoising filter strength (NLM h or Bilateral sigma)
    denoise_h = float(np.clip(2.5 + 5.5 * noise_level, 2.5, 9.0))

    # CLAHE parameters: use less aggressive contrast when noise is prominent
    clahe_clip = float(np.clip(1.15 + 0.65 * (1.0 - noise_level), 1.10, 1.85))
    clahe_mix = float(np.clip(0.20 + 0.25 * (1.0 - noise_level), 0.18, 0.45))

    # Sharpening: controlled to avoid amplifying high-frequency noise
    sharpen = float(np.clip(
        0.015 + 0.055 * (1.0 - noise_level) * (1.0 - 0.5 * darkness),
        0.01, 0.075
    ))

    # Saturation recovery: in low light, sensor loses chroma; boost slightly
    sat_boost = float(np.clip(1.0 + 0.25 * darkness * (1.0 - noise_level), 1.0, 1.30))

    params = stats.copy()
    params.update({
        "gamma": gamma,
        "denoise_h": denoise_h,
        "clahe_clip": clahe_clip,
        "clahe_mix": clahe_mix,
        "sharpen": sharpen,
        "sat_boost": sat_boost,
    })
    return params

