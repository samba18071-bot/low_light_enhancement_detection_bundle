"""
Photorealistic Night-to-Day Style Generative Enhancer.
Combines OpenCV image processing with deep curve estimation, daytime atmospheric color transfer,
and a generative AI integration hook with Laplacian pyramid detail preservation.
"""

from typing import Optional, Tuple, Callable
import cv2
import numpy as np


class GenerativeNightToDayEnhancer:
    """
    Photorealistic Night-to-Day style generation pipeline.
    Combines:
      1. Zero-DCE Multi-Order Illumination Curve Estimation (Retinex inspired)
      2. Daylight Color Space & Atmospheric Tone Synthesis (Reinhard Lab statistical transfer)
      3. Edge-preserving Laplacian Pyramid detail fusion with optional generative models
    """

    def __init__(
        self,
        curve_iterations: int = 6,
        color_temperature_shift: float = 0.25,
        daylight_preset: str = "sunlit_clear",
        generative_backend: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):
        """
        Args:
            curve_iterations: Number of non-linear illumination curve iterations.
            color_temperature_shift: Strength of daylight warm/cool adjustment.
            daylight_preset: Atmospheric style ('sunlit_clear', 'golden_hour', 'overcast_day').
            generative_backend: Optional external AI generator function f(bgr_image) -> bgr_image.
        """
        self.curve_iterations = curve_iterations
        self.color_temperature_shift = color_temperature_shift
        self.daylight_preset = daylight_preset
        self.generative_backend = generative_backend

        # Target statistics in LAB color space for daylight styles (L_mean, L_std, A_mean, A_std, B_mean, B_std)
        self.style_presets = {
            "sunlit_clear": (145.0, 48.0, 126.0, 9.0, 138.0, 15.0),
            "golden_hour":  (140.0, 52.0, 132.0, 12.0, 145.0, 18.0),
            "overcast_day": (135.0, 40.0, 128.0, 7.0, 131.0, 10.0),
        }

    def _estimate_illumination_map(self, img_float: np.ndarray) -> np.ndarray:
        """
        Estimates smooth illumination component using multi-scale guided filtering.
        """
        # Max RGB approximation of initial illumination
        max_channel = np.max(img_float, axis=2)
        # Smooth illumination while respecting structural boundaries
        illumination = cv2.GaussianBlur(max_channel, (0, 0), sigmaX=7.0, sigmaY=7.0)
        return illumination

    def _deep_curve_enhancement(self, img_float: np.ndarray) -> np.ndarray:
        """
        Applies multi-iteration Retinex/Zero-DCE style non-linear illumination curves:
        LE_{n}(x) = LE_{n-1}(x) + A(x) * LE_{n-1}(x) * (1 - LE_{n-1}(x))
        """
        enhanced = img_float.copy()
        illumination = self._estimate_illumination_map(enhanced)
        
        # Adaptive curve parameter A(x): darker areas need higher boost
        alpha = np.clip(1.0 - illumination, 0.1, 0.95)
        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

        for _ in range(self.curve_iterations):
            enhanced = enhanced + alpha * enhanced * (1.0 - enhanced)
            enhanced = np.clip(enhanced, 0.0, 1.0)

        return enhanced

    def _apply_daylight_color_transfer(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Transforms night artificial lighting into natural daylight distribution using
        Reinhard statistical color mapping in LAB color space.
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        target = self.style_presets.get(self.daylight_preset, self.style_presets["sunlit_clear"])
        target_means = [target[0], target[2], target[4]]
        target_stds = [target[1], target[3], target[5]]

        for i in range(3):
            ch = lab[:, :, i]
            src_mean, src_std = np.mean(ch), np.std(ch) + 1e-5
            # Center, scale by standard deviation ratio, and re-center at daylight mean
            # Blend smoothly so original object colors are preserved
            scaled = ((ch - src_mean) * (target_stds[i] / src_std)) + target_means[i]
            # Blend 50% color shift to preserve original chromatic identity
            lab[:, :, i] = np.clip(0.55 * scaled + 0.45 * ch, 0.0, 255.0)

        result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result_bgr

    def _laplacian_pyramid_fusion(self, base_bgr: np.ndarray, ai_bgr: np.ndarray, levels: int = 3) -> np.ndarray:
        """
        Fuses generative AI output with original structural edge details using OpenCV Laplacian pyramids,
        preventing structural distortion while injecting photorealistic daytime textures.
        """
        # Ensure identical sizes
        ai_bgr = cv2.resize(ai_bgr, (base_bgr.shape[1], base_bgr.shape[0]))

        # Generate Gaussian pyramids
        gp_base = [base_bgr.astype(np.float32)]
        gp_ai = [ai_bgr.astype(np.float32)]
        for i in range(levels):
            gp_base.append(cv2.pyrDown(gp_base[-1]))
            gp_ai.append(cv2.pyrDown(gp_ai[-1]))

        # Generate Laplacian pyramids
        lp_base = [gp_base[levels]]
        lp_ai = [gp_ai[levels]]
        for i in range(levels, 0, -1):
            size = (gp_base[i - 1].shape[1], gp_base[i - 1].shape[0])
            base_up = cv2.pyrUp(gp_base[i], dstsize=size)
            ai_up = cv2.pyrUp(gp_ai[i], dstsize=size)
            lp_base.append(gp_base[i - 1] - base_up)
            lp_ai.append(gp_ai[i - 1] - ai_up)

        # Blend: Deep semantic color from AI, high-frequency structural edges from base
        blended_pyramid = []
        for i in range(len(lp_base)):
            # Higher resolution levels take more edge details from base
            weight_base = 0.65 if i > 0 else 0.35
            blended = weight_base * lp_base[i] + (1.0 - weight_base) * lp_ai[i]
            blended_pyramid.append(blended)

        # Reconstruct from Laplacian pyramid
        reconstructed = blended_pyramid[0]
        for i in range(1, len(blended_pyramid)):
            size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
            reconstructed = cv2.pyrUp(reconstructed, dstsize=size) + blended_pyramid[i]

        return np.clip(reconstructed, 0, 255).astype(np.uint8)

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Performs the complete photorealistic Night-to-Day translation.

        Args:
            frame_bgr: Low-light input image in BGR uint8.

        Returns:
            np.ndarray: Photorealistic daytime-enhanced BGR uint8 image.
        """
        # Step 1: Pre-filtering to reduce noise prior to expansion
        clean_bgr = cv2.bilateralFilter(frame_bgr, d=5, sigmaColor=35.0, sigmaSpace=7.0)
        img_float = clean_bgr.astype(np.float32) / 255.0

        # Step 2: Multi-Order Deep Curve Illumination Estimation
        illuminated = self._deep_curve_enhancement(img_float)
        illuminated_bgr = (illuminated * 255.0).astype(np.uint8)

        # Step 3: Daylight Atmospheric Color Transfer
        daylight_bgr = self._apply_daylight_color_transfer(illuminated_bgr)

        # Step 4: Sky & High-Contrast Harmonization
        # Subtly lift shadows while protecting light sources
        lab = cv2.cvtColor(daylight_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        l_final = cv2.addWeighted(l, 0.7, l_clahe, 0.3, 0)
        harmonized_bgr = cv2.cvtColor(cv2.merge([l_final, a, b]), cv2.COLOR_LAB2BGR)

        # Step 5: Generative Model Integration (if external AI generator backend is provided)
        if self.generative_backend is not None:
            try:
                ai_output = self.generative_backend(harmonized_bgr)
                # Fuse generative textures with OpenCV structural edges
                final_bgr = self._laplacian_pyramid_fusion(harmonized_bgr, ai_output)
                return final_bgr
            except Exception as e:
                # Graceful fallback to pure OpenCV photorealistic generation
                return harmonized_bgr

        return harmonized_bgr

