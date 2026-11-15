"""
Image Enhancement and Object Detection Pipeline.
Handles end-to-end processing of still images.
"""

from typing import Dict, Any, Optional
import os
import cv2
import numpy as np

from enhancement.adaptive_enhancer import AdaptiveEnhancer
from enhancement.generative_enhancer import GenerativeNightToDayEnhancer
from enhancement.noise_estimator import select_adaptive_parameters
from detection.detector import ObjectDetector
from detection.evaluator import DetectionEvaluator


class ImagePipeline:
    """
    End-to-end image processing pipeline coordinating low-light enhancement
    and object detection.
    """

    def __init__(
        self,
        enhancer_type: str = "adaptive",
        denoise_method: str = "fastnlmeans",
        detector: Optional[ObjectDetector] = None,
        conf_threshold: float = 0.35,
    ):
        """
        Args:
            enhancer_type: 'adaptive' or 'generative' (photorealistic night-to-day).
            denoise_method: 'fastnlmeans' (high quality) or 'bilateral' (fast).
            detector: Pre-initialized ObjectDetector or None.
            conf_threshold: Object detection confidence threshold.
        """
        self.enhancer_type = enhancer_type.lower()
        self.adaptive_enhancer = AdaptiveEnhancer(denoise_method=denoise_method)
        self.generative_enhancer = GenerativeNightToDayEnhancer()

        if detector is None:
            self.detector = ObjectDetector(conf_threshold=conf_threshold)
        else:
            self.detector = detector

    def process_image(
        self,
        image_input,
        run_detection: bool = True,
        compare_with_raw: bool = True,
        output_path: Optional[str] = None,
        comparison_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes an image through enhancement and object detection.

        Args:
            image_input: File path (str) or BGR numpy array (H, W, 3).
            run_detection: Whether to execute object detection.
            compare_with_raw: Whether to evaluate detection on both raw and enhanced.
            output_path: Optional path to save the enhanced output.
            comparison_path: Optional path to save the side-by-side comparison.

        Returns:
            Dict containing enhanced image, annotated images, detections, and metrics.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Input image not found: {image_input}")
            raw_bgr = cv2.imread(image_input)
            if raw_bgr is None:
                raise ValueError(f"Could not read image file: {image_input}")
        else:
            raw_bgr = image_input.copy()

        # 1. Parameter Analysis
        params = select_adaptive_parameters(raw_bgr)

        # 2. Enhancement
        if self.enhancer_type == "generative":
            enhanced_bgr = self.generative_enhancer.enhance(raw_bgr)
        elif self.enhancer_type == "both":
            # First adaptive, then night-to-day harmonization
            inter = self.adaptive_enhancer.enhance(raw_bgr, params)
            enhanced_bgr = self.generative_enhancer.enhance(inter)
        else:
            enhanced_bgr = self.adaptive_enhancer.enhance(raw_bgr, params)

        raw_detections = []
        enh_detections = []
        raw_annotated = raw_bgr.copy()
        enh_annotated = enhanced_bgr.copy()
        comparison_canvas = None
        eval_metrics = None

        # 3. Object Detection
        if run_detection:
            enh_detections = self.detector.detect(enhanced_bgr)
            enh_annotated = self.detector.draw_detections(enhanced_bgr, enh_detections)

            if compare_with_raw:
                raw_detections = self.detector.detect(raw_bgr)
                raw_annotated = self.detector.draw_detections(raw_bgr, raw_detections)
                eval_metrics = DetectionEvaluator.compare_detections(raw_detections, enh_detections)
                comparison_canvas = DetectionEvaluator.create_comparison_canvas(
                    raw_annotated, enh_annotated, eval_metrics,
                    title_raw="Raw Low-Light",
                    title_enh=f"Enhanced ({self.enhancer_type.capitalize()})"
                )
            else:
                eval_metrics = {"enhanced": DetectionEvaluator.compute_metrics(enh_detections)}

        # 4. Save results if requested
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            to_save = enh_annotated if run_detection else enhanced_bgr
            cv2.imwrite(output_path, to_save)

        if comparison_path and comparison_canvas is not None:
            os.makedirs(os.path.dirname(os.path.abspath(comparison_path)), exist_ok=True)
            cv2.imwrite(comparison_path, comparison_canvas)

        return {
            "raw_image": raw_bgr,
            "enhanced_image": enhanced_bgr,
            "annotated_image": enh_annotated,
            "comparison_canvas": comparison_canvas,
            "params": params,
            "detections": enh_detections,
            "raw_detections": raw_detections,
            "metrics": eval_metrics,
        }

