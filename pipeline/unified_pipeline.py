"""
Unified Pipeline for Image and Video Low-Light Enhancement and Object Detection.
Automatically routes between image and video pipelines based on file type.
"""

from typing import Dict, Any, Optional
import os
import cv2

from .image_pipeline import ImagePipeline
from .video_pipeline import VideoPipeline
from detection.detector import ObjectDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


class UnifiedPipeline:
    """
    Main entry point for processing any media (image or video) with
    low-light enhancement and object detection.
    """

    def __init__(
        self,
        enhancer_type: str = "adaptive",
        denoise_method: str = "bilateral",
        engine: str = "opencv_dnn",
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        max_process_width: int = 1280,
    ):
        """
        Args:
            enhancer_type: 'adaptive' or 'generative'.
            denoise_method: 'bilateral' or 'fastnlmeans'.
            engine: 'opencv_dnn' or 'ultralytics'.
            model_path: Path to object detection model.
            conf_threshold: Object detection confidence threshold.
            max_process_width: Maximum width to resize videos/images.
        """
        self.detector = ObjectDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            engine=engine,
        )
        self.image_pipeline = ImagePipeline(
            enhancer_type=enhancer_type,
            denoise_method=denoise_method,
            detector=self.detector,
            conf_threshold=conf_threshold,
        )
        self.video_pipeline = VideoPipeline(
            enhancer_type=enhancer_type,
            denoise_method=denoise_method,
            max_process_width=max_process_width,
            detector=self.detector,
            conf_threshold=conf_threshold,
        )

    @staticmethod
    def identify_media_type(file_path: str) -> str:
        """Determines whether a file is an image or video."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        if ext in VIDEO_EXTENSIONS:
            return "video"

        # Content-based probe
        img = cv2.imread(file_path)
        if img is not None:
            return "image"
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            cap.release()
            return "video"
        raise ValueError(f"Unsupported or unreadable file format: {file_path}")

    def process(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        run_detection: bool = True,
        save_comparison: bool = True,
    ) -> Dict[str, Any]:
        """
        Processes the input file (image or video).

        Args:
            input_path: Path to media file.
            output_path: Destination path for output.
            run_detection: Run object detection.
            save_comparison: Save side-by-side comparison.

        Returns:
            Dict containing pipeline results.
        """
        media_type = self.identify_media_type(input_path)

        if media_type == "image":
            comp_path = None
            if save_comparison and output_path:
                root, ext = os.path.splitext(output_path)
                comp_path = f"{root}_comparison{ext}"
            return self.image_pipeline.process_image(
                image_input=input_path,
                run_detection=run_detection,
                compare_with_raw=save_comparison,
                output_path=output_path,
                comparison_path=comp_path,
            )
        else:
            if output_path is None:
                root, ext = os.path.splitext(input_path)
                output_path = f"{root}_enhanced.mp4"
            return self.video_pipeline.process_video(
                input_path=input_path,
                output_path=output_path,
                run_detection=run_detection,
                save_comparison=save_comparison,
            )

