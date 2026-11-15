"""
Video Enhancement and Object Detection Pipeline.
Handles streaming processing of video files with temporal smoothing,
scene cut detection, real-time edge-preserving denoising, object detection, and continuous multi-object tracking.
"""

from typing import Dict, Any, Optional, Callable, List
import os
import math
import shutil
import subprocess
import cv2
import numpy as np

from enhancement.adaptive_enhancer import AdaptiveEnhancer
from enhancement.generative_enhancer import GenerativeNightToDayEnhancer
from enhancement.noise_estimator import select_adaptive_parameters
from enhancement.temporal_smoother import TemporalSmoother
from detection.detector import ObjectDetector
from detection.tracker import VideoTracker
from detection.evaluator import DetectionEvaluator


class VideoPipeline:
    """
    End-to-end video processing pipeline with temporal coherence,
    low-light illumination recovery, and robust multi-object tracking.
    """

    def __init__(
        self,
        enhancer_type: str = "adaptive",
        denoise_method: str = "bilateral",
        smoothing_alpha: float = 0.88,
        scene_cut_threshold: float = 0.55,
        max_process_width: Optional[int] = 1280,
        detector: Optional[ObjectDetector] = None,
        conf_threshold: float = 0.25,
        detect_interval: int = 1,
    ):
        """
        Args:
            enhancer_type: 'adaptive' or 'generative'.
            denoise_method: 'bilateral' (recommended for video) or 'fastnlmeans'.
            smoothing_alpha: EMA parameter smoothing factor (0.0 to 1.0).
            scene_cut_threshold: Histogram similarity threshold for scene cuts.
            max_process_width: Maximum width in pixels to process (downsamples wide videos).
            detector: Pre-initialized ObjectDetector or None.
            conf_threshold: Object detection confidence threshold.
            detect_interval: Run detection every N frames (1 = every frame).
        """
        self.enhancer_type = enhancer_type.lower()
        self.adaptive_enhancer = AdaptiveEnhancer(denoise_method=denoise_method)
        self.generative_enhancer = GenerativeNightToDayEnhancer()
        self.smoother = TemporalSmoother(alpha=smoothing_alpha, scene_cut_threshold=scene_cut_threshold)
        self.max_process_width = max_process_width
        self.detect_interval = max(1, detect_interval)

        if detector is None:
            self.detector = ObjectDetector(conf_threshold=conf_threshold)
        else:
            self.detector = detector

        self.enh_tracker = VideoTracker(max_missed=4, iou_threshold=0.25, smooth_alpha=0.70)
        self.raw_tracker = VideoTracker(max_missed=4, iou_threshold=0.25, smooth_alpha=0.70)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        run_detection: bool = True,
        save_comparison: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Processes a video file with low-light enhancement, object detection, and tracking.

        Args:
            input_path: Path to source video file.
            output_path: Path to output enhanced video file.
            run_detection: Whether to execute object detection on frames.
            save_comparison: If True, writes side-by-side (raw vs enhanced) video.
            progress_callback: Optional callback f(current_frame, total_frames).

        Returns:
            Dict containing processing summary statistics and keyframe visualizations.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or not math.isfinite(fps) or fps <= 0:
            fps = 15.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        proc_w, proc_h = orig_w, orig_h
        if self.max_process_width is not None and orig_w > self.max_process_width:
            scale = self.max_process_width / float(orig_w)
            proc_w = int(round(orig_w * scale))
            proc_h = int(round(orig_h * scale))

        # Output resolution depends on whether comparison mode is active
        out_w = proc_w * 2 if save_comparison else proc_w
        out_h = proc_h + 70 if save_comparison else proc_h

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        temp_video = output_path if not output_path.endswith("_temp.mp4") else output_path

        # Prefer H.264 (avc1) for browser, Jupyter, QuickTime and HTML5 playback
        writer = None
        for codec in ["avc1", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            w = cv2.VideoWriter(temp_video, fourcc, fps, (out_w, out_h))
            if w.isOpened():
                writer = w
                break

        if writer is None or not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open VideoWriter for {temp_video}")

        self.smoother.reset()
        self.enh_tracker.reset()
        self.raw_tracker.reset()

        frame_idx = 0
        total_detections_count = 0
        frames_with_detections = 0
        best_det_score = -1.0
        best_frame_idx = 0
        best_raw_frame = None
        best_enh_frame = None
        first_raw_frame = None
        first_enh_frame = None
        sample_keyframes: List[Dict[str, Any]] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame.shape[1], frame.shape[0]) != (proc_w, proc_h):
                    frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

                # 1. Parameter derivation and temporal smoothing
                curr_params = select_adaptive_parameters(frame)
                smoothed_params = self.smoother.update(curr_params, frame)

                # 2. Enhancement
                if self.enhancer_type == "generative":
                    enhanced_frame = self.generative_enhancer.enhance(frame)
                else:
                    enhanced_frame = self.adaptive_enhancer.enhance(frame, smoothed_params)

                # 3. Object Detection & Continuous Tracking
                if run_detection:
                    if frame_idx % self.detect_interval == 0:
                        raw_dets = self.detector.detect(frame)
                        enh_dets = self.detector.detect(enhanced_frame)

                        # Filter spurious whole-frame anomalies
                        enh_dets = [
                            d for d in enh_dets
                            if d["bbox"][2] < proc_w * 0.95 and d["bbox"][3] < proc_h * 0.95
                        ]

                        tracked_raw = self.raw_tracker.update(raw_dets)
                        tracked_enh = self.enh_tracker.update(enh_dets)
                    else:
                        tracked_raw = self.raw_tracker.update([])
                        tracked_enh = self.enh_tracker.update([])

                    if len(tracked_enh) > 0:
                        frames_with_detections += 1
                        total_detections_count += len(tracked_enh)

                    enh_annotated = self.detector.draw_detections(enhanced_frame, tracked_enh)
                    raw_annotated = self.detector.draw_detections(frame, tracked_raw)

                    # Capture first frame
                    if first_raw_frame is None:
                        first_raw_frame = raw_annotated.copy()
                        first_enh_frame = enh_annotated.copy()

                    # Track best keyframe with peak detection confidence
                    det_score = sum(d["confidence"] for d in tracked_enh)
                    if det_score > best_det_score:
                        best_det_score = det_score
                        best_frame_idx = frame_idx
                        best_raw_frame = raw_annotated.copy()
                        best_enh_frame = enh_annotated.copy()

                    # Collect representative keyframe samples
                    if len(tracked_enh) > 0 and len(sample_keyframes) < 4:
                        if not sample_keyframes or (frame_idx - sample_keyframes[-1]["frame_idx"]) >= 8:
                            sample_keyframes.append({
                                "frame_idx": frame_idx,
                                "raw": raw_annotated.copy(),
                                "enh": enh_annotated.copy(),
                                "detections": list(tracked_enh),
                            })

                    if save_comparison:
                        eval_metrics = DetectionEvaluator.compare_detections(
                            tracked_raw, tracked_enh
                        )
                        canvas = DetectionEvaluator.create_comparison_canvas(
                            raw_annotated, enh_annotated, eval_metrics
                        )
                        if (canvas.shape[1], canvas.shape[0]) != (out_w, out_h):
                            canvas = cv2.resize(canvas, (out_w, out_h))
                        writer.write(canvas)
                    else:
                        writer.write(enh_annotated)
                else:
                    if save_comparison:
                        combined = np.hstack([frame, enhanced_frame])
                        writer.write(combined)
                    else:
                        writer.write(enhanced_frame)

                frame_idx += 1
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

        finally:
            cap.release()
            writer.release()

        # Fallback for best frame if no detections occurred
        if best_raw_frame is None:
            best_raw_frame = first_raw_frame
            best_enh_frame = first_enh_frame

        # Audio preservation via ffmpeg if installed
        final_output = output_path
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            audio_out = os.path.splitext(output_path)[0] + "_audio.mp4"
            cmd = [
                ffmpeg_bin, "-y",
                "-i", temp_video,
                "-i", input_path,
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                audio_out
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and os.path.exists(audio_out):
                final_output = audio_out

        return {
            "processed_frames": frame_idx,
            "fps": fps,
            "resolution": (proc_w, proc_h),
            "output_path": final_output,
            "scene_cuts_detected": self.smoother.scene_cut_count,
            "total_detections": total_detections_count,
            "frames_with_detections": frames_with_detections,
            "total_tracks": self.enh_tracker.next_id - 1,
            "best_frame_idx": best_frame_idx,
            "best_raw_frame": best_raw_frame,
            "best_enh_frame": best_enh_frame,
            "sample_keyframes": sample_keyframes,
        }
