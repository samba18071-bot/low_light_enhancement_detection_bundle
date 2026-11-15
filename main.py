#!/usr/bin/env python3
"""
Command Line Interface for Adaptive Low-Light Enhancement & Object Detection.

Usage examples:
  # Image enhancement + OpenCV DNN object detection + side-by-side comparison:
  python main.py --input path/to/dark_image.jpg --output outputs/enhanced.jpg --compare

  # Photorealistic Night-to-Day generative style generation:
  python main.py --input path/to/dark_image.jpg --enhancer generative --output outputs/daylight.jpg --compare

  # Video enhancement with real-time bilateral filtering and object detection:
  python main.py --input path/to/night_video.mp4 --output outputs/enhanced_video.mp4 --compare

  # Run on CPU with pure OpenCV DNN:
  python main.py --input sample.png --detector opencv_dnn --conf 0.35
"""

import argparse
import os
import sys
import time

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.unified_pipeline import UnifiedPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Low-Light Enhancement and Object Detection Pipeline entirely in Python using OpenCV."
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str,
        help="Path to input image or video."
    )
    parser.add_argument(
        "--output", "-o", default=None, type=str,
        help="Path to save enhanced output. Defaults to 'outputs/<input_name>_enhanced.<ext>'."
    )
    parser.add_argument(
        "--enhancer", "-e", choices=["adaptive", "generative", "both"], default="adaptive",
        help="Enhancer algorithm: 'adaptive' (continuous LAB gamma/CLAHE/denoise), 'generative' (photorealistic night-to-day deep curve/color transfer), or 'both'."
    )
    parser.add_argument(
        "--denoiser", choices=["bilateral", "fastnlmeans"], default="bilateral",
        help="Denoising filter: 'bilateral' (fast/real-time) or 'fastnlmeans' (high quality)."
    )
    parser.add_argument(
        "--detector", choices=["opencv_dnn", "ultralytics"], default="opencv_dnn",
        help="Inference engine for object detection: 'opencv_dnn' (pure OpenCV) or 'ultralytics'."
    )
    parser.add_argument(
        "--model", default=None, type=str,
        help="Path to object detector model (.onnx or .pt). Defaults to models/yolov8n.onnx."
    )
    parser.add_argument(
        "--conf", default=0.35, type=float,
        help="Confidence threshold for object detection (default: 0.35)."
    )
    parser.add_argument(
        "--no-detect", dest="run_detect", action="store_false",
        help="Disable object detection and only perform enhancement."
    )
    parser.add_argument(
        "--compare", dest="compare", action="store_true", default=True,
        help="Generate side-by-side comparative visualization (default: True)."
    )
    parser.add_argument(
        "--no-compare", dest="compare", action="store_false",
        help="Disable side-by-side comparative visualization."
    )
    parser.add_argument(
        "--max-width", default=1280, type=int,
        help="Maximum width to scale video/image during processing (default: 1280)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Determine default output path if unspecified
    if args.output is None:
        os.makedirs("outputs", exist_ok=True)
        base = os.path.basename(args.input)
        name, ext = os.path.splitext(base)
        args.output = os.path.join("outputs", f"{name}_enhanced{ext}")

    print("=" * 65)
    print("🚀 ADAPTIVE LOW-LIGHT ENHANCEMENT & OBJECT DETECTION PIPELINE")
    print("=" * 65)
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Enhancer:     {args.enhancer}")
    print(f"Denoiser:     {args.denoiser}")
    print(f"Detector:     {args.detector if args.run_detect else 'Disabled'}")
    print(f"Confidence:   {args.conf}")
    print(f"Comparison:   {args.compare}")
    print("-" * 65)

    start_time = time.time()
    pipeline = UnifiedPipeline(
        enhancer_type=args.enhancer,
        denoise_method=args.denoiser,
        engine=args.detector,
        model_path=args.model,
        conf_threshold=args.conf,
        max_process_width=args.max_width,
    )

    result = pipeline.process(
        input_path=args.input,
        output_path=args.output,
        run_detection=args.run_detect,
        save_comparison=args.compare,
    )

    elapsed = time.time() - start_time
    print("-" * 65)
    print(f"✅ Processing completed in {elapsed:.2f} seconds.")

    if "metrics" in result and result["metrics"]:
        m = result["metrics"]
        if "raw" in m and "enhanced" in m:
            print("\n📊 OBJECT DETECTION PERFORMANCE BENCHMARK:")
            print(f"  • Raw Dark Detections:      {m['raw']['total_count']} objects (Avg Conf: {m['raw']['mean_confidence']:.1%})")
            print(f"  • Enhanced Detections:      {m['enhanced']['total_count']} objects (Avg Conf: {m['enhanced']['mean_confidence']:.1%})")
            print(f"  • Additional Objects Found: +{m['count_gain']} objects")
            print(f"  • Detection Recall Boost:   {'+' if m['count_gain'] > 0 else ''}{m['count_gain']} objects detected after enhancement!")
            if m["enhanced"]["class_counts"]:
                print(f"  • Detected Classes:         {m['enhanced']['class_counts']}")

    if "scene_cuts_detected" in result:
        print(f"\n🎬 Video Scene Cuts Handled: {result['scene_cuts_detected']}")
        print(f"🎞️ Processed Frames:         {result['processed_frames']} frames")

    print(f"\n💾 Saved Enhanced Output:     {args.output}")
    if args.compare:
        root, ext = os.path.splitext(args.output)
        comp_file = f"{root}_comparison{ext}"
        if os.path.exists(comp_file):
            print(f"🖼️ Saved Comparison Canvas:   {comp_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()

