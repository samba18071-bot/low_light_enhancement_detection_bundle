#!/usr/bin/env python3
"""
Streamlit Web Application for Adaptive Low-Light Enhancement and Object Detection.
Interactive UI supporting image uploads, video processing, sample presets, and camera snapshots.
"""

import os
import sys
import time
import tempfile
import json
import glob
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detection.detector import ObjectDetector
from detection.evaluator import DetectionEvaluator
from enhancement.adaptive_enhancer import AdaptiveEnhancer
from enhancement.generative_enhancer import GenerativeNightToDayEnhancer
from enhancement.noise_estimator import select_adaptive_parameters, estimate_noise_sigma, analyze_luminance_and_noise
from pipeline.image_pipeline import ImagePipeline
from pipeline.video_pipeline import VideoPipeline

# Page Configuration
st.set_page_config(
    page_title="Adaptive Low-Light Enhancement & Object Detection",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 50%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        border-radius: 6px;
    }
    .badge-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-yolo {
        background-color: rgba(0, 229, 255, 0.15);
        color: #00e5ff;
        border: 1px solid rgba(0, 229, 255, 0.3);
    }
    .badge-adaptive {
        background-color: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.3);
    }
    .badge-cpu {
        background-color: rgba(255, 171, 0, 0.15);
        color: #ffab00;
        border: 1px solid rgba(255, 171, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# Model Cache Helper
@st.cache_resource(show_spinner=False)
def load_cached_detector(model_choice: str, engine: str, model_path: Optional[str]) -> ObjectDetector:
    """Caches neural network weights to prevent re-reading ONNX/PT models on every interaction."""
    return ObjectDetector(
        model_path=model_path,
        engine=engine,
        conf_threshold=0.25,
        nms_threshold=0.45
    )


def resolve_model_specs(model_choice: str) -> Tuple[str, Optional[str]]:
    """Maps UI model selection string to engine and file path."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    
    if "YOLO11n (ONNX - Pure OpenCV DNN)" in model_choice:
        return "opencv_dnn", os.path.join(models_dir, "yolo11n.onnx")
    elif "YOLO11n (PyTorch - Ultralytics)" in model_choice:
        return "ultralytics", os.path.join(models_dir, "yolo11n.pt")
    elif "YOLOv8n (ONNX - OpenCV DNN)" in model_choice:
        return "opencv_dnn", os.path.join(models_dir, "yolov8n.onnx")
    elif "Open Images V7 (601 Classes - ONNX)" in model_choice:
        return "opencv_dnn", os.path.join(models_dir, "yolov8n-oiv7.onnx")
    elif "Open Images V7 (601 Classes - PyTorch)" in model_choice:
        return "ultralytics", os.path.join(models_dir, "yolov8n-oiv7.pt")
    else:
        return "opencv_dnn", os.path.join(models_dir, "yolo11n.onnx")


def plot_luminance_histogram(raw_bgr: np.ndarray, enh_bgr: np.ndarray):
    """Generates an aesthetic Matplotlib figure comparing luminance distributions before & after."""
    raw_lab = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2LAB)
    enh_lab = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2LAB)
    
    raw_l = raw_lab[:, :, 0].ravel()
    enh_l = enh_lab[:, :, 0].ravel()
    
    fig, ax = plt.subplots(figsize=(7, 2.5), facecolor="#0e1117")
    ax.set_facecolor("#161b22")
    
    ax.hist(raw_l, bins=64, range=(0, 256), color="#ef4444", alpha=0.5, label="Raw Low-Light L*", density=True)
    ax.hist(enh_l, bins=64, range=(0, 256), color="#10b981", alpha=0.6, label="Enhanced L*", density=True)
    
    ax.set_xlim(0, 255)
    ax.tick_params(colors="#94a3b8")
    ax.spines['bottom'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.set_xlabel("Luminance Channel (L* [0 - 255])", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Normalized Density", color="#94a3b8", fontsize=9)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    fig.tight_layout()
    return fig


# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/night-landscape.png", width=64)
    st.title("Settings & Control")
    st.markdown("Configure algorithmic enhancement and object detection parameters.")
    
    st.markdown("### 🧠 Object Detection Model")
    model_choice = st.selectbox(
        "Model Architecture & Backend",
        [
            "YOLO11n (ONNX - Pure OpenCV DNN) [Recommended]",
            "YOLO11n (PyTorch - Ultralytics)",
            "YOLOv8n (ONNX - OpenCV DNN)",
            "Open Images V7 (601 Classes - ONNX)",
            "Open Images V7 (601 Classes - PyTorch)"
        ],
        index=0,
        help="YOLO11n offers the best low-light recall with higher FPS on CPU."
    )
    
    conf_thresh = st.slider("Confidence Threshold", min_value=0.05, max_value=0.90, value=0.20, step=0.05)
    nms_thresh = st.slider("IoU / NMS Threshold", min_value=0.10, max_value=0.80, value=0.45, step=0.05)
    
    st.markdown("---")
    st.markdown("### 🎨 Illumination & Denoising")
    enhancer_type = st.radio(
        "Enhancement Method",
        ["Adaptive (Continuous LAB)", "Generative (Night-to-Day)", "Both (Cascaded)"],
        index=0,
        help="Adaptive selects parameters via noise MAD; Generative performs photorealistic daylight harmonization."
    )
    
    denoiser_method = st.radio(
        "Edge-Preserving Denoiser",
        ["Bilateral (Fast / Real-Time)", "Fast Non-Local Means (High Quality)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Display & Processing Options")
    save_comparison = st.checkbox("Generate Side-by-Side Comparison Canvas", value=True)
    show_labels = st.checkbox("Draw Bounding Box Labels & Confidence", value=True)
    max_video_width = st.select_slider("Video Processing Resolution", options=[480, 640, 720, 1080, 1280], value=720)


# Initialize Engine & Pipeline
engine_name, model_file = resolve_model_specs(model_choice)
try:
    detector = load_cached_detector(model_choice, engine_name, model_file)
    detector.conf_threshold = conf_thresh
    detector.nms_threshold = nms_thresh
except Exception as e:
    st.error(f"Failed to load detector {model_choice}: {e}")
    st.stop()

enhancer_key = "adaptive" if "Adaptive" in enhancer_type else ("generative" if "Generative" in enhancer_type else "both")
denoise_key = "bilateral" if "Bilateral" in denoiser_method else "fastnlmeans"

img_pipeline = ImagePipeline(
    enhancer_type=enhancer_key,
    denoise_method=denoise_key,
    detector=detector,
    conf_threshold=conf_thresh
)


# Header Area
st.markdown('<div class="main-header">🌌 Adaptive Low-Light Enhancement & Object Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    '<span class="badge-tag badge-adaptive">Continuous Adaptive LAB Illumination</span>'
    '<span class="badge-tag badge-yolo">YOLO11n Dual-Engine (COCO + Open Images)</span>'
    '<span class="badge-tag badge-cpu">Pure OpenCV DNN CPU Acceleration</span>'
    '</div>',
    unsafe_allow_html=True
)

# Main Input Selection Tabs
tab_upload, tab_samples, tab_camera = st.tabs(["📁 Upload Image / Video", "🖼️ Curated Low-Light Presets", "📷 Live Webcam Capture"])

selected_input_type = None
input_image_bgr: Optional[np.ndarray] = None
input_video_path: Optional[str] = None
input_name: str = "sample"

# TAB 1: File Uploader
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload dark image or nocturnal video",
        type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "mov", "avi", "mkv"],
        help="Upload nocturnal photos, surveillance clips, or dashcam videos."
    )
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        input_name = os.path.splitext(uploaded_file.name)[0]
        if file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            selected_input_type = "image"
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            input_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            selected_input_type = "video"
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            tfile.write(uploaded_file.read())
            input_video_path = tfile.name

# TAB 2: Sample Presets
with tab_samples:
    st.markdown("Select a real low-light benchmark sample from the bundled datasets:")
    
    preset_options = {
        "ExDark: Pedestrians & Cyclist in Extreme Darkness": "data/exdark/images/2015_00013_jpg.rf.1ccbc5a451029f00e87a4e0d4181841b.jpg",
        "ExDark: Dark Roadside Street & Vehicle": "data/exdark/images/2015_00026_jpg.rf.4ed27f2ff81dd07c2a7b271d8bb85ad3.jpg",
        "ExDark: Dim Outdoor Scene": "data/exdark/images/2015_00074_jpg.rf.a41d7af808b25b6fb23f25bab66985ab.jpg",
        "DICM: Extreme Contrast Low-Light Photo": "data/dicm/01.jpg",
        "LIME: Extreme Low-Light Street": "data/lime/01.bmp",
        "Video: Nocturnal Highway & Traffic Stream": "data/real_night_video.mp4"
    }
    
    preset_choice = st.selectbox("Choose Preset Sample", list(preset_options.keys()))
    preset_path = os.path.join(PROJECT_ROOT, preset_options[preset_choice])
    
    if st.button("Load Selected Preset", type="primary"):
        if preset_path.endswith((".jpg", ".png", ".bmp")):
            selected_input_type = "image"
            input_image_bgr = cv2.imread(preset_path)
            input_name = os.path.basename(preset_path).split(".")[0]
        else:
            selected_input_type = "video"
            input_video_path = preset_path
            input_name = "real_night_video"

# TAB 3: Webcam Capture
with tab_camera:
    cam_picture = st.camera_input("Capture low-light scene with camera")
    if cam_picture is not None:
        selected_input_type = "image"
        cam_bytes = np.frombuffer(cam_picture.read(), np.uint8)
        input_image_bgr = cv2.imdecode(cam_bytes, cv2.IMREAD_COLOR)
        input_name = f"webcam_snapshot_{int(time.time())}"


# =========================================================================
# PROCESSING PIPELINE: IMAGE MODE
# =========================================================================
if selected_input_type == "image" and input_image_bgr is not None:
    st.markdown("---")
    st.subheader("⚡ Processing Results & Real-Time Analytics")
    
    with st.spinner("Processing illumination enhancement and neural detection..."):
        t_start = time.perf_counter()
        results = img_pipeline.process_image(
            image_input=input_image_bgr,
            run_detection=True,
            compare_with_raw=save_comparison
        )
        proc_time = time.perf_counter() - t_start
    
    raw_bgr = results["raw_image"]
    enh_bgr = results["enhanced_image"]
    annotated_bgr = results["annotated_image"]
    comp_canvas = results.get("comparison_canvas")
    params = results.get("params", {})
    metrics = results.get("metrics", {})
    
    # 1. Top Metrics Bar
    col1, col2, col3, col4, col5 = st.columns(5)
    raw_cnt = metrics.get("raw", {}).get("total_count", 0)
    enh_cnt = metrics.get("enhanced", {}).get("total_count", 0)
    gain_cnt = metrics.get("count_gain", enh_cnt - raw_cnt)
    
    col1.metric("Raw Detections", f"{raw_cnt} objects")
    col2.metric("Enhanced Detections", f"{enh_cnt} objects", delta=f"{'+' if gain_cnt >= 0 else ''}{gain_cnt} Recall")
    
    raw_conf = metrics.get("raw", {}).get("mean_confidence", 0.0)
    enh_conf = metrics.get("enhanced", {}).get("mean_confidence", 0.0)
    delta_conf = (enh_conf - raw_conf) * 100
    col3.metric("Avg Confidence", f"{enh_conf:.1%}", delta=f"{delta_conf:+.1f}% vs dark" if raw_cnt > 0 else None)
    
    col4.metric("Noise Sigma (MAD)", f"{params.get('noise_sigma', 0.0):.2f}")
    col5.metric("Adaptive Gamma LUT", f"γ = {params.get('gamma', 1.0):.3f}")

    # 2. Main Visualization Tabs
    vis_tab1, vis_tab2, vis_tab3, vis_tab4 = st.tabs([
        "🔀 Side-by-Side Comparison",
        "✨ Enhanced + Detections",
        "🌑 Original Raw Dark",
        "📈 Luminance Distribution & Histogram"
    ])
    
    with vis_tab1:
        if comp_canvas is not None:
            comp_rgb = cv2.cvtColor(comp_canvas, cv2.COLOR_BGR2RGB)
            st.image(comp_rgb, use_container_width=True, caption=f"Side-by-Side: Raw Low-Light vs Enhanced ({enhancer_type}) with Object Detections")
        else:
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), caption="Raw Low-Light Input", use_container_width=True)
            c2.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), caption="Enhanced + Detected Output", use_container_width=True)
            
    with vis_tab2:
        st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Enhanced Output with {model_choice}")
        
    with vis_tab3:
        raw_annotated = detector.draw_detections(raw_bgr, results.get("raw_detections", []))
        st.image(cv2.cvtColor(raw_annotated, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Raw Dark Image with Initial Detections")
        
    with vis_tab4:
        st.markdown("#### Luminance (L*) Channel Shift")
        st.pyplot(plot_luminance_histogram(raw_bgr, enh_bgr), use_container_width=False)
        st.markdown(f"""
        - **Darkness Factor**: `{params.get('darkness', 0.0):.2f}` (0.0 = bright daylight, 1.0 = pitch black)
        - **Optimal CLAHE Clip Limit**: `{params.get('clahe_clip', 0.0):.2f}`
        - **High-Frequency Sharpening Boost**: `{params.get('sharpen', 0.0):.3f}`
        - **Denoising Filter Strength**: `{params.get('denoise_h', 0.0):.2f}`
        - **Inference Latency**: `{proc_time*1000:.1f} ms`
        """)

    # 3. Detected Objects Table
    st.markdown("### 📋 Detected Objects Breakdown")
    detections = results.get("detections", [])
    if detections:
        det_rows = []
        for idx, d in enumerate(detections, 1):
            bbox = d["bbox"]
            det_rows.append({
                "#": idx,
                "Class": d["class_name"],
                "Confidence": f"{d['confidence']:.1%}",
                "X": bbox[0],
                "Y": bbox[1],
                "Width": bbox[2],
                "Height": bbox[3],
            })
        st.dataframe(det_rows, use_container_width=True)
    else:
        st.info("No objects detected above the confidence threshold.")

    # 4. Downloads
    st.markdown("### 💾 Export & Download")
    dcol1, dcol2, dcol3 = st.columns(3)
    
    # Download Enhanced Image
    _, enh_buf = cv2.imencode(".jpg", annotated_bgr)
    dcol1.download_button(
        label="📥 Download Enhanced Image",
        data=enh_buf.tobytes(),
        file_name=f"{input_name}_enhanced.jpg",
        mime="image/jpeg"
    )
    
    # Download Comparison Canvas
    if comp_canvas is not None:
        _, comp_buf = cv2.imencode(".jpg", comp_canvas)
        dcol2.download_button(
            label="📥 Download Comparison Canvas",
            data=comp_buf.tobytes(),
            file_name=f"{input_name}_comparison.jpg",
            mime="image/jpeg"
        )
    
    # Download JSON Metrics
    clean_metrics = {
        "parameters": params,
        "metrics": metrics,
        "detections_count": len(detections),
        "classes": [d["class_name"] for d in detections]
    }
    dcol3.download_button(
        label="📥 Download Metrics (JSON)",
        data=json.dumps(clean_metrics, indent=2),
        file_name=f"{input_name}_metrics.json",
        mime="application/json"
    )


# =========================================================================
# PROCESSING PIPELINE: VIDEO MODE
# =========================================================================
elif selected_input_type == "video" and input_video_path is not None:
    st.markdown("---")
    st.subheader("🎬 Nocturnal Video Processing & Multi-Object Tracking")
    
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    vcol1.metric("Resolution", f"{w}x{h}")
    vcol2.metric("Total Frames", f"{total_frames}")
    vcol3.metric("Video FPS", f"{fps:.1f}")
    vcol4.metric("Estimated Duration", f"{total_frames / max(1.0, fps):.1f}s")
    
    # Frame limit option for fast testing
    max_frames_to_process = st.slider("Process up to N frames (drag slider to limit for quick preview)", min_value=15, max_value=max(15, total_frames), value=min(60, total_frames))
    
    if st.button("🚀 Start Video Enhancement & Tracking", type="primary"):
        prog_bar = st.progress(0, text="Initializing Video Pipeline...")
        status_text = st.empty()
        
        # Output paths
        out_dir = os.path.join(PROJECT_ROOT, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_video_path = os.path.join(out_dir, f"{input_name}_enhanced.mp4")
        
        video_pipeline = VideoPipeline(
            enhancer_type=enhancer_key,
            denoise_method=denoise_key,
            max_process_width=max_video_width,
            detector=detector,
            conf_threshold=conf_thresh
        )
        
        # Subprocess or chunked reader with progress
        cap = cv2.VideoCapture(input_video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        proc_w, proc_h = orig_w, orig_h
        if max_video_width is not None and orig_w > max_video_width:
            scale = max_video_width / float(orig_w)
            proc_w = int(round(orig_w * scale))
            proc_h = int(round(orig_h * scale))
            
        out_w = proc_w * 2 if save_comparison else proc_w
        out_h = proc_h + 70 if save_comparison else proc_h
        
        # Use avc1 for H.264 browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_video_path, fourcc, fps, (out_w, out_h))
            
        video_pipeline.smoother.reset()
        video_pipeline.enh_tracker.reset()
        video_pipeline.raw_tracker.reset()
        
        t0 = time.time()
        frame_idx = 0
        total_detections_count = 0
        preview_placeholder = st.empty()
        
        while frame_idx < max_frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if (frame.shape[1], frame.shape[0]) != (proc_w, proc_h):
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                
            raw_frame = frame.copy()
            params = select_adaptive_parameters(raw_frame)
            smoothed_params = video_pipeline.smoother.update(raw_frame, params)
            
            if video_pipeline.enhancer_type == "generative":
                enhanced_frame = video_pipeline.generative_enhancer.enhance(raw_frame)
            elif video_pipeline.enhancer_type == "both":
                inter = video_pipeline.adaptive_enhancer.enhance(raw_frame, smoothed_params)
                enhanced_frame = video_pipeline.generative_enhancer.enhance(inter)
            else:
                enhanced_frame = video_pipeline.adaptive_enhancer.enhance(raw_frame, smoothed_params)
                
            enh_dets = detector.detect(enhanced_frame)
            total_detections_count += len(enh_dets)
            tracked_enh = video_pipeline.enh_tracker.update(enh_dets)
            enh_annotated = detector.draw_detections(enhanced_frame, tracked_enh, show_labels=show_labels)
            
            if save_comparison:
                raw_dets = detector.detect(raw_frame)
                tracked_raw = video_pipeline.raw_tracker.update(raw_dets)
                raw_annotated = detector.draw_detections(raw_frame, tracked_raw, show_labels=show_labels)
                
                eval_metrics = DetectionEvaluator.compare_detections(raw_dets, enh_dets)
                canvas = DetectionEvaluator.create_comparison_canvas(
                    raw_annotated, enh_annotated, eval_metrics,
                    title_raw="Raw Nocturnal Video",
                    title_enh=f"Enhanced + Tracking ({model_choice.split()[0]})"
                )
                writer.write(canvas)
                
                # Live thumbnail preview every 10 frames
                if frame_idx % 10 == 0 or frame_idx == 1:
                    preview_placeholder.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Processing Frame {frame_idx}/{max_frames_to_process}")
            else:
                writer.write(enh_annotated)
                if frame_idx % 10 == 0 or frame_idx == 1:
                    preview_placeholder.image(cv2.cvtColor(enh_annotated, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Processing Frame {frame_idx}/{max_frames_to_process}")
                
            prog = int((frame_idx / max_frames_to_process) * 100)
            elapsed = time.time() - t0
            cur_fps = frame_idx / max(0.001, elapsed)
            prog_bar.progress(prog, text=f"Processed frame {frame_idx}/{max_frames_to_process} ({cur_fps:.1f} FPS)")
            
        cap.release()
        writer.release()
        total_time = time.time() - t0
        
        st.success(f"🎉 Video enhancement & tracking completed in {total_time:.1f}s ({frame_idx / total_time:.1f} FPS)!")
        
        # Display Video
        st.video(out_video_path)
        
        # Download Video Button
        with open(out_video_path, "rb") as vf:
            st.download_button(
                label="📥 Download Enhanced Video (.mp4)",
                data=vf.read(),
                file_name=f"{input_name}_enhanced.mp4",
                mime="video/mp4"
            )

else:
    # Empty state guidance
    st.info("👈 Please select an image or video from the tabs above, or choose a preset to begin enhancement and detection.")
