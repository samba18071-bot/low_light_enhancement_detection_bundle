"""
Object Detection Engine using OpenCV DNN (with optional Ultralytics YOLO backend).
Provides pure OpenCV-based deep learning inference for object detection in images and videos.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import cv2
import numpy as np

# Standard 80 COCO classes
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Distinct color palette generator for consistent object colors
np.random.seed(42)
CLASS_COLORS = [
    (int(c[0]), int(c[1]), int(c[2]))
    for c in np.random.randint(50, 255, size=(len(COCO_CLASSES), 3))
]


class ObjectDetector:
    """
    OpenCV DNN-powered object detector supporting YOLOv8 ONNX models.
    Can also fall back to Ultralytics YOLO if requested.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        engine: str = "opencv_dnn",
        target_size: Tuple[int, int] = (640, 640),
    ):
        """
        Args:
            model_path: Path to .onnx model (or .pt for ultralytics).
            conf_threshold: Minimum confidence score to accept detection.
            nms_threshold: Non-Maximum Suppression IoU threshold.
            engine: 'opencv_dnn' (pure OpenCV inference) or 'ultralytics'.
            target_size: Input resolution (width, height) for neural network.
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.engine = engine.lower()
        self.target_size = target_size
        self.classes = COCO_CLASSES

        # Determine default model path if not specified
        if model_path is None:
            default_onnx = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.onnx")
            if os.path.exists(default_onnx):
                self.model_path = os.path.abspath(default_onnx)
            else:
                self.model_path = "yolov8n.onnx"
        else:
            self.model_path = model_path

        self._init_engine()

    def _init_engine(self):
        """Initializes either OpenCV DNN or Ultralytics."""
        if self.engine == "opencv_dnn":
            if not os.path.exists(self.model_path):
                # Check current directory
                if os.path.exists("yolov8n.onnx"):
                    self.model_path = "yolov8n.onnx"
                else:
                    raise FileNotFoundError(
                        f"ONNX model file not found at {self.model_path}. "
                        "Please export or provide a valid yolov8n.onnx file."
                    )
            # Load OpenCV Deep Neural Network
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            except Exception:
                pass
            self.yolo_model = None

            # Check for Open Images or custom class mapping
            oi_json = os.path.join(os.path.dirname(self.model_path), "open_images_classes.json")
            if "oiv7" in self.model_path.lower() and os.path.exists(oi_json):
                import json
                with open(oi_json) as f:
                    data = json.load(f)
                    self.classes = {int(k): v for k, v in data.items()}
        else:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(self.model_path)
                if hasattr(self.yolo_model, "names") and self.yolo_model.names:
                    self.classes = self.yolo_model.names
            except Exception as e:
                print(f"Failed to load ultralytics backend ({e}), falling back to OpenCV DNN.")
                self.engine = "opencv_dnn"
                self._init_engine()

    def _letterbox(
        self,
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Resize image to letterbox format maintaining aspect ratio.
        """
        shape = img.shape[:2]  # [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2.0
        dh = (new_shape[0] - new_unpad[1]) / 2.0

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on the input frame.

        Args:
            frame_bgr: BGR uint8 image (H, W, 3).

        Returns:
            List of detected objects: [
                {
                    'bbox': [x, y, w, h],
                    'box_xyxy': [x1, y1, x2, y2],
                    'confidence': float,
                    'class_id': int,
                    'class_name': str
                }, ...
            ]
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        if self.engine == "ultralytics" and self.yolo_model is not None:
            results = self.yolo_model(frame_bgr, conf=self.conf_threshold, iou=self.nms_threshold, verbose=False)[0]
            detections = []
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                if isinstance(self.classes, dict):
                    name = self.classes.get(cls_id, f"class_{cls_id}")
                elif cls_id < len(self.classes):
                    name = self.classes[cls_id]
                else:
                    name = f"class_{cls_id}"
                detections.append({
                    "bbox": [int(xyxy[0]), int(xyxy[1]), int(w), int(h)],
                    "box_xyxy": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": name
                })
            return detections

        # Pure OpenCV DNN Inference
        letterboxed, ratio, (dw, dh) = self._letterbox(frame_bgr, self.target_size)
        blob = cv2.dnn.blobFromImage(
            letterboxed,
            scalefactor=1.0 / 255.0,
            size=self.target_size,
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)
        output = self.net.forward()  # Shape: (1, 84, 8400)

        # Transpose to (8400, 84)
        preds = output[0].T
        boxes_candidates = preds[:, :4]
        class_scores = preds[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence threshold
        mask = max_scores >= self.conf_threshold
        filtered_boxes = boxes_candidates[mask]
        filtered_scores = max_scores[mask]
        filtered_ids = class_ids[mask]

        if len(filtered_scores) == 0:
            return []

        # Convert letterboxed [cx, cy, w, h] to original frame [x, y, w, h]
        boxes_list = []
        confidences_list = []
        ids_list = []

        for i in range(len(filtered_scores)):
            cx, cy, w, h = filtered_boxes[i]
            # Coordinates in letterboxed 640x640 space
            x1 = (cx - w / 2.0 - dw) / ratio
            y1 = (cy - h / 2.0 - dh) / ratio
            x2 = (cx + w / 2.0 - dw) / ratio
            y2 = (cy + h / 2.0 - dh) / ratio

            # Clip to image boundaries
            x1 = max(0, min(orig_w - 1, int(round(x1))))
            y1 = max(0, min(orig_h - 1, int(round(y1))))
            x2 = max(0, min(orig_w - 1, int(round(x2))))
            y2 = max(0, min(orig_h - 1, int(round(y2))))

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            boxes_list.append([x1, y1, bw, bh])
            confidences_list.append(float(filtered_scores[i]))
            ids_list.append(int(filtered_ids[i]))

        # Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes_list, confidences_list, self.conf_threshold, self.nms_threshold)

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = boxes_list[idx]
                cid = ids_list[idx]
                conf = confidences_list[idx]
                if isinstance(self.classes, dict):
                    cname = self.classes.get(cid, f"class_{cid}")
                elif cid < len(self.classes):
                    cname = self.classes[cid]
                else:
                    cname = f"class_{cid}"
                detections.append({
                    "bbox": [bx, by, bw, bh],
                    "box_xyxy": [bx, by, bx + bw, by + bh],
                    "confidence": conf,
                    "class_id": cid,
                    "class_name": cname
                })

        return detections

    def draw_detections(
        self,
        frame_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draws sleek, modern bounding boxes with colored labels and confidence on the frame.
        """
        annotated = frame_bgr.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            cid = det["class_id"]
            cname = det["class_name"]
            conf = det["confidence"]
            track_id = det.get("track_id")
            color = CLASS_COLORS[(track_id * 37) % len(CLASS_COLORS)] if track_id else CLASS_COLORS[cid % len(CLASS_COLORS)]

            # Draw rounded or semi-transparent bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            if show_labels:
                label = f"#{track_id} {cname} {conf:.0%}" if track_id else f"{cname} {conf:.1%}"
                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y, lh + 6)
                
                # Draw background badge for label readability
                cv2.rectangle(
                    annotated,
                    (x, label_y - lh - 4),
                    (x + lw + 6, label_y + baseline - 2),
                    color,
                    cv2.FILLED
                )
                cv2.putText(
                    annotated,
                    label,
                    (x + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        return annotated
