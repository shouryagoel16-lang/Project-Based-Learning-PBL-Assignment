"""
detector_models.py — Modular Vehicle / Plate Detector
======================================================
Provides a single `VehicleDetector` class whose back-end can be swapped at
init time between three algorithms:

    1. 'yolo'      — YOLOv8 Nano  (ultralytics, best accuracy)
    2. 'mobilenet'  — MobileNet SSD v2 (OpenCV DNN, balanced)
    3. 'haar'       — Haar Cascade  (OpenCV, fastest / lowest accuracy)

Usage:
    detector = VehicleDetector(model='yolo', confidence=0.4)
    boxes    = detector.detect(frame)

Each `detect()` call returns a list of dicts:
    [{"bbox": (x1, y1, x2, y2), "confidence": float, "label": str}, ...]

Edge-Device Optimisations:
  • YOLOv8n is the smallest YOLO variant (<6 MB FP16).
  • MobileNet SSD uses OpenCV's DNN module — avoids heavy framework imports.
  • Haar runs purely on CPU with integral images; zero dependencies.
  • All models resize the input to a small fixed size before inference.
"""

from __future__ import annotations

import os
import logging
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
#  Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
#  Constants — paths are relative to this file's directory
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "models")

# MobileNet SSD config / weights (user must supply or auto-download)
_MOBILENET_PROTO = os.path.join(_MODELS_DIR, "MobileNetSSD_deploy.prototxt")
_MOBILENET_MODEL = os.path.join(_MODELS_DIR, "MobileNetSSD_deploy.caffemodel")

# Haar cascade (ships with OpenCV — fall back to OpenCV data dir)
_HAAR_CASCADE_PLATE = os.path.join(
    _MODELS_DIR, "haarcascade_russian_plate_number.xml"
)

# MobileNet SSD class labels (VOC); we care about class 7 = 'car',
# class 6 = 'bus', class 14 = 'motorbike', class 15 = 'person' (ignored).
_MOBILENET_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
_VEHICLE_CLASS_IDS = {6, 7, 14}  # bus, car, motorbike


# ═══════════════════════════════════════════════════════════════════════════
#  VehicleDetector
# ═══════════════════════════════════════════════════════════════════════════
class VehicleDetector:
    """
    Unified vehicle / plate detector that delegates to one of three backends.

    Parameters
    ----------
    model : str
        One of 'yolo', 'mobilenet', 'haar'.
    confidence : float
        Minimum confidence threshold [0, 1].
    input_size : int
        Square input dimension for the neural-network models (default 320
        keeps inference fast on a Pi).
    """

    SUPPORTED_MODELS = ("yolo", "mobilenet", "haar")

    def __init__(
        self,
        model: str = "yolo",
        confidence: float = 0.40,
        input_size: int = 320,
    ) -> None:
        model = model.lower().strip()
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model}'. "
                f"Choose from {self.SUPPORTED_MODELS}."
            )

        self.model_name = model
        self.confidence = confidence
        self.input_size = input_size
        self._net: Any = None  # lazy-loaded model handle

        # Dispatch table keeps detect() branch-free at runtime.
        self._detect_fn = {
            "yolo":      self._detect_yolo,
            "mobilenet": self._detect_mobilenet,
            "haar":      self._detect_haar,
        }[model]

        # Eagerly load weights so the first frame isn't slow.
        self._load_model()

    # ───────────────────────────────────────────────────────────────────────
    #  Model loading
    # ───────────────────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """Load the selected backend into memory."""
        os.makedirs(_MODELS_DIR, exist_ok=True)

        if self.model_name == "yolo":
            self._load_yolo()
        elif self.model_name == "mobilenet":
            self._load_mobilenet()
        else:
            self._load_haar()

    def _load_yolo(self) -> None:
        """
        Load custom YOLOv8 Plate Detector via the Ultralytics library.
        """
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLOv8. "
                "Install with:  pip install ultralytics"
            ) from exc

        # Path explicitly pointing to the custom Kaggle weights
        yolo_weights = os.path.join(_BASE_DIR, "indian_plates_yolov8.pt")
        if not os.path.isfile(yolo_weights):
            raise FileNotFoundError(f"Custom model not found at {yolo_weights}. Ensure 'indian_plates_yolov8.pt' is in the YOLO folder.")

        self._net = YOLO(yolo_weights)
        # Warm-up with a blank tensor to JIT-compile the graph
        self._net.predict(
            np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8),
            verbose=False,
        )
        logger.info("Custom YOLOv8 Plate Model loaded successfully.")

    def _load_mobilenet(self) -> None:
        """
        Load MobileNet SSD v2 (Caffe) via OpenCV DNN.
        Users must place the .prototxt and .caffemodel in models/.
        """
        if not os.path.isfile(_MOBILENET_PROTO):
            raise FileNotFoundError(
                f"MobileNet prototxt not found at {_MOBILENET_PROTO}.\n"
                "Download from: https://github.com/chuanqi305/MobileNet-SSD"
            )
        if not os.path.isfile(_MOBILENET_MODEL):
            raise FileNotFoundError(
                f"MobileNet caffemodel not found at {_MOBILENET_MODEL}.\n"
                "Download from: https://github.com/chuanqi305/MobileNet-SSD"
            )

        self._net = cv2.dnn.readNetFromCaffe(_MOBILENET_PROTO, _MOBILENET_MODEL)

        # Prefer OpenCL (GPU) if available; falls back to CPU silently.
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("MobileNet SSD loaded (OpenCV DNN).")

    def _load_haar(self) -> None:
        """
        Load the Haar Cascade for Russian / Indian-style plates.
        Falls back to OpenCV's bundled data directory.
        """
        cascade_path = _HAAR_CASCADE_PLATE

        # Fallback: use the cascade bundled with OpenCV
        if not os.path.isfile(cascade_path):
            opencv_data = os.path.join(
                os.path.dirname(cv2.__file__), "data",
                "haarcascade_russian_plate_number.xml",
            )
            if os.path.isfile(opencv_data):
                cascade_path = opencv_data
            else:
                raise FileNotFoundError(
                    f"Haar cascade not found at {cascade_path} or {opencv_data}.\n"
                    "Copy the XML from OpenCV's data/ directory into models/."
                )

        self._net = cv2.CascadeClassifier(cascade_path)
        if self._net.empty():
            raise RuntimeError("Failed to load Haar cascade.")
        logger.info("Haar Cascade loaded for plate detection.")

    # ───────────────────────────────────────────────────────────────────────
    #  Public API
    # ───────────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single BGR frame.

        Returns
        -------
        list[dict]
            Each dict has keys:
              • bbox       — (x1, y1, x2, y2)  absolute pixel coords
              • confidence — float in [0, 1]
              • label      — human-readable class name
        """
        if frame is None or frame.size == 0:
            return []
        return self._detect_fn(frame)

    # ───────────────────────────────────────────────────────────────────────
    #  Back-end implementations
    # ───────────────────────────────────────────────────────────────────────
    def _detect_yolo(self, frame: np.ndarray) -> list[dict]:
        """
        YOLOv8 custom plate model inference.
        """
        # Class 0 is typical for custom single-class datasets (like plates).
        # We also leave 2, 3, 5, 7 in just in case the model is multi-class.
        allowed_classes = {0, 2, 3, 5, 7}
        results = self._net.predict(
            frame,
            imgsz=self.input_size,
            conf=self.confidence,
            verbose=False,
        )

        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id not in allowed_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "bbox":       (x1, y1, x2, y2),
                    "confidence": round(conf, 3),
                    "label":      result.names.get(cls_id, "plate"),
                })
        return detections

    def _detect_mobilenet(self, frame: np.ndarray) -> list[dict]:
        """
        MobileNet SSD inference via OpenCV DNN.
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=0.007843,
            size=(self.input_size, self.input_size),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        output = self._net.forward()  # shape: (1, 1, N, 7)

        detections: list[dict] = []
        for i in range(output.shape[2]):
            conf = float(output[0, 0, i, 2])
            if conf < self.confidence:
                continue
            cls_id = int(output[0, 0, i, 1])
            if cls_id not in _VEHICLE_CLASS_IDS:
                continue

            # De-normalise bounding box back to pixel coordinates
            x1 = max(0, int(output[0, 0, i, 3] * w))
            y1 = max(0, int(output[0, 0, i, 4] * h))
            x2 = min(w, int(output[0, 0, i, 5] * w))
            y2 = min(h, int(output[0, 0, i, 6] * h))

            detections.append({
                "bbox":       (x1, y1, x2, y2),
                "confidence": round(conf, 3),
                "label":      _MOBILENET_CLASSES[cls_id],
            })
        return detections

    def _detect_haar(self, frame: np.ndarray) -> list[dict]:
        """
        Haar Cascade plate detection with CLAHE pre-processing.

        Pre-processing pipeline
        -----------------------
        1. Grayscale conversion.
        2. CLAHE (Contrast Limited Adaptive Histogram Equalisation) with a
           clip limit of 2.0 and an 8×8 tile grid.  Unlike plain histogram
           equalisation, CLAHE adapts contrast *locally*, which neutralises
           screen glare, specular reflections from wet plates, and the uneven
           illumination typical of indoor / phone-screen captures — without
           over-amplifying noise in already-well-lit regions.
        3. Mild Gaussian blur (3×3) to suppress digital LCD pixel artifacts
           before the Haar feature computation.

        detectMultiScale tuning
        -----------------------
        • scaleFactor=1.05  — finer pyramid search; catches more size variants
          at the cost of ~15 % extra CPU per frame (acceptable on M-series).
        • minNeighbors=3    — lower threshold increases recall; the subsequent
          EasyOCR + regex validation stage acts as the precision filter.
        • minSize=(60, 15)  — slightly smaller minimum to catch distant plates.
        """
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CLAHE: locally-adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grey = clahe.apply(grey)

        # Mild blur to remove high-frequency LCD pixel noise
        grey = cv2.GaussianBlur(grey, (3, 3), 0)

        plates = self._net.detectMultiScale(
            grey,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 15),
        )

        detections: list[dict] = []
        if len(plates) == 0:
            return detections

        for (x, y, w, h) in plates:
            detections.append({
                "bbox":       (x, y, x + w, y + h),
                "confidence": 1.0,
                "label":      "plate",
            })
        return detections


if __name__ == "__main__":
    import sys

    model_choice = sys.argv[1] if len(sys.argv) > 1 else "haar"
    print(f"[*] Testing VehicleDetector with model='{model_choice}'")

    try:
        det = VehicleDetector(model=model_choice)
    except Exception as e:
        print(f"[✗] Could not load model: {e}")
        sys.exit(1)

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = det.detect(test_frame)
    print(f"[✓] Detection returned {len(results)} result(s) on blank frame.")