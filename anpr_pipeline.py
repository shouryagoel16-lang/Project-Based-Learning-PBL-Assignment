"""
anpr_pipeline.py — Core ANPR Processing Engine (Production Mode)
EasyOCR-backed pipeline for robust two-line license plate recognition.

Architecture note
-----------------
The `easyocr.Reader` is NOT created here.  It is injected by the caller
(app.py) via the `reader` parameter and kept alive by Streamlit's
`@st.cache_resource` mechanism.  This means:

  • Only ONE PyTorch MPS weight allocation exists across the entire process.
  • Switching the detection model (haar → yolo → mobilenet) never triggers a
    second `easyocr.Reader.__init__`, eliminating the macOS kernel segfaults
    that arise from multiple MPS command queues being destroyed simultaneously.
  • The class remains fully usable in standalone / headless mode — just pass an
    `easyocr.Reader` instance explicitly.
"""

from __future__ import annotations

import os
import re
import time
import logging
import sqlite3
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
try:
    import easyocr
except ImportError:
    raise ImportError("easyocr is required. Install with: pip install easyocr")

from detector_models import VehicleDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_NAME = "anpr_system.db"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_NAME)
DUPLICATE_WINDOW_SEC = 300

# 15 % padding applied to every bounding box before OCR cropping.
# Gives EasyOCR enough structural context to avoid character hallucinations
# when the detector (especially the hyper-precise YOLO model) returns
# extremely tight crops.
BBOX_PAD_RATIO = 0.15

_CLEAN_RE = re.compile(r"[^A-Z0-9]")

# Loosened Regex: Matches standard plates OR highly-fragmented chunks
# to allow for common OCR character substitutions during real-world webcam testing.
_INDIAN_PLATE_RE = re.compile(r"^[A-Z0-9]{4,10}$")


class ANPRPipeline:
    def __init__(
        self,
        model: str = "yolo",
        confidence: float = 0.40,
        db_path: str = DB_PATH,
        reader: Optional["easyocr.Reader"] = None,
    ) -> None:
        """
        Parameters
        ----------
        model      : Detection backend — 'haar' | 'yolo' | 'mobilenet'.
        confidence : Minimum detector confidence threshold.
        db_path    : Absolute path to the SQLite database file.
        reader     : A pre-initialised `easyocr.Reader` instance.
                     When running under Streamlit, pass the object returned by
                     the `@st.cache_resource`-decorated loader so that model
                     weights are only allocated once per process lifetime.
                     When running headlessly, omit this parameter and a fresh
                     reader will be created here.
        """
        self.detector = VehicleDetector(model=model, confidence=confidence)
        self.db_path = db_path
        self._last_seen: dict[str, float] = {}
        self._frame_count: int = 0

        if reader is not None:
            # Caller-injected reader — the hot path for Streamlit operation.
            self.reader = reader
            logger.info(
                "ANPRPipeline initialised [model=%s, conf=%.2f, ocr=EasyOCR(injected)]",
                model, confidence,
            )
        else:
            # Standalone / headless mode: create our own reader.
            # gpu=True picks up MPS on Apple Silicon automatically.
            self.reader = easyocr.Reader(["en"], gpu=True)
            logger.info(
                "ANPRPipeline initialised [model=%s, conf=%.2f, ocr=EasyOCR(owned)]",
                model, confidence,
            )

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _get_db_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous  = NORMAL;")
        return conn

    def insert_log(self, plate_number: str, confidence: float) -> bool:
        now = datetime.now()
        entry_date = now.strftime("%Y-%m-%d")
        entry_time = now.strftime("%H:%M:%S")
        conn = None
        try:
            conn = self._get_db_connection()
            conn.execute(
                "INSERT INTO logs (plate_number, entry_date, entry_time, confidence_score) VALUES (?, ?, ?, ?);",
                (plate_number, entry_date, entry_time, round(confidence, 3))
            )
            conn.commit()
            logger.info("Logged plate %s @ %s %s", plate_number, entry_date, entry_time)
            return True
        except sqlite3.Error as exc:
            logger.error("DB insert failed: %s", exc)
            return False
        finally:
            if conn: conn.close()

    def fetch_owner(self, plate_number: str) -> Optional[str]:
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.execute(
                "SELECT owner_name FROM owners WHERE plate_number = ?;", (plate_number,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.Error:
            return None
        finally:
            if conn: conn.close()

    def fetch_recent_logs(self, limit: int = 20) -> list[dict]:
        conn = None
        try:
            conn = self._get_db_connection()
            cursor = conn.execute(
                """
                SELECT l.log_id, l.plate_number, l.entry_date, l.entry_time, l.confidence_score,
                       COALESCE(o.owner_name, 'Unknown')      AS owner_name,
                       COALESCE(o.vehicle_model, 'Unknown')   AS vehicle_model
                FROM logs l LEFT JOIN owners o ON l.plate_number = o.plate_number
                ORDER BY l.log_id DESC LIMIT ?;
                """,
                (limit,)
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.Error:
            return []
        finally:
            if conn: conn.close()

    # ------------------------------------------------------------------
    # Duplicate suppression
    # ------------------------------------------------------------------

    def _is_duplicate(self, plate_text: str) -> bool:
        now = time.time()
        stale = [p for p, t in self._last_seen.items() if (now - t) > DUPLICATE_WINDOW_SEC]
        for p in stale:
            del self._last_seen[p]
        if plate_text in self._last_seen and (now - self._last_seen[plate_text]) < DUPLICATE_WINDOW_SEC:
            return True
        self._last_seen[plate_text] = now
        return False

    # ------------------------------------------------------------------
    # Image pre-processing  (simplified for EasyOCR)
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_plate(crop: np.ndarray) -> np.ndarray:
        """Aspect-ratio-preserving upscale to ~150 px height + grayscale.

        EasyOCR's CRNN/transformer backbone works best on lightly-processed,
        natural-looking images — aggressive thresholding and morphological ops
        that helped Tesseract actually *hurt* neural-network-based readers.
        """
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop

        # 1. Aspect-ratio-preserving upscale to a target height of 150 px
        target_h = 150
        scale = target_h / h
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2. Convert to greyscale — single-channel is sufficient for OCR and
        #    reduces noise without destroying edge information.
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        return grey

    # ------------------------------------------------------------------
    # OCR execution  (EasyOCR — instance method)
    # ------------------------------------------------------------------

    def _run_ocr(self, processed_plate: np.ndarray) -> str:
        """Run EasyOCR on a pre-processed plate crop and return cleaned text.

        `detail=0`      → returns a flat list of recognised strings (no bboxes).
        `paragraph=True` → merges multi-line results into logical paragraphs,
                           which is ideal for two-line Indian plates.
        """
        try:
            results: list[str] = self.reader.readtext(
                processed_plate, detail=0, paragraph=True
            )

            # Join all recognised fragments and clean
            raw_text = " ".join(results)
            cleaned = _CLEAN_RE.sub("", raw_text.strip().upper())

            if cleaned:
                print(f"--- DEBUG OCR: Cleaned Text -> '{cleaned}' ---")

            return cleaned
        except Exception as exc:
            logger.error("OCR Error: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Frame-level pipeline
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Process a single BGR frame and return (annotated_frame, plates).

        Bounding-box padding
        --------------------
        A 15 % expansion of the detected bbox is applied before cropping so
        that EasyOCR receives a small band of structural context around each
        plate — this eliminates character hallucinations that appear when
        overly-tight crops are fed to the CRNN.  The expansion is clamped to
        the frame boundary so NumPy slicing never goes out-of-range.
        The original (un-padded) coordinates are used for annotation only.
        """
        self._frame_count += 1
        annotated = frame.copy()
        recognised_plates: list[dict] = []
        detections = self.detector.detect(frame)

        fh, fw = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            label = det["label"]

            # ── Clamp raw bbox to frame ──────────────────────────────────
            x1 = max(0, min(fw, x1))
            y1 = max(0, min(fh, y1))
            x2 = max(0, min(fw, x2))
            y2 = max(0, min(fh, y2))

            # ── 15 % geometric padding for OCR crop ──────────────────────
            bw = x2 - x1
            bh = y2 - y1
            pad_x = int(bw * BBOX_PAD_RATIO)
            pad_y = int(bh * BBOX_PAD_RATIO)

            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(fw, x2 + pad_x)
            cy2 = min(fh, y2 + pad_y)

            # ── Crop using the padded coordinates ────────────────────────
            if label == "plate":
                plate_crop = frame[cy1:cy2, cx1:cx2]
            else:
                # For vehicle-level detections (YOLO car / MobileNet car),
                # focus on the lower 40 % of the padded box where the plate lives.
                focus_y = cy1 + int((cy2 - cy1) * 0.6)
                plate_crop = frame[focus_y:cy2, cx1:cx2]

            if plate_crop.size == 0:
                continue

            processed  = self._preprocess_plate(plate_crop)
            plate_text = self._run_ocr(processed)

            if not _INDIAN_PLATE_RE.match(plate_text):
                # Draw amber outline to indicate unvalidated detection
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                continue

            is_new = not self._is_duplicate(plate_text)
            if is_new:
                self.insert_log(plate_text, conf)

            owner_name = self.fetch_owner(plate_text) or "Unknown"
            recognised_plates.append({
                "plate_text": plate_text,
                "confidence": conf,
                "owner_name": owner_name,
                "is_new":     is_new,
            })

            colour = (0, 255, 0) if is_new else (255, 200, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            info_text = f"{plate_text} ({conf:.0%}) | {owner_name}"

            (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), colour, cv2.FILLED)
            cv2.putText(
                annotated, info_text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )

        return annotated, recognised_plates


if __name__ == "__main__":
    import sys

    model_choice = sys.argv[1] if len(sys.argv) > 1 else "haar"
    pipeline = ANPRPipeline(model=model_choice, confidence=0.35)  # standalone: creates own reader

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[✗] Cannot open camera.")
        sys.exit(1)

    print("[*] Press 'q' to quit.")
    try:
        while True:
            ret, test_frame = cap.read()
            if not ret:
                break

            annotated_frame, plates = pipeline.process_frame(test_frame)

            for p in plates:
                status = "NEW" if p["is_new"] else "dup"
                print(
                    f"  [{status}] {p['plate_text']}  "
                    f"conf={p['confidence']:.2f}  owner={p['owner_name']}"
                )

            cv2.imshow("ANPR Pipeline", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as exc:
        print(f"[✗] Pipeline error: {exc}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[✓] Camera released.")