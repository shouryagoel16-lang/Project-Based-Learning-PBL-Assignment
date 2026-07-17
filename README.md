# 🚗 Smart ANPR Dashboard
### AI-Powered Automatic Number Plate Recognition & Logging System
> **Project-Based Learning (PBL) Assignment** · Edge AI · Computer Vision · SQLite · Streamlit

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [The ML Pipeline](#3-the-ml-pipeline)
4. [Detection Models](#4-detection-models)
5. [OCR Engine — EasyOCR](#5-ocr-engine--easyocr)
6. [Frontend — Shadcn UI on Streamlit](#6-frontend--shadcn-ui-on-streamlit)
7. [Database Layer — SQLite](#7-database-layer--sqlite)
8. [Core Optimizations (The "Secret Sauce")](#8-core-optimizations-the-secret-sauce)
9. [Known Bottlenecks & Hardware Notes](#9-known-bottlenecks--hardware-notes)
10. [Repository Structure](#10-repository-structure)
11. [Installation](#11-installation)
12. [How to Run](#12-how-to-run)
13. [Configuration Reference](#13-configuration-reference)
14. [Acknowledgements](#14-acknowledgements)

---

## 1. Project Overview

The **Smart ANPR Dashboard** is a production-grade, real-time Automatic Number Plate Recognition (ANPR) system engineered for deployment on low-cost edge hardware, including Apple Silicon MacBook Pros and single-board computers. It fuses a multi-model computer vision detection pipeline with a neural-network OCR engine and a persistent SQLite logging backend, all exposed through a custom-designed Streamlit web dashboard.

The system is capable of:

- **Real-time plate detection** from a live camera feed using three interchangeable detection backends.
- **Neural-network OCR** via EasyOCR (CRNN + transformer architecture) to extract alphanumeric plate text from noisy, two-line Indian registration plates.
- **Owner lookup** against a structured SQLite `owners` table.
- **Persistent event logging** with timestamp, confidence score, and duplicate suppression.
- **Live analytics** including per-day detection volume, OCR confidence distribution, and a dynamically generated AI performance appraisal.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND (app.py)                     │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Inline Control Center Strip (model selector, confidence, start) │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────┐ │
│  │  TAB 1: Live Capture & Logs │  │  TAB 2: System Analytics        │ │
│  │  ┌───────────┬───────────┐  │  │  • Confidence Timeline          │ │
│  │  │ Camera    │ Detection │  │  │  • Daily Detection Volume       │ │
│  │  │ Feed      │ Log List  │  │  │  • Confidence Distribution      │ │
│  │  └───────────┴───────────┘  │  │  • AI Performance Appraisal    │ │
│  └─────────────────────────────┘  └─────────────────────────────────┘ │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │   ANPRPipeline       │
                     │   (anpr_pipeline.py) │
                     └──────────┬──────────┘
                                │
              ┌─────────────────┼───────────────────┐
              │                 │                   │
   ┌──────────▼──────┐  ┌───────▼──────┐  ┌────────▼───────┐
   │  VehicleDetector│  │   EasyOCR    │  │  SQLite DB     │
   │  (detector_     │  │   Reader     │  │  (anpr_        │
   │   models.py)    │  │  (CRNN/LSTM) │  │   system.db)   │
   └─────────────────┘  └──────────────┘  └────────────────┘
          │
  ┌───────┴────────┐
  │                │
  ▼                ▼
YOLOv8     MobileNet / Haar
 Nano          Cascade
```

### Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| **Streamlit Dashboard** | `app.py` | UI rendering, control flow, session state, KPI strip, tabs, chart rendering, DB wipe |
| **ANPR Pipeline** | `anpr_pipeline.py` | Frame orchestration, pre-processing, OCR execution, bbox padding, duplicate suppression, DB writes |
| **Vehicle Detector** | `detector_models.py` | Wraps all three detection backends behind a single `.detect(frame)` interface |
| **Database** | `anpr_system.db` | SQLite WAL-mode store; `logs` + `owners` tables |

---

## 3. The ML Pipeline

Every video frame passes through the following sequential stages:

```
Raw BGR Frame
      │
      ▼
┌─────────────────────────────────────────────────┐
│  1. OBJECT DETECTION  (VehicleDetector)          │
│     • Returns: [{bbox, confidence, label}]       │
│     • Backend: YOLO / MobileNet / Haar           │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  2. BOUNDING BOX GEOMETRY (15% Padding)          │
│     • Expands detected bbox by BBOX_PAD_RATIO    │
│     • Clamped to frame boundaries (no overflow)  │
│     • Padded crop sent to OCR; original bbox     │
│       used for annotation only                   │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  3. IMAGE PRE-PROCESSING (_preprocess_plate)     │
│     • Aspect-ratio-preserving upscale → 150px h  │
│     • BGR → Grayscale (single-channel for CRNN)  │
│     • No aggressive thresholding (hurts EasyOCR) │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  4. EasyOCR INFERENCE (_run_ocr)                 │
│     • readtext(detail=0, paragraph=True)         │
│     • Merges multi-line text fragments           │
│     • Strips non-alphanumeric characters         │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  5. REGEX VALIDATION (_INDIAN_PLATE_RE)          │
│     • Pattern: ^[A-Z0-9]{4,10}$                 │
│     • Rejects hallucinated / partial strings     │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  6. DUPLICATE SUPPRESSION (_is_duplicate)        │
│     • 300-second rolling dedup window            │
│     • Prevents log flooding for stationary cars  │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  7. DATABASE WRITE + OWNER LOOKUP                │
│     • INSERT INTO logs (plate, date, time, conf) │
│     • SELECT owner_name FROM owners WHERE plate  │
└─────────────────────────────────────────────────┘
```

---

## 4. Detection Models

The system ships three interchangeable detection backends, selectable at runtime from the inline Control Center without restarting the application.

### 4.1 YOLOv8 Nano — `yolo`

| Property | Detail |
|---|---|
| **Architecture** | CSPNet backbone + decoupled detection head (Ultralytics YOLOv8n) |
| **Runtime** | Ultralytics Python SDK; hardware-accelerated via Apple MPS on M-series |
| **Inference Target** | Vehicle + plate bounding boxes |
| **Typical FPS (M-series)** | 10–22 FPS |
| **Accuracy** | Highest among the three backends |
| **Strengths** | Tightest bounding boxes; lowest false-positive rate on Indian plates |
| **Trade-offs** | Highest memory footprint; FPS degrades below 10 on CPU-only hardware |

YOLOv8 Nano is the **recommended production backend** for any deployment where hardware can sustain ≥ 8 FPS. It produces significantly more precise plate crops, directly improving EasyOCR accuracy downstream.

### 4.2 MobileNet SSD v2 — `mobilenet`

| Property | Detail |
|---|---|
| **Architecture** | MobileNetV2 backbone + SSD multi-scale detection heads |
| **Runtime** | OpenCV DNN module (Caffe model format) |
| **Inference Target** | General vehicle bounding boxes |
| **Typical FPS (M-series)** | 8–18 FPS |
| **Accuracy** | Medium |
| **Strengths** | Balanced speed/accuracy; no PyTorch dependency |
| **Trade-offs** | General vehicle detector — lower confidence threshold (≤ 0.40) recommended |

MobileNet SSD is the **best fallback** when YOLOv8 thermal-throttles on long sessions. It operates entirely through OpenCV's DNN runtime, avoiding PyTorch MPS kernel calls entirely.

### 4.3 Haar Cascade — `haar`

| Property | Detail |
|---|---|
| **Architecture** | Viola-Jones integral-image classifier with CLAHE + Gaussian blur pre-processing |
| **Runtime** | OpenCV `CascadeClassifier` (CPU only) |
| **Inference Target** | License plate rectangles directly |
| **Typical FPS (M-series)** | 15–30 FPS |
| **Accuracy** | Low–Medium |
| **Strengths** | Zero GPU dependency; extremely low CPU footprint |
| **Trade-offs** | Sensitive to glare, shadows, and extreme plate angles |

#### Haar Pre-processing Pipeline (CLAHE Revival)

The standard Haar implementation was non-functional due to reliance on a global `equalizeHist`, which amplifies glare artefacts on outdoor plates. The following pre-processing chain was engineered to revive detection reliability:

```python
# 1. Grayscale
grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 2. CLAHE — locally-adaptive contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
grey  = clahe.apply(grey)

# 3. Gaussian blur — removes high-frequency LCD pixel noise
grey = cv2.GaussianBlur(grey, (3, 3), 0)

# 4. Tuned detectMultiScale
plates = cascade.detectMultiScale(
    grey, scaleFactor=1.05, minNeighbors=3, minSize=(60, 15)
)
```

CLAHE divides the image into 8×8 tiles and performs histogram equalisation *locally*, neutralising specular reflections and uneven illumination without over-amplifying already well-lit regions. Reducing `minNeighbors` from 4 → 3 and `scaleFactor` from 1.1 → 1.05 increases recall; the downstream regex validator serves as the precision filter.

---

## 5. OCR Engine — EasyOCR

EasyOCR was selected over Tesseract as the text extraction engine for the following architectural reasons:

| Criterion | Tesseract | EasyOCR |
|---|---|---|
| Architecture | Rule-based LSTM | CRNN + CTC + Transformer |
| Pre-processing required | Heavy (thresholding, morphology) | Minimal (natural image) |
| Two-line plate handling | Poor (requires manual segmentation) | Native (`paragraph=True`) |
| Character substitution | High on noisy images | Low (learned feature space) |
| GPU acceleration | None | MPS / CUDA / CPU |

### EasyOCR Inference Configuration

```python
results = self.reader.readtext(
    processed_plate,
    detail=0,          # Returns flat string list (no bounding boxes)
    paragraph=True     # Merges multi-line fragments into logical paragraphs
)
```

`paragraph=True` is critical for Indian plates that print the state code and series on one line and the numeric identifier on a second line. EasyOCR's internal paragraph merger reconstructs the correct reading order before returning.

### Plate Validation Regex

```python
_INDIAN_PLATE_RE = re.compile(r"^[A-Z0-9]{4,10}$")
```

This loosened pattern (4–10 alphanumeric characters) accounts for common OCR character substitutions (e.g., `0` ↔ `O`, `1` ↔ `I`) while rejecting pure-noise hallucinations shorter than 4 characters.

---

## 6. Frontend — Shadcn UI on Streamlit

The dashboard UI deliberately eschews standard Streamlit widget styling and sidebar navigation in favour of a custom-injected **Shadcn UI design system** applied via `st.markdown(..., unsafe_allow_html=True)`.

### Design System Palette

| Token | Hex | Usage |
|---|---|---|
| Canvas | `#f8f9fa` | `.stApp` background |
| Card | `#ffffff` | All content containers |
| Card border | `#e2e8f0` | `1px solid` border on all cards |
| Primary text | `#0f172a` | Headings, plate text, metric values |
| Muted text | `#64748b` | Labels, sub-headings, metadata |
| CTA fill | `#18181b` | All buttons (dark fill / white text) |
| CTA hover | `#27272a` | Button hover state |

### Layout Structure

```
Page Header (title + subtitle)
    │
    ├── 4-Column KPI Strip
    │       ├── Total Logs (COUNT(*) from DB)
    │       ├── Engine Status (animated live/idle badge)
    │       ├── Active Processor (HAAR / YOLO / MOBILENET pill)
    │       └── Avg FPS (EMA, color-coded by performance tier)
    │
    └── Tabs
            ├── Tab 1: Live Capture & Logs
            │       ├── Inline Control Center (model + conf + start/stop + wipe)
            │       ├── Camera Feed (annotated BGR→RGB frame)
            │       └── Detection Log List (Shadcn "Recent Sales" avatar style)
            │
            └── Tab 2: System Performance Analytics
                    ├── Confidence Timeline (area chart)
                    ├── Daily Detection Volume (bar chart)
                    ├── Confidence Distribution (bucketed bar chart)
                    ├── Top Plates Table (groupby + agg)
                    ├── Automated Performance Appraisal (6-section AI summary)
                    └── Model Capability Reference Table
```

### Inline Control Center (No Sidebar)

The standard Streamlit sidebar is **completely suppressed** via CSS:

```css
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }
```

All pipeline controls are instead rendered as a horizontal 5-column strip at the top of Tab 1, inside a white card container. Model selection and confidence threshold are persisted through `st.session_state` to survive Streamlit's script re-run cycle without resetting mid-capture.

---

## 7. Database Layer — SQLite

The system uses a single **WAL-mode SQLite database** (`anpr_system.db`) with two tables:

```sql
-- Detection event log
CREATE TABLE logs (
    log_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number     TEXT    NOT NULL,
    entry_date       TEXT    NOT NULL,   -- YYYY-MM-DD
    entry_time       TEXT    NOT NULL,   -- HH:MM:SS
    confidence_score REAL    NOT NULL
);

-- Owner registry (pre-populated)
CREATE TABLE owners (
    plate_number    TEXT PRIMARY KEY,
    owner_name      TEXT NOT NULL,
    vehicle_model   TEXT NOT NULL,
    contact_number  TEXT
);
```

**WAL (Write-Ahead Logging) mode** is enabled on every connection:

```python
conn.execute("PRAGMA journal_mode = WAL;")
conn.execute("PRAGMA synchronous  = NORMAL;")
```

WAL allows concurrent readers (Streamlit UI queries) to proceed without blocking the write pipeline (ANPR detection loop), eliminating the read/write contention that caused UI freezes in the initial prototype.

---

## 8. Core Optimizations (The "Secret Sauce")

### 8.1 `_OfflineLogReader` — UI/DB Race Condition Fix

**Problem:** The original implementation accessed the database through the full `ANPRPipeline` object. When the Streamlit script re-ran (e.g., after clicking "Stop"), it attempted to re-instantiate `ANPRPipeline`, which triggered `easyocr.Reader.__init__()`, causing a second MPS weight allocation — resulting in a kernel-level segmentation fault.

**Solution:** A purpose-built lightweight class that owns *only* a database path and a single read query:

```python
class _OfflineLogReader:
    """Reads the SQLite log table without loading any ML model."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def fetch_recent_logs(self, limit: int = 20) -> list[dict]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL;")
        # ... query logs JOIN owners ...
```

`_OfflineLogReader` has **zero ML dependencies**. It never imports `cv2`, `easyocr`, or `torch`. The Streamlit dashboard uses it to populate the detection log list and analytics tab when the pipeline is offline — the previous `ANPRPipeline.__new__()` bypass that left `self.reader` and `self.detector` uninitialised (a latent `AttributeError`) is completely eliminated.

### 8.2 `@st.cache_resource` — Single EasyOCR Singleton

**Problem:** Every time the user changed the detection model (Haar → YOLO → MobileNet), the Streamlit script re-ran, calling `easyocr.Reader(['en'], gpu=True)`. On Apple Silicon, each call allocates a new PyTorch MPS command queue. Destroying the old queue while the new one initialises triggered a kernel segmentation fault.

**Solution:** A `@st.cache_resource`-decorated loader function that Streamlit calls **exactly once per worker process lifetime**:

```python
@st.cache_resource(show_spinner="Loading EasyOCR model weights…")
def _load_easyocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=True)
```

The cached `easyocr.Reader` instance is **injected** into each new `ANPRPipeline` via its constructor parameter:

```python
pipeline = ANPRPipeline(
    model=model_choice,
    confidence=conf_threshold,
    db_path=DB_PATH,
    reader=_load_easyocr_reader(),   # ← zero-cost cache hit after first call
)
```

This decoupling means the Streamlit frontend can switch detection models without ever touching the MPS memory stack.

### 8.3 Thread-Limiting Environment Variables

PyTorch and OpenCV both attempt to spawn their own CPU thread pools at import time. On macOS ARM, this causes memory access violations when both are active simultaneously:

```python
import os
os.environ["OMP_NUM_THREADS"]              = "1"     # OpenMP: single thread
os.environ["OPENBLAS_CORETYPE"]            = "ARMV8" # Force ARM BLAS path
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"     # Disable MSMF backend
```

These **must** be set before any library that touches PyTorch, OpenBLAS, or OpenCV is imported. They are the first executable statements in `app.py`, placed before all `import` statements.

### 8.4 Geometric Bounding Box Padding

Detection models — particularly YOLOv8 — sometimes return excessively tight bounding boxes that clip plate borders. EasyOCR's CRNN backbone requires a band of structural context around the plate to resolve boundary characters correctly.

A **15% geometric expansion** is applied to every detected bounding box before cropping for OCR:

```python
BBOX_PAD_RATIO = 0.15

pad_x = int(bw * BBOX_PAD_RATIO)
pad_y = int(bh * BBOX_PAD_RATIO)
cx1   = max(0,  x1 - pad_x)
cy1   = max(0,  y1 - pad_y)
cx2   = min(fw, x2 + pad_x)
cy2   = min(fh, y2 + pad_y)
```

The padded coordinates are used **only for the OCR crop**. The original tight bbox is retained for drawing annotation rectangles on the display frame.

### 8.5 Camera Lifecycle Safety

`stop_pipeline()` wraps `cap.release()` in a `try/except` to prevent double-release errors propagating as a C++ exception through OpenCV's Python binding:

```python
def stop_pipeline() -> None:
    cap = st.session_state.get("camera")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            logger.warning("Camera release raised — already freed?")
    st.session_state["camera"]   = None
    st.session_state["pipeline"] = None
    st.session_state["running"]  = False
    gc.collect()   # Free YOLO / MobileNet weight tensors before next init
```

`gc.collect()` is called *after* clearing the pipeline reference so that the `VehicleDetector`'s model weight tensors are eligible for collection before the next `start_pipeline()` call, reducing peak RAM usage.

---

## 9. Known Bottlenecks & Hardware Notes

### 9.1 Compilation Environment

> **This project was architected, developed, and validated exclusively on an Apple Silicon (M-Series / ARM64) MacBook Pro running macOS Sequoia.**
>
> All dependency resolution, model loading paths, MPS acceleration flags, and environment variable overrides were tuned for the ARM64 ISA. Evaluators running on Intel x86_64 or Windows machines should follow the platform-specific notes below.

### 9.2 Apple Silicon — `rpds-py` Architecture Conflict

**Symptom:**

```
mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')
```

This fatal error occurs when `rpds-py` (a Rust-compiled Streamlit/PyArrow dependency) was installed from a cached `x86_64` wheel — typically pulled in through a Rosetta 2 pip session or a cached virtual environment.

**Fix (Apple Silicon / ARM64 Mac only):**

```bash
pip install --upgrade --force-reinstall --no-cache-dir rpds-py
```

This forces `pip` to discard any cached binary wheel and compile `rpds-py` from source against the native `arm64` toolchain. Run this **before** installing other requirements.

**Verification:**

```bash
python -c "import rpds; print('rpds-py OK')"
```

### 9.3 Intel / Windows Users

Standard dependency installation works without modification:

```bash
pip install -r requirements.txt
```

On Windows, the MSMF (Microsoft Media Foundation) OpenCV backend may cause camera initialisation delays. This is pre-empted by the environment variable `OPENCV_VIDEOIO_PRIORITY_MSMF=0`, which is already set in `app.py`.

### 9.4 GPU / MPS Availability

| Platform | EasyOCR GPU Backend | Notes |
|---|---|---|
| Apple Silicon (M1/M2/M3) | MPS | `gpu=True` automatically selects MPS |
| NVIDIA (Linux / Windows) | CUDA | `gpu=True` selects CUDA if available |
| Intel CPU only | None | Set `gpu=False` in `_load_easyocr_reader()` |

If MPS is unavailable or produces NaN outputs (rare on M1 edge cases), force CPU inference:

```python
# In app.py — _load_easyocr_reader():
reader = easyocr.Reader(["en"], gpu=False)
```

### 9.5 Performance Baseline (Apple M-Series)

| Model | Avg FPS | CPU % | RAM Δ |
|---|---|---|---|
| Haar Cascade | 20–30 | 18–30% | +80 MB |
| MobileNet SSD | 10–18 | 35–55% | +320 MB |
| YOLOv8 Nano | 10–22 | 40–65% | +480 MB |

> FPS values measured at 640×480 resolution with `FRAME_SKIP=2` on an Apple M2 Pro, 16 GB unified memory.

---

## 10. Repository Structure

```
YOLO/
├── app.py                  # Streamlit dashboard (UI layer)
├── anpr_pipeline.py        # ANPR orchestration engine
├── detector_models.py      # Multi-backend vehicle detector
├── database_setup.py       # DB schema initialisation script
├── anpr_system.db          # SQLite database (auto-created)
├── requirements.txt        # Python dependency manifest
├── README.md               # This document
│
├── models/                 # Pre-trained model weights
│   ├── yolov8n.pt          # YOLOv8 Nano weights
│   ├── MobileNetSSD_deploy.prototxt
│   ├── MobileNetSSD_deploy.caffemodel
│   └── haarcascade_russian_plate_number.xml
│
└── assets/                 # Static UI assets (optional)
```

---

## 11. Installation

### Prerequisites

- Python **3.9 – 3.11** (Python 3.12 has known EasyOCR compatibility gaps)
- `pip` ≥ 23.0
- A working webcam (index 0) or USB camera

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/smart-anpr-dashboard.git
cd smart-anpr-dashboard
```

### Step 2 — Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows
```

### Step 3 — Apple Silicon Only: Fix `rpds-py` Architecture

```bash
pip install --upgrade --force-reinstall --no-cache-dir rpds-py
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**

```
streamlit>=1.35.0
easyocr>=1.7.1
opencv-python>=4.9.0
ultralytics>=8.2.0
torch>=2.2.0
pandas>=2.0.0
psutil>=5.9.0
numpy>=1.24.0
```

### Step 5 — Initialise the Database

```bash
python database_setup.py
```

This creates `anpr_system.db` with the `logs` and `owners` schema and optionally seeds the `owners` table with sample data.

---

## 12. How to Run

### Start the Dashboard

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501` in your default browser.

### Command-Line Options

| Flag | Default | Description |
|---|---|---|
| `--server.port` | `8501` | HTTP port to bind |
| `--server.address` | `localhost` | Bind address |
| `--browser.gatherUsageStats` | `true` | Disable Streamlit telemetry |

**Recommended launch command:**

```bash
streamlit run app.py \
    --server.port 8501 \
    --browser.gatherUsageStats false \
    --server.headless true
```

### Headless / Server Mode

```bash
streamlit run app.py --server.headless true
```

Navigate to the server's IP address from any device on the same network.

### Standalone Pipeline Test (no Streamlit)

```bash
python anpr_pipeline.py haar        # Test with Haar backend
python anpr_pipeline.py yolo        # Test with YOLOv8 backend
python anpr_pipeline.py mobilenet   # Test with MobileNet backend
```

Press `q` to exit the OpenCV preview window.

---

## 13. Configuration Reference

| Constant | File | Default | Description |
|---|---|---|---|
| `CAMERA_INDEX` | `app.py` | `0` | OpenCV camera device index |
| `CAMERA_WIDTH` | `app.py` | `640` | Capture resolution width (px) |
| `CAMERA_HEIGHT` | `app.py` | `480` | Capture resolution height (px) |
| `FRAME_SKIP` | `app.py` | `2` | Process every Nth frame |
| `DEFAULT_CONF` | `app.py` | `0.40` | Default confidence threshold |
| `LOG_REFRESH_ROWS` | `app.py` | `25` | Max rows shown in detection list |
| `BBOX_PAD_RATIO` | `anpr_pipeline.py` | `0.15` | Bounding box expansion (15%) |
| `DUPLICATE_WINDOW_SEC` | `anpr_pipeline.py` | `300` | Dedup window (5 minutes) |

**Reducing `FRAME_SKIP`** to `1` processes every frame — higher accuracy, higher CPU load.  
**Increasing `FRAME_SKIP`** to `3` or `4` significantly reduces CPU utilisation on constrained hardware.

---

## 14. Acknowledgements

| Component | Source |
|---|---|
| EasyOCR | [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| YOLOv8 | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| MobileNet SSD | [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) |
| Haar Cascade | OpenCV contrib — `haarcascade_russian_plate_number.xml` |
| Streamlit | [streamlit/streamlit](https://github.com/streamlit/streamlit) |
| Shadcn UI Design System | [shadcn/ui](https://ui.shadcn.com/) (design reference, CSS-injected) |
| Inter Typeface | [rsms/inter](https://rsms.me/inter/) via Google Fonts |

---

<div align="center">

**Smart ANPR Dashboard** · Built with ❤️ for PBL

*Edge AI · Computer Vision · Apple Silicon · Python*

</div>
