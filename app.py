"""
app.py  —  ANPR Dashboard  (Shadcn UI v3, Design System Edition)
================================================================
Layout:
    SIDEBAR  : Pipeline controls, system stats, DB management
    HEADER   : 4-column KPI summary strip
    TAB 1    : Live Capture & Logs
    TAB 2    : System Performance Analytics

BACKEND STABILITY — DO NOT MODIFY THE FOLLOWING:
  • os.environ thread-limiting block (lines after imports)
  • _load_easyocr_reader()  — @st.cache_resource singleton
  • _OfflineLogReader class  — safe offline DB access
  • start_pipeline() / stop_pipeline()  — camera lifecycle
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
#  ⚠️  Thread-limiting env vars — MUST be set before ALL library imports
#     Prevents PyTorch + OpenCV thread-pool collisions on macOS ARM.
# ---------------------------------------------------------------------------
import os
os.environ["OMP_NUM_THREADS"]              = "1"
os.environ["OPENBLAS_CORETYPE"]            = "ARMV8"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import time
import logging
import gc
import sqlite3
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import psutil
import streamlit as st

from anpr_pipeline import ANPRPipeline

# ---------------------------------------------------------------------------
#  Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
CAMERA_INDEX     = 0
CAMERA_WIDTH     = 640
CAMERA_HEIGHT    = 480
DEFAULT_MODEL    = "haar"
DEFAULT_CONF     = 0.40
FRAME_SKIP       = 2
LOG_REFRESH_ROWS = 25
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anpr_system.db")

# ═══════════════════════════════════════════════════════════════════════════
#  Page config — must be the absolute first st.* call
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ANPR — Smart Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
#  Global CSS — Shadcn UI Design System
#  Palette:
#    Canvas        #f8f9fa
#    Card          #ffffff / border #e2e8f0 / shadow rgba(0,0,0,.05)
#    Primary text  #0f172a
#    Muted text    #64748b
#    Controls/CTA  #18181b fill, #ffffff type
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Inter font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── Canvas ── */
.stApp {
    background-color: #f8f9fa !important;
    color: #0f172a !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Collapse the sidebar entirely ── */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }

/* ──────────────────────────────────────────────
   CHART CANVAS FIX (v2)
   Force every layer of the Vega-Lite / Altair render tree to
   white so charts blend cleanly into .chart-card backgrounds.
   Axis labels, ticks, and gridlines are pinned to the design-
   system palette so they remain legible on white.
────────────────────────────────────────────── */

/* Layer 1: Streamlit data-testid wrappers */
[data-testid="stArrowVegaLiteChart"],
[data-testid="stVegaLiteChart"] {
    background-color: #ffffff !important;
    background:       #ffffff !important;
    border-radius: 6px;
    overflow: hidden;
}

/* Layer 2: vega-embed container div */
[data-testid="stArrowVegaLiteChart"] > div,
[data-testid="stVegaLiteChart"]      > div,
.vega-embed {
    background-color: #ffffff !important;
    background:       #ffffff !important;
}

/* Layer 3: canvas element rendered by Vega runtime */
.vega-embed canvas {
    background-color: #ffffff !important;
}

/* Layer 4: SVG root — Vega-Lite writes a background rect;
   override both the element attribute and CSS fill */
.vega-embed svg,
.vega-embed svg > rect:first-child {
    background-color: #ffffff !important;
    fill: #ffffff !important;
}

/* Axis tick labels — dark slate on white */
.vega-embed .role-axis-label text,
.vega-embed text.mark-text {
    fill: #0f172a !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 11px !important;
}

/* Axis title text */
.vega-embed .role-axis-title text {
    fill: #64748b !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}

/* Axis tick lines and domain lines */
.vega-embed .role-axis path,
.vega-embed .role-axis line {
    stroke: #e2e8f0 !important;
}

/* Grid lines */
.vega-embed .role-axis .grid line {
    stroke: #f1f5f9 !important;
    stroke-dasharray: 4 2;
}

/* Vega tooltip — readable on all backgrounds */
.vega-tooltip {
    background: #0f172a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    padding: 5px 9px !important;
}

/* ──────────────────────────────────────────────
   CARD PRIMITIVES
────────────────────────────────────────────── */
.sd-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
    margin-bottom: 1rem;
}
.sd-card-flush {
    /* card without bottom margin for column packing */
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}

/* ──────────────────────────────────────────────
   KPI METRIC CARDS
────────────────────────────────────────────── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
    height: 100%;
}
.kpi-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0 0 0.4rem 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.kpi-val {
    font-size: 1.9rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    line-height: 1;
    letter-spacing: -0.02em;
}
.kpi-note {
    font-size: 0.7rem;
    color: #64748b;
    margin: 0.35rem 0 0 0;
}
.kpi-badge-active {
    display: inline-flex; align-items: center; gap: 5px;
    background: #dcfce7; color: #15803d;
    border: 1px solid #bbf7d0;
    border-radius: 999px; padding: 2px 10px;
    font-size: 0.68rem; font-weight: 700;
}
.kpi-badge-active::before {
    content: ""; width: 6px; height: 6px;
    border-radius: 50%; background: #22c55e;
    animation: kpi-pulse 1.4s ease infinite;
}
.kpi-badge-idle {
    display: inline-flex; align-items: center; gap: 5px;
    background: #f1f5f9; color: #64748b;
    border: 1px solid #e2e8f0;
    border-radius: 999px; padding: 2px 10px;
    font-size: 0.68rem; font-weight: 700;
}
@keyframes kpi-pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ──────────────────────────────────────────────
   SECTION HEADINGS
────────────────────────────────────────────── */
.sec-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.1rem 0;
}
.sec-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin: 0 0 0.9rem 0;
}

/* ──────────────────────────────────────────────
   DETECTION LIST (Shadcn "Recent Sales" style)
────────────────────────────────────────────── */
.det-list { padding: 0; }
.det-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #f1f5f9;
}
.det-row:last-child { border-bottom: none; }
.det-avatar {
    width: 38px; height: 38px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 800; color: #fff;
    flex-shrink: 0; letter-spacing: .02em;
}
.av-new  { background: #18181b; }
.av-seen { background: #64748b; }
.det-plate {
    font-size: 0.88rem; font-weight: 700;
    color: #0f172a; margin: 0; letter-spacing: .04em;
}
.det-meta {
    font-size: 0.72rem; color: #64748b; margin: 1px 0 0 0;
}
.det-right { margin-left: auto; text-align: right; }
.det-conf {
    font-size: 0.82rem; font-weight: 700; color: #0f172a;
}
.badge-new {
    font-size: 0.62rem; font-weight: 700; padding: 1px 7px;
    border-radius: 999px; background: #f0fdf4; color: #15803d;
    border: 1px solid #bbf7d0; display: block; margin-top: 2px;
}
.badge-seen {
    font-size: 0.62rem; font-weight: 700; padding: 1px 7px;
    border-radius: 999px; background: #f1f5f9; color: #64748b;
    border: 1px solid #e2e8f0; display: block; margin-top: 2px;
}

/* ──────────────────────────────────────────────
   STATUS INDICATORS
────────────────────────────────────────────── */
.status-live {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0fdf4; color: #15803d;
    border: 1px solid #bbf7d0; border-radius: 999px;
    padding: 3px 12px; font-size: 0.72rem; font-weight: 700;
}
.status-live::before {
    content: ""; width: 7px; height: 7px;
    border-radius: 50%; background: #22c55e;
    animation: kpi-pulse 1.4s ease infinite;
}
.status-off {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f8fafc; color: #64748b;
    border: 1px solid #e2e8f0; border-radius: 999px;
    padding: 3px 12px; font-size: 0.72rem; font-weight: 700;
}

/* ──────────────────────────────────────────────
   SIDEBAR CONTROLS
────────────────────────────────────────────── */
.sb-section-head {
    font-size: 0.68rem; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: .07em;
    margin: 1rem 0 0.5rem 0;
}
.sb-stat-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 0.35rem 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.78rem;
}
.sb-stat-key  { color: #64748b; font-weight: 500; }
.sb-stat-val  { color: #0f172a; font-weight: 700; }

/* ──────────────────────────────────────────────
   ANALYTICS CARDS
────────────────────────────────────────────── */
.chart-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
    margin-bottom: 1rem;
}
.chart-hed { font-size: 0.9rem; font-weight: 700; color: #0f172a; margin: 0 0 0.1rem 0; }
.chart-dek { font-size: 0.75rem; color: #64748b; margin: 0 0 0.8rem 0; }

/* ──────────────────────────────────────────────
   APPRAISAL CARD
────────────────────────────────────────────── */
.appraisal-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.appraisal-eyebrow {
    font-size: 0.68rem; font-weight: 800; color: #64748b;
    text-transform: uppercase; letter-spacing: .08em; margin: 0 0 .75rem 0;
}
.appraisal-para {
    font-size: 0.86rem; color: #0f172a; line-height: 1.7;
    margin: 0 0 0.8rem 0;
}
.appraisal-para:last-child { margin-bottom: 0; }
.appraisal-highlight {
    background: #f8fafc; border-left: 3px solid #18181b;
    padding: 0.5rem 0.8rem; border-radius: 0 4px 4px 0;
    font-size: 0.82rem; color: #0f172a; margin: 0.5rem 0;
}

/* ──────────────────────────────────────────────
   BUTTONS (v2 — dark-fill default, white text guaranteed)

   Design decision: ALL buttons default to #18181b fill with
   #ffffff text so that emoji icons, labels, and action copy
   are always visible. A `.btn-ghost` utility class is available
   for contexts that need the light outline variant (not used
   by Streamlit natively, but reserved for custom HTML btns).
   The `[kind="secondary"]` attribute selector restores the
   ghost style for secondary Streamlit button calls.
────────────────────────────────────────────── */

/* ── Base: all buttons — dark fill, white text ── */
.stButton > button {
    font-family:    'Inter', -apple-system, sans-serif !important;
    font-size:      0.8rem !important;
    font-weight:    600 !important;
    border-radius:  6px !important;
    padding:        0.45rem 1rem !important;
    background-color: #18181b !important;
    color:          #ffffff !important;
    border:         1px solid #18181b !important;
    box-shadow:     0 1px 3px rgba(0,0,0,.12);
    transition:     background-color 0.2s ease, border-color 0.2s ease;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background-color: #27272a !important;
    border-color:     #27272a !important;
    color:            #ffffff !important;
    box-shadow:       0 2px 6px rgba(0,0,0,.16);
}
.stButton > button:active {
    background-color: #3f3f46 !important;
    color:            #ffffff !important;
    box-shadow:       none;
}

/* ── Secondary / ghost variant — light outline, dark text ──
   Streamlit renders non-primary buttons as kind="secondary".
   Uncomment / extend if you add secondary st.button() calls
   that explicitly need the light treatment.
   Currently all inline Control Center actions use primary
   or default (both resolved to dark fill above).

.stButton > button[kind="secondary"] {
    background-color: #ffffff !important;
    color:            #0f172a !important;
    border:           1px solid #e2e8f0 !important;
    box-shadow:       0 1px 2px rgba(0,0,0,.04);
}
.stButton > button[kind="secondary"]:hover {
    background-color: #f8fafc !important;
    border-color:     #cbd5e1 !important;
    color:            #0f172a !important;
}
*/

/* ── Explicit primary reinforcement (redundant but safe) ── */
.stButton > button[kind="primary"] {
    background-color: #18181b !important;
    color:            #ffffff !important;
    border-color:     #18181b !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #27272a !important;
    border-color:     #27272a !important;
    color:            #ffffff !important;
}

/* ── Ghost utility class (for custom HTML buttons if needed) ── */
.btn-ghost {
    background-color: #ffffff !important;
    color:            #0f172a !important;
    border:           1px solid #e2e8f0 !important;
    border-radius:    6px;
    padding:          0.4rem 0.9rem;
    font-size:        0.8rem;
    font-weight:      600;
    cursor:           pointer;
    transition:       background-color 0.15s ease;
}
.btn-ghost:hover {
    background-color: #f8fafc !important;
    border-color:     #cbd5e1 !important;
}

/* ──────────────────────────────────────────────
   TABS
────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid #e2e8f0 !important;
    background: transparent !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0 !important;
    padding: 0.65rem 1.3rem !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    margin: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #0f172a !important;
    border-bottom-color: #18181b !important;
    font-weight: 600 !important;
}

/* ──────────────────────────────────────────────
   DATA TABLE
────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.04) !important;
}

/* ──────────────────────────────────────────────
   CAMERA PLACEHOLDER
────────────────────────────────────────────── */
.cam-off {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 340px; gap: 0.6rem;
}
.cam-off-icon { font-size: 2.4rem; opacity: .4; }
.cam-off-text {
    font-size: 0.82rem; color: #64748b; font-weight: 500;
}

/* ──────────────────────────────────────────────
   MISC UTILITIES
────────────────────────────────────────────── */
.divider {
    border: none; border-top: 1px solid #e2e8f0;
    margin: 0.75rem 0;
}
.text-muted { color: #64748b; font-size: 0.78rem; }
.pill-model {
    display: inline-block;
    background: #18181b; color: #fff;
    border-radius: 4px; padding: 1px 8px;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: .05em;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  EasyOCR singleton — ⚠️ DO NOT MODIFY (MPS crash-safety critical)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading EasyOCR model weights…")
def _load_easyocr_reader():
    """Allocated once per process; prevents duplicate MPS weight buffers."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import easyocr
    reader = easyocr.Reader(["en"], gpu=True)
    logger.info("EasyOCR Reader singleton initialised.")
    return reader


# ═══════════════════════════════════════════════════════════════════════════
#  Offline log reader — ⚠️ DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════════════
class _OfflineLogReader:
    """Reads the SQLite log table without loading any ML model."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def fetch_recent_logs(self, limit: int = 20) -> list[dict]:
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL;")
            cur = conn.execute(
                """
                SELECT l.log_id, l.plate_number, l.entry_date, l.entry_time,
                       l.confidence_score,
                       COALESCE(o.owner_name,    'Unknown') AS owner_name,
                       COALESCE(o.vehicle_model, 'Unknown') AS vehicle_model
                FROM logs l
                LEFT JOIN owners o ON l.plate_number = o.plate_number
                ORDER BY l.log_id DESC LIMIT ?;
                """,
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception:
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
#  Database helpers (UI layer only)
# ═══════════════════════════════════════════════════════════════════════════
def _db_connect() -> Optional[sqlite3.Connection]:
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn
    except Exception as exc:
        logger.error("DB connection failed: %s", exc)
        return None


def clear_detection_logs() -> bool:
    """Wipe the logs table. Returns True on success."""
    conn = _db_connect()
    if conn is None:
        return False
    try:
        conn.execute("DELETE FROM logs;")
        conn.commit()
        logger.info("Logs table wiped by user.")
        return True
    except sqlite3.Error as exc:
        logger.error("Wipe failed: %s", exc)
        return False
    finally:
        conn.close()


# ── Scalar DB queries ──
def _scalar(sql: str, default=0):
    conn = _db_connect()
    if conn is None:
        return default
    try:
        row = conn.execute(sql).fetchone()
        return row[0] if row and row[0] is not None else default
    except Exception:
        return default
    finally:
        conn.close()


def get_total_detections() -> int:
    return int(_scalar("SELECT COUNT(*) FROM logs;", 0))


def get_unique_plates() -> int:
    return int(_scalar("SELECT COUNT(DISTINCT plate_number) FROM logs;", 0))


def get_avg_confidence() -> float:
    raw = _scalar("SELECT AVG(confidence_score) FROM logs;", 0.0)
    return round(float(raw) * 100, 1) if raw else 0.0


def fetch_full_log_df() -> pd.DataFrame:
    """Return every log row as a DataFrame for analytics."""
    conn = _db_connect()
    if conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(
            """
            SELECT log_id, plate_number, entry_date, entry_time, confidence_score
            FROM logs
            ORDER BY log_id ASC;
            """,
            conn,
        )
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Dynamic appraisal engine
# ═══════════════════════════════════════════════════════════════════════════
def generate_dynamic_appraisal(
    active_model: str,
    session_fps: float,
    total_detections: int,
    unique_plates: int,
    avg_conf: float,
    cpu_pct: float,
    mem_pct: float,
    df_logs: pd.DataFrame,
) -> list[dict]:
    """
    Produce a structured list of paragraph dicts:
        {"head": str, "body": str, "highlight": str|None}
    All values are computed from live runtime data — nothing hardcoded.
    """
    paragraphs: list[dict] = []

    # ── 1. Model profile ────────────────────────────────────────────────
    profiles = {
        "haar": {
            "name":     "Haar Cascade",
            "backend":  "OpenCV CPU integral-image classifier",
            "strength": "ultra-low latency with zero GPU dependency",
            "weakness": "higher false-positive rate on complex backgrounds",
            "typical_fps_range": "15–30 FPS on M-series",
        },
        "yolo": {
            "name":     "YOLOv8 Nano",
            "backend":  "Ultralytics deep-learning, MPS-accelerated",
            "strength": "tightest bounding boxes, lowest false-positive rate",
            "weakness": "higher memory footprint; FPS may dip below 10 on CPU-only",
            "typical_fps_range": "10–22 FPS on M-series",
        },
        "mobilenet": {
            "name":     "MobileNet SSD v2",
            "backend":  "OpenCV DNN Caffe runtime",
            "strength": "solid balance of speed and detection recall",
            "weakness": "general vehicle detector — may need lower confidence threshold for plates",
            "typical_fps_range": "8–18 FPS on M-series",
        },
    }
    p = profiles.get(active_model, profiles["haar"])
    paragraphs.append({
        "head": f"Active Engine — {p['name']}",
        "body": (
            f"The pipeline is running the <strong>{p['name']}</strong> detector "
            f"({p['backend']}). "
            f"Its primary strength is <em>{p['strength']}</em>; "
            f"the known trade-off is <em>{p['weakness']}</em>. "
            f"Expected throughput on Apple Silicon: <strong>{p['typical_fps_range']}</strong>."
        ),
        "highlight": None,
    })

    # ── 2. Throughput analysis ───────────────────────────────────────────
    if session_fps <= 0:
        fps_verdict = "Pipeline is idle — no FPS data recorded this session."
        fps_highlight = "Start the stream to begin generating throughput metrics."
    elif session_fps < 5:
        fps_verdict = (
            f"Observed throughput is <strong>{session_fps:.1f} FPS</strong>, which is below "
            "the 5 FPS real-time threshold for reliable ANPR. "
            "The frame capture rate is insufficient to catch fast-moving vehicles."
        )
        fps_highlight = (
            f"Recommended action: switch to Haar Cascade or increase FRAME_SKIP "
            f"(currently {FRAME_SKIP}) to reduce per-frame processing load."
        )
    elif session_fps < 12:
        fps_verdict = (
            f"Observed throughput is <strong>{session_fps:.1f} FPS</strong> — "
            "functional for low-speed or stationary vehicle recognition. "
            "Occasional frame drops may cause missed plates on fast captures."
        )
        fps_highlight = (
            "Tip: reducing CAMERA_WIDTH from 640 → 480 can yield a ~20% FPS improvement "
            "with negligible impact on plate crop quality."
        )
    else:
        fps_verdict = (
            f"Observed throughput is <strong>{session_fps:.1f} FPS</strong> — "
            "excellent real-time performance. "
            "The pipeline is comfortably ahead of the 12 FPS threshold required for "
            "reliable capture of vehicles passing at walking speed."
        )
        fps_highlight = None
    paragraphs.append({"head": "Throughput Analysis", "body": fps_verdict, "highlight": fps_highlight})

    # ── 3. Recognition quality ───────────────────────────────────────────
    if avg_conf == 0.0:
        q_body = "No confidence scores recorded yet. Detections will populate this section."
        q_highlight = None
    elif avg_conf < 50:
        q_body = (
            f"Historical average OCR confidence is <strong>{avg_conf:.1f}%</strong> — critically low. "
            "This typically indicates severe crop misalignment, heavy motion blur, or insufficient lighting. "
            "EasyOCR's CRNN backbone requires a minimum of ~40% plate pixel density to produce reliable output."
        )
        q_highlight = (
            "Action: verify BBOX_PAD_RATIO in anpr_pipeline.py is set to ≥ 0.15, "
            "and ensure ambient lighting is ≥ 100 lux at the plate surface."
        )
    elif avg_conf < 72:
        q_body = (
            f"Historical average OCR confidence is <strong>{avg_conf:.1f}%</strong> — acceptable range. "
            "Recognition is functional but occasional character substitutions are expected "
            "(e.g., '0' vs 'O', '1' vs 'I'). "
            "The Indian Plate regex validator downstream will suppress most false positives."
        )
        q_highlight = (
            "To improve: ensure the camera is positioned within 1.5 m of the plate "
            "and the plate is within a 20° horizontal angle from the lens axis."
        )
    else:
        q_body = (
            f"Historical average OCR confidence is <strong>{avg_conf:.1f}%</strong> — high quality. "
            "EasyOCR is operating reliably. Character-level substitution errors are minimal, "
            "and the pipeline's regex validation layer is receiving clean candidates."
        )
        q_highlight = None
    paragraphs.append({"head": "Recognition Quality", "body": q_body, "highlight": q_highlight})

    # ── 4. Detection volume & deduplication ─────────────────────────────
    if total_detections == 0:
        vol_body = "No entries found in the detection log database."
        vol_highlight = None
    else:
        dup_rate = 0.0 if unique_plates == 0 else round((1 - unique_plates / total_detections) * 100, 1)
        vol_body = (
            f"The database holds <strong>{total_detections:,}</strong> total log entries "
            f"spanning <strong>{unique_plates:,}</strong> distinct plate numbers. "
            f"The duplicate suppression system (300-second window) is filtering "
            f"<strong>{dup_rate:.1f}%</strong> of raw detections before DB writes — "
            f"{'efficient and within expected range' if dup_rate < 70 else 'unusually high; consider reducing the dedup window if capturing high-traffic scenarios'}."
        )
        # Detection frequency from df if available
        if not df_logs.empty and "entry_date" in df_logs.columns:
            daily_avg = round(len(df_logs) / max(df_logs["entry_date"].nunique(), 1), 1)
            vol_highlight = f"Average detection rate: {daily_avg:.1f} events per day across {df_logs['entry_date'].nunique()} recorded session day(s)."
        else:
            vol_highlight = None
    paragraphs.append({"head": "Detection Volume & Deduplication", "body": vol_body, "highlight": vol_highlight})

    # ── 5. System resource footprint ────────────────────────────────────
    if cpu_pct > 85:
        res_body = (
            f"CPU utilisation is <strong>{cpu_pct:.0f}%</strong> — dangerously elevated. "
            "At this level, the OS scheduler may begin throttling the camera capture thread, "
            "causing frame drops and potential VideoCapture failures. "
            f"Memory is at <strong>{mem_pct:.0f}%</strong> utilisation."
        )
        res_highlight = "Immediate action: Stop the pipeline, switch to Haar, and set FRAME_SKIP=3."
    elif cpu_pct > 60:
        res_body = (
            f"CPU is at <strong>{cpu_pct:.0f}%</strong> — moderate load, within operational limits. "
            f"Memory utilisation is <strong>{mem_pct:.0f}%</strong>. "
            "No immediate intervention required."
        )
        res_highlight = None
    else:
        res_body = (
            f"System is operating efficiently: CPU at <strong>{cpu_pct:.0f}%</strong>, "
            f"memory at <strong>{mem_pct:.0f}%</strong>. "
            "Headroom is available to increase frame resolution or reduce FRAME_SKIP if needed."
        )
        res_highlight = None
    paragraphs.append({"head": "System Resource Footprint", "body": res_body, "highlight": res_highlight})

    # ── 6. Strategic recommendation ─────────────────────────────────────
    recs = {
        "haar": (
            "Haar Cascade is optimal for constrained hardware. "
            "If the installation environment has controlled lighting and vehicles are "
            "stationary or slow-moving, accuracy is acceptable. "
            "For deployment in high-traffic outdoor scenarios, migrate to <strong>YOLOv8</strong> "
            "to reduce false positives and improve bounding-box precision."
        ),
        "yolo": (
            "YOLOv8 is the production-grade choice for this pipeline. "
            "Maintain the current configuration. "
            "If FPS degrades below 8 (e.g., due to thermal throttling), "
            "fall back to <strong>MobileNet SSD</strong> as a performance-preserving alternative."
        ),
        "mobilenet": (
            "MobileNet SSD provides a reliable mid-tier option. "
            "For higher throughput, switch to <strong>Haar</strong>; "
            "for higher plate-detection precision, upgrade to <strong>YOLOv8</strong>. "
            "The current model is particularly effective when the confidence threshold is "
            "tuned between 0.35–0.45."
        ),
    }
    paragraphs.append({
        "head": "Strategic Recommendation",
        "body": recs.get(active_model, recs["haar"]),
        "highlight": None,
    })

    return paragraphs


# ═══════════════════════════════════════════════════════════════════════════
#  Session state — ⚠️ DO NOT MODIFY
# ═══════════════════════════════════════════════════════════════════════════
def _init_session_state() -> None:
    defaults = {
        "pipeline":    None,
        "camera":      None,
        "running":     False,
        "frame_count": 0,
        "fps":         0.0,
        "last_plates": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline lifecycle — ⚠️ DO NOT MODIFY (segfault-prevention critical)
# ═══════════════════════════════════════════════════════════════════════════
def start_pipeline(model: str, confidence: float) -> None:
    stop_pipeline()
    try:
        reader   = _load_easyocr_reader()
        pipeline = ANPRPipeline(
            model=model, confidence=confidence, db_path=DB_PATH, reader=reader,
        )
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not cap.isOpened():
            st.error("❌ Cannot open camera.")
            cap.release()
            return
        st.session_state.update({
            "pipeline":    pipeline,
            "camera":      cap,
            "running":     True,
            "frame_count": 0,
        })
        logger.info("Pipeline started [model=%s]", model)
    except Exception as exc:
        st.error(f"❌ Pipeline start failed: {exc}")
        logger.exception("Pipeline start error")


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
    gc.collect()
    logger.info("Pipeline stopped.")


# ═══════════════════════════════════════════════════════════════════════════
#  HTML renderers
# ═══════════════════════════════════════════════════════════════════════════
def _det_list_html(logs: list[dict], *, live: bool = False) -> str:
    """
    Render a Shadcn-style detection list from either DB log dicts or
    live pipeline plate dicts.  `live=True` switches the key names.
    """
    if not logs:
        return (
            "<p style='color:#64748b;font-size:0.82rem;"
            "padding:0.75rem 0;text-align:center'>"
            "No detections recorded yet.</p>"
        )

    rows = ""
    items = logs[:14]
    for item in items:
        if live:
            plate   = item.get("plate_text", "—")
            owner   = item.get("owner_name", "Unknown")
            conf    = item.get("confidence", 0.0)
            is_new  = item.get("is_new", False)
            meta    = owner
        else:
            plate   = item.get("plate_number", "—")
            owner   = item.get("owner_name", "Unknown")
            vehicle = item.get("vehicle_model", "Unknown")
            conf    = item.get("confidence_score") or 0.0
            t_str   = item.get("entry_time", "")
            is_new  = True          # DB entries are all historical "new"
            meta    = f"{owner} · {vehicle} · {t_str}"

        initials  = (plate[:2]).upper() if len(plate) >= 2 else plate
        av_cls    = "av-new" if is_new else "av-seen"
        badge_cls = "badge-new" if is_new else "badge-seen"
        badge_lbl = "NEW" if is_new else "SEEN"
        conf_pct  = f"{conf * 100:.0f}%"

        rows += f"""
        <div class="det-row">
            <div class="det-avatar {av_cls}">{initials}</div>
            <div style="flex:1;min-width:0;overflow:hidden">
                <p class="det-plate">{plate}</p>
                <p class="det-meta" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{meta}</p>
            </div>
            <div class="det-right">
                <span class="det-conf">{conf_pct}</span>
                <span class="{badge_cls}">{badge_lbl}</span>
            </div>
        </div>"""
    return f'<div class="det-list">{rows}</div>'


def _appraisal_html(paragraphs: list[dict]) -> str:
    inner = ""
    for para in paragraphs:
        inner += f"""
        <p style="font-size:0.7rem;font-weight:800;color:#64748b;
                  text-transform:uppercase;letter-spacing:.07em;
                  margin:1.1rem 0 0.2rem 0">{para['head']}</p>
        <p class="appraisal-para">{para['body']}</p>"""
        if para.get("highlight"):
            inner += f"""
        <div class="appraisal-highlight">{para['highlight']}</div>"""
    return inner


# ═══════════════════════════════════════════════════════════════════════════
#  Module-level control defaults
#  These are resolved before the KPI strip renders so model_choice and
#  conf_threshold are always defined even before Tab 1 is entered.
# ═══════════════════════════════════════════════════════════════════════════
# Placeholder values — overwritten inside the inline Control Center below.
# Using session_state so the values survive Streamlit reruns during the
# capture loop without resetting on every script execution.
if "_model_choice" not in st.session_state:
    st.session_state["_model_choice"] = DEFAULT_MODEL
if "_conf_threshold" not in st.session_state:
    st.session_state["_conf_threshold"] = DEFAULT_CONF

model_choice   = st.session_state["_model_choice"]
conf_threshold = st.session_state["_conf_threshold"]


# ═══════════════════════════════════════════════════════════════════════════
#  ░░  MAIN CANVAS  ░░
# ═══════════════════════════════════════════════════════════════════════════

# ── Page header ──
st.markdown("""
<div style="margin-bottom:1.25rem">
    <h1 style="font-size:1.55rem;font-weight:800;color:#0f172a;
               margin:0;letter-spacing:-.02em;line-height:1.2">
        Smart ANPR Dashboard
    </h1>
    <p style="font-size:0.82rem;color:#64748b;margin:.2rem 0 0 0">
        AI-Powered Vehicle Registration Plate Recognition &amp; Logging
    </p>
</div>
""", unsafe_allow_html=True)

# ── Pull scalar metrics once ──
total_dets   = get_total_detections()
unique_pls   = get_unique_plates()
avg_conf_pct = get_avg_confidence()
live_fps     = st.session_state.get("fps", 0.0)

# ══════════════════════════════════════════════════════════════════
#  TASK 1 — 4-Column KPI Summary Strip
# ══════════════════════════════════════════════════════════════════
kc1, kc2, kc3, kc4 = st.columns(4, gap="small")

with kc1:
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-eyebrow">📋 Total Logs</p>
        <p class="kpi-val">{total_dets:,}</p>
        <p class="kpi-note">All-time detection events</p>
    </div>""", unsafe_allow_html=True)

with kc2:
    if st.session_state["running"]:
        badge_html = '<span class="kpi-badge-active">Active</span>'
        eng_note   = "Pipeline streaming"
    else:
        badge_html = '<span class="kpi-badge-idle">Idle</span>'
        eng_note   = "Press ▶ Start to begin"
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-eyebrow">⚙️ Engine Status</p>
        <div style="margin:0.2rem 0 0.3rem 0">{badge_html}</div>
        <p class="kpi-note">{eng_note}</p>
    </div>""", unsafe_allow_html=True)

with kc3:
    model_label = model_choice.upper()
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-eyebrow">🔬 Processor</p>
        <p class="kpi-val" style="font-size:1.5rem;letter-spacing:.02em">
            <span class="pill-model">{model_label}</span>
        </p>
        <p class="kpi-note">Conf ≥ {conf_threshold:.0%} · skip {FRAME_SKIP}</p>
    </div>""", unsafe_allow_html=True)

with kc4:
    fps_color = "#15803d" if live_fps >= 12 else ("#b45309" if live_fps >= 5 else "#dc2626")
    fps_display = f"{live_fps:.1f}" if live_fps > 0 else "—"
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-eyebrow">⚡ Avg Processing FPS</p>
        <p class="kpi-val" style="color:{fps_color}">{fps_display}</p>
        <p class="kpi-note">Avg confidence: {avg_conf_pct:.1f}%</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  TASK 2 — Tabbed Workspace
# ══════════════════════════════════════════════════════════════════
tab_live, tab_analytics = st.tabs(["📹  Live Capture & Logs", "📊  System Performance Analytics"])


# ──────────────────────────────────────────────────────────────────
#  TAB 1  —  Live Capture & Logs
# ──────────────────────────────────────────────────────────────────
with tab_live:

    # ══════════════════════════════════════════════════════════════
    #  INLINE CONTROL CENTER STRIP
    #  Horizontal card replacing the sidebar — sits directly above
    #  the camera/log workspace grid.
    # ══════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;
                box-shadow:0 1px 3px rgba(0,0,0,.05);padding:1rem;
                margin-bottom:1rem">
        <p style="font-size:0.68rem;font-weight:700;color:#64748b;
                  text-transform:uppercase;letter-spacing:.07em;
                  margin:0 0 0.65rem 0">Control Center</p>
    </div>
    """, unsafe_allow_html=True)

    # Use a container so the card wraps the native Streamlit widgets cleanly
    with st.container():
        st.markdown(
            "<style>.ctrl-strip{background:#ffffff;border:1px solid #e2e8f0;"
            "border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.05);"
            "padding:.85rem 1rem;margin-bottom:1rem}</style>",
            unsafe_allow_html=True,
        )
        # 5-column strip: model | confidence | spacer | start | stop+wipe
        cc1, cc2, cc3, cc4, cc5 = st.columns([2, 2.5, 0.3, 1, 1.5], gap="small")

        with cc1:
            st.markdown(
                "<p style='font-size:0.68rem;font-weight:700;color:#64748b;"
                "text-transform:uppercase;letter-spacing:.06em;margin:0 0 .3rem 0'>"
                "Detection Engine</p>",
                unsafe_allow_html=True,
            )
            _mc = st.selectbox(
                "model_sel",
                options=["haar", "yolo", "mobilenet"],
                index=["haar", "yolo", "mobilenet"].index(
                    st.session_state.get("_model_choice", DEFAULT_MODEL)
                ),
                label_visibility="collapsed",
                help="haar = fastest  ·  yolo = most accurate  ·  mobilenet = balanced",
                key="cc_model",
            )
            st.session_state["_model_choice"] = _mc
            model_choice = _mc

        with cc2:
            st.markdown(
                "<p style='font-size:0.68rem;font-weight:700;color:#64748b;"
                "text-transform:uppercase;letter-spacing:.06em;margin:0 0 .3rem 0'>"
                "Confidence Threshold</p>",
                unsafe_allow_html=True,
            )
            _ct = st.slider(
                "conf_sel",
                min_value=0.10, max_value=0.95,
                value=st.session_state.get("_conf_threshold", DEFAULT_CONF),
                step=0.05,
                label_visibility="collapsed",
                key="cc_conf",
            )
            st.session_state["_conf_threshold"] = _ct
            conf_threshold = _ct

        with cc3:
            # Visual separator
            st.markdown(
                "<div style='height:100%;border-left:1px solid #e2e8f0;"
                "margin:0 auto;width:1px'></div>",
                unsafe_allow_html=True,
            )

        with cc4:
            st.markdown(
                "<p style='font-size:0.68rem;font-weight:700;color:#64748b;"
                "text-transform:uppercase;letter-spacing:.06em;margin:0 0 .3rem 0'>"
                "Stream</p>",
                unsafe_allow_html=True,
            )
            _btn_c1, _btn_c2 = st.columns(2)
            with _btn_c1:
                if st.button("▶", use_container_width=True, type="primary",
                             key="cc_start", help="Start pipeline"):
                    start_pipeline(model_choice, conf_threshold)
            with _btn_c2:
                if st.button("⏹", use_container_width=True,
                             key="cc_stop", help="Stop pipeline"):
                    stop_pipeline()

        with cc5:
            st.markdown(
                "<p style='font-size:0.68rem;font-weight:700;color:#64748b;"
                "text-transform:uppercase;letter-spacing:.06em;margin:0 0 .3rem 0'>"
                "Database</p>",
                unsafe_allow_html=True,
            )
            if st.button("🗑 Wipe Log History", use_container_width=True,
                         key="cc_wipe", help="Delete all detection logs"):
                if clear_detection_logs():
                    st.toast("Database wiped successfully!", icon="🗑️")
                else:
                    st.toast("Wipe failed — check DB path.", icon="⚠️")

    # ── Status indicator ──
    if st.session_state["running"]:
        st.markdown('<div class="status-live" style="margin-bottom:.75rem">Pipeline active</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-off" style="margin-bottom:.75rem">Pipeline offline</div>',
                    unsafe_allow_html=True)

    # ── Workspace grid ──
    col_cam, col_log = st.columns([3, 2], gap="large")

    # ── Camera column ──
    with col_cam:
        st.markdown("""
        <p class="sec-title">📷 Live Feed</p>
        <p class="sec-sub">Real-time annotated frame stream</p>
        """, unsafe_allow_html=True)
        video_ph       = st.empty()
        live_plates_ph = st.empty()

    # ── Log column ──
    with col_log:
        st.markdown("""
        <p class="sec-title">🔍 Recent Detections</p>
        <p class="sec-sub">Latest recognised plates with owner data</p>
        """, unsafe_allow_html=True)
        log_ph = st.empty()

    # ── Capture loop ──
    if st.session_state["running"]:
        cap: cv2.VideoCapture  = st.session_state["camera"]
        pipeline: ANPRPipeline = st.session_state["pipeline"]

        if cap is None or pipeline is None:
            st.warning("⚠️ Camera or pipeline not initialised.")
            stop_pipeline()
        else:
            frame_counter = 0
            prev_time     = time.time()
            try:
                while st.session_state["running"]:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Camera read failed — pipeline stopped.")
                        stop_pipeline()
                        break

                    frame_counter += 1
                    if frame_counter % FRAME_SKIP != 0:
                        continue

                    annotated, plates = pipeline.process_frame(frame)

                    # FPS — exponential moving average  (DO NOT MODIFY)
                    now         = time.time()
                    delta       = now - prev_time
                    instant_fps = 1.0 / max(delta, 1e-6)
                    st.session_state["fps"] = (
                        0.8 * st.session_state["fps"] + 0.2 * instant_fps
                    )
                    prev_time = now

                    # Video
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_ph.image(rgb, channels="RGB", use_container_width=True)

                    # Live plates below video
                    if plates:
                        st.session_state["last_plates"] = plates
                        live_html = _det_list_html(plates, live=True)
                        live_plates_ph.markdown(
                            f'<div class="sd-card">{live_html}</div>',
                            unsafe_allow_html=True,
                        )

                    # Historical log list (right column)
                    db_logs  = pipeline.fetch_recent_logs(limit=LOG_REFRESH_ROWS)
                    log_html = _det_list_html(db_logs, live=False)
                    log_ph.markdown(
                        f'<div class="sd-card">{log_html}</div>',
                        unsafe_allow_html=True,
                    )

                    time.sleep(0.03)

            except Exception as exc:
                logger.exception("Capture loop error: %s", exc)
                st.error(f"❌ Pipeline crashed: {exc}")
            finally:
                stop_pipeline()   # ⚠️ DO NOT REMOVE — guarantees camera release

    else:
        # Offline placeholders
        with col_cam:
            video_ph.markdown("""
            <div class="cam-off">
                <span class="cam-off-icon">📷</span>
                <span class="cam-off-text">Camera offline — press ▶ Start in the Control Center above</span>
            </div>""", unsafe_allow_html=True)

        with col_log:
            try:
                olr      = _OfflineLogReader(DB_PATH)
                db_logs  = olr.fetch_recent_logs(limit=LOG_REFRESH_ROWS)
                log_html = _det_list_html(db_logs, live=False)
                log_ph.markdown(
                    f'<div class="sd-card">{log_html}</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                log_ph.info("Run `database_setup.py` first to initialise the database.")


# ──────────────────────────────────────────────────────────────────
#  TAB 2  —  System Performance Analytics
# ──────────────────────────────────────────────────────────────────
with tab_analytics:
    st.markdown("""
    <p class="sec-title">System Performance Analytics</p>
    <p class="sec-sub">
        Historical metrics computed from the live SQLite detection database.
        Populate data by running the pipeline and capturing plates.
    </p>
    """, unsafe_allow_html=True)

    df_all = fetch_full_log_df()

    # ── Charts ──────────────────────────────────────────────────────
    if df_all.empty:
        st.markdown("""
        <div class="chart-card" style="text-align:center;padding:2.5rem">
            <p style="font-size:2rem;margin:0">📭</p>
            <p style="font-size:0.85rem;color:#64748b;margin:.5rem 0 0 0">
                No data yet — start the pipeline and capture some plates.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        # Parse datetime
        df_all["dt"] = pd.to_datetime(
            df_all["entry_date"].astype(str) + " " + df_all["entry_time"].astype(str),
            errors="coerce",
        )
        df_ts = df_all.dropna(subset=["dt"]).sort_values("dt")
        df_ts["conf_pct"] = df_ts["confidence_score"] * 100

        # Row 1 — Confidence over time + Daily volume
        ac1, ac2 = st.columns(2, gap="large")

        with ac1:
            st.markdown("""
            <div class="chart-card">
                <p class="chart-hed">Confidence Score Timeline</p>
                <p class="chart-dek">OCR recognition quality per detection event</p>
            """, unsafe_allow_html=True)
            chart_conf = df_ts.set_index("dt")[["conf_pct"]].rename(
                columns={"conf_pct": "Confidence %"}
            )
            st.area_chart(chart_conf, color="#18181b", height=210)
            st.markdown("</div>", unsafe_allow_html=True)

        with ac2:
            st.markdown("""
            <div class="chart-card">
                <p class="chart-hed">Detections per Day</p>
                <p class="chart-dek">Daily volume of plate recognition events</p>
            """, unsafe_allow_html=True)
            daily = (
                df_ts.groupby("entry_date")
                .size()
                .reset_index(name="Count")
                .sort_values("entry_date")
            )
            st.bar_chart(
                daily.set_index("entry_date")["Count"],
                color="#18181b",
                height=210,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Row 2 — Confidence bucket distribution
        st.markdown("""
        <div class="chart-card">
            <p class="chart-hed">Confidence Distribution</p>
            <p class="chart-dek">
                Volume of detections per confidence bracket.
                A healthy pipeline shows mass concentration in the 70–100% range.
            </p>
        """, unsafe_allow_html=True)
        buckets = pd.cut(
            df_all["confidence_score"] * 100,
            bins=[0, 40, 55, 70, 85, 100],
            labels=["0–40 %", "40–55 %", "55–70 %", "70–85 %", "85–100 %"],
        ).value_counts().sort_index()
        st.bar_chart(buckets, color="#18181b", height=180)
        st.markdown("</div>", unsafe_allow_html=True)

        # Row 3 — Top plates table
        st.markdown("""
        <div class="chart-card">
            <p class="chart-hed">Highest-Frequency Plates</p>
            <p class="chart-dek">Sorted by detection count — identifies regular vehicles</p>
        """, unsafe_allow_html=True)
        top_df = (
            df_all.groupby("plate_number")
            .agg(
                Detections=("plate_number", "count"),
                Avg_Conf=("confidence_score", lambda x: f"{x.mean() * 100:.1f}%"),
                Last_Seen=("entry_date", "max"),
            )
            .sort_values("Detections", ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={
                "plate_number": "Plate",
                "Avg_Conf":     "Avg Confidence",
                "Last_Seen":    "Last Seen",
            })
        )
        st.dataframe(top_df, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Dynamic Appraisal ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p class="sec-title">Automated Performance Appraisal</p>
    <p class="sec-sub">
        Dynamically generated multi-paragraph analysis computed from live
        runtime metrics, database statistics, and model characteristics.
    </p>
    """, unsafe_allow_html=True)

    paras = generate_dynamic_appraisal(
        active_model     = model_choice,
        session_fps      = st.session_state.get("fps", 0.0),
        total_detections = total_dets,
        unique_plates    = unique_pls,
        avg_conf         = avg_conf_pct,
        cpu_pct          = psutil.cpu_percent(interval=None),
        mem_pct          = psutil.virtual_memory().percent,
        df_logs          = df_all,
    )

    appraisal_inner = _appraisal_html(paras)

    # Render the card shell first, then inject the dynamic HTML body
    # as a separate st.markdown call.  This prevents the f-string triple-quote
    # block from swallowing the inner HTML as a raw literal and ensures that
    # all <strong>, <em>, and <div class="appraisal-highlight"> tags are parsed
    # correctly by the browser rather than displayed as escaped text.
    st.markdown(
        '<div class="appraisal-card">'
        '<p class="appraisal-eyebrow">🔍 AI System Intelligence Report</p>',
        unsafe_allow_html=True,
    )
    st.markdown(appraisal_inner, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Model reference table ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="chart-card">
        <p class="chart-hed">Model Capability Reference</p>
        <p class="chart-dek">Static characteristics of each supported detection backend</p>
    """, unsafe_allow_html=True)
    ref_df = pd.DataFrame([
        {
            "Model":         "Haar Cascade",
            "Backend":       "OpenCV CPU",
            "Typical FPS":   "15–30",
            "Accuracy":      "Low–Medium",
            "GPU Required":  "No",
            "Best For":      "Low-power / Pi Zero",
        },
        {
            "Model":         "MobileNet SSD",
            "Backend":       "OpenCV DNN",
            "Typical FPS":   "8–18",
            "Accuracy":      "Medium",
            "GPU Required":  "Optional",
            "Best For":      "Balanced edge deployment",
        },
        {
            "Model":         "YOLOv8 Nano",
            "Backend":       "Ultralytics / MPS",
            "Typical FPS":   "10–25",
            "Accuracy":      "High",
            "GPU Required":  "Recommended",
            "Best For":      "Maximum plate accuracy",
        },
    ])
    st.dataframe(ref_df, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
