"""
api/main.py
-----------
FastAPI backend for the Construction Safety Monitor.

Exposes three modes of operation:
  1. POST /analyse          — upload a single image, get a JSON safety report
  2. GET  /stream           — MJPEG stream from webcam with live annotations
  3. GET  /health           — health check (model loaded, GPU status)
  4. GET  /violations/recent — last N violation events logged in memory

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000 in your browser.
"""

import io
import cv2
import time
import base64
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import (
    HTMLResponse, StreamingResponse, JSONResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()

# Add src/ to path so we can import rules.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rules import Detection, check_safety, ViolationReport

# ── Configuration ──────────────────────────────────────────────────────────────

WEIGHTS_PATH    = Path(os.getenv("WEIGHTS_PATH", "models/weights/best.pt"))
CONFIDENCE      = float(os.getenv("CONFIDENCE", 0.40))
CAMERA_INDEX    = int(os.getenv("CAMERA_INDEX", 0))
MAX_LOG_ENTRIES = int(os.getenv("MAX_LOG_ENTRIES", 100))

# BGR colours
C_GREEN  = (56, 197, 34)
C_RED    = (45,  45, 220)
C_WHITE  = (255, 255, 255)
C_BANNER_SAFE   = (34, 120, 34)
C_BANNER_UNSAFE = (30,  30, 180)

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Construction Safety Monitor API",
    description="Real-time PPE detection and safety violation flagging",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")

# ── Global state ───────────────────────────────────────────────────────────────

class AppState:
    model: Optional[YOLO] = None
    camera: Optional[cv2.VideoCapture] = None
    camera_lock = threading.Lock()
    violation_log: deque = deque(maxlen=MAX_LOG_ENTRIES)
    frame_count: int = 0
    start_time: float = time.time()

state = AppState()


# ── Model loading ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_model():
    """Load YOLO model when the server starts."""
    if not WEIGHTS_PATH.exists():
        print(f"[WARN] Weights not found at {WEIGHTS_PATH}")
        print("[WARN] API will start but /analyse and /stream will return 503.")
        print("[WARN] Place best.pt in models/weights/ and restart.")
        return

    print(f"[INFO] Loading model from {WEIGHTS_PATH} ...")
    state.model = YOLO(str(WEIGHTS_PATH))
    print("[INFO] Model loaded. API is ready.")


@app.on_event("shutdown")
async def release_camera():
    if state.camera and state.camera.isOpened():
        state.camera.release()


# ── Response schemas ───────────────────────────────────────────────────────────

class WorkerResult(BaseModel):
    worker_index: int
    is_compliant: bool
    compliance_score: float
    has_hard_hat: bool
    has_vest: bool
    has_harness: bool
    violations: list[str]


class SafetyResult(BaseModel):
    severity: str                   # SAFE | UNSAFE
    scene_confidence: float         # 0.0–1.0
    worker_count: int
    compliant_count: int
    violation_count: int
    workers: list[WorkerResult]
    raw_violations: list[str]
    processing_time_ms: float
    timestamp: str


# ── Drawing helpers (shared between /analyse and /stream) ──────────────────────

def annotate_frame(frame: np.ndarray, report: ViolationReport) -> np.ndarray:
    """
    Draws all bounding boxes and the status banner on a frame (in-place).
    Returns the annotated frame.
    """
    # Worker boxes — colour by compliance
    for ws in report.worker_statuses:
        colour = C_GREEN if ws.is_compliant else C_RED
        x1, y1, x2, y2 = map(int, ws.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        if ws.is_compliant:
            label = f"Worker {ws.worker_index} OK ({ws.compliance_score:.0%})"
        else:
            label = f"Worker {ws.worker_index} VIOLATION"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        badge_y = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, badge_y), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_WHITE, 1, cv2.LINE_AA)

        if not ws.is_compliant:
            for j, v in enumerate(ws.violations):
                cv2.putText(frame, f"! {v}", (x1 + 4, y1 + 20 + j * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_RED, 1, cv2.LINE_AA)

    # Status banner
    h, w = frame.shape[:2]
    colour = C_BANNER_SAFE if report.is_safe else C_BANNER_UNSAFE
    if report.is_safe:
        text = f"SAFE  |  {report.worker_count} worker(s)  |  {report.scene_confidence:.0%} confidence"
    else:
        text = f"UNSAFE  |  {report.violation_count} violation(s)  |  {report.compliant_count}/{report.worker_count} compliant"

    cv2.rectangle(frame, (0, 0), (w, 42), colour, -1)
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_WHITE, 2, cv2.LINE_AA)

    # FPS counter (top right)
    elapsed = time.time() - state.start_time
    fps = state.frame_count / elapsed if elapsed > 0 else 0
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, C_WHITE, 1, cv2.LINE_AA)

    return frame


def run_detection(frame: np.ndarray) -> tuple[ViolationReport, np.ndarray]:
    """Runs YOLO + safety rules on a single frame. Returns (report, annotated_frame)."""
    results = state.model(frame, conf=CONFIDENCE, verbose=False)[0]

    detections = [
        Detection(
            class_name=state.model.names[int(b.cls[0])],
            confidence=float(b.conf[0]),
            box=tuple(b.xyxy[0].tolist()),
        )
        for b in results.boxes
    ]

    report = check_safety("frame", detections)
    annotated = annotate_frame(frame.copy(), report)
    state.frame_count += 1

    # Log violations
    if not report.is_safe:
        state.violation_log.append({
            "timestamp": datetime.now().isoformat(),
            "violation_count": report.violation_count,
            "worker_count": report.worker_count,
            "details": report.raw_violations + [
                v for ws in report.worker_statuses for v in ws.violations
            ],
        })

    return report, annotated


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the live monitoring dashboard."""
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="""
    <html><body>
    <h2>Construction Safety Monitor API</h2>
    <p>Frontend not found. Use the API directly:</p>
    <ul>
      <li><a href="/docs">Swagger UI (API docs)</a></li>
      <li><a href="/stream">Live camera stream (MJPEG)</a></li>
      <li><a href="/health">Health check</a></li>
    </ul>
    </body></html>
    """)


@app.get("/health")
async def health():
    """Returns model and camera status."""
    import torch
    return {
        "status":       "ok",
        "model_loaded": state.model is not None,
        "weights_path": str(WEIGHTS_PATH),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "uptime_seconds": round(time.time() - state.start_time, 1),
        "frames_processed": state.frame_count,
    }


@app.post("/analyse", response_model=SafetyResult)
async def analyse_image(file: UploadFile = File(...)):
    """
    Upload an image and receive a full safety analysis report.

    - Accepts: JPEG, PNG
    - Returns: JSON with severity, per-worker compliance, confidence scores
    """
    if state.model is None:
        raise HTTPException(503, "Model not loaded. Place best.pt in models/weights/ and restart.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected an image file, got {file.content_type}")

    t_start = time.time()

    # Decode uploaded image
    contents = await file.read()
    np_arr   = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image. Make sure it is a valid JPEG/PNG.")

    report, _ = run_detection(frame)
    elapsed_ms = (time.time() - t_start) * 1000

    return SafetyResult(
        severity=report.severity,
        scene_confidence=report.scene_confidence,
        worker_count=report.worker_count,
        compliant_count=report.compliant_count,
        violation_count=report.violation_count,
        workers=[
            WorkerResult(
                worker_index=ws.worker_index,
                is_compliant=ws.is_compliant,
                compliance_score=ws.compliance_score,
                has_hard_hat=ws.has_hard_hat,
                has_vest=ws.has_vest,
                has_harness=ws.has_harness,
                violations=ws.violations,
            )
            for ws in report.worker_statuses
        ],
        raw_violations=report.raw_violations,
        processing_time_ms=round(elapsed_ms, 2),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/analyse/preview")
async def analyse_with_preview(file: UploadFile = File(...)):
    """
    Same as /analyse but also returns the annotated image as a base64 PNG.
    Useful for displaying results in the frontend without a separate request.
    """
    if state.model is None:
        raise HTTPException(503, "Model not loaded.")

    contents = await file.read()
    np_arr   = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    report, annotated = run_detection(frame)

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64   = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "severity":         report.severity,
        "worker_count":     report.worker_count,
        "violation_count":  report.violation_count,
        "scene_confidence": report.scene_confidence,
        "annotated_image":  f"data:image/jpeg;base64,{img_b64}",
    })


def _camera_frame_generator():
    """
    Generator that yields MJPEG frames from the webcam continuously.
    Handles camera open/close and annotates each frame with safety results.
    """
    with state.camera_lock:
        if state.camera is None or not state.camera.isOpened():
            state.camera = cv2.VideoCapture(CAMERA_INDEX)
            state.camera.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not state.camera.isOpened():
        # Yield a simple error frame
        err = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(err, "Camera not found. Check CAMERA_INDEX in api/main.py",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
        _, buf = cv2.imencode(".jpg", err)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        return

    while True:
        ret, frame = state.camera.read()
        if not ret:
            break

        if state.model is not None:
            _, annotated = run_detection(frame)
        else:
            annotated = frame.copy()
            cv2.putText(annotated, "Model not loaded — place best.pt in models/weights/",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)

        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.get("/stream")
async def live_stream():
    """
    MJPEG stream from the webcam with real-time PPE detection overlaid.
    Open this URL in an <img> tag or directly in the browser.

    Example HTML:
        <img src="http://localhost:8000/stream" />
    """
    if state.model is None:
        raise HTTPException(503, "Model not loaded. Place best.pt in models/weights/ and restart.")

    return StreamingResponse(
        _camera_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/violations/recent")
async def recent_violations(limit: int = 20):
    """Returns the most recent N violation events detected during live streaming."""
    events = list(state.violation_log)[-limit:]
    return {
        "total_logged": len(state.violation_log),
        "returned": len(events),
        "events": list(reversed(events)),   # newest first
    }


@app.delete("/violations/clear")
async def clear_violations():
    """Clears the in-memory violation log."""
    state.violation_log.clear()
    return {"status": "cleared"}
