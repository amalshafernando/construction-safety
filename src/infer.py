"""
infer.py
--------
Runs the trained PPE detection model on images, folders, or video.
Draws colour-coded bounding boxes, applies safety rules, and saves
both annotated images and structured violation reports (text + JSON).

Usage:
    python src/infer.py --source path/to/image.jpg
    python src/infer.py --source data/images/test/
    python src/infer.py --source path/to/video.mp4
    python src/infer.py --source data/images/test/ --weights models/weights/best.pt
"""

import argparse
import os
import cv2
import json
import datetime
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

from rules import (
    Detection, ViolationReport, check_safety,
    DIRECT_VIOLATION_CLASSES, WORKER,
    SEVERITY_SAFE, SEVERITY_UNSAFE,
)

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS   = os.getenv("WEIGHTS_PATH", "models/weights/best.pt")
OUTPUT_IMAGE_DIR  = os.getenv("OUTPUT_IMAGE_DIR", "outputs/annotated")
OUTPUT_REPORT_DIR = os.getenv("OUTPUT_REPORT_DIR", "outputs/reports")
CONFIDENCE        = float(os.getenv("CONFIDENCE", 0.40))

# BGR colours (OpenCV uses BGR, not RGB)
C_GREEN   = (56, 197, 34)    # compliant worker / PPE item
C_RED     = (45,  45, 220)   # violation box
C_ORANGE  = (30, 140, 255)   # violation label accent
C_BANNER_SAFE   = (34, 120, 34)
C_BANNER_UNSAFE = (30,  30, 180)
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0)


# ── Drawing helpers ─────────────────────────────────────────────────────────────

def draw_box(image, box, label: str, colour, thickness: int = 2):
    """Draws a rounded-corner bounding box with a filled label badge."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    font_thick = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)

    # Badge sits above the box; clamp to top of image
    badge_y1 = max(0, y1 - th - 8)
    badge_y2 = y1
    cv2.rectangle(image, (x1, badge_y1), (x1 + tw + 6, badge_y2), colour, -1)
    cv2.putText(image, label, (x1 + 3, badge_y2 - 3),
                font, font_scale, C_WHITE, font_thick, cv2.LINE_AA)


def draw_worker_box(image, ws, worker_colour):
    """
    Draws the worker bounding box with a compliance score or violation badge.
    Green = compliant, Red = violation. Shows compliance score if safe.
    """
    x1, y1, x2, y2 = map(int, ws.box)
    cv2.rectangle(image, (x1, y1), (x2, y2), worker_colour, 2)

    if ws.is_compliant:
        label = f"Worker {ws.worker_index} — OK ({ws.compliance_score:.0%})"
    else:
        label = f"Worker {ws.worker_index} — VIOLATION"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    badge_y1 = max(0, y1 - th - 8)
    cv2.rectangle(image, (x1, badge_y1), (x1 + tw + 6, y1), worker_colour, -1)
    cv2.putText(image, label, (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_WHITE, 1, cv2.LINE_AA)

    # If violation, list the specific issues inside the box
    if not ws.is_compliant:
        for j, v_msg in enumerate(ws.violations):
            cv2.putText(image, f"  ! {v_msg}", (x1 + 4, y1 + 18 + j * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_RED, 1, cv2.LINE_AA)


def draw_status_banner(image, report: ViolationReport):
    """Draws a coloured status banner at the top of the image."""
    h, w   = image.shape[:2]
    colour = C_BANNER_SAFE if report.is_safe else C_BANNER_UNSAFE

    if report.is_safe:
        text = f"SAFE  |  {report.worker_count} worker(s)  |  confidence {report.scene_confidence:.0%}"
    else:
        text = (f"UNSAFE  |  {report.violation_count} violation(s)  "
                f"|  {report.compliant_count}/{report.worker_count} workers compliant")

    cv2.rectangle(image, (0, 0), (w, 40), colour, -1)
    cv2.putText(image, text, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, C_WHITE, 2, cv2.LINE_AA)


# ── Per-image processing ────────────────────────────────────────────────────────

def process_image(image, img_name: str, model) -> ViolationReport:
    """
    Runs YOLO on a single image (numpy array), applies safety rules,
    draws annotations in-place, and returns the ViolationReport.
    """
    results = model(image, conf=CONFIDENCE, verbose=False)[0]

    # Parse raw YOLO output into Detection objects
    detections: list[Detection] = []
    for box in results.boxes:
        detections.append(Detection(
            class_name=model.names[int(box.cls[0])],
            confidence=float(box.conf[0]),
            box=tuple(box.xyxy[0].tolist()),
        ))

    # Apply safety rule engine
    report = check_safety(img_name, detections)

    # ── Draw PPE items (non-worker boxes) ──────────────────────────────────────
    for det in detections:
        if det.class_name == WORKER:
            continue   # drawn separately below with worker status
        is_viol = det.class_name in DIRECT_VIOLATION_CLASSES
        colour  = C_RED if is_viol else C_GREEN
        label   = f"{det.class_name} {det.confidence:.0%}"
        draw_box(image, det.box, label, colour)

    # ── Draw worker boxes with per-worker compliance info ──────────────────────
    for ws in report.worker_statuses:
        worker_colour = C_GREEN if ws.is_compliant else C_RED
        draw_worker_box(image, ws, worker_colour)

    # ── Draw status banner ─────────────────────────────────────────────────────
    draw_status_banner(image, report)

    return report


# ── Image folder mode ───────────────────────────────────────────────────────────

def run_on_images(image_paths: list, model, out_img_dir: Path) -> list[ViolationReport]:
    reports = []
    for img_path in image_paths:
        print(f"\n[PROCESSING] {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print("  [WARN] Could not read image, skipping.")
            continue

        report = process_image(image, img_path.name, model)
        reports.append(report)

        # Save annotated image
        out_path = out_img_dir / f"result_{img_path.name}"
        cv2.imwrite(str(out_path), image)
        print(f"  [SAVED]  → {out_path}")
        print(f"  [{report.severity}]  workers:{report.worker_count}  "
              f"violations:{report.violation_count}  "
              f"confidence:{report.scene_confidence:.0%}")

    return reports


# ── Video mode ──────────────────────────────────────────────────────────────────

def run_on_video(video_path: str, model, out_img_dir: Path) -> list[ViolationReport]:
    """
    Processes a video file frame by frame.
    Saves a sample of annotated frames and returns one report per sampled frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = max(1, int(fps))   # analyse one frame per second
    video_name = Path(video_path).stem

    print(f"[VIDEO] {video_name}  |  {total} frames  |  {fps:.1f} fps")
    print(f"[VIDEO] Sampling every {sample_every} frames (~1 per second)")

    reports    = []
    frame_idx  = 0
    saved      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            frame_name = f"{video_name}_frame{frame_idx:06d}.jpg"
            report     = process_image(frame, frame_name, model)
            reports.append(report)

            out_path = out_img_dir / f"result_{frame_name}"
            cv2.imwrite(str(out_path), frame)
            saved += 1

            print(f"  Frame {frame_idx:6d}  [{report.severity}]  "
                  f"workers:{report.worker_count}  violations:{report.violation_count}")

        frame_idx += 1

    cap.release()
    print(f"[VIDEO] Done — {saved} frames saved.")
    return reports


# ── Report saving ───────────────────────────────────────────────────────────────

def save_reports(reports: list[ViolationReport], source: str, out_dir: Path):
    """Saves a plain-text summary report and a machine-readable JSON report."""
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_count  = sum(1 for r in reports if r.is_safe)

    # ── Text report ───────────────────────────────────────────────────────────
    txt_path = out_dir / f"report_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write("Construction Safety Monitor — Violation Report\n")
        f.write(f"Generated : {datetime.datetime.now()}\n")
        f.write(f"Source    : {source}\n")
        f.write(f"Summary   : {safe_count}/{len(reports)} images/frames fully safe\n")
        f.write("=" * 60 + "\n\n")
        for r in reports:
            f.write(r.summary())
            f.write("\n\n")

    # ── JSON report ───────────────────────────────────────────────────────────
    json_path = out_dir / f"report_{timestamp}.json"
    json_data = {
        "generated":    datetime.datetime.now().isoformat(),
        "source":       source,
        "total_images": len(reports),
        "safe_count":   safe_count,
        "unsafe_count": len(reports) - safe_count,
        "results": [
            {
                "image":            r.image_path,
                "severity":         r.severity,
                "scene_confidence": r.scene_confidence,
                "worker_count":     r.worker_count,
                "compliant_count":  r.compliant_count,
                "violation_count":  r.violation_count,
                "raw_violations":   r.raw_violations,
                "workers": [
                    {
                        "index":          ws.worker_index,
                        "is_compliant":   ws.is_compliant,
                        "compliance_score": ws.compliance_score,
                        "has_hard_hat":   ws.has_hard_hat,
                        "has_vest":       ws.has_vest,
                        "has_harness":    ws.has_harness,
                        "violations":     ws.violations,
                    }
                    for ws in r.worker_statuses
                ],
            }
            for r in reports
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n[REPORT]  Text → {txt_path}")
    print(f"[REPORT]  JSON → {json_path}")
    print(f"[SUMMARY] {safe_count}/{len(reports)} images/frames are fully safe.")


# ── Main entry point ────────────────────────────────────────────────────────────

def run_inference(source: str, weights: str = DEFAULT_WEIGHTS):
    print(f"[INFO] Loading model: {weights}")
    model = YOLO(weights)

    out_img_dir = Path(OUTPUT_IMAGE_DIR)
    out_rpt_dir = Path(OUTPUT_REPORT_DIR)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_rpt_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source)
    VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv"}

    if source_path.is_dir():
        # Folder of images
        image_paths = sorted(
            p for p in source_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"[INFO] Found {len(image_paths)} images in {source}")
        reports = run_on_images(image_paths, model, out_img_dir)

    elif source_path.is_file() and source_path.suffix.lower() in VIDEO_EXTS:
        # Video file
        reports = run_on_video(str(source_path), model, out_img_dir)

    elif source_path.is_file():
        # Single image
        reports = run_on_images([source_path], model, out_img_dir)

    else:
        raise FileNotFoundError(f"Source not found: {source}")

    save_reports(reports, source, out_rpt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construction Safety Monitor — Inference")
    parser.add_argument("--source",  required=True,
                        help="Image path, folder of images, or video file (.mp4/.avi)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Path to trained YOLOv8 weights (.pt file)")
    args = parser.parse_args()
    run_inference(source=args.source, weights=args.weights)
