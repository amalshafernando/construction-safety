"""
infer.py
--------
Runs the trained PPE detection model on images or a folder of images.
Draws bounding boxes, applies safety rules, and saves violation reports.

Usage:
    python src/infer.py --source path/to/image.jpg
    python src/infer.py --source data/images/test/
    python src/infer.py --source path/to/video.mp4
"""

import argparse
import os
import cv2
import datetime
from pathlib import Path
from ultralytics import YOLO

# Import our custom safety rule engine
from rules import Detection, check_safety

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS   = "models/weights/best.pt"
OUTPUT_IMAGE_DIR  = "outputs/annotated"
OUTPUT_REPORT_DIR = "outputs/reports"
CONFIDENCE        = 0.40          # Minimum confidence to show a detection

# Colours for bounding boxes: BGR format (OpenCV uses BGR, not RGB)
COLOUR_SAFE      = (34, 197, 94)    # Green  — compliant worker or PPE
COLOUR_VIOLATION = (59, 130, 246)   # Blue   — violation highlight
COLOUR_RED       = (60, 60, 220)    # Red    — violation label background
COLOUR_WHITE     = (255, 255, 255)
COLOUR_BLACK     = (0, 0, 0)


# ── Drawing helpers ─────────────────────────────────────────────────────────────

def draw_box(image, box, label, colour):
    """Draws a bounding box with a label on the image."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness=2)

    # Label background pill
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(image, (x1, y1 - text_h - 8), (x1 + text_w + 6, y1), colour, -1)
    cv2.putText(image, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOUR_WHITE, 1, cv2.LINE_AA)


def draw_status_banner(image, is_safe: bool, violation_count: int):
    """Draws a SAFE / UNSAFE banner at the top of the image."""
    h, w = image.shape[:2]
    banner_h = 38
    colour  = (34, 139, 34) if is_safe else (0, 0, 200)
    label   = "SAFE" if is_safe else f"UNSAFE  —  {violation_count} violation(s) detected"

    cv2.rectangle(image, (0, 0), (w, banner_h), colour, -1)
    cv2.putText(image, label, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOUR_WHITE, 2, cv2.LINE_AA)


# ── Main inference function ─────────────────────────────────────────────────────

def run_inference(source: str, weights: str = DEFAULT_WEIGHTS):
    """Runs inference on a single image, a folder, or a video."""

    print(f"[INFO] Loading model from: {weights}")
    model = YOLO(weights)

    # Collect all image paths
    source_path = Path(source)
    if source_path.is_dir():
        image_paths = list(source_path.glob("*.jpg")) + \
                      list(source_path.glob("*.jpeg")) + \
                      list(source_path.glob("*.png"))
        print(f"[INFO] Found {len(image_paths)} images in {source}")
    elif source_path.is_file():
        image_paths = [source_path]
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)

    all_reports = []

    for img_path in image_paths:
        print(f"\n[PROCESSING] {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Could not read image, skipping.")
            continue

        # Run YOLO detection
        results = model(image, conf=CONFIDENCE, verbose=False)[0]

        # Parse detections into our Detection dataclass
        detections = []
        for box in results.boxes:
            class_id   = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(Detection(
                class_name=class_name,
                confidence=confidence,
                box=(x1, y1, x2, y2),
            ))

        # Apply safety rules
        report = check_safety(str(img_path.name), detections)
        all_reports.append(report)

        # Draw all bounding boxes on image
        for det in detections:
            is_violation_class = det.class_name in {"no-hard-hat", "no-safety-vest"}
            colour = COLOUR_RED if is_violation_class else COLOUR_SAFE
            label  = f"{det.class_name} {det.confidence:.0%}"
            draw_box(image, det.box, label, colour)

        # Draw top status banner
        draw_status_banner(image, report.is_safe, report.violation_count)

        # Save annotated image
        out_img_path = Path(OUTPUT_IMAGE_DIR) / f"result_{img_path.name}"
        cv2.imwrite(str(out_img_path), image)
        print(f"  [SAVED] Annotated image → {out_img_path}")

        # Print report to console
        print(report.summary())

    # Save combined text report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(OUTPUT_REPORT_DIR) / f"report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(f"Construction Safety Monitor — Report\n")
        f.write(f"Generated: {datetime.datetime.now()}\n")
        f.write(f"Source: {source}\n\n")
        for r in all_reports:
            f.write(r.summary())
            f.write("\n\n")

    safe_count = sum(1 for r in all_reports if r.is_safe)
    print(f"\n[SUMMARY] {safe_count}/{len(all_reports)} images are fully safe.")
    print(f"[REPORT]  Saved to {report_path}")


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construction Safety Monitor — Inference")
    parser.add_argument("--source",  required=True, help="Image, folder, or video path")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to model weights .pt file")
    args = parser.parse_args()

    run_inference(source=args.source, weights=args.weights)
