"""
evaluate.py
-----------
Phase 5 — Evaluation & Results

Runs the trained model against the validation/test set and produces:
  1. Standard detection metrics: mAP@0.5, precision, recall per class
  2. Scene-level safety accuracy: how often did we get SAFE/UNSAFE right?
  3. Failure case analysis: images where the model was most wrong
  4. A visual summary plot saved to outputs/

Usage:
    python src/evaluate.py
    python src/evaluate.py --split test          # use test set instead of val
    python src/evaluate.py --weights models/weights/best.pt --split val
"""

import argparse
import json
import os
import cv2
import yaml
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend, safe for server/Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from ultralytics import YOLO
from rules import Detection, check_safety, SEVERITY_SAFE, SEVERITY_UNSAFE
from dotenv import load_dotenv
import os

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = os.getenv("WEIGHTS_PATH", "models/weights/best.pt")
DATA_YAML       = os.getenv("DATA_CONFIG", "data.yaml")
OUTPUT_DIR      = os.getenv("EVAL_OUTPUT_DIR", "outputs/evaluation")
CONFIDENCE      = float(os.getenv("CONFIDENCE", 0.40))
MAX_FAILURES    = int(os.getenv("MAX_FAILURES", 12))


# ── 1. Standard YOLO detection metrics ─────────────────────────────────────────

def run_yolo_evaluation(model, data_yaml: str, split: str) -> dict:
    """
    Uses Ultralytics built-in .val() to compute mAP, precision, recall per class.
    Returns a dict of metrics we can print and save.
    """
    print(f"\n[EVAL] Running YOLO validation on '{split}' split...")

    metrics = model.val(
        data=data_yaml,
        split=split,
        conf=CONFIDENCE,
        verbose=False,
    )

    # Load class names from data.yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get("names", [])
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names)]

    results = {
        "map50":     round(float(metrics.box.map50), 4),
        "map50_95":  round(float(metrics.box.map),   4),
        "precision": round(float(metrics.box.mp),    4),
        "recall":    round(float(metrics.box.mr),    4),
        "per_class": {},
    }

    for i, (ap50, ap) in enumerate(zip(metrics.box.ap50, metrics.box.ap)):
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        results["per_class"][name] = {
            "ap50":   round(float(ap50), 4),
            "ap5095": round(float(ap),   4),
        }

    return results


def print_metrics(metrics: dict):
    """Prints a formatted metrics table to console."""
    print()
    print("=" * 52)
    print("  DETECTION METRICS")
    print("=" * 52)
    print(f"  mAP @ 0.5        :  {metrics['map50']:.4f}  "
          f"({'good' if metrics['map50'] > 0.5 else 'needs improvement'})")
    print(f"  mAP @ 0.5:0.95   :  {metrics['map50_95']:.4f}")
    print(f"  Precision        :  {metrics['precision']:.4f}  "
          f"(how many detections were correct)")
    print(f"  Recall           :  {metrics['recall']:.4f}  "
          f"(how many real objects were found)")
    print()
    print("  Per-class AP @ 0.5:")
    print(f"  {'Class':<22} {'AP50':>6}  {'Bar'}")
    print(f"  {'-'*46}")
    for cls_name, vals in metrics["per_class"].items():
        ap = vals["ap50"]
        bar = "█" * int(ap * 20) + "░" * (20 - int(ap * 20))
        flag = " ← low" if ap < 0.35 else ""
        print(f"  {cls_name:<22} {ap:>6.3f}  {bar}{flag}")
    print("=" * 52)


# ── 2. Scene-level safety accuracy ─────────────────────────────────────────────

def evaluate_scene_safety(model, data_yaml: str, split: str) -> dict:
    """
    For each image in the split, runs inference + safety rules, then checks
    whether the predicted SAFE/UNSAFE matches the ground-truth label
    (derived from whether any violation-class annotation exists in the label file).

    Returns confusion stats: TP, TN, FP, FN at the scene level.
    """
    print(f"\n[EVAL] Running scene-level safety evaluation on '{split}' split...")

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("path", "."))
    # Roboflow exports use 'valid' as the val folder name
    folder_name  = "valid" if split == "val" else split
    img_dir      = dataset_root / folder_name / "images"
    lbl_dir      = dataset_root / folder_name / "labels"

    class_names = cfg.get("names", [])
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names)]

    # Classes that, if present in the label, mean the scene is UNSAFE
    violation_class_names = {"no-hard-hat", "no-safety-vest"}
    violation_class_ids   = {
        i for i, n in enumerate(class_names) if n in violation_class_names
    }

    if not img_dir.exists():
        print(f"  [WARN] Image directory not found: {img_dir}")
        print(f"  Skipping scene-level evaluation.")
        return {}

    image_paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    TP = TN = FP = FN = 0   # scene-level confusion matrix
    failure_cases = []       # (img_path, pred_safe, gt_safe, report)

    for img_path in image_paths:
        # ── Ground truth: is this scene supposed to be unsafe? ──────────────
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_has_violation = False
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                if parts and int(parts[0]) in violation_class_ids:
                    gt_has_violation = True
                    break
        gt_safe = not gt_has_violation

        # ── Prediction ────────────────────────────────────────────────────────
        image   = cv2.imread(str(img_path))
        if image is None:
            continue

        results     = model(image, conf=CONFIDENCE, verbose=False)[0]
        detections  = [
            Detection(
                class_name=model.names[int(b.cls[0])],
                confidence=float(b.conf[0]),
                box=tuple(b.xyxy[0].tolist()),
            )
            for b in results.boxes
        ]
        report      = check_safety(img_path.name, detections)
        pred_safe   = report.is_safe

        # ── Confusion matrix update ───────────────────────────────────────────
        if gt_safe and pred_safe:
            TN += 1
        elif not gt_safe and not pred_safe:
            TP += 1
        elif gt_safe and not pred_safe:
            FP += 1   # false alarm: predicted unsafe but was actually safe
            failure_cases.append((img_path, pred_safe, gt_safe, report))
        else:
            FN += 1   # missed violation: predicted safe but was actually unsafe
            failure_cases.append((img_path, pred_safe, gt_safe, report))

    total = TP + TN + FP + FN
    if total == 0:
        print("  [WARN] No images evaluated.")
        return {}

    accuracy  = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0

    scene_metrics = {
        "total": total, "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
    }

    print()
    print("=" * 52)
    print("  SCENE-LEVEL SAFETY CLASSIFICATION")
    print("=" * 52)
    print(f"  Total images evaluated : {total}")
    print(f"  Correctly flagged UNSAFE (TP) : {TP}")
    print(f"  Correctly cleared SAFE   (TN) : {TN}")
    print(f"  False alarms (FP)        : {FP}  ← model said UNSAFE, was actually SAFE")
    print(f"  Missed violations (FN)   : {FN}  ← model said SAFE, was actually UNSAFE")
    print(f"  Accuracy   : {accuracy:.1%}")
    print(f"  Precision  : {precision:.1%}  (of violations flagged, how many were real)")
    print(f"  Recall     : {recall:.1%}  (of real violations, how many were caught)")
    print("=" * 52)

    if FN > 0:
        print(f"\n  [IMPORTANT] {FN} real violations were MISSED.")
        print("  For a safety system, high recall is critical — missed violations are dangerous.")
        print("  Consider lowering the confidence threshold or adding more training data.")

    return scene_metrics, failure_cases


# ── 3. Failure case analysis ────────────────────────────────────────────────────

def save_failure_cases(failure_cases: list, out_dir: Path, model):
    """
    Saves annotated images of the model's worst mistakes.
    Each image shows what the model predicted vs what the ground truth was.
    """
    if not failure_cases:
        print("\n[EVAL] No failure cases found — model correctly classified all scenes!")
        return

    print(f"\n[EVAL] Saving {min(len(failure_cases), MAX_FAILURES)} failure case images...")

    failures_dir = out_dir / "failure_cases"
    failures_dir.mkdir(parents=True, exist_ok=True)

    for i, (img_path, pred_safe, gt_safe, report) in enumerate(failure_cases[:MAX_FAILURES]):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Draw model's detections
        results = model(image, conf=CONFIDENCE, verbose=False)[0]
        annotated = results.plot()   # built-in plot with all boxes

        # Add failure diagnosis banner
        pred_str = "SAFE" if pred_safe else "UNSAFE"
        gt_str   = "SAFE" if gt_safe   else "UNSAFE"
        error_type = "FALSE ALARM" if pred_safe is False and gt_safe else "MISSED VIOLATION"

        banner_colour = (0, 140, 255) if error_type == "FALSE ALARM" else (0, 0, 180)
        h, w = annotated.shape[:2]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), banner_colour, -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        cv2.putText(annotated,
                    f"{error_type}  |  predicted: {pred_str}  |  actual: {gt_str}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

        out_path = failures_dir / f"failure_{i+1:03d}_{error_type.lower().replace(' ','_')}_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

    print(f"  Saved to: {failures_dir}/")


# ── 4. Summary plot ─────────────────────────────────────────────────────────────

def save_summary_plot(det_metrics: dict, scene_metrics: dict, out_dir: Path):
    """Creates a two-panel summary chart saved as a PNG."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Construction Safety Monitor — Evaluation Summary", fontsize=14, fontweight="bold")

    # Panel 1: Per-class AP50 bar chart
    ax1 = axes[0]
    classes = list(det_metrics["per_class"].keys())
    ap50s   = [det_metrics["per_class"][c]["ap50"] for c in classes]
    colours = ["#16a34a" if ap >= 0.5 else "#dc2626" for ap in ap50s]

    bars = ax1.barh(classes, ap50s, color=colours, edgecolor="white", height=0.6)
    ax1.set_xlim(0, 1.0)
    ax1.axvline(0.5, color="#6b7280", linestyle="--", linewidth=1, label="0.5 threshold")
    ax1.set_xlabel("AP @ 0.5")
    ax1.set_title("Per-class Detection Accuracy (AP50)")
    ax1.legend(fontsize=9)

    for bar, val in zip(bars, ap50s):
        ax1.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9, color="#1f2937")

    # Panel 2: Scene-level confusion matrix summary
    ax2 = axes[1]
    if scene_metrics:
        categories = ["Correct\n(TP+TN)", "False\nAlarms (FP)", "Missed\nViolations (FN)"]
        values  = [
            scene_metrics["TP"] + scene_metrics["TN"],
            scene_metrics["FP"],
            scene_metrics["FN"],
        ]
        colours2 = ["#16a34a", "#f59e0b", "#dc2626"]
        ax2.bar(categories, values, color=colours2, edgecolor="white", width=0.5)
        ax2.set_ylabel("Image count")
        ax2.set_title(f"Scene Safety Classification\n"
                      f"Accuracy {scene_metrics['accuracy']:.1%}  |  "
                      f"Recall {scene_metrics['recall']:.1%}")
        for j, (cat, val) in enumerate(zip(categories, values)):
            ax2.text(j, val + 0.3, str(val), ha="center", fontsize=11, fontweight="bold")

        # Add legend
        patches = [
            mpatches.Patch(color="#16a34a", label="Correct classification"),
            mpatches.Patch(color="#f59e0b", label="False alarm (model over-flagged)"),
            mpatches.Patch(color="#dc2626", label="Missed violation (dangerous!)"),
        ]
        ax2.legend(handles=patches, fontsize=8, loc="upper right")
    else:
        ax2.text(0.5, 0.5, "Scene metrics\nnot available", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, color="gray")

    plt.tight_layout()
    out_path = out_dir / "evaluation_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[EVAL] Summary plot saved → {out_path}")


# ── 5. Save results to JSON ─────────────────────────────────────────────────────

def save_results_json(det_metrics: dict, scene_metrics: dict, split: str, out_dir: Path):
    timestamp = datetime.datetime.now().isoformat()
    payload = {
        "generated":      timestamp,
        "split":          split,
        "detection":      det_metrics,
        "scene_safety":   scene_metrics if scene_metrics else {},
    }
    out_path = out_dir / f"eval_results_{split}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[EVAL] Results JSON  → {out_path}")
    print()
    print("  Copy these numbers into your README.md evaluation table!")


# ── Main ────────────────────────────────────────────────────────────────────────

def main(weights: str, data_yaml: str, split: str):
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model: {weights}")
    model = YOLO(weights)

    # Step 1 — YOLO detection metrics
    det_metrics = run_yolo_evaluation(model, data_yaml, split)
    print_metrics(det_metrics)

    # Step 2 — Scene-level safety accuracy + failure cases
    result = evaluate_scene_safety(model, data_yaml, split)
    scene_metrics, failure_cases = result if result else ({}, [])

    # Step 3 — Save failure cases
    if failure_cases:
        save_failure_cases(failure_cases, out_dir, model)

    # Step 4 — Summary plot
    save_summary_plot(det_metrics, scene_metrics, out_dir)

    # Step 5 — Save JSON
    save_results_json(det_metrics, scene_metrics, split, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construction Safety Monitor — Evaluation")
    parser.add_argument("--weights",   default=DEFAULT_WEIGHTS,
                        help="Path to trained weights (.pt file)")
    parser.add_argument("--data",      default=DATA_YAML,
                        help="Path to data.yaml")
    parser.add_argument("--split",     default="val", choices=["val", "test"],
                        help="Dataset split to evaluate on")
    args = parser.parse_args()
    main(weights=args.weights, data_yaml=args.data, split=args.split)
