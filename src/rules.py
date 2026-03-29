"""
rules.py
--------
Defines the safety rules and violation logic.

Given a list of detections from the YOLO model, this module decides:
  - Which workers are compliant
  - Which workers are in violation
  - What the specific violation is

This is intentionally kept separate from inference so rules can be
changed without touching the model code.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

# ── Class name constants ────────────────────────────────────────────────────────
# These must match the class names in data.yaml exactly.

WORKER        = "worker"
HARD_HAT      = "hard-hat"
SAFETY_VEST   = "safety-vest"
NO_HARD_HAT   = "no-hard-hat"
NO_SAFETY_VEST = "no-safety-vest"
SAFETY_HARNESS = "safety-harness"

# Classes that represent violations when detected directly
DIRECT_VIOLATION_CLASSES = {NO_HARD_HAT, NO_SAFETY_VEST}

# Minimum confidence threshold — detections below this are ignored
CONFIDENCE_THRESHOLD = 0.40


# ── Data structures ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Represents a single bounding box detection from YOLO."""
    class_name: str
    confidence: float
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


@dataclass
class ViolationReport:
    """Summarises the safety status of a single image."""
    image_path: str
    is_safe: bool
    worker_count: int
    violations: List[str] = field(default_factory=list)
    compliant_count: int = 0
    violation_count: int = 0

    def summary(self) -> str:
        status = "SAFE" if self.is_safe else "UNSAFE"
        lines = [
            f"=== Safety Report: {self.image_path} ===",
            f"Status      : {status}",
            f"Workers     : {self.worker_count}",
            f"Compliant   : {self.compliant_count}",
            f"Violations  : {self.violation_count}",
        ]
        if self.violations:
            lines.append("Violation details:")
            for v in self.violations:
                lines.append(f"  - {v}")
        return "\n".join(lines)


# ── IoU helper ──────────────────────────────────────────────────────────────────

def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """
    Computes Intersection over Union between two bounding boxes.
    Used to check if a PPE item overlaps with a worker box.
    IoU > 0.1 is enough to say "this helmet is on this worker".
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ── Core rule engine ────────────────────────────────────────────────────────────

def check_safety(image_path: str, detections: List[Detection]) -> ViolationReport:
    """
    Applies safety rules to a list of detections and returns a ViolationReport.

    Strategy:
    1. If the model directly detects 'no-hard-hat' or 'no-safety-vest', flag immediately.
    2. Otherwise, for each 'worker' box, check if a 'hard-hat' and 'safety-vest'
       bounding box overlaps it (IoU > threshold). If not, flag a violation.
    """

    # Filter out low-confidence detections
    detections = [d for d in detections if d.confidence >= CONFIDENCE_THRESHOLD]

    workers      = [d for d in detections if d.class_name == WORKER]
    hard_hats    = [d for d in detections if d.class_name == HARD_HAT]
    vests        = [d for d in detections if d.class_name == SAFETY_VEST]
    violations_direct = [d for d in detections if d.class_name in DIRECT_VIOLATION_CLASSES]

    violation_messages = []

    # Rule A: direct violation classes detected by model
    for v in violations_direct:
        label = "Missing hard hat" if v.class_name == NO_HARD_HAT else "Missing safety vest"
        violation_messages.append(
            f"{label} (confidence: {v.confidence:.0%})"
        )

    # Rule B: overlap-based check — worker has no PPE box overlapping them
    IOU_OVERLAP_THRESHOLD = 0.10  # Low threshold: any overlap counts

    compliant_count = 0
    for i, worker in enumerate(workers):
        has_helmet = any(compute_iou(worker.box, h.box) > IOU_OVERLAP_THRESHOLD for h in hard_hats)
        has_vest   = any(compute_iou(worker.box, v.box) > IOU_OVERLAP_THRESHOLD for v in vests)

        worker_violations = []
        if not has_helmet:
            worker_violations.append("no hard hat")
        if not has_vest:
            worker_violations.append("no safety vest")

        if worker_violations:
            violation_messages.append(
                f"Worker {i+1}: {', '.join(worker_violations)}"
            )
        else:
            compliant_count += 1

    is_safe = len(violation_messages) == 0

    return ViolationReport(
        image_path=image_path,
        is_safe=is_safe,
        worker_count=len(workers),
        violations=violation_messages,
        compliant_count=compliant_count,
        violation_count=len(violation_messages),
    )
