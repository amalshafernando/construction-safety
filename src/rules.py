"""
rules.py
--------
Defines the safety rules and violation logic.

Given a list of detections from the YOLO model, this module decides:
  - Which workers are compliant
  - Which workers are in violation, and what rule they broke
  - A confidence score for each verdict (not just a binary flag)
  - A severity level: SAFE | WARNING | UNSAFE

This is intentionally kept separate from inference so rules can be
changed without touching the model code.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import os

load_dotenv()

# ── Class name constants ────────────────────────────────────────────────────────
# These must match the class names in data.yaml exactly.

WORKER         = "worker"
HARD_HAT       = "hard-hat"
SAFETY_VEST    = "safety-vest"
NO_HARD_HAT    = "no-hard-hat"
NO_SAFETY_VEST = "no-safety-vest"
SAFETY_HARNESS = "safety-harness"

# Classes that directly signal a violation when the model detects them
DIRECT_VIOLATION_CLASSES = {NO_HARD_HAT, NO_SAFETY_VEST}

# Minimum detection confidence — anything below is discarded
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.40))

# Minimum IoU for a PPE box to "belong" to a worker box
IOU_OVERLAP_THRESHOLD = float(os.getenv("IOU_OVERLAP_THRESHOLD", 0.10))

# Severity levels
SEVERITY_SAFE    = "SAFE"
SEVERITY_WARNING = "WARNING"   # harness missing at height (softer rule)
SEVERITY_UNSAFE  = "UNSAFE"    # hard PPE rule violated


# ── Data structures ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Represents a single bounding box detection from YOLO."""
    class_name: str
    confidence: float
    box: Tuple[float, float, float, float]   # (x1, y1, x2, y2) absolute pixels


@dataclass
class WorkerStatus:
    """Compliance status for a single detected worker."""
    worker_index: int
    box: Tuple[float, float, float, float]
    has_hard_hat: bool
    has_vest: bool
    has_harness: bool
    hat_confidence: float    # confidence of the overlapping hat detection (0 if absent)
    vest_confidence: float
    is_compliant: bool
    violations: List[str] = field(default_factory=list)

    @property
    def compliance_score(self) -> float:
        """
        A 0–1 score expressing how confident we are this worker is fully compliant.
        Derived from the confidence of each detected PPE item.
        0.0 = definite violation, 1.0 = fully compliant with high confidence.
        """
        if not self.is_compliant:
            return 0.0
        # Average of hat and vest detection confidences
        scores = [s for s in [self.hat_confidence, self.vest_confidence] if s > 0]
        return round(sum(scores) / len(scores), 3) if scores else 0.0


@dataclass
class ViolationReport:
    """Summarises the safety status of a single image."""
    image_path: str
    severity: str                              # SAFE | WARNING | UNSAFE
    worker_count: int
    compliant_count: int
    violation_count: int
    worker_statuses: List[WorkerStatus] = field(default_factory=list)
    scene_confidence: float = 0.0             # overall scene-level safety confidence
    raw_violations: List[str] = field(default_factory=list)  # direct model detections

    @property
    def is_safe(self) -> bool:
        return self.severity == SEVERITY_SAFE

    def summary(self) -> str:
        lines = [
            f"=== Safety Report: {self.image_path} ===",
            f"Status           : {self.severity}",
            f"Scene confidence : {self.scene_confidence:.0%}",
            f"Workers detected : {self.worker_count}",
            f"Compliant        : {self.compliant_count}",
            f"Violations       : {self.violation_count}",
        ]
        if self.raw_violations:
            lines.append("Direct model detections:")
            for v in self.raw_violations:
                lines.append(f"  [DIRECT]  {v}")
        for ws in self.worker_statuses:
            status_str = f"COMPLIANT (score: {ws.compliance_score:.0%})" if ws.is_compliant else "VIOLATION"
            lines.append(f"  Worker {ws.worker_index}: {status_str}")
            for v in ws.violations:
                lines.append(f"    ⚠  {v}")
        return "\n".join(lines)


# ── IoU helper ──────────────────────────────────────────────────────────────────

def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """
    Intersection over Union between two (x1,y1,x2,y2) bounding boxes.
    Used to decide if a PPE detection overlaps with a worker box.
    Any IoU above IOU_OVERLAP_THRESHOLD means "this PPE belongs to this worker".
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union   = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def best_overlapping_ppe(
    worker_box: Tuple,
    ppe_detections: List[Detection],
) -> Optional[Detection]:
    """
    Returns the PPE detection with the highest IoU overlap with the given worker box,
    or None if no overlap exceeds the threshold.
    """
    best: Optional[Detection] = None
    best_iou = IOU_OVERLAP_THRESHOLD  # minimum bar to clear

    for ppe in ppe_detections:
        iou = compute_iou(worker_box, ppe.box)
        if iou > best_iou:
            best_iou = iou
            best = ppe

    return best


# ── Core rule engine ────────────────────────────────────────────────────────────

def check_safety(image_path: str, detections: List[Detection]) -> ViolationReport:
    """
    Applies all safety rules to a list of YOLO detections.

    Two-pass strategy:
      Pass A — direct: if the model itself detected 'no-hard-hat' or 'no-safety-vest',
               log those immediately (highest confidence signal).
      Pass B — overlap: for each 'worker' box, find whether a 'hard-hat' and a
               'safety-vest' box overlaps it. If either is missing, flag a violation.

    Returns a ViolationReport with per-worker statuses and a scene confidence score.
    """
    # Discard detections below confidence threshold
    detections = [d for d in detections if d.confidence >= CONFIDENCE_THRESHOLD]

    workers    = [d for d in detections if d.class_name == WORKER]
    hard_hats  = [d for d in detections if d.class_name == HARD_HAT]
    vests      = [d for d in detections if d.class_name == SAFETY_VEST]
    harnesses  = [d for d in detections if d.class_name == SAFETY_HARNESS]
    direct_viol = [d for d in detections if d.class_name in DIRECT_VIOLATION_CLASSES]

    raw_violation_msgs: List[str] = []

    # ── Pass A: direct violation classes ──────────────────────────────────────
    for v in direct_viol:
        label = "Missing hard hat" if v.class_name == NO_HARD_HAT else "Missing safety vest"
        raw_violation_msgs.append(f"{label} ({v.confidence:.0%} confidence)")

    # ── Pass B: per-worker overlap check ──────────────────────────────────────
    worker_statuses: List[WorkerStatus] = []
    compliant_count = 0

    for i, worker in enumerate(workers, start=1):
        # Find best-overlapping PPE for this worker
        hat_det     = best_overlapping_ppe(worker.box, hard_hats)
        vest_det    = best_overlapping_ppe(worker.box, vests)
        harness_det = best_overlapping_ppe(worker.box, harnesses)

        has_hat     = hat_det is not None
        has_vest    = vest_det is not None
        has_harness = harness_det is not None

        worker_violations: List[str] = []
        if not has_hat:
            worker_violations.append("No hard hat detected")
        if not has_vest:
            worker_violations.append("No safety vest detected")

        is_compliant = len(worker_violations) == 0
        if is_compliant:
            compliant_count += 1

        worker_statuses.append(WorkerStatus(
            worker_index=i,
            box=worker.box,
            has_hard_hat=has_hat,
            has_vest=has_vest,
            has_harness=has_harness,
            hat_confidence=hat_det.confidence if hat_det else 0.0,
            vest_confidence=vest_det.confidence if vest_det else 0.0,
            is_compliant=is_compliant,
            violations=worker_violations,
        ))

    # ── Severity decision ──────────────────────────────────────────────────────
    total_violations = len(raw_violation_msgs) + sum(
        1 for ws in worker_statuses if not ws.is_compliant
    )

    if total_violations == 0:
        severity = SEVERITY_SAFE
    else:
        severity = SEVERITY_UNSAFE

    # ── Scene confidence score ─────────────────────────────────────────────────
    # If safe: average of all worker compliance scores (how confident are detections)
    # If unsafe: 0.0 (definite violation)
    if severity == SEVERITY_SAFE and worker_statuses:
        scene_confidence = round(
            sum(ws.compliance_score for ws in worker_statuses) / len(worker_statuses), 3
        )
    elif severity == SEVERITY_SAFE and not worker_statuses:
        scene_confidence = 1.0   # no workers in frame → trivially safe
    else:
        scene_confidence = 0.0

    return ViolationReport(
        image_path=image_path,
        severity=severity,
        worker_count=len(workers),
        compliant_count=compliant_count,
        violation_count=total_violations,
        worker_statuses=worker_statuses,
        scene_confidence=scene_confidence,
        raw_violations=raw_violation_msgs,
    )
