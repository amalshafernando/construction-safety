# Safety Rules Definition

This document formally defines every safety rule enforced by the Construction Safety Monitor.
It is a required submission deliverable.

---

## Rule 1 — Hard Hat Required

**Description**: Every worker present on the construction site must wear a hard hat (safety helmet)
at all times while in an active work zone.

**Violation condition**: A `worker` bounding box is detected with no `hard-hat` bounding box
overlapping it (IoU < 0.10), OR the model directly detects a `no-hard-hat` class.

**Severity**: High — head injuries are a leading cause of fatalities on construction sites.

**Example violations**:
- Worker visible with bare head
- Worker wearing a baseball cap instead of a hard hat
- Hard hat resting on a surface rather than on the worker's head

**Not a violation**:
- Worker is inside a site office / welfare hut (zone-based rule — not yet enforced)
- Hard hat clearly visible and worn correctly

---

## Rule 2 — High-Visibility Safety Vest Required

**Description**: Every worker in an active work zone must wear a high-visibility (hi-vis) vest
to remain visible to moving machinery and vehicle operators.

**Violation condition**: A `worker` bounding box is detected with no `safety-vest` bounding box
overlapping it (IoU < 0.10), OR the model directly detects a `no-safety-vest` class.

**Severity**: High — being struck by plant or vehicles is a leading cause of site fatalities.

**Example violations**:
- Worker wearing a plain dark jacket with no hi-vis vest over it
- Worker with vest unzipped or removed and tucked under arm
- Worker with a vest that is too worn/faded to be clearly high-visibility

**Not a violation**:
- Worker is a site visitor escorted to a safe viewing area (not yet modelled)
- Vest is clearly worn and fastened

---

## Rule 3 — Fall Protection (Harness) at Height

**Description**: Workers operating at height (scaffolding, elevated platforms) should be wearing
a visible safety harness.

**Status**: Bonus rule — implemented as an optional detection class (`safety-harness`).
Violation logic for this rule is flagged in reports but treated as a lower-severity warning
rather than a hard block, as harness visibility from a camera angle is unreliable.

**Violation condition**: Worker detected in an elevated zone (heuristic: upper portion of frame)
with no `safety-harness` detection overlapping their bounding box.

---

## Confidence and Edge Cases

| Scenario | System behaviour |
|----------|-----------------|
| Worker partially occluded | Detection attempted; low-confidence detections below 40% are discarded |
| Worker very far from camera (small box) | Detection attempted; may miss PPE on small figures |
| Multiple workers overlapping | Each worker box evaluated independently |
| Poor lighting / heavy shadow | Model may miss detections; flagged as a known limitation |
| Worker wearing a white hard hat | Detected — model trained on multiple helmet colours |

---

## Violation Severity Levels

| Level | Colour in output | Meaning |
|-------|-----------------|---------|
| Unsafe (hard) | Red banner | At least one definite PPE violation |
| Safe | Green banner | All detected workers are compliant |
| Warning (future) | Amber banner | Harness rule or uncertain detection |
