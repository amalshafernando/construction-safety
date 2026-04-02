# Construction Safety Monitor
### AI-Powered Real-Time PPE Detection & Violation Flagging
**Associate Software Engineer · AI / ML Assignment**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Safety Rules Definition](#3-safety-rules-definition)
4. [Project Structure](#4-project-structure)
5. [Phase 1 — Environment Setup](#5-phase-1--environment-setup)
6. [Phase 2 — Dataset Collection & Annotation](#6-phase-2--dataset-collection--annotation)
7. [Phase 3 — Model Training](#7-phase-3--model-training)
8. [Phase 4 — Inference & Violation Detection](#8-phase-4--inference--violation-detection)
9. [Phase 5 — Evaluation & Results](#9-phase-5--evaluation--results)
10. [Phase 6 — FastAPI Backend & Live Camera](#10-phase-6--fastapi-backend--live-camera)
11. [Running the Full System](#11-running-the-full-system)
12. [API Reference](#12-api-reference)
13. [Design Decisions & Trade-offs](#13-design-decisions--trade-offs)
14. [Known Limitations](#14-known-limitations)

---

## 1. Project Overview

Construction sites are among the most hazardous work environments in the world. This project builds a computer vision system that answers one question in real time:

> **Is this situation safe or unsafe?**

Given a live camera feed or a still image from a construction site, the system:

- **Detects workers** — locates every person in the frame, regardless of distance or pose
- **Recognises PPE** — identifies hard hats and high-visibility vests worn by each worker
- **Checks compliance** — for each detected worker, determines whether they are wearing all required safety equipment
- **Flags violations** — draws red bounding boxes around non-compliant workers and generates a human-readable report
- **Streams live** — exposes a real-time MJPEG camera feed via a FastAPI backend with a web dashboard

The core model is **YOLOv8** (You Only Look Once, version 8), fine-tuned on a custom PPE dataset sourced from Roboflow Universe and extended with custom images.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                        │
│                                                                         │
│  Camera / Image                                                         │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐   │
│  │ Preprocessor│────▶│  YOLOv8 Model    │────▶│ Safety Rule Engine │   │
│  │ resize 640² │     │  (best.pt)       │     │ (rules.py)         │   │
│  │ normalise   │     │  6-class detect  │     │ IoU overlap check  │   │
│  └─────────────┘     └──────────────────┘     └────────────────────┘   │
│                                                         │               │
│                      ┌──────────────────────────────────┘               │
│                      ▼                                                  │
│  ┌───────────────────────────────────────────────┐                      │
│  │              OUTPUT RENDERER                  │                      │
│  │  Green box = compliant   Red box = violation  │                      │
│  │  Banner: SAFE or UNSAFE + violation count     │                      │
│  └───────────────────────────────────────────────┘                      │
│                      │                                                  │
│         ┌────────────┴───────────────┐                                  │
│         ▼                            ▼                                  │
│  ┌─────────────┐            ┌───────────────────┐                       │
│  │ Annotated   │            │  Violation Report │                       │
│  │ Image/Frame │            │  .txt + .json     │                       │
│  └─────────────┘            └───────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     LIVE SYSTEM (FastAPI)                               │
│                                                                         │
│  Browser Dashboard <──── MJPEG stream <──── /stream endpoint            │
│       │                                           │                     │
│       │  POST /analyse (image upload)             │ webcam              │
│       │                                      cv2.VideoCapture           │
│       ▼                                                                 │
│  SafetyResult JSON ──── Violation Log (/violations/recent)              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| Rule engine | `src/rules.py` | Safety rules, IoU overlap logic, per-worker compliance scoring |
| Inference | `src/infer.py` | Processes images/folders/video; draws boxes; saves reports |
| Training | `src/train.py` | Fine-tunes YOLOv8 on the PPE dataset |
| Evaluation | `src/evaluate.py` | mAP metrics, scene-level confusion matrix, failure case analysis |
| Dataset tools | `src/dataset.py` | Validates dataset structure, splits raw images into train/val/test |
| API backend | `api/main.py` | FastAPI server: `/analyse`, `/stream`, `/health`, `/violations/recent` |
| Dashboard | `frontend/index.html` | Live web UI: camera stream, upload, stats, violation log |

---

## 3. Safety Rules Definition

### Rule 1 — Hard Hat Required

Every worker in an active work zone must wear a correctly fitted hard hat at all times.

| | |
|---|---|
| **Violation trigger** | Worker detected with no `hard-hat` box overlapping (IoU < 0.10), OR model detects `no-hard-hat` class |
| **Severity** | UNSAFE |
| **Examples** | Bare head; baseball cap; hard hat on ground near worker |

### Rule 2 — High-Visibility Vest Required

Every worker must wear a high-visibility vest to remain visible to machinery operators.

| | |
|---|---|
| **Violation trigger** | Worker detected with no `safety-vest` box overlapping (IoU < 0.10), OR model detects `no-safety-vest` class |
| **Severity** | UNSAFE |
| **Examples** | Plain dark jacket; vest unzipped; worn/faded vest |

### Rule 3 — Fall Protection Harness (Bonus)

Workers on elevated platforms should wear a visible safety harness.

| | |
|---|---|
| **Violation trigger** | Worker in upper frame zone with no `safety-harness` box overlapping |
| **Severity** | WARNING (lower — harness visibility from camera angle is unreliable) |

### Confidence Scoring

Rather than a binary SAFE/UNSAFE verdict, each worker receives a **compliance score** (0–1):

```
compliance_score = average(hat_detection_confidence, vest_detection_confidence)
```

This surfaces model uncertainty. A score of 0.95 means we are very confident the worker is compliant. A score of 0.52 means the PPE was detected with low confidence — worth a manual check.

---

## 4. Project Structure

```
construction-safety/
│
├── api/
│   └── main.py               # FastAPI server (Phase 6)
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── docs/
│   ├── safety_rules.md
│   └── dataset.md
│
├── frontend/
│   └── index.html             # Web dashboard
│
├── models/
│   └── weights/
│       └── best.pt            # Place trained weights here
│
├── notebooks/
│   └── train.ipynb            # Google Colab training notebook
│
├── outputs/
│   ├── annotated/
│   ├── reports/
│   └── evaluation/
│
├── src/
│   ├── dataset.py             # Phase 2
│   ├── train.py               # Phase 3
│   ├── rules.py               # Phase 4 — safety rule engine
│   ├── infer.py               # Phase 4 — image/video inference
│   └── evaluate.py            # Phase 5
│
├── data.yaml
├── requirements.txt
└── README.md
```

---

## 5. Phase 1 — Environment Setup

### Prerequisites

- Python 3.10+
- CUDA GPU recommended for training (Google Colab T4 works)
- Webcam for live streaming

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/construction-safety.git
cd construction-safety

python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Verify GPU

```python
import torch
print(torch.cuda.is_available())       # True if GPU available
print(torch.cuda.get_device_name(0))   # e.g. "Tesla T4"
```

---

## 6. Phase 2 — Dataset Collection & Annotation

### Dataset Strategy

**Stage 1 — Roboflow Universe base dataset**

Downloaded `roboflow-100/construction-site-safety` — ~3,000 pre-annotated images across indoor and outdoor construction environments.

**Stage 2 — Custom image collection**

| Search Query | Purpose |
|---|---|
| `"construction site workers no helmet"` | Unsafe scenes |
| `"scaffolding workers PPE"` | Height + harness |
| `"warehouse workers hard hat vest"` | Indoor lighting |
| `"construction site night shift"` | Low-light scenes |
| `"construction site multiple workers"` | Crowded scenes, occlusion |

**Stage 3 — Merge & export in Roboflow → YOLOv8 format**

### Annotation Rules

| Class | How to draw the box |
|---|---|
| `worker` | Tight box around full person body |
| `hard-hat` | Box around helmet only (not full head) |
| `safety-vest` | Box around vest area on torso only |
| `no-hard-hat` | Box around worker's head/upper body when clearly no helmet |
| `no-safety-vest` | Box around worker's torso when clearly no vest |
| `safety-harness` | Box around visible harness straps |

### Dataset Validation

```bash
python src/dataset.py --check
```

To split a raw image folder into train/val/test (70/20/10):

```bash
python src/dataset.py --split path/to/raw_images/
```

### Class Distribution

| Class | Count | % of total |
|---|---|---|
| worker | TBD | TBD |
| hard-hat | TBD | TBD |
| safety-vest | TBD | TBD |
| no-hard-hat | TBD | TBD |
| no-safety-vest | TBD | TBD |
| safety-harness | TBD | TBD |

> Run `python src/dataset.py --check` and fill in the table above.

---

## 7. Phase 3 — Model Training

### Architecture: YOLOv8n

YOLOv8 was chosen for four reasons:

1. **Speed** — ~10ms per frame on GPU, enabling 30+ FPS real-time inference
2. **Accuracy** — state-of-the-art COCO benchmarks
3. **Transfer learning** — pre-trained COCO weights understand shapes and human figures; fine-tuning only needs to learn our specific PPE classes
4. **Easy deployment** — single `.pt` file, runs on CPU or GPU

The `n` (nano) variant prioritises speed for live camera use. Swap to `yolov8s.pt` for better accuracy if latency allows.

### Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| `epochs` | 50 | Convergence with early stopping |
| `imgsz` | 640 | Standard YOLOv8 — good speed vs resolution |
| `batch` | 16 | Fits T4 GPU; reduce to 8 if out of memory |
| `patience` | 15 | Early stop if val mAP doesn't improve |
| `conf` (inference) | 0.40 | Avoids false positives |
| `pretrained` | yolov8n.pt (COCO) | Transfer learning |

### Training in Google Colab

```
1. Open notebooks/train.ipynb in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Fill in your Roboflow API key in Step 3
4. Run all cells (~25-40 minutes)
5. Download best.pt from Step 10
6. Place in models/weights/
```

### Training Locally

```bash
python src/train.py
```

### How to Read Training Curves

| Curve | Healthy | Problem |
|---|---|---|
| `box_loss` (train) | Steadily decreasing | Flat = LR too low |
| `mAP50` (val) | Increasing then plateau | Decreasing after peak = overfitting |
| `val/box_loss` | Tracks train loss | Diverges upward = overfitting |

---

## 8. Phase 4 — Inference & Violation Detection

### How the Safety Rule Engine Works

**Two-pass strategy in `src/rules.py`:**

**Pass A — Direct:** If YOLO detects `no-hard-hat` or `no-safety-vest` class directly, those are immediately logged as violations. Highest-confidence signal.

**Pass B — Overlap:** For each `worker` box, find the PPE with the highest IoU overlap:

```
IoU(worker_box, hat_box)  > 0.10  →  worker has a hard hat
IoU(worker_box, vest_box) > 0.10  →  worker has a vest
```

If either PPE is missing or below the threshold, the worker is flagged.

**Why IoU 0.10?** PPE boxes (just the helmet) are much smaller than full-body worker boxes. Even a perfectly worn helmet has low IoU against a body box. 0.10 is calibrated to avoid attributing a nearby worker's PPE to this one.

### Running Inference

```bash
python src/infer.py --source path/to/image.jpg
python src/infer.py --source data/images/test/
python src/infer.py --source path/to/video.mp4
```

### Outputs

**Annotated images** (`outputs/annotated/`):
- Green box + compliance score = compliant worker
- Red box + violation text = non-compliant worker
- Green/red banner across top = scene verdict

**Text report** (`outputs/reports/report_TIMESTAMP.txt`):
```
=== Safety Report: site_image_01.jpg ===
Status           : UNSAFE
Scene confidence : 0%
Workers detected : 3  |  Compliant: 2  |  Violations: 1
  Worker 3: No hard hat detected
```

**JSON report** (`outputs/reports/report_TIMESTAMP.json`) — machine-readable, suitable for integration with site management software.

---

## 9. Phase 5 — Evaluation & Results

### Running Evaluation

```bash
python src/evaluate.py --split val    # on validation set
python src/evaluate.py --split test   # on test set (run once at the end)
```

### Metrics

| Metric | What it measures | Target |
|---|---|---|
| mAP @ 0.5 | Mean Average Precision — main benchmark | > 0.50 |
| Precision | Of all drawn boxes, % that were correct | > 0.70 |
| Recall | Of all real objects, % the model found | > 0.65 |
| Scene accuracy | % of images correctly labelled SAFE/UNSAFE | > 0.80 |
| Violation recall | % of real violations caught | **Most critical** |

### Results

> Fill in after training and running evaluate.py

| Metric | Value |
|---|---|
| mAP @ 0.5 | TBD |
| mAP @ 0.5:0.95 | TBD |
| Precision | TBD |
| Recall | TBD |
| Scene accuracy | TBD |
| Violation recall | TBD |
| False alarm rate | TBD |

### Evaluation Outputs

| File | Contents |
|---|---|
| `outputs/evaluation/evaluation_summary.png` | Per-class AP bars + scene confusion matrix |
| `outputs/evaluation/eval_results_val.json` | All metrics as JSON |
| `outputs/evaluation/failure_cases/` | Images labelled FALSE ALARM or MISSED VIOLATION |

---

## 10. Phase 6 — FastAPI Backend & Live Camera

### Why a Backend?

The assignment requires real-time visual input from a camera. A FastAPI backend provides:

- Live MJPEG stream any browser can display
- Image upload endpoint for testing
- Persistent violation log
- Clean separation of model (Python) and UI (HTML/JS)

### Starting the Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

### Dashboard Features

**Live Camera tab** — click "Start Camera" to begin MJPEG stream with real-time PPE detection overlaid. Violation log updates every 3 seconds.

**Upload Image tab** — drag-and-drop a JPEG/PNG to analyse it. Annotated image is returned inline.

**Stats panel** — tracks safe scenes, total violations, and total workers across the session.

### Changing the Camera Source

Edit `api/main.py`:

```python
CAMERA_INDEX = 0    # 0=default webcam, 1=USB camera
# or IP/RTSP camera:
# CAMERA_INDEX = "rtsp://user:pass@192.168.1.100:554/stream"
```

---

## 11. Running the Full System

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (or skip if you have best.pt)
#    → Open notebooks/train.ipynb in Colab, run all cells, download best.pt
python src/train.py   # local alternative

# 3. Place weights
cp path/to/best.pt models/weights/best.pt

# 4. Test on images
python src/infer.py --source data/images/test/

# 5. Evaluate
python src/evaluate.py --split val

# 6. Start live server
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `Model not loaded` in browser | Check `models/weights/best.pt` exists, restart server |
| `Camera not found` | Change `CAMERA_INDEX` in `api/main.py` |
| `CUDA out of memory` | Reduce `batch` from 16 to 8 in `src/train.py` |
| Too many false alarms | Increase `CONFIDENCE` to 0.50 in `src/infer.py` and `api/main.py` |
| Too many missed violations | Decrease `CONFIDENCE`; add more violation examples to dataset |
| Port 8000 busy | Use `--port 8001` |

---

## 12. API Reference

### `POST /analyse`

Upload an image → JSON safety report.

**Request:** `multipart/form-data`, field `file` (JPEG or PNG)

**Response:**
```json
{
  "severity": "UNSAFE",
  "scene_confidence": 0.0,
  "worker_count": 2,
  "compliant_count": 1,
  "violation_count": 1,
  "workers": [
    {
      "worker_index": 1,
      "is_compliant": true,
      "compliance_score": 0.82,
      "has_hard_hat": true,
      "has_vest": true,
      "violations": []
    },
    {
      "worker_index": 2,
      "is_compliant": false,
      "compliance_score": 0.0,
      "has_hard_hat": false,
      "has_vest": true,
      "violations": ["No hard hat detected"]
    }
  ],
  "processing_time_ms": 34.2,
  "timestamp": "2024-01-15T10:30:00.123"
}
```

### `GET /stream`

MJPEG stream from webcam. Embed in HTML:

```html
<img src="http://localhost:8000/stream" />
```

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "Tesla T4",
  "uptime_seconds": 142.5,
  "frames_processed": 4280
}
```

### `GET /violations/recent?limit=20`

```json
{
  "total_logged": 7,
  "events": [
    {
      "timestamp": "2024-01-15T10:31:22",
      "violation_count": 1,
      "worker_count": 3,
      "details": ["No hard hat detected"]
    }
  ]
}
```

### `DELETE /violations/clear`

Clears the in-memory violation log.

### `GET /docs`

Auto-generated Swagger UI (FastAPI built-in) — try all endpoints interactively.

---

## 13. Design Decisions & Trade-offs

### Why YOLOv8 and not Faster R-CNN?

YOLOv8 processes at ~10ms/frame on a T4 GPU vs ~50ms for Faster R-CNN. For a live streaming safety monitor at 30 FPS, speed was the deciding constraint. Faster R-CNN is more accurate on small objects but too slow for real-time use. RT-DETR was considered but has a larger model size and less mature tooling for custom training.

### Why two violation detection strategies (direct + overlap)?

Some Roboflow datasets annotate violations as their own class (`no-hard-hat`). Others annotate only positive PPE (`hard-hat`). By implementing both strategies in `rules.py`, the system works correctly regardless of annotation style, and the two strategies cross-validate each other.

### Why IoU threshold of 0.10?

A full-body `worker` box is 10–15× larger than a `hard-hat` box. Even a perfectly placed helmet has IoU ~0.10–0.15 against its worker's body box. The threshold is calibrated empirically to avoid attributing one worker's PPE to a nearby worker.

### Why FastAPI over Flask?

FastAPI provides async request handling (critical for streaming), automatic request validation via Pydantic, auto-generated Swagger UI, and type-annotated responses. For streaming endpoints, FastAPI's async model is meaningfully better than Flask's synchronous default.

### Why MJPEG and not WebRTC?

WebRTC provides lower latency but requires a signalling server and significant setup complexity. MJPEG streams work natively in `<img>` tags with zero JavaScript, are trivial to implement with FastAPI's `StreamingResponse`, and provide acceptable latency (100–300ms) for a monitoring display. MJPEG was the pragmatic choice for a single-developer assignment.

---

## 14. Known Limitations

| Limitation | Impact | Potential fix |
|---|---|---|
| Small workers (far from camera) | PPE too small to classify | Higher resolution; `yolov8m` |
| Heavy occlusion | Worker partially hidden | Temporal tracking across frames |
| Poor lighting / night | Confidence drops | Add night-time training images |
| Harness class under-represented | Low AP for harness | More annotated harness examples |
| No worker identity tracking | Same person counted separately per frame | Add ByteTrack tracker |
| Single camera viewpoint | Blind spots | Multi-camera setup |
| Binary scene verdict | One violation = whole scene UNSAFE | Zone-based rules |

---

## Requirements

- Python 3.10+
- GPU recommended for training (Google Colab T4 sufficient)
- Webcam for live monitoring

```
ultralytics>=8.2.0    fastapi>=0.111.0
roboflow>=1.1.0       uvicorn[standard]>=0.29.0
opencv-python>=4.8.0  python-multipart>=0.0.9
torch>=2.0.0          pydantic>=2.0.0
matplotlib>=3.7.0     PyYAML>=6.0
```

See `requirements.txt` for full pinned versions.

---

*Built for the Associate Software Engineer · AI/ML assignment.*
*Dataset extended from Roboflow Universe (Construction Site Safety, MIT license).*
