# Construction Safety Monitor

An AI-powered computer vision system that monitors construction site images and detects
safety violations in real time using YOLOv8 object detection.

---

## What It Does

- Detects workers in a scene (regardless of distance or pose)
- Recognises Personal Protective Equipment (PPE): hard hats and high-visibility vests
- Flags workers who are missing required safety gear
- Generates annotated images and human-readable violation reports

---

## Safety Rules Enforced

| Rule | Violation Condition |
|------|---------------------|
| Hard hat required | Worker detected without a helmet in frame |
| High-vis vest required | Worker detected without a safety vest |
| No partial PPE | Helmet detected but not worn on head counts as violation |

See `docs/safety_rules.md` for full definitions with example images.

---

## Project Structure

```
construction-safety/
├── data/
│   ├── images/
│   │   ├── train/          # Training images
│   │   ├── val/            # Validation images
│   │   └── test/           # Test images
│   └── labels/
│       ├── train/          # YOLO format .txt label files
│       ├── val/
│       └── test/
├── notebooks/
│   └── train.ipynb         # Google Colab training notebook
├── models/
│   └── weights/            # Saved .pt model files go here
├── src/
│   ├── train.py            # Training script
│   ├── infer.py            # Run inference on an image or video
│   ├── rules.py            # Safety rule logic
│   └── dataset.py          # Dataset preparation helpers
├── outputs/
│   ├── annotated/          # Output images with bounding boxes
│   └── reports/            # Violation report text files
├── docs/
│   └── safety_rules.md     # Formal safety rule definitions
├── data.yaml               # YOLO dataset config
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/construction-safety.git
cd construction-safety
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset
- Place images in `data/images/train/` and `data/images/val/`
- Place YOLO label files in `data/labels/train/` and `data/labels/val/`
- Update `data.yaml` if you change class names

### 4. Train the model
```bash
python src/train.py
```
Or open `notebooks/train.ipynb` in Google Colab for GPU-accelerated training.

### 5. Run inference on an image
```bash
python src/infer.py --source path/to/image.jpg
```

### 6. Run inference on a folder
```bash
python src/infer.py --source data/images/test/
```

---

## Model

- **Architecture**: YOLOv8n (nano) — fast and suitable for real-time inference
- **Pre-trained on**: COCO dataset (transfer learning)
- **Fine-tuned on**: Custom construction site PPE dataset

---

## Evaluation Results

> Fill this section in after training is complete.

| Metric | Value |
|--------|-------|
| mAP@0.5 | TBD |
| Precision | TBD |
| Recall | TBD |

### Known Limitations
- Performance degrades when workers are heavily occluded
- Distant workers (small bounding boxes) are harder to classify
- Lighting conditions (night / strong shadows) reduce confidence

---

## Dataset

- Source: Extended from [Roboflow Universe — Construction Site Safety](https://universe.roboflow.com/roboflow-100/construction-site-safety)
- Custom additions: [describe your own images here]
- Total images: TBD
- Class distribution: TBD

See `docs/dataset.md` for full details.

---

## Requirements

- Python 3.10+
- CUDA-compatible GPU recommended for training (Google Colab works)
- See `requirements.txt` for Python packages
