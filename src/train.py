"""
train.py
--------
Trains a YOLOv8 model on the construction safety dataset.

Usage (local):
    python src/train.py

Usage (Colab):
    Run the equivalent cells in notebooks/train.ipynb
"""

from ultralytics import YOLO
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_CONFIG  = os.getenv("DATA_CONFIG", "data.yaml")
MODEL_BASE   = os.getenv("MODEL_BASE", "yolov8n.pt")
PROJECT_DIR  = os.getenv("PROJECT_DIR", "outputs/runs")
RUN_NAME     = os.getenv("RUN_NAME", "ppe_detector_v1")

# Training hyperparameters
EPOCHS       = int(os.getenv("EPOCHS", 50))
IMAGE_SIZE   = int(os.getenv("IMAGE_SIZE", 640))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 16))
PATIENCE     = int(os.getenv("PATIENCE", 10))
DEVICE       = os.getenv("DEVICE", "0")

# ── Training ───────────────────────────────────────────────────────────────────

def train():
    print(f"[INFO] Loading base model: {MODEL_BASE}")
    model = YOLO(MODEL_BASE)

    print(f"[INFO] Starting training for {EPOCHS} epochs...")
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,          # Overwrite if run name already exists
        save=True,              # Save best and last weights
        plots=True,             # Save training curve plots
        val=True,               # Run validation after each epoch
        verbose=True,
    )

    best_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    print(f"\n[DONE] Training complete.")
    print(f"[INFO] Best weights saved to: {best_weights}")
    print(f"[INFO] Copy best.pt to models/weights/ before running inference.")

    return results


if __name__ == "__main__":
    train()
