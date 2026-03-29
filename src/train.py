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

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_CONFIG  = "data.yaml"          # Path to dataset config
MODEL_BASE   = "yolov8n.pt"         # Start from YOLOv8 nano (pre-trained on COCO)
                                    # Options: yolov8n, yolov8s, yolov8m, yolov8l
PROJECT_DIR  = "outputs/runs"       # Where training results are saved
RUN_NAME     = "ppe_detector_v1"    # Name for this training run

# Training hyperparameters
EPOCHS       = 50                   # How many times to loop through all training data
IMAGE_SIZE   = 640                  # Input image size (YOLOv8 default)
BATCH_SIZE   = 16                   # Images per gradient update (lower if GPU runs out of memory)
PATIENCE     = 10                   # Stop early if no improvement after N epochs
DEVICE       = "0"                  # "0" = first GPU, "cpu" = CPU only

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
