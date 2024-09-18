import torch
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
print(f'Using: {device}')
results = model.train(
    data="config.yaml",
    epochs=50,
    device=device,
    patience=10,  # Early stopping patience
    lr0=0.01,  # Initial learning rate
    lrf=0.01,  # Final learning rate as a fraction of initial learning rate
)
val_results = model.val()
print(val_results)