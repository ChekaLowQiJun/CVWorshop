from ultralytics import YOLO

# Paths
model_path = "/Users/cheka/Documents/Projects/Workshop/runs/detect/train3/weights/last.pt"
data_path = "/Users/cheka/Documents/Projects/Workshop/RawDataset/YOLOData/data.yaml"

# Load model
model = YOLO(model_path)

# Training parameters
results = model.train(
    resume=True,
    data=data_path,
    epochs=100,
    batch=8,
    imgsz=640,
    device='cpu'
)
