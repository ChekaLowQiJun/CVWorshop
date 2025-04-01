from ultralytics import YOLO

model = YOLO("../models/yolov8n.pt")

#Replace with the absolute path 
model.train(data="/Users/cheka/Documents/Projects/Workshop/RawDataset/YOLOData/data.yaml", epochs=100, patience=20, batch=32, save=True, save_period=5, seed=28)
