from ultralytics import YOLO

model = YOLO("../models/yolov8n.pt")

model.train(data="/Users/cheka/Documents/Projects/Workshop/RawDataset/HandDetection.v1i.yolov8/data.yaml", epochs=100, patience=20, batch=32, save=True, save_period=5, seed=28)