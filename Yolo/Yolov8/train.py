from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data = "data.yaml", imgsz =  640, batch = 8, epochs = 100, workers = 0, device="0")

model.save("yolov8-hand-sign.pt")