from ultralytics import YOLO

model = YOLO("yolov8-hand-sign.pt")

model.predict(source = "0", show=True, save=True, conf=0.6, line_width = 2, save_crop = False, save_txt = False, show_labels = True, show_conf= True)