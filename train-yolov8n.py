import ultralytics
from ultralytics import YOLO
model = YOLO("yolov8n-obb.yaml")
model.train(data="DOTAv1.yaml", epochs=60, imgsz=640,lr0=0.001,batch=2,device=[0],project='runs/dota/yolov8n',save_period=1,iou=0.5,workers=8)