import ultralytics
from ultralytics import YOLO
model = YOLO("/data/zhaodexu/ultralytics/runs/dota/yolov9n/train/weights/last.pt")
model.train(data="DOTAv1.yaml", epochs=60, imgsz=640,lr0=0.001,batch=2,device=[1],project='runs/dota/yolov9n',save_period=1,iou=0.5,workers=8,resume=True)