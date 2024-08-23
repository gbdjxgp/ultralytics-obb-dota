import ultralytics
from ultralytics import YOLO
model = YOLO("runs/dota/yolov10n/train3/weights/last.pt")
model.train(data="DOTAv1.yaml", epochs=60, imgsz=640,lr0=0.001,batch=2,device=[7],project='runs/dota/yolov10n',save_period=1,iou=0.5,workers=8,task="obb",val=False,resume=True)