import numpy as np
from ultralytics import YOLO
from typing import Dict


class YOLODetector:
    """YOLO v8 object detector for football analysis"""

    def __init__(self, model_path: str = "yolo26x.pt", device: str = "cpu", conf: float = 0.5, iou: float = 0.45):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou

    def detect(self, frame: np.ndarray) -> Dict:
        results = self.model(frame, conf=self.conf, iou=self.iou, device=self.device, nms=False)
        
        detections = {
            "players": [],
            "goalkeepers": [],
            "balls": [],
            "referees": []
        }
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = result.names.get(cls, str(cls)).lower()
                
                bbox = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": conf_score,
                    "class": cls,
                    "class_name": class_name,
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                }
                
                # Classify detected objects
                if class_name in {"person", "player"}:
                    detections["players"].append(bbox)
                elif class_name == "goalkeeper":
                    detections["goalkeepers"].append(bbox)
                elif class_name in {"ball", "sports ball"}:
                    detections["balls"].append(bbox)
                elif class_name == "referee":
                    detections["referees"].append(bbox)
        
        return detections
    
