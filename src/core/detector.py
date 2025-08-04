from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple


class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        self.model = YOLO(model_path)
        self.device = device
        
    def detect(self, frame: np.ndarray, classes: List[int] = None, conf_threshold: float = 0.5) -> List[Dict]:
        results = self.model(frame, device=self.device, conf=conf_threshold, classes=classes)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': self.model.names[int(box.cls[0])]
                    }
                    detections.append(detection)
        
        return detections
    
    def get_class_names(self) -> Dict[int, str]:
        return self.model.names