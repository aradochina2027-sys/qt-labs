import time
from typing import List, Dict, Any
from PIL import Image
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> Dict[str, Any]:
        start_time = time.perf_counter()

        results = self.model.predict(source=image, conf=conf_threshold, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            class_name = result.names[cls_id]
            confidence = float(box.conf[0].item())

            xyxy = box.xyxy[0].tolist()

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 4),
                "bbox": [round(coord, 2) for coord in xyxy]
            })

        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "count": len(detections),
            "inference_time_ms": execution_time_ms,
            "objects": detections
        }