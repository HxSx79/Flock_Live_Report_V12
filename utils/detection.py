import cv2
from ultralytics import YOLO
from typing import Dict, List
from .line_detector import LineDetector
from .production import ProductionTracker

class ObjectDetector:
    def __init__(self):
        self.model = YOLO('best.pt')
        self.line_detector = LineDetector()
        self.production_tracker = ProductionTracker()
        self.confidence_threshold = 0.95
        self.frame_width = 1020
        self.frame_height = 600
        self.names = self.model.names

    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        if frame is None:
            return frame
            
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        results = self.model.track(frame, persist=True)
        
        detections = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = self.names[class_id]
                x1, y1, x2, y2 = box
                
                detection = {
                    'class_name': class_name,
                    'track_id': track_id,
                    'box': box
                }
                detections.append(detection)
                
                color = (0, 255, 0) if class_name.endswith('_OK') else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{track_id} - {class_name}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Process line crossings and update production data
        frame = self.line_detector.process_detections(frame, detections)
        
        return frame