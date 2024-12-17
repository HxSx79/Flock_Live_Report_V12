import cv2
from typing import Dict, List
from .flock_report import FlockReport

class LineDetector:
    def __init__(self):
        self.line_position = 0.5  # Middle of the frame
        self.previous_positions = {}
        self.flock_report = FlockReport()
        self.counted_ids = set()

    def process_detections(self, frame: cv2.Mat, detections: List[Dict]) -> cv2.Mat:
        height, width = frame.shape[:2]
        line_x = int(width * self.line_position)
        
        # Draw counting line
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 2)
        
        for detection in detections:
            track_id = detection['track_id']
            if track_id in self.counted_ids:
                continue
                
            x1, y1, x2, y2 = detection['box']
            center_x = (x1 + x2) / 2
            
            if track_id in self.previous_positions:
                prev_x = self.previous_positions[track_id]
                
                # Check if crossed the line
                if (prev_x < line_x and center_x >= line_x) or \
                   (prev_x > line_x and center_x <= line_x):
                    self.counted_ids.add(track_id)
                    self.flock_report.record_crossing(detection['class_name'])
            
            self.previous_positions[track_id] = center_x
        
        return frame

    def reset(self):
        self.previous_positions.clear()
        self.counted_ids.clear()