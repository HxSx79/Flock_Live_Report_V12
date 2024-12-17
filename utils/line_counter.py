"""Module for counting parts crossing the line"""
from typing import Dict, List
from .line_detector import LineDetector, LineCrossing
from .crossing_recorder import CrossingRecorder

class LineCounter:
    def __init__(self):
        self.line_detector = LineDetector()
        self.crossing_recorder = CrossingRecorder()
        self.counted_ids = set()
        self.counts = {'line1': 0, 'line2': 0}
    
    def update_counts(self, detections: List[Dict]) -> None:
        """Process detections and update counts"""
        # Detect new crossings
        crossings = self.line_detector.detect_crossings(detections)
        
        # Filter out already counted tracks
        new_crossings = [
            crossing for crossing in crossings 
            if crossing.track_id not in self.counted_ids
        ]
        
        if new_crossings:
            # Update counts
            for crossing in new_crossings:
                if 'Line1' in crossing.class_name:
                    self.counts['line1'] += 1
                elif 'Line2' in crossing.class_name:
                    self.counts['line2'] += 1
                self.counted_ids.add(crossing.track_id)
            
            # Record new crossings
            self.crossing_recorder.record_crossings(new_crossings)
    
    def draw_counting_line(self, frame):
        """Draw the counting line on the frame"""
        height, width = frame.shape[:2]
        line_x = int(width * self.line_detector.line_position)
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 2)
        return frame
    
    def get_counts(self) -> Dict[str, int]:
        """Get current counts for both lines"""
        return self.counts.copy()
    
    def reset(self) -> None:
        """Reset counter state"""
        self.counted_ids.clear()
        self.counts = {'line1': 0, 'line2': 0}
        self.line_detector.reset()