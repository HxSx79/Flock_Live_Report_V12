import cv2
import time
from typing import Generator, Optional
from werkzeug.datastructures import FileStorage

class VideoStream:
    def __init__(self):
        self.cap = None
        self.test_video = None
        self.frame_interval = 1.0 / 25  # 25 FPS
        self.last_frame = None

    def set_test_video(self, video_file: FileStorage) -> None:
        try:
            temp_path = "/tmp/test_video.mp4"
            video_file.save(temp_path)
            
            if self.test_video is not None:
                self.test_video.release()
            
            self.test_video = cv2.VideoCapture(temp_path)
            if not self.test_video.isOpened():
                raise ValueError("Failed to open video file")
                
            print("Test video loaded successfully")
            
        except Exception as e:
            print(f"Error setting test video: {e}")
            raise

    def read_frame(self):
        if self.test_video is not None:
            ret, frame = self.test_video.read()
            if not ret:
                self.test_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.test_video.read()
            return ret, frame
        
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 25)
        
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.test_video is not None:
            self.test_video.release()

    def generate_frames(self, detector):
        while True:
            success, frame = self.read_frame()
            if not success:
                break

            if frame is not None and detector is not None:
                frame = detector.process_frame(frame)

            if frame is not None:
                try:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {e}")
                    continue