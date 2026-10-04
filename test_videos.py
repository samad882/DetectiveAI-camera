import cv2
import os
import sys
sys.path.append('src')
from detection import Detector

model = Detector('models/bestonnx.onnx')
videos = ['videos/Normal_Videos_289_x264.mp4', 'videos/Shooting021_x264.mp4']

for vid in videos:
    print(f"\n--- Testing {vid} ---")
    cap = cv2.VideoCapture(vid)
    frame_count = 0
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret: break
        
        # Test detection
        dets = model.detect(frame, conf_threshold=0.40)
        
        # Are there any weird detections?
        weird = [d for d in dets if d[5] not in ('person', 'man')]
        if weird:
            print(f"Frame {frame_count}: {weird}")
            
        frame_count += 1
    cap.release()
