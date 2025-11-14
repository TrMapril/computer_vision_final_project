import numpy as np
from ultralytics import YOLO

from .mediapipe_detector import MediaPipeGestureDetector
from .config import CONFIDENCE_THRESHOLD, MP_CONFIDENCE_THRESHOLD

class HybridGestureRecognition:
    """Class kết hợp YOLO và MediaPipe"""
    
    def __init__(self, model_path):
        self.yolo_model = YOLO(model_path)
        self.mp_detector = MediaPipeGestureDetector()
        
        self.yolo_weight = 0.55
        self.mediapipe_weight = 0.45
        
    def combine_results(self, yolo_gesture, yolo_conf, mp_gesture, mp_conf):
        if yolo_gesture is None and mp_gesture is None:
            return None, 0.0, "none"
        
        if yolo_gesture is not None and mp_gesture is None:
            if yolo_conf >= CONFIDENCE_THRESHOLD:
                return yolo_gesture, yolo_conf, "yolo_only"
            return None, 0.0, "low_conf"
        
        if yolo_gesture is None and mp_gesture is not None:
            if mp_conf >= MP_CONFIDENCE_THRESHOLD:
                return mp_gesture, mp_conf, "mp_only"
            return None, 0.0, "low_conf"
        
        if yolo_gesture == mp_gesture:
            combined_conf = (yolo_conf * self.yolo_weight + mp_conf * self.mediapipe_weight)
            boosted_conf = min(combined_conf * 1.25, 1.0)
            return yolo_gesture, boosted_conf, "both_agree"
        else:
            if yolo_conf >= 0.65:
                return yolo_gesture, yolo_conf * 0.95, "yolo_priority"
            elif mp_conf >= 0.85:
                return mp_gesture, mp_conf * 0.90, "mp_priority"
            elif yolo_conf > mp_conf * 0.9:
                return yolo_gesture, yolo_conf * 0.85, "yolo_priority"
            else:
                return None, 0.0, "conflict"
    
    def process_frame(self, frame):
        yolo_gesture = None
        yolo_conf = 0.0
        bbox = None
        
        results = self.yolo_model(frame, stream=True, verbose=False)
        for r in results:
            if len(r.boxes) > 0:
                box = r.boxes[0]
                cls_id = int(box.cls[0])
                yolo_gesture = self.yolo_model.names[cls_id]
                yolo_conf = float(box.conf[0])
                bbox = box.xyxy[0]
                break
        
        mp_gesture, mp_conf, hand_landmarks = self.mp_detector.process_frame(frame)
        
        final_gesture, final_conf, source = self.combine_results(
            yolo_gesture, yolo_conf, mp_gesture, mp_conf
        )
        
        return {
            'gesture': final_gesture,
            'confidence': final_conf,
            'source': source,
            'yolo_gesture': yolo_gesture,
            'yolo_conf': yolo_conf,
            'mp_gesture': mp_gesture,
            'mp_conf': mp_conf,
            'bbox': bbox,
            'hand_landmarks': hand_landmarks
        }
    
    def release(self):
        self.mp_detector.release()
