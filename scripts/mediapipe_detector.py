import cv2
import numpy as np
import mediapipe as mp

class MediaPipeGestureDetector:
    """Class xử lý nhận diện cử chỉ bằng MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def count_extended_fingers(self, hand_landmarks):
        """Đếm số ngón tay duỗi thẳng"""
        landmarks = hand_landmarks.landmark
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]

        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        
        thumb_dist_tip = np.sqrt((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)
        thumb_dist_ip = np.sqrt((thumb_ip.x - wrist.x)**2 + (thumb_ip.y - wrist.y)**2)
        thumb_extended = thumb_dist_tip > thumb_dist_ip * 1.1
        
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_pips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        extended_fingers = []
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            tip = landmarks[tip_id]
            pip = landmarks[pip_id]
            if tip.y < pip.y - 0.02:
                extended_fingers.append(True)
            else:
                extended_fingers.append(False)
        
        return thumb_extended, extended_fingers
    
    def recognize_gesture(self, hand_landmarks):
        """Nhận diện cử chỉ dựa trên landmarks của MediaPipe"""
        thumb_extended, fingers_extended = self.count_extended_fingers(hand_landmarks)
        num_extended = sum(fingers_extended)
        
        landmarks = hand_landmarks.landmark
        
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        thumb_pointing_up = thumb_tip.y < wrist.y - 0.05
        thumb_above_fingers = thumb_tip.y < index_mcp.y
        
        if thumb_extended and num_extended == 0 and thumb_pointing_up and thumb_above_fingers:
            return "thumbs_up", 0.92
        
        if thumb_extended and num_extended == 4:
            return "open_palm", 0.88
        
        if not thumb_extended and num_extended == 0:
            return "fist", 0.88
        
        if num_extended == 2 and fingers_extended[0] and fingers_extended[1]:
            return "v-sign", 0.85
        
        if num_extended == 1 and fingers_extended[0]:
            return "pointing", 0.85
        
        thumb_tip_coords = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        distance = np.sqrt((thumb_tip_coords.x - index_tip.x)**2 + 
                           (thumb_tip_coords.y - index_tip.y)**2)
        
        if distance < 0.06 and num_extended >= 2:
            return "ok_sign", 0.82
        
        return None, 0.0
    
    def process_frame(self, frame):
        """Xử lý frame và trả về cử chỉ nhận diện được"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                gesture, confidence = self.recognize_gesture(hand_landmarks)
                return gesture, confidence, hand_landmarks
        
        return None, 0.0, None
    
    def release(self):
        self.hands.close()
