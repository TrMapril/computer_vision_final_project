import cv2
import time
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Turn off all YOLO logs
LOGGER.setLevel(50)

# Map gesture -> command
GESTURE_COMMAND = {
    "fist": "bật TV",
    "open_palm": "tắt TV",
    "ok_sign": "chuyển kênh sau",
    "v-sign": "chuyển kênh trước",
    "pointing": "tăng âm lượng 5dv",
    "thumbs_up": "giảm âm lượng 5dv"
}

HOLD_TIME = 1  # seconds

def main():
    model_path = r"D:\Mon_hoc\TGMT\computer_vision_final_project\runs\train\hand_gesture_model5\weights\best.pt"
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return

    prev_time = 0

    # Tracking variables
    last_gesture = None
    gesture_start_time = 0
    action_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        results = model(frame, stream=True)

        detected_gesture = None

        # Detect gesture
        for r in results:
            if len(r.boxes) > 0:
                box = r.boxes[0]  # lấy detection đầu tiên
                cls_id = int(box.cls[0])
                detected_gesture = model.names[cls_id]
                # Draw bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{detected_gesture} {conf:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

        # === Gesture holding logic ===
        current_time = time.time()

        if detected_gesture is not None:
            if detected_gesture == last_gesture:
                # Check if gesture held long enough
                if not action_triggered and (current_time - gesture_start_time >= HOLD_TIME):
                    action_triggered = True
                    if detected_gesture in GESTURE_COMMAND:
                        print(">>>", GESTURE_COMMAND[detected_gesture])
            else:
                # Gesture changed → reset
                last_gesture = detected_gesture
                gesture_start_time = current_time
                action_triggered = False
        else:
            # Không detect → reset
            last_gesture = None
            action_triggered = False

        # === FPS ===
        curr = time.time()
        fps = 1 / (curr - prev_time) if prev_time != 0 else 0
        prev_time = curr

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Hand Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
