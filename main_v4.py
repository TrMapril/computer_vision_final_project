import cv2
import time

from scripts.hybrid_gesture import HybridGestureRecognition
from scripts.config import (
    GESTURE_COMMAND,
    HOLD_TIME,
    CONFIDENCE_THRESHOLD
)
from scripts.drawing_utils_2 import draw_ui   # UI mới
from scripts.tv_logic import TVController   # Logic TV


def main():
    model_path = r"D:\Mon_hoc\TGMT\cv_final_project\runs\train\hand_gesture_model3\weights\best.pt"
    print(f"Loading YOLO model from: {model_path}")
    print("Initializing MediaPipe...")

    # Hybrid system
    hybrid_system = HybridGestureRecognition(model_path)

    # TV controller
    tv = TVController()

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0

    last_gesture = None
    gesture_start_time = 0
    action_triggered = False

    print("✅ System ready! Press 'q' to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Flip horizontally
        frame = cv2.flip(frame, 1)

        # Process frame with hybrid detector
        result = hybrid_system.process_frame(frame)
        detected_gesture = result["gesture"]

        # Current time
        current_time = time.time()

        # ============================
        #  GESTURE HOLD LOGIC
        # ============================
        if detected_gesture is not None and result['confidence'] >= CONFIDENCE_THRESHOLD:
            if detected_gesture == last_gesture:
                hold_duration = current_time - gesture_start_time

                if not action_triggered and hold_duration >= HOLD_TIME:
                    action_triggered = True

                    if detected_gesture in GESTURE_COMMAND:
                        # Apply TV logic
                        tv_result = tv.apply_command(detected_gesture)

                        print(
                            f">>> {tv_result} | Gesture: {detected_gesture} | "
                            f"Conf: {result['confidence']:.2f} | Source: {result['source']}"
                        )

                # Draw progress bar
                progress = min(hold_duration / HOLD_TIME, 1.0)
                bar_width = 200
                bar_height = 20
                bar_x = (frame.shape[1] - bar_width) // 2
                bar_y = frame.shape[0] - 50

                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_width, bar_y + bar_height),
                              (255, 255, 255), 2)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + int(bar_width * progress), bar_y + bar_height),
                              (0, 255, 0), -1)
                cv2.putText(frame, "Hold to execute",
                            (bar_x, bar_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

            else:
                last_gesture = detected_gesture
                gesture_start_time = current_time
                action_triggered = False
        else:
            last_gesture = None
            action_triggered = False

        # ============================
        #  FPS
        # ============================
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # ============================
        #  DRAW UI (giao diện mới)
        # ============================
        canvas = draw_ui(frame, result, fps, tv)
        cv2.imshow("Hybrid Hand Gesture Detection (YOLO + MediaPipe)", canvas)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hybrid_system.release()
    print("\n✅ System closed successfully")


if __name__ == "__main__":
    main()
