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
CONFIDENCE_THRESHOLD = 0.55  # Ngưỡng confidence cho YOLO (giảm để YOLO tham gia nhiều hơn)
MP_CONFIDENCE_THRESHOLD = 0.75  # Ngưỡng confidence cho MediaPipe (tăng để chặt chẽ hơn)
