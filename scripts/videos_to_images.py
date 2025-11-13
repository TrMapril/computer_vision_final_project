import cv2
import os

# Thư mục chứa video và ảnh đầu ra
video_dir = r"E:\Nhan_Dang_Cu_Chi_Tay\video\videos"
output_dir = r"E:\Nhan_Dang_Cu_Chi_Tay\data"

# Danh sách video cử chỉ
gestures = [
    "fist_1", "fist_2", "fist_3",
    "ok_sign_1", "ok_sign_2", "ok_sign_3",
    "open_palm_1", "open_palm_2", "open_palm_3",
    "pointing_1", "pointing_2", "pointing_3",
    "thumbs_up_1", "thumbs_up_2", "thumbs_up_3",
    "v_sign_1", "v_sign_2", "v_sign_3"
]

# Số lượng ảnh muốn trích xuất mỗi video
frames_to_extract = 50

for gesture in gestures:
    # Lấy phần "gốc" của tên cử chỉ (vd: 'fist_1' → 'fist')
    gesture_root = gesture.split('_')[0]

    # Tạo thư mục chung theo gốc
    gesture_dir = os.path.join(output_dir, gesture_root)
    os.makedirs(gesture_dir, exist_ok=True)

    # Đường dẫn video
    video_path = os.path.join(video_dir, f"{gesture}.mp4")

    # Mở video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // frames_to_extract)
    count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            img_name = f"{gesture}_{count}.jpg"
            img_path = os.path.join(gesture_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_path}")
            count += 1

        frame_id += 1

    cap.release()
    print(f"✅ Done extracting {count} images for {gesture} → stored in {gesture_root}")

print("\n🎉 Hoàn thành trích xuất toàn bộ video!")
