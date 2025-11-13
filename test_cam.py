import cv2

def main():
    # Mở webcam (0 là camera mặc định)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở webcam")
        return

    while True:
        # Đọc frame từ webcam
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được frame từ webcam")
            break

        # Hiển thị frame
        cv2.imshow("Webcam Test", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
