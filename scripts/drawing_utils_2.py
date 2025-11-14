import cv2
import numpy as np

def draw_ui(frame, result, fps, tv):
    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # TẠO KHUNG MỚI: VIDEO (bên trái) + PANEL (bên phải)
    # --------------------------------------------------------
    panel_width = 260
    new_w = w + panel_width

    # canvas mới (h x (w+panel))
    canvas = np.zeros((h, new_w, 3), dtype=np.uint8)

    # đưa video vào trái canvas
    canvas[:, :w] = frame

    # panel nền xám đậm
    cv2.rectangle(canvas, (w, 0), (new_w, h), (40, 40, 40), -1)

    # ==================================================
    # 1) TV STATUS PANEL
    # ==================================================
    block_x = w + 10
    block_y = 20
    block_w = panel_width - 20
    block_h = 90

    cv2.rectangle(canvas, 
                  (block_x, block_y),
                  (block_x + block_w, block_y + block_h),
                  (70, 70, 70), -1)
    cv2.rectangle(canvas,
                  (block_x, block_y),
                  (block_x + block_w, block_y + block_h),
                  (0, 200, 255), 2)

    cv2.putText(canvas, f"TV: {'ON' if tv.is_on else 'OFF'}",
                (block_x + 10, block_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,0) if tv.is_on else (0,100,255), 2)

    cv2.putText(canvas, f"Channel: {tv.channel}",
                (block_x + 10, block_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.putText(canvas, f"Volume: {tv.volume}",
                (block_x + 125, block_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # ==================================================
    # 2) SYSTEM STATUS PANEL
    # ==================================================
    block_y += block_h + 20
    block_h2 = 120

    cv2.rectangle(canvas,
                  (block_x, block_y),
                  (block_x + block_w, block_y + block_h2),
                  (70, 70, 70), -1)
    cv2.rectangle(canvas,
                  (block_x, block_y),
                  (block_x + block_w, block_y + block_h2),
                  (255,255,255), 1)

    cv2.putText(canvas, f"FPS: {fps:.2f}",
                (block_x + 10, block_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)

    cv2.putText(canvas, f"YOLO: {result['yolo_gesture'] or 'None'} ({result['yolo_conf']:.2f})",
                (block_x + 10, block_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,165,0), 1)

    cv2.putText(canvas, f"MP: {result['mp_gesture'] or 'None'} ({result['mp_conf']:.2f})",
                (block_x + 10, block_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    cv2.putText(canvas, f"Source: {result['source']}",
                (block_x + 10, block_y + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,255,180), 1)

    # ==================================================
    # 3) GESTURE PANEL
    # ==================================================
    block_y += block_h2 + 20
    block_h3 = 80

    cv2.rectangle(canvas,
                  (block_x, block_y),
                  (block_x + block_w, block_y + block_h3),
                  (70, 70, 70), -1)
    cv2.rectangle(canvas,
                  (block_x, block_y),
                  (block_x + block_w, block_y + block_h3),
                  (0, 255, 100), 2)

    cv2.putText(canvas, f"G: {result['gesture'] or 'None'}",
                (block_x + 10, block_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(canvas, f"Conf: {result['confidence']:.2f}",
                (block_x + 10, block_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    return canvas
