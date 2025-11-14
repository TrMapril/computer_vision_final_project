class TVController:
    """
    Bộ điều khiển TV mô phỏng:
    - TV bật / tắt
    - Chuyển kênh (1–10)
    - Tăng/giảm âm lượng (0–100)
    - Không cho xử lý cử chỉ khác khi TV đang tắt
    """

    def __init__(self):
        self.is_on = False          # trạng thái TV
        self.volume = 50            # âm lượng mặc định
        self.channel = 1            # kênh mặc định
        self.total_channels = 10    # số lượng kênh

    # =========================
    #  XỬ LÝ HÀNH ĐỘNG TỪ CỬ CHỈ
    # =========================
    def apply_command(self, gesture):
        """
        Nhận cử chỉ (gesture) và xử lý logic TV.
        Trả về nội dung hành động đã thực hiện (string).
        """

        # ===== Nếu TV đang tắt → chỉ cho phép bật =====
        if not self.is_on:
            if gesture == "fist":  # bật TV
                return self.turn_on()
            else:
                return "⚠️ TV đang tắt — không thể thực hiện lệnh!"

        # ===== TV đang bật — có thể thực hiện tất cả lệnh =====
        if gesture == "open_palm":
            return self.turn_off()
        elif gesture == "pointing":
            return self.volume_up()
        elif gesture == "thumbs_up":
            return self.volume_down()
        elif gesture == "ok_sign":
            return self.next_channel()
        elif gesture == "v-sign":
            return self.previous_channel()
        elif gesture == "fist":
            return "⚠️ TV đã bật rồi!"
        else:
            return "⚠️ Cử chỉ không hợp lệ!"

    # =========================
    #  CÁC HÀNH ĐỘNG TV CỤ THỂ
    # =========================

    def turn_on(self):
        if not self.is_on:
            self.is_on = True
            return "📺 TV đã bật!"
        return "⚠️ TV đã bật rồi!"

    def turn_off(self):
        if self.is_on:
            self.is_on = False
            return "📺 TV đã tắt!"
        return "⚠️ TV đang tắt rồi!"

    def volume_up(self):
        self.volume = min(100, self.volume + 5)
        return f"🔊 Tăng âm lượng → {self.volume}"

    def volume_down(self):
        self.volume = max(0, self.volume - 5)
        return f"🔉 Giảm âm lượng → {self.volume}"

    def next_channel(self):
        self.channel = 1 if self.channel == self.total_channels else self.channel + 1
        return f"📺 Chuyển sang kênh {self.channel}"

    def previous_channel(self):
        self.channel = self.total_channels if self.channel == 1 else self.channel - 1
        return f"📺 Trở về kênh {self.channel}"
