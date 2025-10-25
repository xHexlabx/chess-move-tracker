"""
State 1: Board Localisation
ทำหน้าที่ค้นหากระดานและ Warp ภาพให้ตรง

[UPDATE 7]:
- แก้ไข Bug ร้ายแรงใน `_order_corners` ที่ทำให้เกิดภาพเทา
- เปลี่ยนไปใช้กลยุทธ์ "Sort by X-coordinate"
- Logic นี้จะคืนค่า 4 มุมที่ถูกต้อง (ไม่ซ้ำ)
  แม้ภาพจะหมุน 45 องศา
"""
import cv2
import numpy as np
from typing import Tuple

from src.core.typing import ImageBGR, WarpedImage, HomographyMatrix
from src.core.exceptions import BoardNotFoundException
from src.utils import image_utils

class BoardLocator:
    """
    คลาสสำหรับค้นหาและสกัดภาพกระดานหมากรุก
    """
    def __init__(self, warped_size: int = 600):
        self.warped_size = warped_size
        self.debug_mask: np.ndarray = None 

    def _get_avg_lightness(self, image: ImageBGR) -> float:
        h, w = image.shape[:2]
        crop = image[h//4:h*3//4, w//4:w*3//4]
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        return np.mean(l_channel)

    def _isolate_board_mask(self, image: ImageBGR) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        lower_white_square = np.array([0, 0, 180])
        upper_white_square = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white_square, upper_white_square)
        
        combined_mask = cv2.bitwise_or(green_mask, white_mask)

        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_closed = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        self.debug_mask = mask_closed 

        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise BoardNotFoundException("ไม่พบ Contours จากการ Mask สี")
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        return largest_contour

    def _find_corners_from_contour(self, contour: np.ndarray) -> np.ndarray:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        epsilon = 0.02 * perimeter
        approx_corners = cv2.approxPolyDP(hull, epsilon, True)
        
        num_corners = len(approx_corners)
        attempts = 0
        while num_corners > 4 and attempts < 10:
            attempts += 1
            epsilon = (0.02 + 0.01 * attempts) * perimeter
            approx_corners = cv2.approxPolyDP(hull, epsilon, True)
            num_corners = len(approx_corners)

        if num_corners == 4:
            return approx_corners.reshape(4, 2)
        else:
            raise BoardNotFoundException(
                f"ไม่สามารถประมาณค่าเป็น 4 มุมได้ (พบ {num_corners} มุม)"
            )

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        [NEW - UPDATE 7] เรียงลำดับมุม 4 มุม (TL, TR, BR, BL)
        ใช้วิธี Sort by X-coordinate ที่ทนทานกว่า
        """
        # 1. เรียงลำดับ corners ตามพิกัด x (จากซ้ายไปขวา)
        x_sorted = corners[np.argsort(corners[:, 0]), :]
        
        # 2. แยกเป็น "คู่ซ้าย" (2 จุดแรก) และ "คู่ขวา" (2 จุดหลัง)
        left_corners = x_sorted[:2, :]
        right_corners = x_sorted[2:, :]
        
        # 3. เรียง "คู่ซ้าย" ตามพิกัด y (บนไปล่าง)
        left_corners = left_corners[np.argsort(left_corners[:, 1]), :]
        (tl, bl) = left_corners
        
        # 4. เรียง "คู่ขวา" ตามพิกัด y (บนไปล่าง)
        # (หมายเหตุ: สำหรับภาพหมุน 45 องศา, BR อาจมี y น้อยกว่า TR
        #  เราจึงใช้ np.argmax(y) สำหรับ BR และ np.argmin(y) สำหรับ TR)
        
        # หามุม Top-Right (TR) และ Bottom-Right (BR) จากคู่ขวา
        if right_corners[0][1] < right_corners[1][1]:
            (tr, br) = right_corners
        else:
            (br, tr) = right_corners
            
        # (วิธีที่ปลอดภัยกว่าคือ argmin/argmax)
        tr = right_corners[np.argmin(right_corners[:, 1]), :]
        br = right_corners[np.argmax(right_corners[:, 1]), :]

        ordered = np.array([tl, tr, br, bl], dtype="float32")
        return ordered


    def _fix_rotation_by_piece_color(self, warped_image: WarpedImage) -> WarpedImage:
        """
        [UPDATE 6] กลยุทธ์: ตรวจสอบสีของ "หมาก" ไม่ใช่ "ช่อง"
        """
        
        hsv = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
        
        # Mask สีหมากขาว
        lower_white_piece = np.array([0, 0, 190])
        upper_white_piece = np.array([180, 40, 255])
        white_piece_mask = cv2.inRange(hsv, lower_white_piece, upper_white_piece)
        
        # Mask สีหมากดำ
        lower_black_piece = np.array([0, 0, 0])
        upper_black_piece = np.array([180, 255, 60])
        black_piece_mask = cv2.inRange(hsv, lower_black_piece, upper_black_piece)
        
        # กำหนด "โซน"
        h, w = warped_image.shape[:2]
        zone_height = h // 4 # 2 แถว
        zone_width = w // 4  # 2 คอลัมน์

        top_zone = (0, zone_height)
        bottom_zone = (h - zone_height, h)
        left_zone = (0, zone_width)
        right_zone = (w - zone_width, w)
        
        # นับ pixel สีขาว/ดำ ในแต่ละโซน
        white_in_top = np.sum(white_piece_mask[top_zone[0]:top_zone[1], :])
        black_in_top = np.sum(black_piece_mask[top_zone[0]:top_zone[1], :])
        
        white_in_bottom = np.sum(white_piece_mask[bottom_zone[0]:bottom_zone[1], :])
        black_in_bottom = np.sum(black_piece_mask[bottom_zone[0]:bottom_zone[1], :])
        
        white_in_left = np.sum(white_piece_mask[:, left_zone[0]:left_zone[1]])
        black_in_left = np.sum(black_piece_mask[:, left_zone[0]:left_zone[1]])
        
        white_in_right = np.sum(white_piece_mask[:, right_zone[0]:right_zone[1]])
        black_in_right = np.sum(black_piece_mask[:, right_zone[0]:right_zone[1]])
        
        # คำนวณ "คะแนน" ของแต่ละแกน
        score_0_deg = white_in_bottom + black_in_top
        score_180_deg = white_in_top + black_in_bottom
        score_90_cw = white_in_left + black_in_right # 90CW = ขาวซ้าย, ดำขวา
        score_90_ccw = white_in_right + black_in_left # 90CCW = ขาวขวา, ดำซ้าย

        scores = {
            "0": score_0_deg,
            "180": score_180_deg,
            "90cw": score_90_cw,
            "90ccw": score_90_ccw
        }

        # ค้นหาทิศทางที่คะแนนสูงสุด
        # (เพิ่ม 1 เข้าไปใน score_0_deg เพื่อเป็น "default" หากคะแนนเท่ากันหมด)
        scores["0"] += 1 
        orientation = max(scores, key=scores.get)

        print(f"Orientation Check: Found '{orientation}' configuration (Scores: {scores})")

        # สั่งหมุนภาพ
        if orientation == "0":
            return warped_image
        elif orientation == "180":
            return cv2.rotate(warped_image, cv2.ROTATE_180)
        elif orientation == "90cw":
            # ภาพหมุน 90CW (ขวา) -> เราต้องหมุนกลับ 90CCW (ซ้าย)
            return cv2.rotate(warped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif orientation == "90ccw":
            # ภาพหมุน 90CCW (ซ้าย) -> เราต้องหมุนกลับ 90CW (ขวา)
            return cv2.rotate(warped_image, cv2.ROTATE_90_CLOCKWISE)
            
        return warped_image

    def find_and_warp(self, image: ImageBGR) -> Tuple[WarpedImage, HomographyMatrix]:
        """
        Main method:
        1. Mask, Contour, Hull, approxPolyDP
        2. [NEW] Order corners using robust X-Sort
        3. Warp
        4. Fix rotation by piece color
        """
        
        largest_contour = self._isolate_board_mask(image)
        corners = self._find_corners_from_contour(largest_contour)
        
        # [NEW] เรียกใช้ _order_corners ที่แก้ไขแล้ว
        ordered_src_pts = self._order_corners(corners)

        dst_pts = np.array([
            [0, 0],
            [self.warped_size - 1, 0],
            [self.warped_size - 1, self.warped_size - 1],
            [0, self.warped_size - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(ordered_src_pts, dst_pts)
        initial_warp = image_utils.warp_image(image, matrix, self.warped_size)
        
        final_warped_image = self._fix_rotation_by_piece_color(initial_warp)
        
        return final_warped_image, matrix