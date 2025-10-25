"""
ฟังก์ชัน Helpers ที่ใช้บ่อยๆ เกี่ยวกับ OpenCV (Image Processing)
"""
import cv2
import numpy as np
from typing import Optional, List # [FIX] เพิ่ม List

from src.core.typing import (
    ImageBGR, ImageGreyscale, WarpedImage, HomographyMatrix, 
    LineSegment, Lines, IntersectionPoint, SquareImage # [FIX] เพิ่ม SquareImage
)

# --- (โค้ด State 1 ของเดิม) ---
def preprocess_for_lines(image: ImageBGR) -> ImageGreyscale:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def find_hough_lines(edges: ImageGreyscale) -> Lines:
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=100, minLineLength=100, maxLineGap=10
    )
    if lines is None: return []
    return [tuple(line[0]) for line in lines]

def warp_image(image: ImageBGR, matrix: HomographyMatrix, size: int) -> WarpedImage:
    return cv2.warpPerspective(image, matrix, (size, size))

def calculate_intersection(line1: LineSegment, line2: LineSegment) -> Optional[IntersectionPoint]:
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0: return None
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
    t = t_num / den
    u = u_num / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1)))
    return None
# --- (จบโค้ด State 1) ---


def crop_squares_from_warped(warped_image: WarpedImage, context_ratio: float) -> List[SquareImage]:
    """
    [NEW] ตัดภาพ WarpedImage ออกเป็น 64 ช่องเล็ก
    
    ปฏิบัติตาม Paper:
    "The squares are cropped with a 50% increase in width and height 
    to include contextual information."
    
    :param warped_image: ภาพกระดานที่ Warp แล้ว
    :param context_ratio: อัตราส่วนขยายขอบ (เช่น 0.5 คือ 50%)
    :return: List 64 ภาพ (SquareImage)
    """
    height, width = warped_image.shape[:2]
    if height != width:
        raise ValueError("Warped image must be square")
        
    sq_size = height // 8
    
    # คำนวณขนาดขอบที่เพิ่มเข้ามา
    context_padding = int((sq_size * context_ratio) / 2)
    
    squares = []
    for r in range(8): # row
        for c in range(8): # col
            
            # 1. หาพิกัดช่องแบบปกติ
            y1, y2 = r * sq_size, (r + 1) * sq_size
            x1, x2 = c * sq_size, (c + 1) * sq_size
            
            # 2. ขยายขอบด้วย Context Padding
            y1_pad = max(0, y1 - context_padding)
            y2_pad = min(height, y2 + context_padding)
            x1_pad = max(0, x1 - context_padding)
            x2_pad = min(width, x2 + context_padding)
            
            # 3. Crop
            square_with_context = warped_image[y1_pad:y2_pad, x1_pad:x2_pad]
            squares.append(square_with_context)
            
    return squares