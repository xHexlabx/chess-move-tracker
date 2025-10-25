"""
ฟังก์ชัน Helpers ที่ใช้บ่อยๆ เกี่ยวกับ OpenCV (Image Processing)
"""
import cv2
import numpy as np
from typing import Optional  # <-- [FIX 1] เพิ่ม import นี้

# [FIX 2] เพิ่ม 'Lines' และ 'IntersectionPoint' เข้าไปใน import
from src.core.typing import (
    ImageBGR, ImageGreyscale, WarpedImage, HomographyMatrix, 
    LineSegment, Lines, IntersectionPoint
)

def preprocess_for_lines(image: ImageBGR) -> ImageGreyscale:
    """
    เตรียมภาพสำหรับ Canny และ Hough Line Detection
    (ตาม Paper)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # ใช้ Canny
    # ค่า Thresholds (50, 150) เป็นค่าที่นิยมใช้ อาจต้องปรับแก้
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def find_hough_lines(edges: ImageGreyscale) -> Lines:
    """
    ค้นหาเส้นตรงจากภาพ Edges โดยใช้ Probabilistic Hough Transform
    (Paper ใช้ Standard Hough, แต่ P-Hough มักจะเร็วกว่า)
    """
    # ค่า Thresholds (minLineLength, maxLineGap) อาจต้องปรับแก้
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )
    
    if lines is None:
        return []
        
    return [tuple(line[0]) for line in lines]

def warp_image(image: ImageBGR, matrix: HomographyMatrix, size: int) -> WarpedImage:
    """
    Warp ภาพโดยใช้ Homography Matrix ที่ได้มา
    """
    return cv2.warpPerspective(image, matrix, (size, size))

def calculate_intersection(line1: LineSegment, line2: LineSegment) -> Optional[IntersectionPoint]:
    """
    คำนวณหาจุดตัดของเส้น 2 เส้น (x1,y1,x2,y2)
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # คำนวณ denominator
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None  # เส้นขนาน

    # คำนวณ numerator
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))

    # คำนวณ t และ u
    t = t_num / den
    u = u_num / den

    # ถ้าจุดตัดอยู่บนทั้งสองส่วนของเส้น
    if 0 <= t <= 1 and 0 <= u <= 1:
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))
        return (px, py)
        
    return None # จุดตัดไม่อยู่บนส่วนของเส้น