"""
กำหนด Type Hints ที่ใช้บ่อยในโปรเจกต์
"""
from typing import List, Tuple, Any, Optional
import numpy as np

# ประเภทของภาพ
ImageBGR = np.ndarray         # ภาพ OpenCV (channels BGR)
ImageGreyscale = np.ndarray   # ภาพ 1 channel
WarpedImage = np.ndarray      # ภาพกระดานที่ถูก warp แล้ว
HomographyMatrix = np.ndarray # 3x3 matrix จาก cv2.findHomography

# ประเภทของข้อมูลประมวลผล
# (x1, y1, x2, y2)
LineSegment = Tuple[int, int, int, int] 
Lines = List[LineSegment]
IntersectionPoint = Tuple[int, int]