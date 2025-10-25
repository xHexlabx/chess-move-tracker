"""
กำหนด Type Hints ที่ใช้บ่อยในโปรเจกต์
"""
from typing import List, Tuple, Any, Optional
import numpy as np

# ประเภทของภาพ
ImageBGR = np.ndarray         # ภาพ OpenCV (channels BGR)
ImageGreyscale = np.ndarray   # ภาพ 1 channel
ImageRGB = np.ndarray         # ภาพสำหรับ PyTorch (channels RGB)
WarpedImage = np.ndarray      # ภาพกระดานที่ถูก warp แล้ว
HomographyMatrix = np.ndarray # 3x3 matrix
SquareImage = np.ndarray      # ภาพ 1 ช่องที่ถูกตัดออกมา

# ประเภทของข้อมูลประมวลผล
LineSegment = Tuple[int, int, int, int] 
Lines = List[LineSegment]
IntersectionPoint = Tuple[int, int]

# ประเภทของผลลัพธ์
OccupancyGrid = List[bool]    # List 64 ค่า (True=Occupied, False=Empty)

# [NEW] ประเภทของผลลัพธ์ State 3
PieceGrid = List[str]         # List 64 (e.g., 'empty', 'w_Pawn', 'b_Knight')
FEN_String = str              # FEN standard string