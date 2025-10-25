"""
สคริปต์สำหรับทดสอบ State 2: Occupancy Classification
"""
import cv2
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# เพิ่ม src/ เข้าไปใน path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.s1_board_locator import BoardLocator
from src.pipeline.s2_occupancy_model import OccupancyModel
from src.utils import image_utils 
from src.core.exceptions import BoardNotFoundException

def main():
    # --- 1. ตั้งค่า ---
    IMAGE_PATH = "data/raw/images/single_test/test_image_1.jpg"
    
    # Path ไปยังไฟล์ checkpoint ของ Lightning
    MODEL_CHECKPOINT_PATH = "models/occupancy/occupancy_model.ckpt" 
    
    # ตั้งเป็น True เพื่อใช้ Dummy Model (เพราะเรายังไม่ได้เทรน)
    USE_DUMMY_MODEL = True
    
    if not os.path.exists(IMAGE_PATH):
        print(f"!!! ข้อผิดพลาด: ไม่พบไฟล์ '{IMAGE_PATH}'")
        return

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"ไม่สามารถอ่านไฟล์ภาพที่: {IMAGE_PATH}")
        return
        
    print(f"--- กำลังประมวลผล: {IMAGE_PATH} ---")

    # --- 2. รัน State 1 (Board Localisation) ---
    locator = BoardLocator(warped_size=600)
    try:
        warped_board, matrix = locator.find_and_warp(image)
        print("State 1: ค้นหากระดานและหมุนภาพสำเร็จ")
    except BoardNotFoundException as e:
        print(f"!!! State 1 ล้มเหลว: {e}")
        return

    # --- 3. รัน State 2 (Occupancy Classification) ---
    
    # [cite_start]3.1 ตัด 64 ช่อง (ตาม Paper [cite: 321])
    try:
        squares = image_utils.crop_squares_from_warped(
            warped_board, 
            context_ratio=0.5 
        )
        print(f"State 2: ตัด 64 ช่องสำเร็จ (ด้วย context 50%)")
    except Exception as e:
        print(f"!!! State 2 (Crop) ล้มเหลว: {e}")
        return
        
    # 3.2 โหลดโมเดล
    occupancy_model = OccupancyModel(MODEL_CHECKPOINT_PATH, use_dummy=USE_DUMMY_MODEL)
    
    # 3.3 ทำนาย
    occupancy_grid = occupancy_model.predict(squares)
    print(f"State 2: ทำนาย Occupancy สำเร็จ (ใช้ Dummy: {USE_DUMMY_MODEL})")

    # --- 4. แสดงผลลัพธ์ ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("State 2: Occupancy Classification Pipeline", fontsize=12)

    # (Col 0) Original
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original: {os.path.basename(IMAGE_PATH)}", fontsize=10)
    axes[0].axis("off")

    # (Col 1) Warped
    axes[1].imshow(cv2.cvtColor(warped_board, cv2.COLOR_BGR2RGB))
    axes[1].set_title("State 1: Warped & Rotated Board", fontsize=10)
    axes[1].axis("off")

    # (Col 2) Occupancy Result
    result_img = warped_board.copy()
    sq_size = warped_board.shape[0] // 8
    
    for i in range(64):
        is_occupied = occupancy_grid[i]
        
        row, col = i // 8, i % 8
        x1, y1 = col * sq_size, row * sq_size
        
        text = "O" if is_occupied else "E"
        color = (0, 0, 255) if is_occupied else (0, 255, 0) # แดง / เขียว
        
        text_x = x1 + int(sq_size * 0.3)
        text_y = y1 + int(sq_size * 0.6)
        
        cv2.putText(result_img, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    axes[2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("State 2: Occupancy Result (O=Occupied, E=Empty)", fontsize=10)
    axes[2].axis("off")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98], h_pad=3.0)
    plt.show()

if __name__ == "__main__":
    main()