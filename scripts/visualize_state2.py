"""
สคริปต์สำหรับทดสอบ State 2: Occupancy Classification
[UPDATED]
- รันแบบ Batch (หลายภาพ)
- สร้าง Subplots แบบ (N, 3)
- บันทึกผลลัพธ์ทั้งหมดลงในไฟล์ .png ไฟล์เดียว
"""
import matplotlib
matplotlib.use('Agg') # ใช้ 'Agg' (สำหรับเซิร์ฟเวอร์/Colab)
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
    IMAGE_DIR = "data/raw/images/single_test/"
    MODEL_CHECKPOINT_PATH = "models/occupancy/occupancy_model_best.ckpt" 
    
    # [NEW] รายการภาพทั้งหมดที่จะทดสอบ
    IMAGE_NAMES = [
        "test_image_1.jpg",
        "test_image_2.jpg",
        "test_image_3.jpg",
        "test_image_4.jpg", # (ตรวจสอบว่าคุณมีไฟล์นี้)
        "test_image_5.jpg"  # (ตรวจสอบว่าคุณมีไฟล์นี้)
    ]
    
    # [NEW] ชื่อไฟล์ผลลัพธ์ใหม่
    OUTPUT_FILENAME = "state2_visualization_batch.png"
    
    # ตั้งเป็น False เพื่อใช้โมเดลจริงที่เทรนแล้ว
    USE_DUMMY_MODEL = False 
    
    if not os.path.exists(MODEL_CHECKPOINT_PATH) and not USE_DUMMY_MODEL:
        print(f"!!! ข้อผิดพลาด: ไม่พบโมเดลที่ '{MODEL_CHECKPOINT_PATH}'")
        print("โปรดรัน 'scripts/train_occupancy.py' ก่อน หรือตั้ง USE_DUMMY_MODEL = True")
        return

    # --- 2. สร้าง Subplots ---
    num_images = len(IMAGE_NAMES)
    fig, axes = plt.subplots(num_images, 3, figsize=(20, 7 * num_images))
    fig.suptitle("State 2: Occupancy Classification Pipeline (Batch Test)", fontsize=12)
    
    # จัดการกรณีมีภาพเดียว
    if num_images == 1:
        axes = np.array([axes])

    # --- 3. โหลดโมเดล (ครั้งเดียว) ---
    locator = BoardLocator(warped_size=600)
    occupancy_model = OccupancyModel(MODEL_CHECKPOINT_PATH, use_dummy=USE_DUMMY_MODEL)

    # --- 4. วน Loop ประมวลผลทีละภาพ ---
    for i, img_name in enumerate(IMAGE_NAMES):
        
        print(f"\n--- กำลังประมวลผล: {img_name} ---")
        
        ax_orig = axes[i, 0]
        ax_warped = axes[i, 1]
        ax_result = axes[i, 2]
        
        image_path = os.path.join(IMAGE_DIR, img_name)

        if not os.path.exists(image_path):
            print(f"!!! ไม่พบไฟล์: {image_path}")
            ax_orig.set_title(f"Original: {img_name} (NOT FOUND)", fontsize=10)
            ax_orig.text(0.5, 0.5, "File Not Found", ha='center', va='center', color='red')
            ax_orig.axis("off")
            ax_warped.axis("off")
            ax_result.axis("off")
            continue

        image = cv2.imread(image_path)
        
        # (Col 0) Original
        ax_orig.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_orig.set_title(f"Original: {img_name}", fontsize=10)
        ax_orig.axis("off")

        try:
            # --- รัน State 1 ---
            warped_board, matrix = locator.find_and_warp(image)
            print("State 1: ค้นหากระดานและหมุนภาพสำเร็จ")
            
            # (Col 1) Warped
            ax_warped.imshow(cv2.cvtColor(warped_board, cv2.COLOR_BGR2RGB))
            ax_warped.set_title("State 1: Warped & Rotated Board", fontsize=10)
            ax_warped.axis("off")

            # --- รัน State 2 ---
            squares = image_utils.crop_squares_from_warped(
                warped_board, 
                context_ratio=0.5 
            )
            occupancy_grid = occupancy_model.predict(squares)
            print(f"State 2: ทำนาย Occupancy สำเร็จ (ใช้ Dummy: {USE_DUMMY_MODEL})")

            # (Col 2) Occupancy Result
            result_img = warped_board.copy()
            sq_size = warped_board.shape[0] // 8
            
            for j in range(64):
                is_occupied = occupancy_grid[j]
                row, col = j // 8, j % 8
                x1, y1 = col * sq_size, row * sq_size
                text = "O" if is_occupied else "E"
                color = (0, 0, 255) if is_occupied else (0, 255, 0) # แดง / เขียว
                text_x = x1 + int(sq_size * 0.3)
                text_y = y1 + int(sq_size * 0.6)
                cv2.putText(result_img, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            ax_result.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            ax_result.set_title("State 2: Occupancy Result", fontsize=10)
            ax_result.axis("off")

        except Exception as e:
            # [NEW] Error Handling ต่อภาพ
            print(f"!!! State 1/2 ล้มเหลว: {e}")
            ax_warped.set_title("State 1: Failed", fontsize=10)
            ax_warped.text(0.5, 0.5, f"PROCESSING FAILED\n({e})", 
                           ha='center', va='center', color='red', fontsize=10)
            ax_warped.axis("off")
            ax_result.set_title("State 2: N/A", fontsize=10)
            ax_result.axis("off")
            if locator.debug_mask is not None:
                # แสดง Mask ที่ล้มเหลว (เผื่อช่วย Debug)
                ax_warped.imshow(locator.debug_mask, cmap='gray')

    # --- 5. บันทึกไฟล์ (ครั้งเดียว) ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.98], h_pad=3.0)
    plt.savefig(OUTPUT_FILENAME)
    plt.close(fig) # ปิด Figure เพื่อประหยัด memory
    print(f"\nบันทึกภาพผลลัพธ์ทั้งหมดไปที่: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()