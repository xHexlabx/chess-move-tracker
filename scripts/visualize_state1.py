"""
สคริปต์สำหรับทดสอบ State 1: Board Localisation

[UPDATE 3]:
- ปรับลดขนาด Font และเพิ่มระยะห่าง (h_pad)
"""
import cv2
import matplotlib.pyplot as plt
import sys
import os
import numpy as np # Import ที่ขาดไปในเวอร์ชันก่อน

# เพิ่ม src/ เข้าไปใน path เพื่อให้ import module ของเราได้
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.s1_board_locator import BoardLocator
from src.core.exceptions import BoardNotFoundException, LinesNotFoundException

def main():
    # --- 1. ตั้งค่า ---
    IMAGE_DIR = "data/raw/images/single_test/test/"
    
    IMAGE_NAMES = [
        "test_1.jpg",
        "test_2.jpg",
        "test_3.jpg"
    ]
    
    image_paths = []
    for name in IMAGE_NAMES:
        path = os.path.join(IMAGE_DIR, name)
        if not os.path.exists(path):
            print(f"!!! คำเตือน: ไม่พบไฟล์ '{path}', จะข้ามภาพนี้")
        else:
            image_paths.append(path)

    if not image_paths:
        print(f"!!! ข้อผิดพลาด: ไม่พบไฟล์ภาพใดๆ ใน List ที่ '{IMAGE_DIR}'")
        return

    # --- 2. สร้าง Subplots ---
    num_images = len(image_paths)
    fig, axes = plt.subplots(num_images, 3, figsize=(20, 7 * num_images))
    
    if num_images == 1:
        axes = np.array([axes])
        
    # [FIX] ลดขนาด Font
    fig.suptitle("Board Localisation Pipeline Test (Multi-Image)", fontsize=12) 

    # --- 3. รัน Pipeline (วน Loop) ---
    locator = BoardLocator(warped_size=600)

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            print(f"ไม่สามารถอ่านไฟล์ภาพที่: {image_path}")
            continue
            
        print(f"--- กำลังประมวลผล: {image_path} ---")

        # --- แกน Original Image (Col 0) ---
        ax_orig = axes[i, 0]
        ax_orig.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_orig.set_title(f"Original: {os.path.basename(image_path)}", fontsize=10) # [FIX]
        ax_orig.axis("off")
        
        # --- แกน Mask (Col 1) ---
        ax_mask = axes[i, 1]
        
        # --- แกน Warped (Col 2) ---
        ax_warped = axes[i, 2]

        try:
            warped_board, matrix = locator.find_and_warp(image)
            print("ค้นหากระดานสำเร็จ!")

            # แสดง Mask (Col 1)
            ax_mask.imshow(locator.debug_mask, cmap='gray')
            ax_mask.set_title("Debug: Isolated Board Mask", fontsize=10) # [FIX]
            ax_mask.axis("off")

            # แสดง Warped (Col 2)
            ax_warped.imshow(cv2.cvtColor(warped_board, cv2.COLOR_BGR2RGB))
            ax_warped.set_title("State 1: Warped Board", fontsize=10) # [FIX]
            ax_warped.axis("off")

        except (BoardNotFoundException, LinesNotFoundException) as e:
            print(f"!!! การประมวลผลล้มเหลว: {e}")
            
            if locator.debug_mask is not None:
                ax_mask.imshow(locator.debug_mask, cmap='gray')
                ax_mask.set_title("Debug: Failed Mask", fontsize=10) # [FIX]
            else:
                ax_mask.set_title("Debug: Mask Failed", fontsize=10) # [FIX]
            ax_mask.axis("off")

            ax_warped.text(0.5, 0.5, f"PROCESSING FAILED\n({e})", 
                           ha='center', va='center', color='red', fontsize=10) # [FIX]
            ax_warped.set_title("State 1: Failed", fontsize=10) # [FIX]
            ax_warped.axis("off")

    # [FIX] เพิ่ม h_pad=3.0 เพื่อเว้นระยะห่างแนวตั้ง
    plt.tight_layout(rect=[0, 0.03, 1, 0.98], h_pad=3.0) 
    plt.show()

if __name__ == "__main__":
    main()