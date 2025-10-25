"""
[UPDATED]
สคริปต์สำหรับ "เตรียมข้อมูล Few-Shot อัตโนมัติ"

[NEW LOGIC]:
1. สแกน 'data/raw/images/single_test/' ทั้งหมด
2. ถ้าชื่อไฟล์ตรงกับ `TRAIN_PREFIXES` -> บันทึกลง '.../train/...'
3. ถ้าชื่อไฟล์ไม่ตรง -> บันทึกลง '.../val/...'
4. "Auto-label" ภาพทั้งหมดโดยใช้กฎ 'starting position'
"""
import cv2
import sys
import os
import shutil
import numpy as np

# เพิ่ม src/ เข้าไปใน path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.s1_board_locator import BoardLocator
from src.utils import image_utils
from src.core.exceptions import BoardNotFoundException

# --- (สำคัญ!) ตั้งค่า ---
# Prefix ของไฟล์ที่จะใช้เป็น "Train Set"
TRAIN_PREFIXES = ["white_view_", "black_view_"]

# โฟลเดอร์ที่สแกนหาภาพ
IMAGE_DIR = "data/raw/images/single_test/"
# โฟลเดอร์ Output หลัก
OUTPUT_DATA_DIR = "data/processed/occupancy_dataset"
# -------------------------

def main():
    # 1. สร้าง/ล้างโฟลเดอร์ Output (ทั้ง Train และ Val)
    train_dir = os.path.join(OUTPUT_DATA_DIR, "train")
    val_dir = os.path.join(OUTPUT_DATA_DIR, "val")
    
    # ล้างข้อมูลเก่า
    if os.path.exists(OUTPUT_DATA_DIR):
        print(f"ล้างข้อมูลเก่าใน: {OUTPUT_DATA_DIR}")
        shutil.rmtree(OUTPUT_DATA_DIR)
        
    # สร้างโครงสร้างใหม่
    os.makedirs(os.path.join(train_dir, "0_empty"))
    os.makedirs(os.path.join(train_dir, "1_occupied"))
    os.makedirs(os.path.join(val_dir, "0_empty"))
    os.makedirs(os.path.join(val_dir, "1_occupied"))
    
    print(f"สร้างโฟลเดอร์ Train/Val ที่: {OUTPUT_DATA_DIR}")
    
    locator = BoardLocator(warped_size=600)
    img_counter = 0

    # สแกนหาภาพทั้งหมดในไดเรกทอรี
    image_names = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"สแกนพบ {len(image_names)} ภาพใน '{IMAGE_DIR}'...")
    
    for img_name in image_names:
        img_path = os.path.join(IMAGE_DIR, img_name)
        image = cv2.imread(img_path)
        
        # 2. ตรวจสอบว่าเป็น Train หรือ Val
        is_train = any(img_name.startswith(p) for p in TRAIN_PREFIXES)
        
        if is_train:
            dataset_part = "train"
            dest_empty_dir = os.path.join(train_dir, "0_empty")
            dest_occupied_dir = os.path.join(train_dir, "1_occupied")
        else:
            dataset_part = "val"
            dest_empty_dir = os.path.join(val_dir, "0_empty")
            dest_occupied_dir = os.path.join(val_dir, "1_occupied")
            
        print(f"\n--- กำลังประมวลผล: {img_name} (-> {dataset_part}) ---")

        try:
            # 3. รัน State 1 (Warp & Rotate)
            warped_board, _ = locator.find_and_warp(image)
            
            # 4. ตัด 64 ช่อง
            squares = image_utils.crop_squares_from_warped(warped_board, context_ratio=0.5)
            
            # 5. "ติดป้าย" (Label) อัตโนมัติ และบันทึก
            for i, sq_img in enumerate(squares):
                row = i // 8
                
                # ตรรกะการ Label อัตโนมัติ (สมมติว่าทุกภาพเป็น Sstarting Position)
                if row in [0, 1, 6, 7]: # 2 แถวบน, 2 แถวล่าง
                    save_dir = dest_occupied_dir
                else: # 4 แถวกลาง
                    save_dir = dest_empty_dir
                    
                filename = f"{dataset_part}_{img_counter:04d}_{img_name.split('.')[0]}_sq{i}.jpg"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, sq_img)
                img_counter += 1
                
            print(f"✅ สำเร็จ: {img_name} -> 64 ช่อง (บันทึกลง {dataset_part})")

        except BoardNotFoundException as e:
            print(f"❌ ล้มเหลว: {img_name} -> {e}")

    print("-" * 30)
    print(f"บันทึกภาพทั้งหมด {img_counter} ภาพ เรียบร้อย")
    print("ตอนนี้คุณพร้อมที่จะรัน 'scripts/train_occupancy.py' แล้ว!")

if __name__ == "__main__":
    main()