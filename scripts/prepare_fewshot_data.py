"""
[UPDATED]
สคริปต์สำหรับ "เตรียมข้อมูล Few-Shot อัตโนมัติ" (Occupancy)

[NEW LOGIC]:
- ลบการเช็ค TRAIN_PREFIXES ออก
- วน Loop อ่านภาพจาก data/raw/images/[train/val/test]/
- บันทึก Crop ลงใน data/processed/occupancy/[train/val/test]/
- "Auto-label" ภาพทั้งหมดโดยใช้กฎ 'starting position'
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
RAW_IMAGE_DIR = "data/raw/images/"
OUTPUT_DATA_DIR = "data/processed/occupancy"
DATA_SPLITS = ["train", "val", "test"] # โฟลเดอร์ที่จะประมวลผล
# -------------------------

def main():
    # 1. สร้าง/ล้างโฟลเดอร์ Output
    if os.path.exists(OUTPUT_DATA_DIR):
        print(f"ล้างข้อมูลเก่าใน: {OUTPUT_DATA_DIR}")
        shutil.rmtree(OUTPUT_DATA_DIR)

    for split in DATA_SPLITS:
        os.makedirs(os.path.join(OUTPUT_DATA_DIR, split, "0_empty"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DATA_DIR, split, "1_occupied"), exist_ok=True)

    print(f"สร้างโฟลเดอร์ Train/Val/Test ที่: {OUTPUT_DATA_DIR}")

    locator = BoardLocator(warped_size=600)
    total_processed_images = 0

    # --- วน Loop ตาม Split (train, val, test) ---
    for split in DATA_SPLITS:
        current_raw_dir = os.path.join(RAW_IMAGE_DIR, split)
        current_out_dir = os.path.join(OUTPUT_DATA_DIR, split)
        dest_empty_dir = os.path.join(current_out_dir, "0_empty")
        dest_occupied_dir = os.path.join(current_out_dir, "1_occupied")

        if not os.path.exists(current_raw_dir):
            print(f"\nคำเตือน: ไม่พบโฟลเดอร์ {current_raw_dir}, จะข้าม split นี้")
            continue

        image_names = [f for f in os.listdir(current_raw_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"\n--- กำลังประมวลผล Split '{split}' ({len(image_names)} ภาพ) ---")
        split_img_counter = 0

        for img_name in image_names:
            img_path = os.path.join(current_raw_dir, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f"❌ ไม่สามารถอ่าน: {img_name}")
                continue

            print(f"   Processing: {img_name}...")

            try:
                # 3. รัน State 1 (Warp & Rotate)
                warped_board, _ = locator.find_and_warp(image)

                # 4. ตัด 64 ช่อง
                squares = image_utils.crop_squares_from_warped(warped_board, context_ratio=0.5)

                # 5. "ติดป้าย" (Label) อัตโนมัติ และบันทึก
                for i, sq_img in enumerate(squares):
                    row = i // 8

                    # ตรรกะการ Label อัตโนมัติ (สมมติว่าทุกภาพเป็น Starting Position)
                    if row in [0, 1, 6, 7]: # 2 แถวบน, 2 แถวล่าง
                        save_dir = dest_occupied_dir
                    else: # 4 แถวกลาง
                        save_dir = dest_empty_dir

                    filename = f"{split}_{split_img_counter:04d}_{img_name.split('.')[0]}_sq{i}.jpg"
                    save_path = os.path.join(save_dir, filename)
                    cv2.imwrite(save_path, sq_img)
                    split_img_counter += 1

                print(f"   ✅ สำเร็จ: {img_name} -> 64 ช่อง")

            except BoardNotFoundException as e:
                print(f"   ❌ ล้มเหลว (State 1): {img_name} -> {e}")
            except Exception as e:
                 print(f"   ❌ ล้มเหลว (อื่นๆ): {img_name} -> {e}")


        total_processed_images += split_img_counter

    print("-" * 30)
    print(f"บันทึกภาพ Occupancy ทั้งหมด {total_processed_images} ภาพ เรียบร้อย")
    print("ตอนนี้คุณพร้อมที่จะรัน 'scripts/train_occupancy.py' แล้ว!")

if __name__ == "__main__":
    main()