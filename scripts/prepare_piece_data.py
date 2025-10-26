"""
[UPDATED]
สคริปต์สำหรับ "เตรียมข้อมูล Few-Shot อัตโนมัติ" สำหรับ State 3 (Piece)

[NEW] เพิ่ม Horizontal Flipping สำหรับหมากฝั่งซ้าย
[NEW] เพิ่ม Resize ภาพ Crop ให้ใหญ่ขึ้นก่อนบันทึก
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
from src.utils.fen_utils import PIECE_MAP # Map คำตอบ 64 ช่อง
from src.core.exceptions import BoardNotFoundException
from src.models.piece.piece_lit_model import PieceLitModel # List คลาส

# --- (สำคัญ!) ตั้งค่า ---
RAW_IMAGE_DIR = "data/raw/images/"
OUTPUT_DATA_DIR = "data/processed/piece"
DATA_SPLITS = ["train", "val", "test"] # โฟลเดอร์ที่จะประมวลผล
CLASSES = PieceLitModel.CLASSES # ['b_Bishop', ..., 'empty']
TARGET_CROP_SIZE = (224, 224) # (Width, Height) ขนาดใหม่ของภาพ Crop ที่จะบันทึก
# -------------------------

def main():
    # 1. สร้าง/ล้างโฟลเดอร์ Output
    if os.path.exists(OUTPUT_DATA_DIR):
        print(f"ล้างข้อมูลเก่าใน: {OUTPUT_DATA_DIR}")
        shutil.rmtree(OUTPUT_DATA_DIR)

    for split in DATA_SPLITS:
        for cls_name in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DATA_DIR, split, cls_name), exist_ok=True)

    print(f"สร้างโฟลเดอร์ {len(CLASSES)} คลาส x {len(DATA_SPLITS)} splits ที่: {OUTPUT_DATA_DIR}")

    locator = BoardLocator(warped_size=600)
    total_processed_images = 0

    # --- วน Loop ตาม Split (train, val, test) ---
    for split in DATA_SPLITS:
        current_raw_dir = os.path.join(RAW_IMAGE_DIR, split)
        current_out_dir = os.path.join(OUTPUT_DATA_DIR, split)

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

                # 4. ตัด 64 ช่อง (ด้วยฟังก์ชัน Crop ของ State 3)
                squares = image_utils.crop_piece_squares(warped_board)

                # 5. "ติดป้าย" (Label) อัตโนมัติ, Flip, Resize และบันทึก
                for i, sq_img in enumerate(squares):
                    # หา Label จาก Map คำตอบ
                    label = PIECE_MAP[i] # e.g., 'b_Rook', 'empty'

                    # --- Horizontal Flipping ---
                    col = i % 8 # หาคอลัมน์ (0-7)
                    if col <= 3: # ถ้าเป็นคอลัมน์ a, b, c, d (ฝั่งซ้าย)
                        flipped_sq_img = cv2.flip(sq_img, 1) # กลับภาพแนวนอน
                        flip_suffix = "_flip"
                    else:
                        flipped_sq_img = sq_img
                        flip_suffix = ""
                    # -------------------------

                    # --- [NEW] Resize Image ---
                    # ใช้ INTER_LINEAR ซึ่งเป็นค่า default ที่ดีสำหรับการย่อ/ขยายทั่วไป
                    resized_sq_img = cv2.resize(flipped_sq_img, TARGET_CROP_SIZE, interpolation=cv2.INTER_LINEAR)
                    # -------------------------


                    # หา Path ที่จะบันทึก
                    save_dir = os.path.join(current_out_dir, label)

                    filename = f"{split}_{split_img_counter:04d}_{img_name.split('.')[0]}_sq{i}{flip_suffix}.jpg"
                    save_path = os.path.join(save_dir, filename)

                    # ตรวจสอบว่ามี directory ก่อนบันทึก
                    if not os.path.exists(save_dir):
                         print(f"      Warning: Directory {save_dir} not found, skipping save for {filename}")
                         continue

                    cv2.imwrite(save_path, resized_sq_img) # บันทึกภาพที่ Resize แล้ว
                    split_img_counter += 1

                print(f"   ✅ สำเร็จ: {img_name} -> 64 ช่อง (ขนาด {TARGET_CROP_SIZE[0]}x{TARGET_CROP_SIZE[1]})")

            except BoardNotFoundException as e:
                print(f"   ❌ ล้มเหลว (State 1): {img_name} -> {e}")
            except Exception as e:
                 print(f"   ❌ ล้มเหลว (อื่นๆ): {img_name} -> {e}")


        total_processed_images += split_img_counter

    print("-" * 30)
    print(f"บันทึกภาพ Piece ทั้งหมด {total_processed_images} ภาพ เรียบร้อย (ขนาด {TARGET_CROP_SIZE[0]}x{TARGET_CROP_SIZE[1]})")
    print("ตอนนี้คุณพร้อมที่จะรัน 'scripts/train_piece.py' แล้ว!")

if __name__ == "__main__":
    main()