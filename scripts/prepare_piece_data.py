"""
[NEW FILE]
สคริปต์สำหรับ "เตรียมข้อมูล Few-Shot อัตโนมัติ" สำหรับ State 3
1. โหลดภาพกระดานตั้งต้น
2. รัน State 1 (Warp & Rotate)
3. ตัด 64 ช่อง (ด้วย Crop แบบใหม่)
4. "ติดป้าย" (Label) อัตโนมัติตาม PIECE_MAP (13 คลาส)
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
from src.utils.fen_utils import PIECE_MAP # [NEW] Import Map คำตอบ
from src.core.exceptions import BoardNotFoundException
from src.models.piece.piece_lit_model import PieceLitModel # [NEW] Import เพื่อเอา List คลาส

# --- (สำคัญ!) ตั้งค่า ---
TRAIN_PREFIXES = ["white_view_", "black_view_"]
IMAGE_DIR = "data/raw/images/single_test/"
OUTPUT_DATA_DIR = "data/processed/piece_dataset"
# -------------------------

def main():
    # 1. สร้าง/ล้างโฟลเดอร์ Output (Train และ Val)
    train_dir = os.path.join(OUTPUT_DATA_DIR, "train")
    val_dir = os.path.join(OUTPUT_DATA_DIR, "val")
    
    if os.path.exists(OUTPUT_DATA_DIR):
        print(f"ล้างข้อมูลเก่าใน: {OUTPUT_DATA_DIR}")
        shutil.rmtree(OUTPUT_DATA_DIR)
        
    # สร้างโฟลเดอร์ Class 13 คลาส
    CLASSES = PieceLitModel.CLASSES 
    for part in [train_dir, val_dir]:
        for cls_name in CLASSES:
            os.makedirs(os.path.join(part, cls_name))
    
    print(f"สร้างโฟลเดอร์ {len(CLASSES)} คลาส ที่: {OUTPUT_DATA_DIR}")
    
    locator = BoardLocator(warped_size=600)
    img_counter = 0

    image_names = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"สแกนพบ {len(image_names)} ภาพใน '{IMAGE_DIR}'...")
    
    for img_name in image_names:
        img_path = os.path.join(IMAGE_DIR, img_name)
        image = cv2.imread(img_path)
        
        is_train = any(img_name.startswith(p) for p in TRAIN_PREFIXES)
        dataset_part = "train" if is_train else "val"
        
        print(f"\n--- กำลังประมวลผล: {img_name} (-> {dataset_part}) ---")

        try:
            # 3. รัน State 1
            warped_board, _ = locator.find_and_warp(image)
            
            # 4. ตัด 64 ช่อง (ด้วยฟังก์ชันใหม่)
            squares = image_utils.crop_piece_squares(warped_board)
            
            # 5. "ติดป้าย" (Label) อัตโนมัติ และบันทึก
            for i, sq_img in enumerate(squares):
                # หา Label จาก Map
                label = PIECE_MAP[i]
                
                # หา Path ที่จะบันทึก
                save_dir = os.path.join(OUTPUT_DATA_DIR, dataset_part, label)
                    
                filename = f"{dataset_part}_{img_counter:04d}_{img_name.split('.')[0]}_sq{i}.jpg"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, sq_img)
                img_counter += 1
                
            print(f"✅ สำเร็จ: {img_name} -> 64 ช่อง (บันทึกลง {dataset_part})")

        except BoardNotFoundException as e:
            print(f"❌ ล้มเหลว: {img_name} -> {e}")

    print("-" * 30)
    print(f"บันทึกภาพทั้งหมด {img_counter} ภาพ เรียบร้อย")
    print("ตอนนี้คุณพร้อมที่จะรัน 'scripts/train_piece.py' แล้ว!")

if __name__ == "__main__":
    main()