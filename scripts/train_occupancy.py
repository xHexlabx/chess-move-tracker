"""
สคริปต์สำหรับ "เทรน" โมเดล Occupancy (State 2)

[FIX]
- แก้ไขการทับซ้อนของไฟล์:
  - ให้ ModelCheckpoint บันทึกลงใน default_root_dir (log dir)
  - คัดลอก (copy) ไฟล์ที่ดีที่สุดไปยัง 'OUTPUT_CHECKPOINT' ตอนจบ
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import sys
import os
import shutil

# เพิ่ม src/ เข้าไปใน path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.occupancy.occupancy_lit_model import OccupancyLitModel
from src.data.occupancy_datamodule import OccupancyDataModule

def train():
    print("Starting Occupancy Model Training Script...")
    
    # --- 1. ตั้งค่า ---
    DATA_DIR = "data/processed/occupancy_dataset"
    
    # 1. ที่เก็บ Artifacts ระหว่างเทรน (logs, checkpoints ทั้งหมด)
    CHECKPOINT_DIR = "models/occupancy/checkpoints/"
    # 2. ไฟล์โมเดล "ที่ดีที่สุด" สำหรับนำไปใช้งาน (Inference)
    OUTPUT_CHECKPOINT = "models/occupancy/occupancy_model_best.ckpt"
    
    # ตรวจสอบว่ามีข้อมูลเทรน
    if not os.path.exists(os.path.join(DATA_DIR, "train", "0_empty")):
        print(f"!!! ข้อผิดพลาด: ไม่พบข้อมูลเทรนที่ {DATA_DIR}/train/0_empty")
        print("โปรดรัน 'scripts/prepare_fewshot_data.py' ก่อน")
        return

    # --- 2. สร้าง DataModule ---
    datamodule = OccupancyDataModule(data_dir=DATA_DIR, batch_size=64)

    # --- 3. สร้างโมเดล ---
    model = OccupancyLitModel(learning_rate=1e-4)

    # --- 4. สร้าง Callbacks (ตัวช่วย) ---
    
    # [FIX] ให้ Checkpoint บันทึกลงใน CHECKPOINT_DIR
    # โดยการไม่ระบุ dirpath และใช้ filename แบบไดนามิก
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        # dirpath=None (จะใช้ default_root_dir ของ Trainer)
        filename="occupancy-best-{epoch:02d}-{val_acc:.3f}", 
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    # --- 5. สร้าง Trainer ---
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto", 
        default_root_dir=CHECKPOINT_DIR, # ที่เก็บ Log และ Checkpoint
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # --- 6. เริ่มเทรน ---
    print("\n--- Starting Training ---")
    trainer.fit(model, datamodule)
    print("--- Training Complete ---")
    
    # --- 7. คัดลอกโมเดลที่ดีที่สุดไปใช้งาน ---
    print(f"\nTraining finished. Best model path is: {checkpoint_callback.best_model_path}")
    
    # [FIX] เพิ่มการคัดลอกไฟล์
    try:
        if os.path.exists(checkpoint_callback.best_model_path):
            # ตรวจสอบว่า Source และ Dest ไม่ใช่ไฟล์เดียวกัน (เผื่อกรณีรันซ้ำ)
            if os.path.abspath(checkpoint_callback.best_model_path) != os.path.abspath(OUTPUT_CHECKPOINT):
                shutil.copy(checkpoint_callback.best_model_path, OUTPUT_CHECKPOINT)
                print(f"✅ Successfully copied best model to: {OUTPUT_CHECKPOINT}")
            else:
                print(f"โมเดลที่ดีที่สุดอยู่ที่ {OUTPUT_CHECKPOINT} อยู่แล้ว (ไม่ต้องคัดลอก)")
                
            print(f"คุณสามารถใช้โมเดลนี้ได้ใน visualize_state2.py")
        else:
            print(f"!!! Error: Best model path not found at {checkpoint_callback.best_model_path}")

    except Exception as e:
        print(f"!!! Error copying best model: {e}")
        print(f"Please copy it manually from: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    train()