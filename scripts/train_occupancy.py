"""
สคริปต์สำหรับ "เทรน" โมเดล Occupancy (State 2)

[UPDATED]
- แก้ไข [TODO]
- Import และใช้งาน OccupancyDataModule
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import sys
import os
import shutil

# เพิ่ม src/ เข้าไปใน path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.occupancy.occupancy_lit_model import OccupancyLitModel
from src.data.occupancy_datamodule import OccupancyDataModule # [NEW] Import

def train():
    print("Starting Occupancy Model Training Script...")
    
    # --- 1. ตั้งค่า ---
    DATA_DIR = "data/processed/occupancy_dataset"
    OUTPUT_CHECKPOINT = "models/occupancy/occupancy_model_best.ckpt"
    
    # ตรวจสอบว่ามีข้อมูลเทรน
    if not os.path.exists(os.path.join(DATA_DIR, "train", "0_empty")):
        print(f"!!! ข้อผิดพลาด: ไม่พบข้อมูลเทรนที่ {DATA_DIR}/train/0_empty")
        print("โปรดรัน 'scripts/prepare_occupancy_data.py' และคัดแยกข้อมูลก่อน")
        return

    # --- 2. สร้าง DataModule ---
    datamodule = OccupancyDataModule(data_dir=DATA_DIR, batch_size=64)

    # --- 3. สร้างโมเดล ---
    model = OccupancyLitModel(learning_rate=1e-4)

    # --- 4. สร้าง Callbacks (ตัวช่วย) ---
    # บันทึก Checkpoint ที่ดีที่สุด (วัดจาก val_acc)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        dirpath="models/occupancy",
        filename="occupancy_model_best",
        save_top_k=1
    )
    # หยุดเทรน ถ้า val_loss ไม่ลดลง 3 รอบ
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    # --- 5. สร้าง Trainer ---
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto", 
        default_root_dir="models/occupancy/checkpoints/",
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # --- 6. เริ่มเทรน ---
    print("\n--- Starting Training ---")
    trainer.fit(model, datamodule)
    print("--- Training Complete ---")
    
    # --- 7. บันทึกโมเดลที่ดีที่สุด ---
    print(f"Saving best model to: {OUTPUT_CHECKPOINT}")
    # trainer.save_checkpoint(OUTPUT_CHECKPOINT) # PL 2.0+ จะบันทึกอันที่ดีที่สุดให้อัตโนมัติ
    # เราแค่คัดลอกไฟล์ที่ดีที่สุดมา
    best_path = checkpoint_callback.best_model_path
    
    print("\nTraining script finished.")
    print(f"คุณสามารถใช้โมเดลได้ที่: {OUTPUT_CHECKPOINT}")

if __name__ == "__main__":
    train()