"""
สคริปต์สำหรับ "เทรน" โมเดล Occupancy (State 2)

[UPDATED]
- ใช้ OccupancyDataModule ที่โหลด train/val/test แยกกัน
- เพิ่มการเรียก trainer.test() (ถ้ามี Test Set)
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import sys
import os
import shutil

# เพิ่ม src/ เข้าไปใน path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.occupancy_lit_model import OccupancyLitModel
from src.data.occupancy_datamodule import OccupancyDataModule

def train():
    print("Starting Occupancy Model Training Script...")

    # --- 1. ตั้งค่า ---
    DATA_DIR = "data/processed/occupancy"
    CHECKPOINT_DIR = "models/occupancy/checkpoints/"
    OUTPUT_CHECKPOINT = "models/occupancy/occupancy_model_best.ckpt"

    # ตรวจสอบว่ามีข้อมูลเทรน
    if not os.path.exists(os.path.join(DATA_DIR, "train")):
        print(f"!!! ข้อผิดพลาด: ไม่พบโฟลเดอร์ {DATA_DIR}/train")
        print("โปรดรัน 'scripts/prepare_fewshot_data.py' ก่อน")
        return

    # --- 2. สร้าง DataModule ---
    datamodule = OccupancyDataModule(data_dir=DATA_DIR, batch_size=64)
    # datamodule.setup() # Setup is called automatically by trainer.fit/test

    # --- 3. สร้างโมเดล ---
    model = OccupancyLitModel(learning_rate=1e-4)

    # --- 4. สร้าง Callbacks (ตัวช่วย) ---
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
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
        default_root_dir=CHECKPOINT_DIR,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # --- 6. เริ่มเทรน ---
    print("\n--- Starting Training ---")
    trainer.fit(model, datamodule=datamodule)
    print("--- Training Complete ---")

    # --- 7. [NEW] ทดสอบกับ Test Set (ถ้ามี) ---
    if os.path.exists(os.path.join(DATA_DIR, "test")):
        print("\n--- Starting Testing ---")
        # โหลดโมเดลที่ดีที่สุดที่ถูกบันทึกไว้
        trainer.test(datamodule=datamodule, ckpt_path="best")
        print("--- Testing Complete ---")
    else:
        print("\nSkipping testing: Test set not found.")


    # --- 8. คัดลอกโมเดลที่ดีที่สุดไปใช้งาน ---
    print(f"\nTraining finished. Best model path: {checkpoint_callback.best_model_path}")

    try:
        if os.path.exists(checkpoint_callback.best_model_path):
            if os.path.abspath(checkpoint_callback.best_model_path) != os.path.abspath(OUTPUT_CHECKPOINT):
                shutil.copy(checkpoint_callback.best_model_path, OUTPUT_CHECKPOINT)
                print(f"✅ Successfully copied best model to: {OUTPUT_CHECKPOINT}")
            else:
                print(f"โมเดลที่ดีที่สุดอยู่ที่ {OUTPUT_CHECKPOINT} อยู่แล้ว")
        else:
            print(f"!!! Error: Best model path not found.")

    except Exception as e:
        print(f"!!! Error copying best model: {e}")

if __name__ == "__main__":
    train()