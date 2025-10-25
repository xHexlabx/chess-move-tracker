"""
[NEW FILE]
สคริปต์สำหรับ "เทรน" โมเดล Piece Classification (State 3)
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import sys
import os
import shutil

# เพิ่ม src/ เข้าไปใน path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.piece.piece_lit_model import PieceLitModel
from src.data.piece_datamodule import PieceDataModule

def train():
    print("Starting Piece Model Training Script...")
    
    # --- 1. ตั้งค่า ---
    DATA_DIR = "data/processed/piece_dataset"
    CHECKPOINT_DIR = "models/piece/checkpoints/"
    OUTPUT_CHECKPOINT = "models/piece/piece_model_best.ckpt"
    
    if not os.path.exists(os.path.join(DATA_DIR, "train", "w_Pawn")):
        print(f"!!! ข้อผิดพลาด: ไม่พบข้อมูลเทรนที่ {DATA_DIR}/train/w_Pawn")
        print("โปรดรัน 'scripts/prepare_piece_data.py' ก่อน")
        return

    # --- 2. สร้าง DataModule ---
    datamodule = PieceDataModule(data_dir=DATA_DIR, batch_size=32)

    # --- 3. สร้างโมเดล ---
    model = PieceLitModel(learning_rate=1e-4)

    # --- 4. สร้าง Callbacks (ตัวช่วย) ---
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc", # เราจะใช้ val_acc
        mode="max",
        filename="piece-best-{epoch:02d}-{val_acc:.3f}", 
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5, # ให้ 5 epochs (เพราะเทรนยากกว่า)
        mode="min"
    )

    # --- 5. สร้าง Trainer ---
    trainer = pl.Trainer(
        max_epochs=50, # เทรนนานขึ้น เพราะ 13 คลาส + Augmentation หนัก
        accelerator="auto", 
        default_root_dir=CHECKPOINT_DIR,
        callbacks=[checkpoint_callback, early_stop_callback],
        # InceptionV3 ทำงานได้ดีกับ precision ที่ผสมกัน
        precision="16-mixed" 
    )

    # --- 6. เริ่มเทรน ---
    print("\n--- Starting Training (Piece Model) ---")
    trainer.fit(model, datamodule)
    print("--- Training Complete ---")
    
    # --- 7. คัดลอกโมเดลที่ดีที่สุดไปใช้งาน ---
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