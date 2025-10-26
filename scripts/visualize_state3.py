"""
[NEW FILE]
สคริปต์สำหรับทดสอบ Pipeline S1 -> S2 -> S3 (End-to-End)
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

# [NEW] Import Orchestrator หลัก
from src.pipeline.state_recognizer import StateRecognizer
from src.utils.fen_utils import PIECE_TO_FEN # Import เพื่อใช้แปลงชื่อเต็ม -> ตัวย่อ

def main():
    # --- 1. ตั้งค่า ---
    IMAGE_DIR = "data/raw/images/single_test/test/"
    IMAGE_NAMES = [
        "test_1.jpg",
        "test_2.jpg",
        "test_3.jpg",
    ]
    
    # Path ไปยังโมเดลที่เทรนแล้ว
    OCCUPANCY_MODEL_PATH = "models/occupancy/occupancy_model_best.ckpt" 
    PIECE_MODEL_PATH = "models/piece/piece_model_best.ckpt"
    
    OUTPUT_FILENAME = "state3_visualization_batch.png"
    
    # ตรวจสอบว่ามีโมเดล
    if not os.path.exists(OCCUPANCY_MODEL_PATH):
        print(f"!!! Error: Occupancy model not found at {OCCUPANCY_MODEL_PATH}")
        return
    if not os.path.exists(PIECE_MODEL_PATH):
        print(f"!!! Error: Piece model not found at {PIECE_MODEL_PATH}")
        return

    # --- 2. สร้าง Subplots ---
    num_images = len(IMAGE_NAMES)
    fig, axes = plt.subplots(num_images, 2, figsize=(15, 7 * num_images)) # [CHANGED] ใช้แค่ 2 คอลัมน์
    fig.suptitle("State 3: End-to-End Piece Recognition Pipeline", fontsize=12)
    
    if num_images == 1:
        axes = np.array([axes])

    # --- 3. โหลด Pipeline Orchestrator (ครั้งเดียว) ---
    recognizer = StateRecognizer(
        occupancy_model_path=OCCUPANCY_MODEL_PATH,
        piece_model_path=PIECE_MODEL_PATH,
        use_dummy_occupancy=False # ใช้โมเดล Occupancy จริง
    )

    # --- 4. วน Loop ประมวลผลทีละภาพ ---
    for i, img_name in enumerate(IMAGE_NAMES):
        
        print(f"\n--- Processing: {img_name} ---")
        
        ax_orig = axes[i, 0]
        ax_result = axes[i, 1]
        
        image_path = os.path.join(IMAGE_DIR, img_name)

        if not os.path.exists(image_path):
            print(f"!!! File not found: {image_path}")
            ax_orig.set_title(f"Original: {img_name} (NOT FOUND)", fontsize=10)
            ax_orig.text(0.5, 0.5, "File Not Found", ha='center', va='center', color='red')
            ax_orig.axis("off")
            ax_result.axis("off")
            continue

        image = cv2.imread(image_path)
        
        # (Col 0) Original
        ax_orig.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_orig.set_title(f"Original: {img_name}", fontsize=10)
        ax_orig.axis("off")

        try:
            # --- [NEW] รัน Pipeline ทั้งหมด ---
            board_state = recognizer.recognize(image)
            
            if board_state is None:
                raise Exception("Recognizer returned None (likely S1 failed)")

            # (Col 1) Piece Recognition Result
            result_img = board_state.warped_image.copy()
            sq_size = result_img.shape[0] // 8
            piece_grid = board_state.piece_grid
            
            for j in range(64):
                piece_name = piece_grid[j]
                if piece_name == 'empty':
                    continue # ไม่ต้องวาดช่องว่าง
                    
                row, col = j // 8, j % 8
                x1, y1 = col * sq_size, row * sq_size
                
                # แปลงชื่อเต็ม -> ตัวย่อ FEN (เช่น 'w_Pawn' -> 'P')
                fen_char = PIECE_TO_FEN.get(piece_name, '?') 
                
                # กำหนดสีตามฝ่าย (ขาว/ดำ)
                color = (255, 255, 255) if piece_name.startswith('w_') else (50, 50, 50) 
                bg_color = (0,0,0) if piece_name.startswith('w_') else (200, 200, 200) # พื้นหลัง Text
                
                # คำนวณตำแหน่งกลางช่อง
                text_x = x1 + int(sq_size * 0.3)
                text_y = y1 + int(sq_size * 0.7)
                
                # วาดพื้นหลังก่อน
                (w, h), _ = cv2.getTextSize(fen_char, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(result_img, (text_x - 5, text_y - h - 5), (text_x + w + 5, text_y + 5), bg_color, -1)
                
                cv2.putText(result_img, fen_char, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            ax_result.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            # [NEW] แสดง FEN String ใน Title
            ax_result.set_title(f"State 3: Piece Recognition\nFEN: {board_state.fen}", fontsize=10) 
            ax_result.axis("off")

        except Exception as e:
            # Error Handling
            print(f"!!! Pipeline Failed for {img_name}: {e}")
            ax_result.set_title("Pipeline Failed", fontsize=10)
            ax_result.text(0.5, 0.5, f"ERROR:\n{e}", 
                           ha='center', va='center', color='red', fontsize=10, wrap=True)
            ax_result.axis("off")


    # --- 5. บันทึกไฟล์ (ครั้งเดียว) ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.98], h_pad=3.0)
    plt.savefig(OUTPUT_FILENAME)
    plt.close(fig) 
    print(f"\nบันทึกภาพผลลัพธ์ทั้งหมดไปที่: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()