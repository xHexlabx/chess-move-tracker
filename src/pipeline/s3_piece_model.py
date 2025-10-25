"""
[NEW FILE]
State 3: Piece Classification (Inference Wrapper)

โหลดโมเดล PieceLitModel (.ckpt) เพื่อทำนายชนิดหมาก (13 คลาส)
"""
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from typing import List
from src.core.typing import SquareImage, OccupancyGrid, ImageRGB, WarpedImage, PieceGrid

# Import LightningModule ของ State 3
from src.models.piece.piece_lit_model import PieceLitModel
from src.utils import image_utils # Import ฟังก์ชัน crop_piece_squares

class PieceModel:
    """
    Wrapper สำหรับโมเดล Piece Classification (State 3)
    """
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        try:
            # โหลด Checkpoint ของ State 3
            self.model = PieceLitModel.load_from_checkpoint(
                checkpoint_path=model_path,
                map_location=self.device
            )
            self.model.eval() # ตั้งเป็นโหมดประเมินผล
            print(f"[PieceModel] Loaded Lightning checkpoint from: {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"!!! Error: Piece model checkpoint not found at {model_path}")
        except Exception as e:
            raise RuntimeError(f"!!! Error loading piece model: {e}")

        # Transform มาตรฐานสำหรับ InceptionV3 (299x299)
        self.transform = T.Compose([
            T.ToTensor(),
            # [IMPORTANT] ใช้ Resize(antialias=True) เพื่อคุณภาพที่ดีขึ้น
            T.Resize((299, 299), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # เก็บ List ชื่อคลาสจาก Model (สำคัญมาก)
        self.class_names = PieceLitModel.CLASSES

    def _preprocess(self, squares: List[SquareImage]) -> torch.Tensor:
        """
        แปลง List ของภาพ (OpenCV BGR) เป็น Batch Tensor (PyTorch RGB)
        """
        batch = torch.stack([
            self.transform(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
            for sq in squares
        ])
        return batch.to(self.device)

    def predict(self, warped_board: WarpedImage, occupancy_grid: OccupancyGrid) -> PieceGrid:
        """
        Input:
            - warped_board: ภาพกระดานที่ Warp แล้ว
            - occupancy_grid: List 64 bools จาก State 2
        Output:
            - PieceGrid: List 64 strings (เช่น 'empty', 'w_Pawn', 'b_Knight')
        """
        if len(occupancy_grid) != 64:
            raise ValueError("Occupancy grid must have 64 elements")

        # 1. ตัดภาพ 64 ช่อง (โดยใช้ฟังก์ชัน crop แบบใหม่สำหรับ State 3)
        # [cite_start]ฟังก์ชันนี้ใช้ Heuristic ขยาย BBox แนวตั้งตาม Paper [cite: 221-223]
        piece_crops = image_utils.crop_piece_squares(warped_board)

        # 2. คัดเลือกเฉพาะช่องที่ "มีหมาก" (Occupied)
        occupied_indices = [i for i, occupied in enumerate(occupancy_grid) if occupied]
        
        # ถ้าไม่มีหมากเลย ก็คืนค่า empty ทั้งหมด
        if not occupied_indices:
            return ['empty'] * 64
            
        occupied_crops = [piece_crops[i] for i in occupied_indices]

        # 3. ทำนายผลเฉพาะช่องที่มีหมาก
        with torch.no_grad():
            batch = self._preprocess(occupied_crops)
            logits = self.model(batch) # เรียก .forward()
            preds_indices = torch.argmax(logits, dim=1)
            
            # แปลง index กลับเป็นชื่อคลาส
            predicted_class_names = [self.class_names[p.item()] for p in preds_indices]

        # 4. สร้าง PieceGrid สุดท้าย
        final_grid: PieceGrid = ['empty'] * 64
        for idx, class_name in zip(occupied_indices, predicted_class_names):
            # ไม่เอา class 'empty' ที่โมเดลอาจทายผิด
            if class_name != 'empty':
                final_grid[idx] = class_name
                
        return final_grid