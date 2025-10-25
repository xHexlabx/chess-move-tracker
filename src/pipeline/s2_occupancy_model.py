"""
State 2: Occupancy Classification (Inference Wrapper)

[REFACTORED for PyTorch Lightning]
คลาสนี้จะ "โหลด" LightningModule ที่เทรนแล้ว (.ckpt)
เพื่อใช้ในการทำนายผล (Inference)
"""
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from typing import List
from src.core.typing import SquareImage, OccupancyGrid, ImageRGB

# Import LightningModule ของเรา
from src.models.occupancy_lit_model import OccupancyLitModel

class OccupancyModel:
    """
    Wrapper สำหรับโมเดล Occupancy (State 2)
    """
    def __init__(self, model_path: str, use_dummy: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_dummy = use_dummy
        self.model = None

        if self.use_dummy:
            print("[OccupancyModel] Using DUMMY model for testing.")
        else:
            try:
                # ใช้วิธีโหลด Checkpoint ของ Lightning
                self.model = OccupancyLitModel.load_from_checkpoint(
                    checkpoint_path=model_path,
                    map_location=self.device
                )
                self.model.eval() # ตั้งเป็นโหมดประเมินผล
                print(f"[OccupancyModel] Loaded Lightning checkpoint from: {model_path}")
            except FileNotFoundError:
                print(f"!!! Error: Model checkpoint not found at {model_path}")
                print("!!! Switching to DUMMY model.")
                self.use_dummy = True
            except Exception as e:
                print(f"!!! Error loading model: {e}")
                print("!!! Switching to DUMMY model.")
                self.use_dummy = True

        # [cite_start]Transform มาตรฐาน (Paper ใช้ 100x100) [cite: 322]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((100, 100), antialias=True), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _preprocess(self, squares: List[SquareImage]) -> torch.Tensor:
        """
        แปลง List ของภาพ (OpenCV BGR) เป็น Batch Tensor (PyTorch RGB)
        """
        batch = torch.stack([
            self.transform(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)) 
            for sq in squares
        ])
        return batch.to(self.device)

    def _run_dummy_model(self) -> OccupancyGrid:
        """
        [DUMMY] สร้างผลลัพธ์จำลอง (แถว 0, 1 และ 6, 7 มีหมาก)
        """
        grid = [False] * 64
        for i in range(64):
            if (i // 8) in [0, 1, 6, 7]: # 2 แถวบน, 2 แถวล่าง
                grid[i] = True
        return grid

    def predict(self, squares: List[SquareImage]) -> OccupancyGrid:
        """
        Input: List 64 ภาพของช่อง (BGR)
        Output: List 64 booleans (True = Occupied)
        """
        if not squares or len(squares) != 64:
            return [False] * 64
            
        if self.use_dummy:
            return self._run_dummy_model()

        # --- Logicสำหรับโมเดลจริง ---
        with torch.no_grad():
            batch = self._preprocess(squares)
            logits = self.model(batch) 
            preds = torch.argmax(logits, dim=1) # 0=Empty, 1=Occupied
            
        return [bool(p.item()) for p in preds]