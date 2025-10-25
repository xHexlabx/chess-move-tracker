"""
[UPDATED]
คลาสหลักที่ทำหน้าที่เป็น State Machine (S1 -> S2 -> S3)
รับ ImageBGR -> คืนค่า BoardState (รวม FEN)
"""
from typing import Optional
from src.core.typing import ImageBGR, FEN_String, PieceGrid, WarpedImage
from src.core.exceptions import BoardNotFoundException # Import เพิ่ม
from src.pipeline.s1_board_locator import BoardLocator
from src.pipeline.s2_occupancy_model import OccupancyModel
from src.pipeline.s3_piece_model import PieceModel # [NEW] Import State 3
from src.utils import image_utils, fen_utils # [NEW] Import fen_utils

# [NEW] สร้าง Dataclass เก็บผลลัพธ์
from dataclasses import dataclass
@dataclass
class BoardState:
    fen: FEN_String
    piece_grid: PieceGrid
    warped_image: WarpedImage

class StateRecognizer:
    """
    คลาสหลักที่รัน Pipeline S1->S2->S3
    """
    def __init__(self,
                 occupancy_model_path: str,
                 piece_model_path: str,
                 use_dummy_occupancy: bool = False):

        print("Initializing StateRecognizer Pipeline...")
        # 1. โหลด State 1
        self.locator = BoardLocator()
        
        # 2. โหลด State 2
        self.occupancy_model = OccupancyModel(
            occupancy_model_path,
            use_dummy=use_dummy_occupancy
        )
        
        # 3. โหลด State 3
        # (State 3 ไม่ควรใช้ Dummy เพราะสำคัญมาก)
        self.piece_model = PieceModel(piece_model_path)
        print("StateRecognizer Pipeline Initialized.")


    def recognize(self, image: ImageBGR) -> Optional[BoardState]:
        """
        รัน Pipeline ทั้งหมดบนภาพเดียว (End-to-End)
        Input: ImageBGR
        Output: BoardState (เก็บ FEN, Grid, Warped Image) หรือ None ถ้าล้มเหลว
        """
        try:
            # === STATE 1: BOARD LOCALISATION ===
            warped_board, matrix = self.locator.find_and_warp(image)
            print("S1: Board located and warped.")
            
            # === STATE 2: OCCUPANCY CLASSIFICATION ===
            # ใช้ Crop แบบ State 2 (มี Context)
            occupancy_squares = image_utils.crop_squares_from_warped(
                warped_board,
                context_ratio=0.5
            )
            occupancy_grid = self.occupancy_model.predict(occupancy_squares)
            num_occupied = sum(1 for occupied in occupancy_grid if occupied)
            print(f"S2: Occupancy predicted ({num_occupied}/64 occupied).")

            # === STATE 3: PIECE CLASSIFICATION ===
            # ใช้ Crop แบบ State 3 (taller boxes) และ Occupancy Grid
            piece_grid = self.piece_model.predict(
                warped_board,
                occupancy_grid
            )
            num_pieces = sum(1 for p in piece_grid if p != 'empty')
            print(f"S3: Piece classification complete ({num_pieces} pieces found).")

            # === FINAL STEP: FEN GENERATION ===
            fen = fen_utils.convert_grid_to_fen(piece_grid)
            print(f"FEN Generated: {fen}")

            return BoardState(fen=fen, piece_grid=piece_grid, warped_image=warped_board)

        except BoardNotFoundException as e:
            print(f"!!! Pipeline Error (S1): {e}")
            return None
        except Exception as e:
            print(f"!!! Pipeline Error (S2/S3/FEN): {e}")
            # อาจจะคืนค่าบางส่วน หรือ None ก็ได้
            return None