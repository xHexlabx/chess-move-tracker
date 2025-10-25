"""
[NEW FILE]
ฟังก์ชันช่วยเหลือสำหรับจัดการ FEN (Forsyth-Edwards Notation)
และเก็บ Map มาตรฐานของกระดาน
"""
from src.core.typing import PieceGrid, FEN_String

# Map มาตรฐานของกระดาน (index 0-63)
# (จำเป็นสำหรับ prepare_piece_data.py)
PIECE_MAP = [
    'b_Rook', 'b_Knight', 'b_Bishop', 'b_Queen', 'b_King', 'b_Bishop', 'b_Knight', 'b_Rook',
    'b_Pawn', 'b_Pawn', 'b_Pawn', 'b_Pawn', 'b_Pawn', 'b_Pawn', 'b_Pawn', 'b_Pawn',
    'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty',
    'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty',
    'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty',
    'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty',
    'w_Pawn', 'w_Pawn', 'w_Pawn', 'w_Pawn', 'w_Pawn', 'w_Pawn', 'w_Pawn', 'w_Pawn',
    'w_Rook', 'w_Knight', 'w_Bishop', 'w_Queen', 'w_King', 'w_Bishop', 'w_Knight', 'w_Rook'
]

# Map ชื่อย่อสำหรับ FEN
PIECE_TO_FEN = {
    'b_Rook': 'r', 'b_Knight': 'n', 'b_Bishop': 'b', 'b_Queen': 'q', 'b_King': 'k', 'b_Pawn': 'p',
    'w_Rook': 'R', 'w_Knight': 'N', 'w_Bishop': 'B', 'w_Queen': 'Q', 'w_King': 'K', 'w_Pawn': 'P',
    'empty': '1' # '1' คือ 1 ช่องว่าง
}

def convert_grid_to_fen(grid: PieceGrid) -> FEN_String:
    """
    แปลง List 64 ช่อง ('empty', 'w_Pawn'...) ให้เป็น FEN string
    (ยังไม่รวมข้อมูล 'w KQkq - 0 1')
    """
    fen = ""
    empty_count = 0
    
    for i in range(64):
        piece_name = grid[i]
        fen_char = PIECE_TO_FEN.get(piece_name, '1')

        if fen_char == '1':
            empty_count += 1
        else:
            if empty_count > 0:
                fen += str(empty_count)
                empty_count = 0
            fen += fen_char
            
        # เมื่อจบคอลัมน์
        if (i + 1) % 8 == 0:
            if empty_count > 0:
                fen += str(empty_count)
            if i < 63: # ถ้ายังไม่จบท้ายสุด
                fen += '/'
            empty_count = 0
            
    return fen