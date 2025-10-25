# Chess Move Tracker Project

โปรเจกต์นี้มีเป้าหมายเพื่อสร้าง Pipeline สำหรับติดตามการเดินหมากรุกจากวิดีโอ และแปลงเป็น FEN/PGN โดยใช้ Computer Vision และ Deep Learning (PyTorch Lightning)

---

## 🚀 เริ่มต้นใช้งาน (ทดสอบ State 1 + 2)

นี่คือขั้นตอนสำหรับรันทดสอบ Pipeline ปัจจุบัน (State 1: Board Localisation + State 2: Occupancy Classification)

### 1. ติดตั้ง Environment

โปรเจกต์นี้ใช้ `conda` ในการจัดการ dependencies

```bash
# สร้าง environment ใหม่ชื่อ chess_tracker
conda env create -f environment.yml

# เปิดใช้งาน environment
conda activate chess_tracker
````

### 2\. ทดสอบ Pipeline (ด้วยโมเดลที่เทรนแล้ว)

สคริปต์นี้จะรันภาพทั้งหมดใน `IMAGE_NAMES` (ที่กำหนดไว้ในสคริปต์) ผ่าน State 1 (Warp) และ State 2 (Classify) และบันทึกผลลัพธ์ลงในไฟล์ `.png`

1.  **เตรียมภาพ:** ตรวจสอบว่าคุณมีภาพ (เช่น `test_image_1.jpg`, `test_image_2.jpg` ฯลฯ) ใน `data/raw/images/single_test/`
2.  **เตรียมโมเดล:** ตรวจสอบว่าคุณมีโมเดลที่เทรนแล้วที่ `models/occupancy/occupancy_model_best.ckpt`
3.  **รันสคริปต์:**
    ```bash
    python scripts/visualize_state2.py
    ```
4.  **(สำหรับ Colab/Server) ดูผลลัพธ์:** รันเซลล์ใหม่เพื่อแสดงภาพ
    ```python
    from IPython.display import Image, display
    display(Image('state2_visualization_batch.png'))
    ```

ผลลัพธ์ที่ได้จะเป็นภาพสูงๆ ที่แสดงผลการวิเคราะห์ "Original", "Warped", และ "Occupancy Result (O/E)" ของภาพทดสอบแต่ละภาพ

-----

## 🤖 ขั้นตอนการเทรน (Workflow)

หากคุณต้องการเทรนโมเดล State 2 ใหม่ (เช่น เพิ่มข้อมูล, เปลี่ยนสถาปัตยกรรม) ให้ทำตามขั้นตอนดังนี้:

### 1\. เตรียมข้อมูล (Few-Shot Auto-Labeling)

สคริปต์นี้จะสร้างชุดข้อมูล (Dataset) โดยอัตโนมัติจากภาพ "กระดานตั้งต้น" (Starting Position)

1.  **เตรียมภาพ:**
      * **Train Set:** นำภาพตั้งต้นที่ใช้เทรนไปไว้ที่ `data/raw/images/single_test/` (เช่น `white_view_1.jpg`, `black_view_1.jpg`)
      * **Validation Set:** นำภาพตั้งต้นอื่นๆ ที่ใช้ทดสอบไปไว้ที่เดียวกัน (เช่น `test_image_1.jpg`)
2.  **แก้ไขสคริปต์:** เปิด `scripts/prepare_fewshot_data.py` และตั้งค่า `TRAIN_PREFIXES` ให้ตรงกับชื่อไฟล์ Train Set ของคุณ
3.  **รันสคริปต์:**
    ```bash
    python scripts/prepare_fewshot_data.py
    ```
    (สคริปต์จะสร้างข้อมูลที่ติดป้ายแล้วใน `data/processed/occupancy_dataset/` โดยแบ่งเป็น `train` และ `val` ให้อัตโนมัติ)

### 2\. รันการเทรน (PyTorch Lightning)

เมื่อข้อมูลพร้อม รันสคริปต์เทรน:

```bash
python scripts/train_occupancy.py
```

  * สคริปต์นี้จะโหลดข้อมูล, ใช้ Data Augmentation, เริ่มเทรน, และบันทึกโมเดลที่ดีที่สุด (วัดจาก `val_acc`) ไปที่ `models/occupancy/occupancy_model_best.ckpt`

-----

## 🎯 Pipeline Overview

### State 1: Board Localisation (✅ เสร็จแล้ว)

  * **รับ:** ภาพดิบ (BGR)
  * **ทำ:** Mask สี $\rightarrow$ หา Contour/Hull $\rightarrow$ ประมาณ 4 มุม $\rightarrow$ Warp $\rightarrow$ ตรวจสอบสีหมาก $\rightarrow$ หมุนภาพ (ขาวล่าง-ดำบน)
  * **คืนค่า:** `WarpedImage`

### State 2: Occupancy Classification (✅ เสร็จแล้ว)

  * **รับ:** `WarpedImage`
  * **ทำ:** ตัด 64 ช่อง (พร้อม Context) $\rightarrow$ ป้อนเข้าโมเดล `OccupancyLitModel` (ResNet18)
  * **คืนค่า:** `OccupancyGrid` (List 64 bools)

### State 3: Piece Classification (🔵 ขั้นต่อไป)

  * **รับ:** `WarpedImage` และ `OccupancyGrid`
  * **เป้าหมาย:** สร้างโมเดลจำแนกชนิดหมาก (12-class CNN: wP, wN, wB, wR, wQ, wK, bP, ..., bK)
  * **Logic:**
    1.  ตัดภาพช่องอีกครั้ง (Paper แนะนำให้ใช้ Bounding Box ที่ "สูงกว่า" สำหรับ State นี้)
    2.  ส่งเฉพาะภาพช่องที่ `OccupancyGrid` เป็น `True` (มีหมาก) เข้าโมเดล
    3.  รวมผลลัพธ์กลับเป็น `PieceGrid` (List 64 strings: 'wP', 'bN', 'empty', ...)

<!-- end list -->
