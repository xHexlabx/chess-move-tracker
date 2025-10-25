# Chess Move Tracker Project

โปรเจกต์นี้มีเป้าหมายเพื่อสร้าง Pipeline สำหรับติดตามการเดินหมากรุกจากวิดีโอ และแปลงเป็น PGN โดยใช้ Computer Vision และ Deep Learning (PyTorch Lightning)

---

## 🚀 เริ่มต้นใช้งาน (ทดสอบ State 1 + 2)

นี่คือขั้นตอนสำหรับรันทดสอบ Pipeline (State 1: Board Localisation + State 2: Occupancy Classification)

### 1. ติดตั้ง Environment

โปรเจกต์นี้ใช้ `conda` ในการจัดการ dependencies

```bash
# สร้าง environment ใหม่ชื่อ chess_tracker
conda env create -f environment.yml

# เปิดใช้งาน environment
conda activate chess_tracker
````

### 2\. เตรียมโมเดล (Dummy)

สคริปต์ `visualize_state2.py` ถูกตั้งค่าเริ่มต้นให้ใช้ **Dummy Model** (โมเดลจำลอง) คุณจึงสามารถรันเพื่อทดสอบโครงสร้างได้ทันที

### 3\. รันสคริปต์ทดสอบ

รันสคริปต์ `visualize_state2.py` เพื่อทดสอบการทำงานของ State 1 (Warping) และ State 2 (Dummy Classification)

1.  **เตรียมภาพ:** ตรวจสอบว่าคุณมีภาพ (เช่น `test_image_1.jpg`) ใน `data/raw/images/single_test/`
2.  **รันสคริปต์:**
    ```bash
    python scripts/visualize_state2.py
    ```

หากทำงานสำเร็จ คุณจะเห็นหน้าต่าง Matplotlib แสดง 3 ภาพ:

1.  **Original Image**: ภาพต้นฉบับ
2.  **State 1: Warped Board**: ภาพที่ถูก Warp และหมุนทิศทาง (ขาวล่าง-ดำบน)
3.  **State 2: Occupancy Result**: ภาพที่มีการวาด "O" (Occupied) และ "E" (Empty) ทับ (ผลลัพธ์จาก Dummy Model)

-----

## 🤖 ขั้นตอนที่ 2: การเทรนโมเดล State 2 (Occupancy)

หากคุณต้องการเทรนโมเดล AI ของคุณเอง (แทนการใช้ Dummy Model) ให้ทำตามขั้นตอนดังนี้:

### ขั้นตอน 2.1: เตรียมข้อมูล (Few-Shot Auto-Labeling)

เราจะสร้างชุดข้อมูล (Dataset) โดยอัตโนมัติจากภาพ "กระดานตั้งต้น" (Starting Position)

1.  **เตรียมภาพ:**

      * นำภาพ "กระดานตั้งต้น" ที่คุณต้องการใช้ **เทรน** ไปไว้ใน `data/raw/images/single_test/` และตั้งชื่อให้ขึ้นต้นด้วย `white_view_` หรือ `black_view_` (เช่น `white_view_1.jpg`)
      * นำภาพ "กระดานตั้งต้น" อื่นๆ ที่คุณต้องการใช้ **วัดผล (Validation)** ไปไว้ในโฟลเดอร์เดียวกัน (เช่น `test_image_1.jpg`, `test_image_2.jpg`)

2.  **รันสคริปต์เตรียมข้อมูล:**

    ```bash
    python scripts/prepare_fewshot_data.py
    ```

      * สคริปต์นี้จะสแกนภาพ, รัน State 1, ตัด 64 ช่อง, และบันทึกภาพที่ติดป้าย (Label) แล้วลงใน `data/processed/occupancy_dataset/` (แบ่ง `train` / `val` ตามชื่อไฟล์)

### ขั้นตอน 2.2: รันการเทรน (PyTorch Lightning)

เมื่อข้อมูลพร้อม รันสคริปต์เทรน:

```bash
python scripts/train_occupancy.py
```

  * สคริปต์นี้จะ:
    1.  โหลดข้อมูลจาก `.../train/` และ `.../val/` (โดย `OccupancyDataModule`)
    2.  ใช้ **Data Augmentation** อย่างหนักกับชุดข้อมูล Train
    3.  เริ่มเทรนโมเดล `OccupancyLitModel` (ResNet18)
    4.  แสดงผล `val_acc` (ความแม่นยำในการวัดผล)
    5.  บันทึกโมเดลที่ดีที่สุด (Best Checkpoint) ไปที่ `models/occupancy/occupancy_model_best.ckpt`

### ขั้นตอน 2.3: ทดสอบโมเดลจริง

1.  เปิดไฟล์ `scripts/visualize_state2.py`

2.  แก้ไข 2 บรรทัดนี้:

    ```python
    # Path ไปยังไฟล์ checkpoint ที่เทรนเสร็จ
    MODEL_CHECKPOINT_PATH = "models/occupancy/occupancy_model_best.ckpt" 

    # ปิด Dummy Model
    USE_DUMMY_MODEL = False
    ```

3.  รันสคริปต์อีกครั้ง:

    ```bash
    python scripts/visualize_state2.py
    ```

    ตอนนี้ ภาพผลลัพธ์ "State 2" จะแสดงผลการทำนายจาก AI จริงที่คุณเพิ่งเทรน\!

-----

## 🎯 Pipeline Overview

### State 1: Board Localisation (✅ เสร็จแล้ว)

  * **รับ:** ภาพดิบ (BGR)
  * **ทำ:** Mask สี $\rightarrow$ หา Contour $\rightarrow$ หา Convex Hull $\rightarrow$ ประมาณ 4 มุม $\rightarrow$ Warp ภาพ $\rightarrow$ ตรวจสอบสีหมาก (ขาว/ดำ) $\rightarrow$ หมุนภาพให้ถูกทิศ
  * **คืนค่า:** `WarpedImage` (ภาพกระดาน 600x600 ที่มี "หมากขาว" อยู่ล่าง)

### State 2: Occupancy Classification (✅ เสร็จแล้ว)

  * **รับ:** `WarpedImage`
  * **ทำ:** ตัด 64 ช่อง (พร้อม Context) $\rightarrow$ ป้อนเข้าโมเดล `OccupancyLitModel` (ResNet18)
  * **คืนค่า:** `OccupancyGrid` (List 64 bools)

### State 3: Piece Classification (ขั้นต่อไป)

  * **รับ:** `WarpedImage` และ `OccupancyGrid`
  * **ทำ:** ตัดเฉพาะช่องที่มีหมาก (`True`) $\rightarrow$ ป้อนเข้าโมเดล (12-class CNN)
  * **คืนค่า:** `PieceGrid` (List 64 strings: 'wP', 'bN', 'empty', ...)

<!-- end list -->
