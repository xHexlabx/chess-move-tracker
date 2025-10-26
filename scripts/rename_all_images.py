import os
import sys

# เพิ่ม src/ เข้าไปใน path (เผื่อกรณีรันจาก root project)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- ตั้งค่า ---
# [NEW] กำหนด Directory และ Prefix ที่ต้องการ Rename
TARGETS = [
    {"directory": "data/raw/images/train/", "prefix": "train_"},
    {"directory": "data/raw/images/val/",   "prefix": "val_"},
    {"directory": "data/raw/images/test/",  "prefix": "test_"}
]
# ---------------

def rename_files_in_directory(directory_path, prefix):
    """
    Renames all .jpg, .png, .jpeg files in the specified directory
    to prefix_1.ext, prefix_2.ext, ...
    """
    print(f"\n--- Processing directory: {directory_path} ---")

    # ตรวจสอบว่า directory มีอยู่จริง
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return 0 # คืนค่าจำนวนไฟล์ที่ rename = 0

    # 1. List ไฟล์ทั้งหมด
    try:
        all_files = os.listdir(directory_path)
    except OSError as e:
        print(f"Error listing directory contents: {e}")
        return 0

    # 2. กรองเฉพาะไฟล์ภาพ (jpg, png, jpeg - ตัวพิมพ์เล็ก/ใหญ่)
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in all_files if f.lower().endswith(image_extensions)]

    # 3. เรียงลำดับ
    image_files.sort()

    print(f"Found {len(image_files)} image files to potentially rename.")

    # 4. วน Loop และ Rename
    counter = 1
    renamed_count = 0
    skipped_count = 0
    error_count = 0

    for old_filename in image_files:
        # ดึงนามสกุลไฟล์เดิม
        _, extension = os.path.splitext(old_filename)
        extension = extension.lower() # ใช้นามสกุลตัวพิมพ์เล็ก

        # สร้างชื่อไฟล์ใหม่
        new_filename = f"{prefix}{counter}{extension}"

        # สร้าง Path เต็ม
        old_path = os.path.join(directory_path, old_filename)
        new_path = os.path.join(directory_path, new_filename)

        # ตรวจสอบว่าชื่อไฟล์ใหม่ซ้ำกับไฟล์ที่มีอยู่ (ที่ไม่ใช่ตัวเอง) หรือไม่
        # (แปลงเป็น lowercase เพื่อเทียบ)
        existing_files_lower = {f.lower() for f in all_files}
        if new_filename.lower() in existing_files_lower and old_filename.lower() != new_filename.lower():
            print(f"   Skipping rename for '{old_filename}': Target name '{new_filename}' already exists.")
            skipped_count += 1
            # Note: We don't increment the counter here if skipping due to conflict
            continue # ข้ามไฟล์นี้ไป

        # Rename
        try:
            # ไม่ rename ถ้าชื่อเหมือนเดิมแล้ว (เทียบแบบ case-insensitive)
            if old_filename.lower() != new_filename.lower():
                os.rename(old_path, new_path)
                print(f"   Renamed: '{old_filename}' -> '{new_filename}'")
                renamed_count += 1
            else:
                 print(f"   Skipping: '{old_filename}' already has the correct name.")
                 skipped_count += 1

            counter += 1 # เพิ่ม counter เฉพาะเมื่อ rename สำเร็จ หรือชื่อถูกต้องอยู่แล้ว

        except OSError as e:
            print(f"   Error renaming '{old_filename}' to '{new_filename}': {e}")
            error_count += 1

    print(f"Finished processing. Renamed: {renamed_count}, Skipped: {skipped_count}, Errors: {error_count}")
    return renamed_count

if __name__ == "__main__":
    total_renamed = 0
    print("Starting file renaming process...")
    # วน Loop ตาม TARGETS ที่กำหนด
    for target in TARGETS:
        total_renamed += rename_files_in_directory(target["directory"], target["prefix"])

    print("-" * 30)
    print(f"Total files renamed across all directories: {total_renamed}")