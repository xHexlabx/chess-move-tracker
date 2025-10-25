# Chess Move Tracker Project

โปรเจกต์นี้มีเป้าหมายเพื่อสร้าง Pipeline สำหรับติดตามการเดินหมากรุกจากวิดีโอ และแปลงเป็น PGN

## 🚀 เริ่มต้นใช้งาน (State 1)

นี่คือขั้นตอนสำหรับรันทดสอบ **State 1: Board Localisation**

### 1. ติดตั้ง Environment

โปรเจกต์นี้ใช้ `conda` ในการจัดการ dependencies

```bash
# สร้าง environment ใหม่ชื่อ chess_tracker
conda env create -f environment.yml

# เปิดใช้งาน environment
conda activate chess_tracker