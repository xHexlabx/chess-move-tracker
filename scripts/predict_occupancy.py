import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.models.occupancy.occupancy_lit_model import OccupancyLitModel  # <-- Import โมเดลของคุณ

# --- 1. กำหนดค่าพื้นฐาน ---
MODEL_PATH = "./models/occupancy/occupancy_model_best.ckpt"
CLASS_NAMES = ['0_empty', '1_occupied']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 🚨 (สำคัญมาก) กำหนด Transforms ---
# Transforms เหล่านี้ "ต้อง" เหมือนกับ Validation/Test transforms
# ที่คุณใช้ใน DataModule ตอนเทรนทุกประการ
# (ResNet มักใช้ 224x224 และ ImageNet normalization)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image_path):
    print(f"Loading model from: {MODEL_PATH}")
    # --- 3. โหลดโมเดลจาก Checkpoint ---
    # เราใช้ .load_from_checkpoint() ของ Pytorch Lightning
    try:
        model = OccupancyLitModel.load_from_checkpoint(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'src.models.occupancy.occupancy_lit_model' is correct.")
        return

    model.to(DEVICE)  # ย้ายโมเดลไป GPU
    model.eval()      # ❗ ตั้งเป็นโหมด Evaluation (ปิด Dropout ฯลฯ)

    print(f"Loading and processing image: {image_path}")
    # --- 4. โหลดและแปลงภาพ ---
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
        
    # แปลงภาพด้วย transforms และเพิ่ม "batch dimension" (มิติที่ 0)
    input_tensor = data_transform(image).unsqueeze(0).to(DEVICE)

    # --- 5. ทำนายผล ---
    with torch.no_grad():  # ❗ ปิดการคำนวณ Gradient เพื่อประหยัด Memory
        output = model(input_tensor)
        
        # แปลง output (logits) เป็น probabilities
        probabilities = torch.softmax(output, dim=1)
        
        # หา class ที่มีความมั่นใจสูงสุด
        confidence, predicted_class_idx = torch.max(probabilities, 1)

    # --- 6. แสดงผล ---
    predicted_label = CLASS_NAMES[predicted_class_idx.item()]
    confidence_percent = confidence.item() * 100

    print("\n--- Prediction Result ---")
    print(f"Image:     {image_path}")
    print(f"Prediction:  {predicted_label}")
    print(f"Confidence:  {confidence_percent:.2f}%")
    print("--------------------------")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.predict_occupancy <path_to_image>")
    else:
        predict(sys.argv[1])