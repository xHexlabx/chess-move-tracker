"""
[UPDATED]
สร้าง DataModule สำหรับ Occupancy Dataset

[NEW LOGIC]:
- ไม่ใช้ random_split
- โหลด '.../train/' เป็น train_dataset (ใช้ Heavy Augmentation)
- โหลด '.../val/' เป็น val_dataset (ใช้ Normal Transform)
"""
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from typing import Optional
import os

class OccupancyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # 1. [Heavy Augmentation] สำหรับ Training
        self.train_transform = T.Compose([
            T.Resize((100, 100)),
            # --- Augmentations ---
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # ---------------------
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. Transform ธรรมดาสำหรับ Validation
        self.val_transform = T.Compose([
            T.Resize((100, 100)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        โหลดข้อมูลจาก 'train' และ 'val' แยกกัน
        """
        train_root = os.path.join(self.data_dir, "train")
        val_root = os.path.join(self.data_dir, "val")
        
        if not os.path.exists(train_root):
            raise FileNotFoundError(f"ไม่พบไดเรกทอรี Train: {train_root}")
        if not os.path.exists(val_root):
            raise FileNotFoundError(f"ไม่พบไดเรกทอรี Val: {val_root}")
            
        # 1. โหลด Train Set (พร้อม Augmentation หนักๆ)
        self.train_dataset = ImageFolder(root=train_root, transform=self.train_transform)
        print(f"DataModule: โหลด Train Set: {len(self.train_dataset)} ภาพ")
        print(f"DataModule: Train Classes: {self.train_dataset.classes}")
        
        # 2. โหลด Val Set (ใช้ Transform ธรรมดา)
        self.val_dataset = ImageFolder(root=val_root, transform=self.val_transform)
        print(f"DataModule: โหลด Val Set: {len(self.val_dataset)} ภาพ")
        print(f"DataModule: Val Classes: {self.val_dataset.classes}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 2,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() // 2,
            pin_memory=True
        )