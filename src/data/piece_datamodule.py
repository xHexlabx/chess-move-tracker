"""
[UPDATED]
สร้าง DataModule สำหรับ Piece Classification Dataset (State 3)

[NEW LOGIC]:
- ไม่ใช้ random_split
- โหลด '.../train/' เป็น train_dataset (ใช้ Heavy Augmentation)
- โหลด '.../val/' เป็น val_dataset (ใช้ Normal Transform)
- โหลด '.../test/' เป็น test_dataset (ใช้ Normal Transform)
"""
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from typing import Optional
import os

class PieceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # InceptionV3 ต้องการ Input 299x299
        INPUT_SIZE = (299, 299)

        # 1. [Heavy Augmentation] สำหรับ Training
        self.train_transform = T.Compose([
            T.Resize(INPUT_SIZE),
            # --- Augmentations ---
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            # ---------------------
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. Transform ธรรมดาสำหรับ Validation/Test
        self.eval_transform = T.Compose([
            T.Resize(INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None # [NEW]

    def setup(self, stage: Optional[str] = None):
        """
        โหลดข้อมูลจาก 'train', 'val', 'test' แยกกัน
        """
        train_root = os.path.join(self.data_dir, "train")
        val_root = os.path.join(self.data_dir, "val")
        test_root = os.path.join(self.data_dir, "test") # [NEW]

        # 1. โหลด Train Set
        if os.path.exists(train_root):
            self.train_dataset = ImageFolder(root=train_root, transform=self.train_transform)
            print(f"PieceDataModule: โหลด Train Set: {len(self.train_dataset)} ภาพ")
            print(f"PieceDataModule: Train Classes: {self.train_dataset.classes}")
        else:
             print(f"PieceDataModule: คำเตือน: ไม่พบ Train Set ที่ {train_root}")

        # 2. โหลด Val Set
        if os.path.exists(val_root):
            self.val_dataset = ImageFolder(root=val_root, transform=self.eval_transform)
            print(f"PieceDataModule: โหลด Val Set: {len(self.val_dataset)} ภาพ")
            print(f"PieceDataModule: Val Classes: {self.val_dataset.classes}")
        else:
            print(f"PieceDataModule: คำเตือน: ไม่พบ Val Set ที่ {val_root}")

        # 3. โหลด Test Set [NEW]
        if os.path.exists(test_root):
            self.test_dataset = ImageFolder(root=test_root, transform=self.eval_transform)
            print(f"PieceDataModule: โหลด Test Set: {len(self.test_dataset)} ภาพ")
            print(f"PieceDataModule: Test Classes: {self.test_dataset.classes}")
        else:
            print(f"PieceDataModule: คำเตือน: ไม่พบ Test Set ที่ {test_root}")


    def train_dataloader(self) -> DataLoader:
        if self.train_dataset:
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=os.cpu_count() // 2, pin_memory=True
            )
        return None

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset:
            return DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=os.cpu_count() // 2, pin_memory=True
            )
        return None

    def test_dataloader(self) -> DataLoader: # [NEW]
        if self.test_dataset:
            return DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=os.cpu_count() // 2, pin_memory=True
            )
        return None