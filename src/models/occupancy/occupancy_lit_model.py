"""
[NEW FILE]
นิยาม "พิมพ์เขียว" ของโมเดล Occupancy (State 2) 
โดยใช้ PyTorch Lightning
"""
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torchmetrics

class OccupancyLitModel(pl.LightningModule):
    """
    LightningModule สำหรับ Occupancy Classification (2 Classes: Empty, Occupied)
    
    [cite_start]เราจะใช้ ResNet18 ตามที่ Paper แนะนำ [cite: 345, 409]
    """
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        
        # บันทึก hyperparameter (เช่น lr)
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # 1. โหลดสถาปัตยกรรม (Backbone)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. แก้ไข Layer สุดท้าย (Classifier Head)
        num_ftrs = self.backbone.fc.in_features
        # เราต้องการ Output แค่ 2 (Empty=0, Occupied=1)
        self.backbone.fc = nn.Linear(num_ftrs, 2)

        # 3. กำหนด Loss และ Metrics
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        รันโมเดล (Inference)
        """
        return self.backbone(x)

    def _shared_step(self, batch, batch_idx, step_type: str):
        """
        Logic ที่ใช้ร่วมกันระหว่าง Training และ Validation
        """
        x, y = batch # (images, labels)
        logits = self.forward(x)
        
        # คำนวณ Loss
        loss = self.criterion(logits, y)
        
        # คำนวณ Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{step_type}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer