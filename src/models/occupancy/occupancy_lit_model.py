"""
[UPDATED]
นิยาม "พิมพ์เขียว" ของโมเดล Occupancy (State 2)
โดยใช้ PyTorch Lightning

[FIX] ลบ circular import ที่ผิดพลาดออก
"""
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torchmetrics

class OccupancyLitModel(pl.LightningModule):
    """
    LightningModule สำหรับ Occupancy Classification (2 Classes: Empty, Occupied)
    """
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 2)

        self.criterion = nn.CrossEntropyLoss()
        # สร้าง metric แยกสำหรับแต่ละ step (best practice)
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _shared_step(self, batch, batch_idx):
        """Logic calculation ที่ใช้ร่วมกัน"""
        x, y = batch # (images, labels)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        acc = self.train_accuracy(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        acc = self.val_accuracy(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Logic สำหรับการทดสอบ (เหมือน validation_step มาก)
        """
        loss, preds, y = self._shared_step(batch, batch_idx)
        acc = self.test_accuracy(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer