"""
[UPDATED]
นิยาม "พิมพ์เขียว" ของโมเดล Piece Classification (State 3)
โดยใช้ PyTorch Lightning (12 Classes)

[FIX] เพิ่ม test_step() method เพื่อให้ trainer.test() ทำงานได้
"""
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torchmetrics
from typing import List

class PieceLitModel(pl.LightningModule):
    """
    LightningModule สำหรับ Piece Classification (13 Classes including empty)
    """

    CLASSES: List[str] = [
        'b_Bishop', 'b_King', 'b_Knight', 'b_Pawn', 'b_Queen', 'b_Rook',
        'w_Bishop', 'w_King', 'w_Knight', 'w_Pawn', 'w_Queen', 'w_Rook',
        'empty'
    ]
    
    NUM_CLASSES = len(CLASSES) # 13

    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.backbone = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

        # Main classifier
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, self.NUM_CLASSES)

        # Aux classifier
        # Only modify if it exists (it does for default inception_v3)
        if hasattr(self.backbone, 'AuxLogits') and self.backbone.AuxLogits is not None:
             num_ftrs_aux = self.backbone.AuxLogits.fc.in_features
             self.backbone.AuxLogits.fc = nn.Linear(num_ftrs_aux, self.NUM_CLASSES)
        else:
             # Handle cases where aux logits might be turned off
             print("Warning: AuxLogits not found or disabled in InceptionV3 model.")


        self.criterion = nn.CrossEntropyLoss()
        # [NEW] สร้าง metric แยกสำหรับแต่ละ step
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)


    def forward(self, x: torch.Tensor):
        outputs = self.backbone(x)
        return outputs

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)

        # InceptionV3 Logic (handle training vs eval output)
        if self.training and isinstance(outputs, models.inception.InceptionOutputs):
            logits, aux_logits = outputs
            loss1 = self.criterion(logits, y)
            loss2 = self.criterion(aux_logits, y)
            loss = loss1 + 0.4 * loss2 # Standard InceptionV3 loss weighting
            final_logits = logits # Use main logits for accuracy calculation
        else: # During eval or if aux_logits are off
            logits = outputs
            loss = self.criterion(logits, y)
            final_logits = logits

        preds = torch.argmax(final_logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        acc = self.train_accuracy(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch, batch_idx)
        acc = self.val_accuracy(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    # --- [NEW] test_step Method ---
    def test_step(self, batch, batch_idx):
        """
        Logic for the testing phase (identical to validation).
        """
        loss, preds, y = self._shared_step(batch, batch_idx)
        acc = self.test_accuracy(preds, y)
        # Log test metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    # --- End of test_step ---

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer