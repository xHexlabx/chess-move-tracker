"""
[NEW FILE]
นิยาม "พิมพ์เขียว" ของโมเดล Piece Classification (State 3) 
โดยใช้ PyTorch Lightning (12 Classes)
"""
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torchmetrics
from typing import List

class PieceLitModel(pl.LightningModule):
    """
    LightningModule สำหรับ Piece Classification (12 Classes)
    
    Paper  แนะนำ InceptionV3
    """
    
    # [NEW] กำหนด Class 12 คลาส + 1 (Empty)
    # เราจะเทรน 13 คลาส (รวม 'empty') เพื่อให้โมเดลทนทาน
    # แม้ว่า State 2 จะกรองมาแล้วก็ตาม
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

        # 1. โหลด InceptionV3 (ต้องการ Input 299x299)
        self.backbone = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        
        # 2. แก้ไข Layer สุดท้าย (Classifier Head)
        # InceptionV3 มี 2 outputs (main, aux)
        # 2a. Main classifier
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, self.NUM_CLASSES)
        
        # 2b. Aux classifier
        num_ftrs_aux = self.backbone.AuxLogits.fc.in_features
        self.backbone.AuxLogits.fc = nn.Linear(num_ftrs_aux, self.NUM_CLASSES)

        # 3. กำหนด Loss และ Metrics
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)

    def forward(self, x: torch.Tensor):
        # InceptionV3 คืนค่า logits 2 ตัว (main, aux) ตอน .train()
        # และคืนค่า main logits ตอน .eval()
        outputs = self.backbone(x)
        return outputs

    def _shared_step(self, batch, batch_idx, step_type: str):
        x, y = batch
        outputs = self.forward(x)
        
        # InceptionV3 Logic (ตอน .train() จะคืนค่าเป็น Tuple)
        if self.training:
            logits, aux_logits = outputs
            loss1 = self.criterion(logits, y)
            loss2 = self.criterion(aux_logits, y)
            loss = loss1 + 0.4 * loss2 # ตามเอกสารของ InceptionV3
        else: # ตอน .eval()
            logits = outputs
            loss = self.criterion(logits, y)
        
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