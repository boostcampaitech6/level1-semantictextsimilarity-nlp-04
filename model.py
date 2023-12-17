import argparse
import random

import pandas as pd

import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from torchmetrics.functional import pearson_corrcoef


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.lr = lr
        self.loss_func = torch.nn.L1Loss()

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs.logits.squeeze(-1)

    def step(self, batch):
        inputs = {key: val for key, val in batch.items() if key!= 'labels'}
        labels = batch['labels']
        predictions = self(**inputs)
        loss = self.loss_func(predictions, labels.float())
        pearson = pearson_corrcoef(predictions, labels.float())
        return loss, pearson

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pearson = self.step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_pearson", pearson, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, pearson = self.step(batch)
        self.log("test_pearson", pearson)
        
    def predict_step(self, batch, batch_idx):
        inputs = {key: val for key, val in batch.items() if key!= 'labels'}
        return self(**inputs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)