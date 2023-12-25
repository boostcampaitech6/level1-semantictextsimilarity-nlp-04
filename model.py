import argparse
import random

import pandas as pd

import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
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
        pearson = pearson_corrcoef(predictions, labels.to(torch.float64))
        return loss, pearson

    def training_step(self, batch, batch_idx):
        train_loss, pearson = self.step(batch)
        self.log("train_loss", train_loss, logger=True)
        self.log("train_pearson", pearson, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss, pearson = self.step(batch)
        self.log("val_loss", val_loss, logger=True)
        self.log("val_pearson", pearson, logger=True)

    def test_step(self, batch, batch_idx):
        inputs = {key: val for key, val in batch.items() if key!= 'labels'}
        labels = batch['labels']
        predictions = self(**inputs)
        self.predictions.append(predictions.detach().cpu())
        pearson = pearson_corrcoef(predictions, labels.to(torch.float64))
        self.log("test_pearson", pearson, logger=True)
            
    def predict_step(self, batch, batch_idx):
        inputs = {key: val for key, val in batch.items() if key!= 'labels'}
        return self(**inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def setup(self, stage=None):
        if stage =='test':
            self.predictions = []


