import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, get_linear_schedule_with_warmup
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.classification import BinaryF1Score


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        # Models
        self.model = AutoModel.from_pretrained(model_name)
        # Heads
        self.classification_head = nn.Linear(self.model.config.hidden_size, 1)
        self.regression_head_0 = nn.Linear(self.model.config.hidden_size, 1)
        self.regression_head_1 = nn.Linear(self.model.config.hidden_size, 1)
        # Loss functions
        self.classifier_loss_fn = nn.BCEWithLogitsLoss()
        self.regressor_loss_fn = nn.L1Loss()
        # Evaluation metrics
        self.f1_score = BinaryF1Score()
        self.pearson_corrcoef = PearsonCorrCoef()
        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in [self.classification_head, self.regression_head_0, self.regression_head_1]:
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs.get('pooler_output', outputs.last_hidden_state[:, 0])
        classification_output = self.classification_head(pooled_output)
        classifiaction_probs = torch.sigmoid(classification_output)
        binary = (classifiaction_probs > 0.5).int()
        regression_output = torch.where(binary == 1, self.regression_head_1(pooled_output), self.regression_head_0(pooled_output))
        return classification_output.squeeze(), regression_output.squeeze()

    def step(self, batch):
        # Feedforward
        input_ids, attention_mask, token_type_ids, regression_labels, binary_labels = batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['regression_label'], batch['binary_label']
        classification_output, regression_output = self(input_ids, attention_mask, token_type_ids)

        # Validation 과정에서 dimension 체크
        if classification_output.ndim == 0:
            classification_output = classification_output.unsqueeze(-1)

        if regression_output.ndim == 0:
            regression_output = regression_output.unsqueeze(-1)

        # Loss functions
        classification_loss = self.classifier_loss_fn(classification_output, binary_labels.float())
        regression_loss = self.regressor_loss_fn(regression_output, regression_labels)
        
        # Loss
        combined_loss = 0.5*classification_loss + 0.5*regression_loss
        
        # Evaluation metrics
        classification_probs = torch.sigmoid(classification_output)
        classification_preds = classification_probs > 0.5
        self.f1_score(classification_preds, binary_labels.float())
        self.pearson_corrcoef(regression_output, regression_labels.float())

        return regression_output, combined_loss

    def training_step(self, batch, batch_idx):
        _, combined_loss = self.step(batch)
        self.log_dict({
            'train_loss': combined_loss, 
            'train_pearson': self.pearson_corrcoef,
            'train_f1': self.f1_score}, 
            logger=True)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        _, combined_loss = self.step(batch)
        self.log_dict({
            'val_loss': combined_loss, 
            'val_pearson': self.pearson_corrcoef,
            'val_f1': self.f1_score}, 
            logger=True)
        
    def test_step(self, batch, batch_idx):
        regression_output, combined_loss = self.step(batch)
        self.predictions.append(regression_output.detach().cpu())
        self.log_dict({
            'test_loss': combined_loss, 
            'test_pearson': self.pearson_corrcoef,
            'test_f1': self.f1_score}, 
            logger=True)
            
    def predict_step(self, batch, batch_idx):
        self.eval()
        input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
        _, predictions = self(input_ids, attention_mask, token_type_ids)
        return predictions.detach().cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps= 0.05 * (self.total_steps),
                num_training_steps=self.total_steps),
            'interval': 'step'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def setup(self, stage='fit'):
        if stage =='fit':
            self.total_steps=self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())
        elif stage == 'test':
            self.predictions = []