import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, get_linear_schedule_with_warmup
from torchmetrics.functional import pearson_corrcoef
from torchmetrics.classification import BinaryF1Score



class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(model_name)
        self.classification_head = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.regression_head = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.stacking_head = torch.nn.Linear(2, 1)
        self.lr = lr
        self.f1 = BinaryF1Score()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids, attention_mask, token_type_ids)
        if 'pooler_output' in outputs:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0]
        classification_output = self.classification_head(pooled_output)
        regression_output = self.regression_head(pooled_output)
        combined_output = torch.concat((classification_output, regression_output), dim=1)
        final_output = self.stacking_head(combined_output)
        return classification_output.squeeze(), regression_output.squeeze(), final_output.squeeze()

    def step(self, batch):
        input_ids, attention_mask, token_type_ids, regression_labels, binary_labels = batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['regression_label'], batch['binary_label']
        classification_output, regression_output, final_output = self(input_ids, attention_mask, token_type_ids)

        classification_loss = F.binary_cross_entropy_with_logits(classification_output, binary_labels.float())
        classification_probs = torch.sigmoid(classification_output)
        classification_preds = (classification_probs > 0.5).int()
        self.f1(classification_preds, binary_labels.int())
        
        regression_loss = F.l1_loss(regression_output, regression_labels)

        final_loss = F.mse_loss(final_output, regression_labels)
        pearson = pearson_corrcoef(final_output, regression_labels.to(torch.float64))

        combined_loss = classification_loss + regression_loss + final_loss

        return final_output, pearson, combined_loss

    def training_step(self, batch, batch_idx):
        _, pearson, combined_loss = self.step(batch)
        self.log_dict({
            'train_loss': combined_loss, 
            'train_pearson': pearson,
            'train_f1': self.f1}, 
            logger=True)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        _, pearson, combined_loss = self.step(batch)
        self.log_dict({
            'val_loss': combined_loss, 
            'val_pearson': pearson,
            'val_f1': self.f1}, 
            logger=True)
        
    def test_step(self, batch, batch_idx):
        final_output, pearson, combined_loss = self.step(batch)
        self.predictions.append(final_output.detach().cpu())
        self.log_dict({
            'test_loss': combined_loss, 
            'test_pearson': pearson,
            'test_f1': self.f1}, 
            logger=True)
            
    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
        _, _, predictions = self(input_ids, attention_mask, token_type_ids)
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps= 0.1 * (self.total_steps),
                num_training_steps=self.total_steps),
            'interval': 'step'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def setup(self, stage='fit'):
        if stage =='fit':
            self.total_steps=self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())
        elif stage == 'test':
            self.predictions = []


