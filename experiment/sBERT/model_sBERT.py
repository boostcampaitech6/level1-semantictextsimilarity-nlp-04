import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction,EarlyStoppingCallback
import pandas as pd
from scipy.stats import pearsonr
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SimilarityDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=128):
        self.dataframe = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        record = self.dataframe.iloc[idx]
        inputs = self.tokenizer.encode_plus(
            str(record['sentence_1']), str(record['sentence_2']),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        inputs = {key: val.squeeze(0).to('cpu') for key, val in inputs.items()} # Ensure tensors are on CPU
        inputs["label"] = torch.tensor(record['label'], dtype=torch.float)
        return inputs

class SimilarityDatasetWithoutLabels(Dataset):
    def __init__(self, filename, tokenizer, max_length=128):
        self.dataframe = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        record = self.dataframe.iloc[idx]
        inputs = self.tokenizer.encode_plus(
            str(record['sentence_1']), str(record['sentence_2']),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {key: val.squeeze(0).to('cpu') for key, val in inputs.items()}
        return inputs

def compute_metrics(p: EvalPrediction):
    predictions = p.predictions
    if len(predictions.shape) == 2:
        predictions = predictions[:, 0]  # Extract the first column if predictions are 2D
    return {"pearson": pearsonr(predictions, p.label_ids)[0]}

# snunlp/KR-SBERT-V40K-klueNLI-augSTS
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
train_dataset = SimilarityDataset("/data/ephemeral/home/becky/data/train.csv", tokenizer)
dev_dataset = SimilarityDataset("/data/ephemeral/home/becky/data/dev.csv", tokenizer)
test_dataset = SimilarityDatasetWithoutLabels('/data/ephemeral/home/becky/data/test.csv', tokenizer)  # For prediction
# , hidden_dropout_prob=0.3
model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-SBERT-V40K-klueNLI-augSTS", num_labels=1)
model.to(device)
training_args = TrainingArguments(
    output_dir='./lucid',
    num_train_epochs=40,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    warmup_steps = 600,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
    do_eval=True,
    save_total_limit=2,
    report_to="wandb",
    run_name="lucid",
    save_strategy='steps',
    evaluation_strategy='steps',
    save_steps=500,
    eval_steps=500,
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

trainer.train()

def predict(model, tokenizer, test_dataset):
    model.eval()  # Set the model to evaluation mode
    predictions = []

    # Iterate over the test dataset
    for item in test_dataset:
        # Prepare the input data and move it to the same device as the model
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in item.items()}

        with torch.no_grad():  # Disable gradient calculation
            outputs = model(**inputs)  # Get model predictions
            logits = outputs.logits
            # Store predictions; you might want to apply a softmax here if needed
            predictions.append(logits.squeeze().cpu().numpy())

    return predictions

test_predictions = predict(model, tokenizer, test_dataset)

submission_df = pd.read_csv('/data/ephemeral/home/becky/data/test.csv')[['id']]
submission_df['target'] = test_predictions
submission_df.to_csv('/data/ephemeral/home/level1-semantictextsimilarity-nlp-04/lucid/submission.csv', index=False)