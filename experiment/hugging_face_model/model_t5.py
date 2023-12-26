import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, EvalPrediction, EarlyStoppingCallback
import pandas as pd
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimilarityDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=512):
        self.dataframe = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        record = self.dataframe.iloc[idx]
        input_text = f"similarity: {record['sentence_1']} [SEP] {record['sentence_2']}"
        inputs = self.tokenizer.encode_plus(
            input_text, 
            add_special_tokens=True, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0).long()  # Convert to long
        attention_mask = inputs['attention_mask'].squeeze(0).long()  # Convert to long

        # Assuming label is a floating point number for a regression task
        label_str = str(record['label'])
        label_tokens = self.tokenizer.encode(label_str, add_special_tokens=False)
        label_length = 5  # Choose an appropriate length
        label_tokens += [self.tokenizer.pad_token_id] * (label_length - len(label_tokens))

        labels = torch.tensor(label_tokens, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class SimilarityDatasetWithoutLabels(Dataset):
    def __init__(self, filename, tokenizer, max_length=512):
        self.dataframe = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        record = self.dataframe.iloc[idx]
        input_text = f"similarity: {record['sentence_1']} [SEP] {record['sentence_2']}"
        inputs = self.tokenizer.encode_plus(
            input_text, 
            add_special_tokens=True, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0).long()  # Convert to long
        attention_mask = inputs['attention_mask'].squeeze(0).long()  # Convert to long


        return {"input_ids": input_ids, "attention_mask": attention_mask}

# Define the compute_metrics function



print(f"Using device: {device}")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
train_dataset = SimilarityDataset('/data/ephemeral/home/becky/data/train.csv', tokenizer)
dev_dataset = SimilarityDataset('/data/ephemeral/home/becky/data/dev.csv', tokenizer)
test_dataset = SimilarityDatasetWithoutLabels('/data/ephemeral/home/becky/data/test.csv', tokenizer)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False, pin_memory=True)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
#t5-base -> t5-large, or google/flan-t5-xl
model.to(device)

training_args = TrainingArguments(
    output_dir='./results_flan_xl_t5',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    do_eval=True,
    save_total_limit=2,
    report_to="wandb",
    run_name="test_flan_t5",
    save_strategy='steps',
    evaluation_strategy='steps',  
    save_steps=100,
    eval_steps=100,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()

def predict(model, tokenizer, test_dataset):
    model.eval()
    predictions = []
    for item in test_dataset:
        inputs = {k: v.to(device).unsqueeze(0) for k, v in item.items()}
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = model.generate(input_ids)
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            try:
                prediction = float(decoded_output)  # Convert string prediction back to float
            except:
                print("errors")
                prediction = 0.0
            predictions.append(prediction)
    return predictions

test_predictions = predict(model, tokenizer, test_dataset)

submission_df = pd.read_csv('/data/ephemeral/home/becky/data/test.csv')[['id']]
submission_df['target'] = test_predictions
submission_df.to_csv('submission.csv', index=False)
