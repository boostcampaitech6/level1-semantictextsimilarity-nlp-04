import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback)
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import random
import os
from tqdm.auto import tqdm

# Function to set seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

# Function to join sentences with [SEP] token
def sep_join(dataframe):
    data = []
    for idx, item in tqdm(dataframe.iterrows()):
        sen1 = item['sentence_1']
        sen2 = item['sentence_2']
        totaltext = '[SEP]'.join([sen1, sen2])
        data.append(totaltext)
    return data

# Function to categorize labels
def categ(df):
    df['label_cat'] = df['label'].apply(lambda x: int(x // 0.5))
    return df

# Load your datasets
train = pd.read_csv("/data/ephemeral/home/becky/data/no_swap/train.csv")
dev = pd.read_csv("/data/ephemeral/home/becky/data/no_swap/val.csv")
test = pd.read_csv("/data/ephemeral/home/becky/data/test.csv")

# Process and categorize labels
train = categ(train)
dev = categ(dev)


# Process datasets
train['text'] = sep_join(train)
dev['text'] = sep_join(dev)
test['text'] = sep_join(test)

# Use integer labels directly
train['label'] = train['label_cat']
dev['label'] = dev['label_cat']


# Create Hugging Face datasets
df_train = pd.DataFrame({"text": train["text"], 'labels': train["label"]})
dataset_train = Dataset.from_pandas(df_train)
df_val = pd.DataFrame({"text": dev["text"], 'labels': dev["label"]})
dataset_val = Dataset.from_pandas(df_val)
df_test = pd.DataFrame({"text": test["text"]})  # 'labels' column if available
dataset_test = Dataset.from_pandas(df_test)

# Model and Tokenizer setup
model_checkpoint = "beomi/KcELECTRA-base-v2022"
num_labels = train['label'].nunique()  # Number of unique labels
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=100)

tokenized_train_datasets = dataset_train.map(tokenize_function, batched=True)
tokenized_val_datasets = dataset_val.map(tokenize_function, batched=True)
tokenized_test_datasets = dataset_test.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=100,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_dir="logs",
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Metric computation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_val_datasets,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()

# Optionally, save the model
#model.save_pretrained("your_model_directory")
