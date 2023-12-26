import pandas as pd
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback,
                          Trainer, TrainingArguments)
from torch.nn.functional import softmax
import numpy as np
import random
import os
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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

# Function to one-hot encode labels
def one_hot_encode_labels(labels):
    labels = torch.tensor(labels).to(torch.int64)  # Ensure labels are integers
    return torch.nn.functional.one_hot(labels, num_classes=2).tolist()

# Load your datasets
train = pd.read_csv("/data/ephemeral/home/becky/data/train_trans_downs.csv")
dev = pd.read_csv("/data/ephemeral/home/becky/data/dev.csv")
test = pd.read_csv("/data/ephemeral/home/becky/data/test.csv")

# Process datasets
train['text'] = sep_join(train)
dev['text'] = sep_join(dev)
test['text'] = sep_join(test)

# One-hot encode labels
train['binary-label'] = one_hot_encode_labels(train['binary-label'])
dev['binary-label'] = one_hot_encode_labels(dev['binary-label'])

# Create Hugging Face datasets
df_train = pd.DataFrame({"title": train["text"], 'label': train["binary-label"]})
dataset_train = Dataset.from_pandas(df_train)
df_val = pd.DataFrame({"title": dev["text"], 'label': dev["binary-label"]})
dataset_val = Dataset.from_pandas(df_val)
df_test = pd.DataFrame({"title": test["text"]})
dataset_test = Dataset.from_pandas(df_test)

# Model and Tokenizer setup
num_labels = 2  # For binary classification with two output logits
#model_checkpoint = "team-lucid/deberta-v3-base-korean"
model_checkpoint = 'snunlp/KR-ELECTRA-discriminator'
wd = 0.01
lr = 2e-5
epochs = 50
task = "binary_classification"
label_list = [0,1]

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels,  hidden_dropout_prob=0.05)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["title"], padding="max_length", truncation=True, max_length=100)

tokenized_train_datasets = dataset_train.map(tokenize_function, batched=True)
tokenized_val_datasets = dataset_val.map(tokenize_function, batched=True)
tokenized_test_datasets = dataset_test.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    f"test-{task}",
    learning_rate= lr,#2e-5
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=epochs,
    seed = 42,
    weight_decay=wd,
    load_best_model_at_end=True,
    warmup_steps = 600,
    logging_steps=500,
    do_eval=True,
    save_total_limit=2,
    report_to="wandb",
    save_strategy='steps',
    evaluation_strategy='steps',
    save_steps=500,
    eval_steps=500
    )


# Metric computation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Convert one-hot encoded labels to single integer labels
    labels_single = labels.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_single, preds, average='macro')
    acc = accuracy_score(labels_single, preds)
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

# Evaluate the model
#trainer.evaluate()

# or call with pretrained model
# model = LightningTransformer.load_from_checkpoint(PATH)

test_predictions = trainer.predict(tokenized_test_datasets)


# Apply softmax to convert logits to probabilities (for binary, use index 1 for class 1)
predicted_labels = np.argmax(test_predictions.predictions, axis=-1)


# Save to CSV
submission_df = pd.read_csv('/data/ephemeral/home/becky/data/test.csv')[['id']]
submission_df['binary-label-pred'] = predicted_labels
submission_df.to_csv('/data/ephemeral/home/becky/data/test_binary_result_final.csv', index=False)