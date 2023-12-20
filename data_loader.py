import argparse
import random

import pandas as pd

import pytorch_lightning as pl
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path, is_test=False):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        sentence_1 = self.data.iloc[idx]['sentence_1']
        sentence_2 = self.data.iloc[idx]['sentence_2']
        encoded = self.tokenizer(sentence_1, sentence_2, truncation=True, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        if not self.is_test:
            item['labels'] = torch.tensor(self.data.iloc[idx]['label'])
        return item


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, val_path, predict_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.val_path = val_path
        self.predict_path = predict_path

        self.collate_fn = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors='pt')

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_dataset = Dataset(self.tokenizer, self.train_path)
            self.dev_dataset = Dataset(self.tokenizer, self.dev_path)
        else:
            self.val_dataset = Dataset(self.tokenizer, self.val_path)
            self.predict_dataset = Dataset(self.tokenizer, self.predict_path, is_test=True)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dev_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)