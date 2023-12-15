import argparse
import yaml

import random
import re
import shutil
import warnings
import pyprnt

import pandas as pd

from tqdm.auto import tqdm
import pytorch_lightning as pl
import transformers
import torch
import torchmetrics

from data_loader import Dataloader, Dataset
from model import Model


# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

# 경고 제거
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*TensorBoard support*")
warnings.filterwarnings("ignore", ".*target is close to zero*")

if __name__ == '__main__':

    # 하이퍼 파라미터 등 각종 설정값을 컨픽에서 입력받습니다
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    print("-------------------")
    pyprnt.prnt(config)
    print("-------------------")

    # dataloader와 model을 생성합니다.
    # mdoel_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path 지정
    dataloader = Dataloader(config['model']['model_name'], 
                            config['train']['batch_size'], config['train']['shuffle'],
                            config['data']['train_path'], config['data']['dev_path'],
                            config['data']['test_path'], config['data']['predict_path'])
    
    # model_name, learning_rate 지정
    model = Model(config['model']['model_name'], float(config['train']['learning_rate']))
    
    # 모델 저장을 위한 이름 지정, /경로를 언더바로 변환 및 에포크를 하나로
    model_name = re.sub('/', '_', config['model']['model_name'])
    epoch = config['train']['max_epochs']

    # early stopping
    early_stopping_callbacks = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    # model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath= config['data']['checkpoint_path'],
        filename=f'{model_name}_{epoch:02d}_{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=epoch, log_every_n_steps=1, callbacks=[early_stopping_callbacks, checkpoint_callback])

    # # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, f'output/{model_name}_{epoch}.pt')

    # 데이터 파일을 복사합니다.
    output_path = config['data']['output_path']
    shutil.copyfile(config['data']['train_path'], f'{output_path}/{model_name}_{epoch}.csv')