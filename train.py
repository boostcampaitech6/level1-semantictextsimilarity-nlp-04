import argparse
import yaml

import random
import os
import re
import shutil
import warnings
import pyprnt

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import transformers

from data_loader import Dataloader
from model import Model

import pytz
from datetime import datetime


def set_seed(seed:int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def warning_block() -> None:
    # 경고 제거, 함수의 반환은 없습니다.
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*TensorBoard support*")
    warnings.filterwarnings("ignore", ".*target is close to zero*")
    warnings.filterwarnings("ignore", ".*exists and is not*")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, help='model_name overriding')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--shuffle', type=bool)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--train_path')
    parser.add_argument('--dev_path')
    parser.add_argument('--val_path')
    parser.add_argument('--predict_path')

    return parser.parse_args()

def train(config: dict) -> None:
    '''
    컨픽을 받아와서 넣어주는 역할을 합니다.
    반환 없이 함수 내에서 파일을 생성하는 것으로 종료합니다.
    '''

    # 시드 고정
    set_seed(0)
    # 경고 제거
    warning_block()

    # 깔끔한 코드를 위한 변수명 지정
    model_name = config['model']['model_name']
    learning_rate = float(config['train']['learning_rate'])
    batch_size = config['train']['batch_size']
    max_epoch = config['train']['max_epoch']
    shuffle = config['train']['shuffle']
    train_path = config['data']['train_path']
    dev_path = config['data']['dev_path']
    val_path = config['data']['val_path']
    predict_path = config['data']['predict_path']
    output_path = config['data']['output_path']
    checkpoint_path = config['data']['checkpoint_path']
    saved_name = re.sub('/', '_', config['model']['model_name'])
    
    

    # dataloader와 model을 생성합니다.
    # mdoel_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path 지정
    dataloader = Dataloader(model_name, batch_size, shuffle, train_path, dev_path, val_path, predict_path)
    
    # model_name, learning_rate 지정
    model = Model(model_name, learning_rate)

    # early stopping
    early_stopping_callbacks = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    # model checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath= checkpoint_path,
        filename=f'{saved_name}_{max_epoch:02d}_{{val_loss:.2f}}',
        save_top_k=1,
        mode='min'
    )
    
    # wandb
    experiment_name = f"{model_name}_{max_epoch:02d}_{learning_rate}_{datetime.now(pytz.timezone('Asia/Seoul')):%y%m%d%H%M}"
    wandb_logger = WandbLogger(name=experiment_name, project='monitor', entity='level1-semantictextsimilarity-nlp-04', log_model=True)

    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epoch,
        log_every_n_steps=1,
        callbacks=[early_stopping_callbacks, checkpoint_callback],
        logger = wandb_logger
        )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)

    # Validation part
    trainer.test(model=model, datamodule=dataloader)
    valid_predictions = torch.cat(model.predictions, dim=0).numpy()
    validation_df = pd.read_csv(val_path)
    validation_df['prediction'] = valid_predictions

    # wandb에 dataframe을 업로드
    validation_table = wandb.Table(dataframe=validation_df)
    wandb.log({'validation_data': validation_table})
    
    wandb_logger.experiment.finish()

    # 모델 저장을 위한 이름 지정, /경로를 언더바로 변환 및 에포크를 하나로
    saved_name = re.sub('/', '_', config['model']['model_name'])

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, f'{output_path}{saved_name}_{max_epoch}.pt')

    # 데이터 파일을 복사합니다.
    shutil.copyfile(train_path, f'{output_path}{saved_name}_{max_epoch}.csv')


if __name__ == '__main__':

    # 하이퍼 파라미터 등 각종 설정값을 컨픽에서 입력받습니다
    with open('config/config.yaml') as f:
        configs = yaml.safe_load(f)

    # python3 train.py -> 컨픽에 적용된 설정 그대로 
    # python3 train.py --model_name="your_model" --batch_size=16 등으로 커스터마이즈 가능
    args = parse_args()
    
    if args:
    # args가 있다면 덮어 씌우기 진행, 없으면 패스.
        if args.model_name:
            configs['model']['model_name'] = args.model_name
        if args.batch_size:
            configs['train']['batch_size'] = args.batch_size
        if args.max_epoch:
            configs['train']['max_epoch'] = args.max_epoch
        if args.shuffle:
            configs['train']['shuffle'] = args.shuffle
        if args.learning_rate:
            configs['train']['learning_rate'] = args.learning_rate
        if args.train_path:
            configs['data']['train_path'] = args.train_path        
        if args.val_path:
            configs['data']['val_path'] = args.val_path
        if args.dev_path:
            configs['data']['dev_path'] = args.dev_path
        if args.predict_path:
            configs['data']['predict_path'] = args.predict_path

        # config 파일에 덮어씌우고 저장
        with open('config/config.yaml', 'w') as f:
            yaml.dump(configs, f)
    

    print("---------------------------------------------------------------")
    pyprnt.prnt(configs)
    print("---------------------------------------------------------------")

    train(configs)