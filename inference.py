import argparse
import yaml

import re
import warnings
import pyprnt

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from data_loader import Dataloader, Dataset
from model import Model


# 경고 제거
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*TensorBoard support*")
warnings.filterwarnings("ignore", ".*target is close to zero*")


if __name__ == '__main__':

    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 inference.py --model='klue/roberta-small' 혹은
    # 실행 시 --epoch 15 같이 입력하셔야 합니다. 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=config['train']['max_epochs'], type=int)
    parser.add_argument('--model', default=config['model']['model_name'], type=str)

    args = parser.parse_args()

    print("-------------------")
    pyprnt.prnt(config)
    print("-------------------")

    # argparser 이걸로 모델이름이랑, 시간 받아와서
    
    # dataloader와 model을 생성합니다.
    # mdoel_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path
    dataloader = Dataloader(args.model, 
                            config['train']['batch_size'], config['train']['shuffle'],
                            config['data']['train_path'], config['data']['dev_path'],
                            config['data']['test_path'], config['data']['predict_path'])
    
    # mdoel_name, learning_rate
    model = Model(args.model, float(config['train']['learning_rate']))

    # early stopping
    early_stopping_callbacks = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.epoch, log_every_n_steps=1, callbacks=[early_stopping_callbacks])

    # 모델 저장을 위한 이름 지정, /경로를 언더바로 변환 및 에포크를 하나로
    model_name = re.sub('/', '_', args.model)
    epoch = args.epoch

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    output_path = config['data']['output_path']
    model = torch.load(f'{output_path}{model_name}_{epoch}.pt')
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    # default = '/data/ephemeral/home/sub_mission.csv'
    
    output = pd.read_csv(config['data']['submission_path'])

    output['target'] = predictions
    output.to_csv(f'{output_path}output.csv', index=False)