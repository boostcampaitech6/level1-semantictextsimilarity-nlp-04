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

from data_loader import Dataloader
from model import Model

def warning_block() -> None:
    # 경고 제거, 함수의 반환은 없습니다.
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*TensorBoard support*")
    warnings.filterwarnings("ignore", ".*target is close to zero*")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, help='model_name overriding')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--shuffle', type=bool)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--train_path')
    parser.add_argument('--dev_path')
    parser.add_argument('--test_path')

    return parser.parse_args()

def inference(config: dict) -> None:

    # 깔끔한 코드를 위한 변수명 지정
    model_name = config['model']['model_name']
    learning_rate = float(config['train']['learning_rate'])
    batch_size = config['train']['batch_size']
    max_epoch = config['train']['max_epoch']
    shuffle = config['train']['shuffle']
    train_path = config['data']['train_path']
    dev_path = config['data']['dev_path']
    test_path = config['data']['test_path']
    submission_path = config['data']['submission_path']
    output_path = config['data']['output_path']
    saved_name = re.sub('/', '_', config['model']['model_name'])
    
    # dataloader와 model을 생성합니다.
    # model_name, batch_size, shuffle, train_path, dev_path, test_path
    dataloader = Dataloader(model_name, batch_size, shuffle, train_path, dev_path, test_path)
    
    # model_name, learning_rate
    model = Model(model_name, learning_rate)

    # early stopping
    early_stopping_callbacks = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )

    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epoch, log_every_n_steps=1)

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load(f'{output_path}{saved_name}_{max_epoch}.pt')
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    # default = '/data/ephemeral/home/sub_mission.csv'
    output = pd.read_csv(submission_path)

    output['target'] = predictions
    output.to_csv(f'{output_path}{saved_name}_{max_epoch}_{learning_rate}_output.csv', index=False)
 

if __name__ == '__main__':

    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 inference.py --model='klue/roberta-small' 혹은
    # 실행 시 --epoch 15 같이 입력하셔서 컨픽을 덮어쓸 수 있씁니다. 인자를 입력하지 않으면 컨픽 값이 기본으로
    with open('config/config.yaml') as f:
        configs = yaml.safe_load(f)

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
        if args.dev_path:
            configs['data']['dev_path'] = args.dev_path
        if args.test_path:
            configs['data']['test_path'] = args.test_path

        # config 파일에 덮어씌우고 저장
        with open('config/config.yaml', 'w') as f:
            yaml.dump(configs, f)

    print("---------------------------------------------------------------")
    pyprnt.prnt(configs)
    print("---------------------------------------------------------------")

    warning_block()
    inference(configs)