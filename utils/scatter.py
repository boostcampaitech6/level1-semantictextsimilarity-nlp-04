import re
import sys
import yaml
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import transformers
import torch
import pytorch_lightning as pl
from data_loader import Dataloader
from model import Model


def warning_block() -> None:
    # 경고 제거, 함수의 반환은 없습니다.
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*TensorBoard support*")
    warnings.filterwarnings("ignore", ".*target is close to zero*")

def main(config):
    # 깔끔한 코드를 위한 변수명 지정
    model_name = config['model']['model_name']
    learning_rate = float(config['train']['learning_rate'])
    batch_size = config['train']['batch_size']
    max_epoch = config['train']['max_epoch']
    shuffle = config['train']['shuffle']
    train_path = config['data']['train_path']
    dev_path = config['data']['dev_path']
    test_path = config['data']['train_path']
    submission_path = config['data']['train_path']
    output_path = config['data']['output_path']
    saved_name = re.sub('/', '_', config['model']['model_name'])

    # dataloader와 model을 생성합니다.
    # model_name, batch_size, shuffle, train_path, dev_path, test_path
    dataloader = Dataloader(model_name, batch_size, shuffle, train_path, dev_path, test_path)
    
    # model_name, learning_rate
    model = Model(model_name, learning_rate)

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
    
    label = output['label'].to_list()

    return label, predictions

def sactter(x, y, file_name, file_path):
    plt.figure(figsize=(10, 7))

    colors = np.random.rand(len(x))
    plt.title(file_name)
    plt.scatter(x, y, s=5, c=colors)

    # y=x 직선 생성(보조선)
    plt.plot([0, 5], [0, 5], color='black')
    plt.xlabel('labels')
    plt.ylabel('predictions')
    plt.savefig(file_path, dpi=300)
    plt.show()

if __name__ == '__main__':

    with open('./config/config.yaml') as f:
        configs = yaml.safe_load(f)

    title = configs['data']['train_path'].split('/')[-1]
    save_path = configs['data']['output_path'] + title.split('.')[0] + '.png'

    sactter(*main(configs), file_name=title, file_path=save_path)