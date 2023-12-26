import argparse
import yaml

import random
import re
import shutil
import warnings
import pyprnt

import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import transformers

from data_loader import Dataloader
from model import Model

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
    parser.add_argument('--test_path')
    parser.add_argument('--predict_path')

    return parser.parse_args()

if __name__ == '__main__':

    # 하이퍼 파라미터 등 각종 설정값을 컨픽에서 입력받습니다
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    # python3 train.py -> 컨픽에 적용된 설정 그대로 
    # python3 train.py --model_name="your_model" --batch_size=16 등으로 커스터마이즈 가능
    args = parse_args()

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

    model = Model(model_name, learning_rate)
    dataloader = Dataloader(model_name, batch_size, shuffle, train_path, dev_path, val_path, predict_path)
    
    df = pd.read_csv(predict_path)
    
    '''
    토크나이저는 데이터로더에서 불러오는 토크나이저를 사용함.
    데이터로더 모듈에서 함수를 작성하여 불러옴.
    토큰을 추가했을 때는 모델에서 토큰을 추가하고 model.resize_token_embeddings(len(dataloader.tokenizer)) 을 진행해주어야 함.
    '''

    # sentence_1, sentence_2 데이터 토큰화 진행 후 저장
    dataloader.get_me_csv(df)


    '''
    # # <PERSON> 토큰 추가 및 토크나이저 임베딩
    # new_tokens = ['<PERSON>']
    # dataloader.token_add(new_tokens)
    # model.model.resize_token_embeddings(len(dataloader.tokenizer))

    # 전체 unique vocab
    # vocab = dataloader.get_me_vocab(df)
    # print(vocab.most_common())
    
    # 데이터셋 문장 vocab
    # dataloader.get_me_sentence_vocab(df)
    

    # [UNK] 토큰 발생 확인 및 제거
    # def find_unk(sent1, sent2):
    #     t1 = dataloader.tokenizer.tokenize(sent1)
    #     t2 = dataloader.tokenizer.tokenize(sent2)
    #     return '[UNK]' in t1 or 'UNK' in t2
    
    # df['UNK'] = df.apply(lambda row: find_unk(row['sentence_1'], row['sentence_2']), axis=1)
    # new_data = df[~df['UNK']]
    '''