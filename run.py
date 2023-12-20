import yaml

import pyprnt

from data_loader import Dataloader
from model import Model
from inference import inference
from train import set_seed, warning_block, parse_args, train

def main(configs: dict) -> None:

    print("---------------------------------------------------------------")
    pyprnt.prnt(configs)
    print("---------------------------------------------------------------")

    train(configs)
    print("***** Train done *****")
    
    inference(configs)
    print("***** inference done *****")


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
        if args.dev_path:
            configs['data']['dev_path'] = args.dev_path
        if args.val_path:
            configs['data']['val_path'] = args.val_path
        if args.predict_path:
            configs['data']['predict_path'] = args.predict_path

        # config 파일에 덮어씌우고 저장
        with open('config/config.yaml', 'w') as f:
            yaml.dump(configs, f)
    
    main(configs)