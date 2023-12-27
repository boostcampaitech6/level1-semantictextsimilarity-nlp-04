import re
import yaml
import pandas as pd

abs_path = abs_path = '/data/ephemeral/home/level1-semantictextsimilarity-nlp-04'

with open(abs_path + '/config/config.yaml') as f:
    configs = yaml.safe_load(f)

file_path = abs_path + configs['data']['train_path'][1:]
file_name = file_path.split('/')[-1]

df = pd.read_csv(file_path) # load

# 중복제거
df = df.drop_duplicates(subset='id')

# id 순으로 sort
df['num'] = df['id'].apply(lambda x: int(re.findall(r'\d+', x)[1])) 
df = df.sort_values(by='num') 
df.drop('num', axis=1, inplace=True)

df.to_csv(file_path[:-len(file_name)]+'sorted_'+file_name, index=False)