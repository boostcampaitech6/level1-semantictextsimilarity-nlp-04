import os
import wget
import pandas as pd

# download
train_url = "https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorSTS/sts-train.tsv"
dev_url = "https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorSTS/sts-dev.tsv"
test_url = "https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorSTS/sts-test.tsv"

wget.download(train_url)
wget.download(dev_url)
wget.download(test_url)

# load dataframe
orginal_train = pd.read_csv('./data/train.csv')
orginal_dev = pd.read_csv('./data/dev.csv')
orginal_test = pd.read_csv('./data/test.csv')

additional_train = pd.read_csv('./sts-train.tsv', sep=r'\t')
additional_dev = pd.read_csv('./sts-dev.tsv', sep=r'\t')
additional_test = pd.read_csv('./sts-test.tsv', sep=r'\t')

# rename columns
columns = {'genre' : 'source',
           'score' : 'label',
           'sentence1' : 'sentence_1',
           'sentence2' : 'sentence_2'}

additional_train.rename(columns=columns,inplace=True)
additional_dev.rename(columns=columns,inplace=True)
additional_test.rename(columns=columns,inplace=True)

# concatenate
new_train = pd.concat([orginal_train, additional_train], join='inner', axis=0, ignore_index=True)
new_dev = pd.concat([orginal_dev, additional_dev], join='inner', axis=0, ignore_index=True)
new_test = pd.concat([orginal_test, additional_test], join='inner', axis=0, ignore_index=True)

# save new file
new_train.to_csv('./data/concat_train.csv', sep=',', encoding='utf-8', index=False)
new_dev.to_csv('./data/concat_dev.csv', sep=',', encoding='utf-8', index=False)
new_test.to_csv('./data/concat_test.csv', sep=',', encoding='utf-8', index=False)

# remove downloaded file
os.remove('./sts-train.tsv')
os.remove('./sts-dev.tsv')
os.remove('./sts-test.tsv')