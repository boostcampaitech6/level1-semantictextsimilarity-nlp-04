import re
import pandas as pd
from pykospacing import Spacing
from hanspell import spell_checker

# 중복 글자 제거. 세 번 이상 반복되는 글자를 두 개로 줄이는 함수

def replace_repeated_chars(text):
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # text = spacing(text)
    # text = spell_checker.check(text).checked
    # print(text)
    if text:
        return text
    else:
        return ' '

spacing = Spacing()
df = pd.read_csv('./data/trainc_copy.csv')
df['sentence_1'] = df['sentence_1'].apply(replace_repeated_chars)
df['sentence_2'] = df['sentence_2'].apply(replace_repeated_chars)
df.to_csv('trainc_copy_pre.csv', index=False)