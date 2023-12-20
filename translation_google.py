# !pip3 uninstall googletrans
# !pip3 install googletrans==3.1.0a0  <- 안정적인 버전
# pip install --upgrade googletrans==4.0.0-rc1 <- 최신 버전

import pandas as pd
from googletrans import Translator
from tqdm.auto import tqdm

test = pd.read_csv('/data/ephemeral/home/becky/data/test.csv')
print(test)
translator = Translator()

def google_ko2en2ko(ko_text, translator):
    ko_en_result = translator.translate(ko_text, dest = 'en').text
    en_ko_result = translator.translate(ko_en_result, dest = 'ko').text
    return en_ko_result

test_text = "나는지금코드를보고있는중이다."
print(google_ko2en2ko(test_text, translator))
# >>> 지금 코드를 보고 있는 중이에요.
"""
sen1_list = test['sentence_1'] 
sen2_list = test['sentence_2'] 

for idx, sentence in enumerate(tqdm(sen1_list)):
    # print("[original] ", sentence)
    result =  google_ko2en2ko(sentence, translator)
    test.loc[idx,'sentence_1'] = result

for idx, sentence in enumerate(tqdm(sen2_list)):
    # print("[original] ", sentence)
    result =  google_ko2en2ko(sentence, translator)
    test.loc[idx,'sentence_2'] = result

print(test)
test.to_csv('/data/ephemeral/home/becky/data/trans_pre_test.csv', index= False)
"""




"""
from googletrans import Translator
translator = Translator()

a = translator.translate('오마이가뜨지져스크롸이스트휏', dest = 'en').text
print(a)
b = translator.translate(a, dest = 'ko').text
print(b)
from pykospacing import Spacing
spacing = Spacing()
c = spacing('오마이가뜨지져스크롸이스트휏')
print(c)
d = translator.translate(c, dest = 'en').text
print(d)
e = translator.translate(d, dest = 'ko').text
print(e)
"""