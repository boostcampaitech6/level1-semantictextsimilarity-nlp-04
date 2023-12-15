import re
from pykospacing import Spacing
spacing = Spacing()


def preprocess(text: str):
    '''
    특수문자 제거 + spacing 적용
    '''
    p = re.compile("[a-zA-Z가-힣0-9]+")
    text = p.findall(text)
    text = ' '.join(text)
    text = spacing(text)
    if text: return text
    else: return ' '


def copied(sentence: str):
    '''
    문장 복사 -> 전처리 후 리턴
    '''
    return preprocess(sentence)


def copy_data(data_path: str, df:pd.DataFrame):
    '''
    문장 복사 -> 전처리 후 데이터프레임 반환
    '''
    copied_sentence_1 = []
    copied_sentence_2 = []
    id_list_1 = []
    id_list_2 = []
    source_list_1 = []
    source_list_2 = []
    label_list = [5.0]*len(df)
    binary_label = [1.0]*len(df)
    for i in range(len(df)):
      copied_sentence_1.append(copied(df.loc[i].sentence_1))
      copied_sentence_2.append(copied(df.loc[i].sentence_2))
      id_list_1.append(df.loc[i].id+'_copy')
      id_list_2.append(df.loc[i].id+'_copy')
      source_list_1.append(df.loc[i].source+'_copy')
      source_list_2.append(df.loc[i].source+'_copy')
    df_copy_1 = pd.DataFrame({
        'id': id_list_1,
        'source': source_list_1,
        'sentence_1': df.sentence_1,
        'sentence_2': copied_sentence_1,
        'label': label_list,
        'binary-label': binary_label
    })
    df_copy_2 = pd.DataFrame({
        'id': id_list_2,
        'source': source_list_2,
        'sentence_1': copied_sentence_2,
        'sentence_2': df.sentence_2,
        'label': label_list,
        'binary-label': binary_label
    })
    return concat_data(data_path, df_copy_1, df_copy_2)


def swap(df: pd.DataFrame, label_min: float, label_max: float):
    '''
    특정 범위 내의 라벨에 해당하는 행의 문장 순서를 바꾼 데이터프레임 반환
    '''
    swap_sentence1 = []
    swap_sentence2 = []
    id_list = []
    source_list = []
    label_list = []
    binary_label = []
    for i in range(len(df)):
      if df.loc[i].label >= label_min and df.loc[i].label <= label_max:
        swap_sentence1.append(df.loc[i].sentence_2)
        swap_sentence2.append(df.loc[i].sentence_1)
        id_list.append(df.loc[i].id+'_swap')
        source_list.append(df.loc[i].source+'_swap')
        label_list.append(df.loc[i].label)
        binary_label.append(df.loc[i]['binary-label'])
    df_swap = pd.DataFrame({
        'id': id_list,
        'source': source_list,
        'sentence_1': swap_sentence1,
        'sentence_2': swap_sentence2,
        'label': label_list,
        'binary-label': binary_label
    })
    return df_swap


def concat_data(data_path: str, *dataframes: pd.DataFrame):
    """
    데이터프레임을 합쳐서 csv 파일로 저장하는 함수
    """
    result = pd.concat(dataframes)
    result.to_csv(data_path, index=False)
