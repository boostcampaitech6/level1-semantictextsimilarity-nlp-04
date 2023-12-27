import pandas as pd
import numpy as np


output_path = './output'

outputs = ['beomi_KcELECTRA-base-v2022_10_1e-05_50.csv',
           'snunlp_KR-ELECTRA-discriminator_10_1e-05_output.csv']

df = pd.read_csv(f'{output_path}/{outputs[0]}')
df['target'] = df['target'].clip(lower=0.0, upper=5.0)

for i in range(1, len(outputs)):
    temp = (pd.read_csv(f'{output_path}/{outputs[i]}')['target']).clip(lower=0.0, upper=5.0)
    df['target'] += temp
df['target'] = np.round(df['target']/len(outputs),1)

new_name = ''
for output in outputs:
    new_name += output.split('_')[0] + '_'

df.to_csv(f'{output_path}/{new_name}.csv', index=False)