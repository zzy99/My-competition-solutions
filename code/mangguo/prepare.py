import os
import numpy as np
import pandas as pd
from tqdm import tqdm

train1 = pd.read_csv('必选数据.txt', header=None, sep=' ', names=['filename', 'time'])
train2 = pd.read_csv('补充数据.txt', header=None, sep=' ', names=['filename', 'time'])

train = pd.concat([train1, train2])

train['s/e'] = train['filename'].apply(lambda x: x.split('_')[-1][0])

train.to_csv('train.csv', index=False)

test = os.listdir('data_A/test')

test = pd.DataFrame(test)
test.columns = ['filename']
test['s/e'] = test['filename'].apply(lambda x: x.split('_')[-1][0])
test['time'] = 0
test.loc[test['s/e'] == 's', 'time'] = 70.502
test.loc[test['s/e'] == 'e', 'time'] = 131.242

test.to_csv('test_a.csv', index=False, header=None)

test = os.listdir('data_C')

test = pd.DataFrame(test)
test.columns = ['filename']
test['s/e'] = test['filename'].apply(lambda x: x.split('_')[-1][0])
test['time'] = 0
test.loc[test['s/e'] == 's', 'time'] = 70.502
test.loc[test['s/e'] == 'e', 'time'] = 131.242
test

test.to_csv('test_b.csv', index=False, header=None)
