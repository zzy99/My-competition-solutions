import pandas as pd
import numpy as np

sub1 = pd.read_csv('submission_4.33.csv', header=None, names=['filename', 's/e', 'time'])
sub2 = pd.read_csv('submission_4.25.csv', header=None, names=['filename', 's/e', 'time'])

pred = []
for i in range(len(sub1)):
    t1 = sub1['time'].values[i]
    t2 = sub2['time'].values[i]

    # if abs(t1 - t2) >= 10:
    #     pred.append(round(t1, 3))
    # else:
    pred.append(round(t1 * 0.5 + t2 * 0.5, 3))

sub1['time'] = pred

sub1.to_csv('submission.csv', index=False, header=None)
