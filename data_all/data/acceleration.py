import os
import sys
import re
import pandas as pd

data_list = ['Up_down', 'Click', 'Rock', 'Relax', 'Ok_pose', 'Chinese_seven']

base_root = os.path.abspath('.')
print(base_root)
resList = []
for filename in os.listdir(base_root):
    if filename[-3:] =="csv":
        new_path = base_root + "\\"+ filename
        resList.append(new_path)
        df = pd.read_csv(filename , delimiter=',', header=0, skiprows=0,
                                   error_bad_lines=False)
        df1 = df.shift(1)
        df1.fillna(value=0)
        diff = df1[1:] - df[1:]
        diff.to_csv('./diff/' + filename[:-4] +'_diff' + '.csv',
                            mode='a',
                            header=False,
                            index=False)