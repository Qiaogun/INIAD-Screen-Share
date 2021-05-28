import os
import sys
import re
import pandas as pd

data_list = ['Up_down', 'Click', 'Rock', 'Relax', 'Ok_pose', 'Chinese_seven']

base_root = os.path.abspath('.')
#print(base_root)
resList = []
for filename in os.listdir(base_root):
    if filename[0:4] =="data":
        new_path = base_root + "\\"+ filename
        resList.append(new_path)
#print(resList)
for fpath in resList:
    for csvfile in os.listdir(fpath):
        new_path = fpath + "\\" + csvfile
        #print(new_path)
        df = pd.read_csv(new_path)
        df.to_csv('./data/' + csvfile + '.csv',
                            mode='a',
                            header=False,
                            index=False)

