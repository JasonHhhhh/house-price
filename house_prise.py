import pandas as pd
import numpy as np
import math
df=pd.read_csv('train.csv')
for i in df:
    '''
    change str to int
    maybe one-hot
    '''
    if type(df[i][0]) is str:
        keylist=[]
        for j in df[i]:
            if j not in keylist:
                keylist.append(j)
        print(keylist)

    '''
    fill nan by mean 
    '''
    df[i]=df[i].fillna(df[i][df[i].notnull()].mean())
    print(df[i][13])