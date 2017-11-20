# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
from scipy.sparse import coo_matrix

path1 = "/Users/wendyti/PycharmProjects/5001ProjectTransfer/tst_user_20170128.json"
path2 = "/Users/wendyti/PycharmProjects/5001ProjectTransfer/tst_article_20170128.json"
filename = "/Users/wendyti/PycharmProjects/5001ProjectTransfer/tr_data_20170128.json"

repUID = []
for line2 in open(path1):
    data_line2 = json.loads(line2)
    repUID.append(data_line2['user_id'])

uID = []
aID = []
dT = []
for line in open(filename):
    data_line = json.loads(line)
    if 'user_id' in data_line:
        if data_line['user_id'] in repUID:
            uID.append(data_line['user_id'])
            aID.append(data_line['article_contentid'])

            if 'article_dwelltime' in data_line:
                dT.append(data_line['article_dwelltime'])
            else:
                dT.append(0)

i = 0
while i < len(dT):
    if dT[i] < 5 or dT[i] > 250:
        dT[i] = 0
    i += 1

for item in repUID:
    if item in uID:
        continue
    else:
        uID.append(item)

dfUID = pd.DataFrame(uID, columns=['user_id'])

dfTime = pd.DataFrame(dT, columns=['article_dwelltime'])

for line1 in open(path2):
    data_line1 = json.loads(line1)
    if 'article_contentid' in data_line1:
        if data_line1['article_contentid'] in aID:
            continue
        else:
            aID.append(data_line1['article_contentid'])

dfAID = pd.DataFrame(aID, columns=['article_id'])
#设置articleID编码
dfAID.article_id = pd.Categorical(dfAID.article_id)
dfAID['article_code'] = dfAID.article_id.cat.codes
#设置usreID编码
dfUID.user_id = pd.Categorical(dfUID.user_id)
dfUID['user_code'] = dfUID.user_id.cat.codes

user_len = dfUID.drop_duplicates('user_id')
arti_len = dfAID.drop_duplicates('article_id')

df2 = pd.concat([dfUID, dfAID, dfTime], axis=1, join_axes=[dfAID.index])

addID = df2.iloc[:, 3]
df2.iloc[:, 3] = addID.fillna(addID.iloc[0])

d = arti_len.iloc[:, 1].drop_duplicates()
x = d.as_matrix()
row = np.array(df2.iloc[:, 3])
col = np.array(df2.iloc[:, 1])
data = np.array(df2.iloc[:, 4])

matrix = coo_matrix((data, (row, col)), shape=(len(user_len), len(arti_len))).toarray()
df3 = pd.DataFrame(matrix, columns=x)
df3 = df3.replace(0, np.nan)  #dataframe格式的trian矩阵
df3.to_csv('finaltest.csv')
