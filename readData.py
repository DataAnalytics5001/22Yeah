# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json
import time

start = time.clock()
filename = "/Users/wendyti/PycharmProjects/5001ProjectTransfer/tr_data_20170128.json"
data = pd.DataFrame()
file = open(filename, 'r')
for line in file:
    data_line = json.loads(line)
        #data.append之后必须再赋值给data
    data = data.append(pd.Series(data_line), ignore_index=True)




