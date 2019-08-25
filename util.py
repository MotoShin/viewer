import numpy as np
import pandas as pd
import sys
import os

def make_dir(dir_name):
    file_path = os.path.dirname(os.getcwd() + dir_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def dfToNumpy(data):
    data.drop("Unnamed: 0", axis=1)
    columns = list(data.columns)
    print(columns)
    #np.arrayに変換
    npdata=np.array(data.values.flatten())
    #npdataの形を、pandaで読み込んだデータフレームの形に変形する
    nparray=np.reshape(npdata,(data.shape[0],data.shape[1]))
    nparray = np.rot90(nparray, k=-1)
    nparray = np.flip(nparray, axis=1)

    # print(nparray)

    dataDict = {}
    for i in range(len(columns)):
        if columns[i] == "Unnamed: 0":
            nparray[i] = nparray[i] + 1
            dataDict.update({"times": nparray[i]})
        else:
            dataDict.update({columns[i]: nparray[i]})

    return dataDict