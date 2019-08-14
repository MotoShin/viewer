import numpy as np
import csv
import seaborn as sns
import pandas as pd
import sys
import os

from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.font_manager as fm

def make_dir(dir_name):
    file_path = os.path.dirname(os.getcwd() + dir_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def result_plot(times, agent_result, dir_name, file_name, limit=True, max_num=9):
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    agents_order = ["Q-Learning", "RS+GRC", "RS Oblivion", "SL", "Max.Ent.IRL", "Max Reward"]

    for i in agents_order:
        if i in agent_result:
            if i == "Max Reward":
                ax.plot(agent_result[i], '--', label=i)
            else:
                ax.plot(agent_result[i], label=i)
        else:
            continue
    
    ax.set_xlabel("Episode", fontsize=20)
    ax.set_ylabel(file_name, fontsize=20)
    if limit:
        plt.ylim([0, max_num])
    plt.legend(loc=9, ncol=2, frameon=False, fontsize=18) # 各凡例が横に並ぶ数（default: 1）
    font = fm.FontProperties(size=20)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font)
    fig.savefig(dir_name+"{}.png".format(file_name))
    plt.close()

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

if __name__ == '__main__':
    args = sys.argv
    sns.set()

    if len(args) == 2:
        dir_png_name = "png/"+args[1]+"/"
        dir_csv_name = "csv/"+args[1]+"/"
    else:
        dir_png_name = "png/"
        dir_csv_name = "csv/"
    make_dir("/"+dir_png_name)
    make_dir("/"+dir_csv_name)

    rewards = pd.read_csv(dir_csv_name+"rewards.csv")
    regrets = pd.read_csv(dir_csv_name+"regrets.csv")
    notGreedyCounts = pd.read_csv(dir_csv_name+"notGreedyCounts.csv")

    agent_rewards = dfToNumpy(rewards)
    agent_regrets = dfToNumpy(regrets)
    agent_notGreedyCounts = dfToNumpy(notGreedyCounts)

    # print(agent_rewards)

    result_plot(agent_rewards["times"], agent_rewards, dir_png_name, "Rewards", max_num=8)
    result_plot(agent_regrets["times"], agent_regrets, dir_png_name, "Regrets", limit=False)
    result_plot(agent_notGreedyCounts["times"], agent_notGreedyCounts, dir_png_name, "NotGreedyCounts", max_num=1.2)
