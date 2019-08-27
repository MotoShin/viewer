import numpy as np
import pandas as pd
import seaborn as sns
import sys

from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.font_manager as fm

from util import dfToNumpy, make_dir

def field_heatmap(name, field, nrow, ncol, ndim, dir_name, num=0):
    fig, ax = plt.subplots(2, 2, figsize=(10, 3))

    temp_field = np.zeros(nrow * 3 * ncol * 3)
    temp_field = temp_field.reshape(nrow * 3, ncol * 3)
    plot_field = {d: temp_field.copy() for d in range(ndim)}

    def to_s(row, col, dim):
        return row*ncol + col + dim*(ncol*nrow)

    for dim in range(ndim):
        for i in range(nrow):
            center_x = 1 + 3 * i
            for l in range(ncol):
                center_y = 1 + 3 * l
                temp = field[to_s(i, l, dim)]

                plot_field[dim][center_x - 1][center_y - 1] = np.mean(temp)
                plot_field[dim][center_x][center_y - 1] = temp[0] #UP
                plot_field[dim][center_x + 1][center_y - 1] = np.mean(temp)
                plot_field[dim][center_x - 1][center_y] = temp[3] #LEFT
                plot_field[dim][center_x][center_y] = max(temp)
                plot_field[dim][center_x + 1][center_y] = temp[1] #RIGHT
                plot_field[dim][center_x - 1][center_y + 1] = np.mean(temp)
                plot_field[dim][center_x][center_y + 1] = temp[2] #DWON
                plot_field[dim][center_x + 1][center_y + 1] = np.mean(temp)

    sns.heatmap(plot_field[0], ax=ax[0, 0])
    sns.heatmap(plot_field[1], ax=ax[0, 1])
    sns.heatmap(plot_field[2], ax=ax[1, 0])
    sns.heatmap(plot_field[3], ax=ax[1, 1])
    ### ヒートマップ上にグリッドを引いている ###
    vertices = []
    codes = []
    codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    a = 0
    b = 2.95
    vertices = [(a, a), (b, a), (b, b), (a, b), (0, 0)]
    for t in range(0, ncol):
        for i in range(0, nrow):
            codes += [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
            vertices += [(a+t*3, a+i*3), (b+t*3, a+i*3), (b+t*3, b+i*3), (a+t*3, b+i*3), (0, 0)]
    vertices = np.array(vertices, float)
    path = Path(vertices, codes)
    pathpatch1 = PathPatch(path, facecolor='None', edgecolor='white', linewidth=0.5)
    pathpatch2 = PathPatch(path, facecolor='None', edgecolor='white', linewidth=0.5)
    pathpatch3 = PathPatch(path, facecolor='None', edgecolor='white', linewidth=0.5)
    pathpatch4 = PathPatch(path, facecolor='None', edgecolor='white', linewidth=0.5)
    ax[0, 0].add_patch(pathpatch1)
    ax[0, 1].add_patch(pathpatch2)
    ax[1, 0].add_patch(pathpatch3)
    ax[1, 1].add_patch(pathpatch4)
    ### ここまで ###
    ### 軸の数字と目盛りを消している ###
    ax[0, 0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax[0, 1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax[1, 0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax[1, 1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ### ここまで ###

    # 余白を設定
    plt.subplots_adjust(wspace=0.05, hspace=1.2)
    # タイトルを設定
    ax[0,0].set_title("Q-values in 999 epi of Q-Learning", fontsize=15)
    ax[0,1].set_title("Q-values in 9999 epi of Q-Learning", fontsize=15)
    ax[1,0].set_title("Q-values in 999 epi of RS+GRC", fontsize=15)
    ax[1,1].set_title("Q-values in 9999 epi of RS+GRC", fontsize=15)
    if num == 0:
        fig.savefig(dir_name+"%s_Heatmap.png" % name)
    else:
        fig.savefig(dir_name+"Heatmap_{}.png".format(num))
    plt.close()

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

    # Q-Learningの前半のデータを抽出
    qlQDataFirstPattern = pd.read_csv(dir_csv_name+"QL/midMap_Q_999.csv")
    # Q-Learningの後半のデータを抽出
    qlQDataFinalPattern = pd.read_csv(dir_csv_name+"QL/midMap_Q_9999.csv")
    # RS+GRCの前半のデータを抽出
    rsQDataFirstPattern = pd.read_csv(dir_csv_name+"RS+GRC/midMap_Q_999.csv")
    # RS+GRCの後半のデータを抽出
    rsQDataFinalPattern = pd.read_csv(dir_csv_name+"RS+GRC/midMap_Q_9999.csv")

    # 特定の状態だけ抽出
    qlQDataFirstPattern = qlQDataFirstPattern.loc[134:138] 
    qlQDataFinalPattern = qlQDataFinalPattern.loc[134:138]
    rsQDataFirstPattern = rsQDataFirstPattern.loc[134:138]
    rsQDataFinalPattern = rsQDataFinalPattern.loc[134:138]

    # 変数dataに結合
    dataDfList = []
    dataDfList.append(qlQDataFirstPattern.loc[:,['0','1','2','3']])
    dataDfList.append(qlQDataFinalPattern.loc[:,['0','1','2','3']])
    dataDfList.append(rsQDataFirstPattern.loc[:,['0','1','2','3']])
    dataDfList.append(rsQDataFinalPattern.loc[:,['0','1','2','3']])

    dataDf = pd.concat(dataDfList, ignore_index=True)
    
    # numpy配列に変換
    data = np.array(dataDf.values.flatten())    
    data = data.reshape(20, 4)
    print(data)

    field_heatmap("compareQValue", data, 1, 5, 4, dir_png_name)
