
import os
import re
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import datetime

import pandas as pd
import numpy as np
import warnings
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib


warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


# 画图
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

import semi_auto_writer as sw

set_color = sw.set_color


def eda_num(data, cols):
    '''
    给定数据集和列，返回EDA结果graph_base.html
    '''
    eda = pd.DataFrame(data[cols].describe())
    eda.loc["cv", :] = eda.loc["std", :] / eda.loc["mean", :]

    for i in eda.columns:
        print("*-*-*-*-*-*-*-*-*-*{}探索数据分析*-*-*-*-*-*-*-*-*-*".format(i))
        print("                  ")
        print("集中趋势：{}的均值是{:.1f}".format(i, eda.loc["mean", i]))
        print(
            "离中趋势：{}的标准差是{:.1f},变异系数是{:.2f}".format(
                i, eda.loc["std", i], eda.loc["cv", i]
            )
        )
        print("                  ")

    if len(eda.columns) >= 2:
        corr(data[cols])
    return eda



def corr(data, cols=None, color="Blues"):
    '''
    计算相关系数的矩阵
    data：数据集
    cols：要计算相关性的列，默认None;如果默认，则对数据集中所有列计算相关系数
    '''
    if cols == None:
        corr = data.corr()
    else:
        corr = data[cols].corr()

    fig = plt.figure(figsize=(corr.shape[0] * 2, corr.shape[0] * 2))
  
    matplotlib.rcParams["font.family"] = ["Heiti TC"]
    sns.heatmap(
        corr, annot=True, annot_kws={"fontsize": "x-large"}, fmt=".2g", cmap=color
    )
    plt.yticks(fontsize="x-large")
    plt.xticks(fontsize="x-large")
    plt.title("相关系数矩阵", fontsize="x-large")
    print("*-*-*-*-*-*-*-*-*-*相关系数情况*-*-*-*-*-*-*-*-*-*")
    for i in range(1, corr.shape[0]):
        for j in range(0, i):
            print(
                "{}与{}的相关系数是{:.2f}".format(
                    corr.index[i], corr.columns[j], corr.iloc[i, j]
                )
            )

    return corr


def draw_pie(data, feature, values_col, func=np.mean, color=None):
    """
    data:数据集
    feature：用来分类的字段
    values_col:用来计算的字段
    func：计算的函数，如果要计数的话 写 len
    """
    data_pie = data.groupby(feature)[values_col].apply(func)
    data_pie = data_pie.sort_values(ascending=False)

    if color == None:
        colors = set_color(num=len(data_pie))
    else:
        colors = color[: len(data_pie)]

    patches, l_text, p_text = plt.pie(
        data_pie,
        labels=data_pie.index,
        autopct="%1.1f%%",
        # labeldistance=1.5,
        shadow=True,
        colors=colors,
        explode=[0.05] * 3,
        startangle=90,
        counterclock=False,
    )
    for t in p_text:
        t.set_size("x-large")
    for l in l_text:
        l.set_size("x-large")

    plt.title("{}的结构占比".format(feature), fontsize="x-large")

    for i in data_pie.index:
        print("{}的占比为{:.2%}".format(i, data_pie[i] / data_pie.sum()))


def draw_bar_line(data, y1_col, y2_col, if_y2_p=False, color=None):
    '''
    data:数据集，要求 index是双轴柱线图横轴所需要标签
    y1_col：柱状图所需要的列名
    y2_col: 折线图所需要的列名
    if_y2_p 折线图是否表现为百分比
    
    '''

    fig = plt.figure(figsize=(len(data.index) * (2 if len(data.index) <= 3 else 1), 5))
    ax1 = fig.add_subplot(111)
    plt.yticks(fontsize="x-large")
    if color == None:
        colors = set_color(num=2)
    else:
        colors = color[:2]

    matplotlib.rcParams["font.family"] = ["Heiti TC"]

    plt.bar(data.index, data[y1_col], color=colors[0], label=y1_col)
    ax2 = ax1.twinx()
    plt.plot(data.index, data[y2_col], "-yo", label=y2_col)  # , color=colors[1]

    import matplotlib.transforms as transforms

    # trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)

    for i in data.index:
        # print(i, y1_col)
        # print(data.loc[i, y1_col])
        ax1.text(
            i,
            data.loc[i, y1_col] * 0.5,
            s=round(data.loc[i, y1_col], 2),
            size="x-large",
            horizontalalignment="center",
            verticalalignment="center",
            # transform=trans,
            c="w",
        )

        s_lable = (
            round(data.loc[i, y2_col], 1)
            if if_y2_p == False
            else "{:.1%}".format(data.loc[i, y2_col])
        )

        ax2.text(
            i,
            data.loc[i, y2_col] + 0.02,
            s=s_lable,
            size="x-large",
            horizontalalignment="center",
            verticalalignment="center",
            # transform=trans,
            c="k",
        )

    ax1.legend(loc=2, bbox_to_anchor=(1.1, 0.5), borderaxespad=0.0, fontsize="x-large")
    ax2.legend(loc=2, bbox_to_anchor=(1.1, 0.6), borderaxespad=0.0, fontsize="x-large")
    ax1.set_xticklabels(labels=data.index, fontsize="x-large")

    # ax2.set_yticklabels(labels=[], fontsize="x-large") y的标签还没有设置好


def four_quadrant(
    data,
    x_col,
    y_col,
    method=np.mean,
    text_type="all",
    text_quadrant=["I"],
    labels=None,
    hue=None,
    color=None,
):
    """
    参数：
    data: 数据集
    x_col: 横轴的名字
    y_col：纵轴的名字
    method: 用什么方法分象限，默认为 均值 np.mean;75分位数 lambda x:np.quantile(x,0.75)
    text_type: 打标签的种类,默认为all 即所有标签都打,"quadrant"为只打某几个象限的标签,None不打标签
    text_quadrant: 列表,


    """
    data = data.copy()

    x_line = method(data[x_col])
    y_line = method(data[y_col])

    x_max = max(data[x_col])
    y_max = max(data[y_col])

    data["quadrant"] = ""
    data.loc[(data[x_col] >= x_line) & (data[y_col] >= y_line), "quadrant"] = "I"
    data.loc[(data[x_col] < x_line) & (data[y_col] >= y_line), "quadrant"] = "II"
    data.loc[(data[x_col] < x_line) & (data[y_col] < y_line), "quadrant"] = "III"
    data.loc[(data[x_col] >= x_line) & (data[y_col] < y_line), "quadrant"] = "IV"

    if color == None:
        colors = set_color(num=4)
    else:
        colors = color[:4]

    plt.figure(figsize=(15, 15))
    # 中文


    matplotlib.rcParams["font.family"] = ["Heiti TC"]
    sns.scatterplot(x_col, y_col, data=data, palette="Set2", hue=hue)

    # 打标功能
    if text_type == None:
        pass
    elif labels == None:
        print("请输入标签字段!")
    elif text_type == "all":
        for i in data.index:
            plt.text(
                data.loc[i, x_col],
                data.loc[i, y_col] * 1.05,
                s=data.loc[i, labels],
            )
    elif text_type == "quadrant":
        if type(text_quadrant) != list:
            print("text_quadrant需要为list形式!")
        elif len(text_quadrant) == 0:
            print("text_quadrant需要填写 ‘I’、‘II’、‘III’、‘IV’!")
        elif (
            len(
                [
                    text_quadrant[i]
                    for i in range(len(text_quadrant))
                    if text_quadrant[i] in ("I", "II", "III", "IV")
                ]
            )
            == 0
        ):
            print("text_quadrant需要填写 ‘I’、‘II’、‘III’、‘IV’!")

        data["text_row"] = data["quadrant"].map(lambda x: x in text_quadrant)
        for i in data.loc[data.text_row == True, :].index:
            plt.text(
                data.loc[i, x_col], data.loc[i, y_col] * 1.05, s=data.loc[i, labels]
            )

    plt.fill_between(
        np.linspace(x_line * 1.01, x_max),
        y_line * 1.01,
        y_max,
        alpha=0.1,
        color=colors[0],
    )

    plt.fill_between(
        np.linspace(0, x_line * 0.99),
        y_line * 1.01,
        y_max,
        alpha=0.1,
        color=colors[1],
    )

    plt.fill_between(
        np.linspace(0, x_line * 0.99),
        0,
        y_line * 0.99,
        alpha=0.1,
        color=colors[2],
    )

    # 第四象限
    plt.fill_between(
        np.linspace(x_line * 1.01, x_max),
        0,
        y_line * 0.99,
        alpha=0.1,
        color=colors[3],
    )

    plt.xticks(fontsize="x-large")
    plt.xlim(0, x_max * 1.01)
    plt.ylim(0, y_max * 1.01)
    plt.yticks(fontsize="x-large")
    plt.xlabel(x_col, fontsize="x-large")
    plt.ylabel(y_col, fontsize="x-large")
    plt.axvline(x_line, 0, 1, color="grey", linestyle="--")
    plt.axhline(y_line, 0, 1, color="grey", linestyle="--")

    return data

