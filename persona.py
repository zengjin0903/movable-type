import pandas as pd
import numpy as np
import chinese_calendar as calendar  # 导入日历包
from chinese_calendar import is_holiday  ##是否节假日
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys
import semi_auto_writer as sw

plt.style.use("seaborn-whitegrid")
import matplotlib

set_color = sw.set_color


def tgi(data, base, compare, large=1, dim_name="",color=None):
    """
    计算和两个人群在某维度上的对比,返回TGI数据集和相关描述以及图像
    data,数据集，该数据集的index必须为需要比较维度的枚举项，如性别中的“男”和“女”，列为需要比较维度，值是比例
    base:需要比较的维度,字符串
    compare:对比的维度，字符串
    lager: 一个阈值，默认为1。如果大于这个阈值，就认为是显著特征

    数据表形式：
    需要的数据表应该是数据透视表的形式，形式如下：

           |  维度1 |  维度2  |
    枚举项1 |   12  |   23    |
    枚举项2 |   12  |   34    |
    枚举项3 |   45  |   23    |



    """
    tgi_df = data[compare] / data[base]
    tgi_df = tgi_df.sort_values(ascending=False)
    tgi_large = tgi_df[tgi_df > large]
    head_str = dim_name + "维度上," if dim_name != "" else ""
    print(
        head_str + "{}相对于{}而言，在".format(compare, base),
        "'、'".join(tgi_large.index),
        "'维度上占比更大,TGI更高",
    )

    print(
        "TGI为",
        "、".join([str(round(tgi_large.values[i], 2)) for i in range(len(tgi_large))]),
    )
    

    if color == None:
        colors = set_color()[0]
    else:
        colors = color[0]

    fig = plt.figure(figsize=(14, 5))

    matplotlib.rcParams["font.family"] = ["Heiti TC"]
    plt.plot(
        tgi_df,
        "-o",
        color=colors
    )
    for i in tgi_df.index:
        plt.text(i, tgi_df[i] + 0.05, s=round(tgi_df[i], 2), size="x-large")

    plt.title("{}TGI".format(dim_name), size="x-large")
    plt.xticks(size="x-large")

    return tgi_df


def p_stackplot(data, dim_name="",color=None):
    '''
    绘制两组数据的百分比堆积图
    data: 数据集，index 比较的枚举项，列为比较的对象
    dim_name 图表名称,默认为空
    数据表形式：
    需要的数据表应该是数据透视表的形式，形式如下：

           |  维度1 |  维度2  |
    枚举项1 |   12  |   23    |
    枚举项2 |   12  |   34    |
    枚举项3 |   45  |   23    |
    
    '''
    data_cum = data.cumsum()
    

    fig = plt.figure(figsize=(10, 10))
    ax=fig.add_subplot(111)
    matplotlib.rcParams["font.family"] = ["Heiti TC"]  # 绘图
     # x轴坐标刻度字体大小
    plt.yticks([0,0.2,0.4,0.6,0.8,1],[0,'20%','40%','60%','80%','100%'],fontsize=15)  # y轴坐标刻度字体大小
    plt.ylim(0, 1)  # 设置y轴范围
    bottom_values = [0] * len(data.columns)
    # 颜色设置
    if color==None:
        colors=set_color(num=len(data.index))
    else:
        colors=color[:len(data.index)]
    #画图
    k=0
    for i in data.index:
        plt.bar(
            x=data.columns,
            height=data.loc[i],
            bottom=bottom_values,
            alpha=0.7,
            tick_label=i,
            label=i,
            color=colors[k]
        )
        k+=1

        #加标签
        for j in range(len(data.columns)):
            if data.loc[i].iloc[j]>0.03:
                plt.text(x=data.columns[j],y=(data_cum.loc[i].iloc[j]-bottom_values[j])*0.5+bottom_values[j],
                         s = "{:.2%}".format(data.loc[i].iloc[j]),c="k",size="x-large",
                         horizontalalignment="center",verticalalignment="center")
        bottom_values += data.loc[i]
    plt.legend(loc=2, bbox_to_anchor=(1.05,0.5),borderaxespad = 0.)
    plt.title(dim_name,fontsize='x-large')
    plt.xticks(data.columns,data.columns,fontsize='x-large') 





def v_compare(data, if_p=False, dim_name="", color=None):
    """
    横向柱图比较，
    data: 数据集，index 比较的枚举项，列为比较的对象
    dim_name 图表名称,默认为空
    if_p 是否百分比，默认为不是
    数据表形式：
    需要的数据表应该是数据透视表的形式，形式如下：
           |  维度1 |  维度2  |
    枚举项1 |   12  |   23    |
    枚举项2 |   12  |   34    |
    枚举项3 |   45  |   23    |
    """
    fig = plt.figure(figsize=(data.shape[0] * data.shape[1], 8))
    ax = fig.add_subplot(111)

    matplotlib.rcParams["font.family"] = ["Heiti TC"] 
    x = np.arange(data.shape[0])
    plt.xticks(x, data.index, fontsize="x-large")
    plt.yticks(fontsize="x-large")
    widths = 0.8 / data.shape[1]

    # 设置颜色
    if color == None:
        colors = set_color(num=len(data.columns))
    else:
        colors = color[: len(data.columns)]
    k = 0

    ## 绘图
    for i in data.columns:
        plt.bar(x, data[i], width=widths, alpha=0.7, label=i, color=colors[k])
        for j in range(len(x)):
            y_loc = data.loc[data.index[j], i]
            if if_p:
                ss = "{:.1%}".format(y_loc)
                size_s = "x-large"
            elif len(str(y_loc)) < 3:
                ss = y_loc
                size_s = "x-large"
            elif len(str(y_loc)) < 5:
                ss = y_loc
                size_s = "large"
            elif len(str(y_loc)) < 8:
                ss = y_loc
                size_s = "medium"
            else:
                ss = ""
                size_s = "medium"

            plt.text(
                x=x[j],
                y=(y_loc) * 0.5,
                s=ss,
                c="k",
                size=size_s,
                horizontalalignment="center",
                verticalalignment="center",
            )

        x = x + widths
        k += 1
    plt.legend(loc=2, bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0)
