
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

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# 画图
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

import semi_auto_writer as sw
set_color = sw.set_color


def trend_describe(
    data,
    value_cols,
    time_range=None,
    time_col="imp_date",
    holiday_col="is_holiday",
    p=0.05,
    color=None
):
    """
    判断趋势的函数,返回一段描述
    data:数据集,
    time_range:时间区间的列表或元组，需要有起止两个时间,时间必须是datetime格式的
    value_col:需要关注的序列列名，字符串或者列表、或者元组
    time_col:代表时间的列，默认“imp_date”，格式需要时“yyyy-MM-dd”的时间戳
    holiday_col：代表是否节假日的表示，格式需为 0或1 的数值
    p:显著性水平要求，默认为0.05

    数据要求：
    最好是经过semi_auto_writer.time_handel包调整过的数据框。
    需要有值列（value_cols）,时间列（time_col）和节假日标识列（holiday_col）
    ...  | time_col  |  value_col1  | value_col2 | holiday_col |
     0   | 20210101  |    34523     |    43212   |     1       |
     1   | 20210102  |    34343     |    43512   |     1       |
     2   | 20210103  |    34343     |    43512   |     0       |


    """
    # 如果时间序列未输入，则基于全部数据进行判断
    data = data.sort_values(time_col)
    colors=color

    if time_range == None:
        data_new = data
    else:
        data_new = data.loc[
            (data[time_col] >= time_range[0]) & (data[time_col] <= time_range[1]), :
        ]
    data_new["x"] = range(data_new.shape[0])
    

    # 判断如果是字符串，则转化成list,后面可以迭代循环
    if str(type(value_cols)) == "<class 'str'>":
        value_cols = [value_cols]
    else:
        value_cols = value_cols

    describe_df = pd.DataFrame(
        {"dim": value_cols, "slope": 0, "sig.": 0, "describe": 0}
    )

    for i in value_cols:
        formu = i + "~" + "x"
        fit = sm.formula.ols(formu, data=data_new).fit()
        
        describe_df.loc[describe_df.dim == i, "slope"] = fit.params[1]
        describe_df.loc[describe_df.dim == i, "sig."] = fit.pvalues[1]
        if fit.pvalues[1] >= p:
            des = "变化趋势不明显"
        elif fit.params[1] > data_new[i].mean() * 0.1 / data_new.shape[0]:
            des = "呈现上升趋势"
        elif fit.params[1] < -1 * data_new[i].mean() * 0.1 / data_new.shape[0]:
            des = "呈现下降趋势"
        else:
            des = "呈现稳定趋势"

        describe_df.loc[describe_df.dim == i, "describe"] = i + des
        print(describe_df.loc[describe_df.dim == i, "describe"].values[0] + "/n")
        draw_trend(data_new, i, time_col, holiday_col, color=colors)
    return describe_df


def draw_trend(data, value_cols, time_col, holiday_col, color=None):
    """
    画时序的图,
    data数据集,
    value_col:需要关注的序列列名，字符串或者列表、或者元组
    time_col:代表时间的列，默认“imp_date”
    holiday_col：代表是否节假日的表示，格式需为 0或1 的数值

    数据要求：
    最好是经过semi_auto_writer.time_handel包调整过的数据框。
    需要有值列（value_cols）,时间列（time_col）和节假日标识列（holiday_col）
    ...  | time_col  |  value_col1  | value_col2 | holiday_col |
     0   | 20210101  |    34523     |    43212   |     1       |
     1   | 20210102  |    34343     |    43512   |     1       |
     2   | 20210103  |    34343     |    43512   |     0       |

    """
    ## 用数据确定周末节假日的起止时间
    data["holiday_no"] = [1] + [
        (data[holiday_col].iloc[i - 1] * 1 + data[holiday_col].iloc[i] * 1)
        for i in range(1, data.shape[0])
    ]
    data["holiday_no"]
    data["holiday_no2"] = 0
    for i in range(1, data.shape[0]):
        if data.holiday_no.iloc[i] == 1:
            data.holiday_no2.iloc[i] = data.holiday_no2.iloc[i - 1] + 1
        else:
            data.holiday_no2.iloc[i] = data.holiday_no2.iloc[i - 1]

    
    fig = plt.figure(figsize=(14, 5))
    matplotlib.rcParams["font.family"] = ["Heiti TC"]

    if color == None:
        colors = set_color()[:2]
    else:
        colors = color[:2]

    plt.plot(
        data[time_col],
        data[value_cols],
        "-",
        color=colors[0]
    )

    plt.title("{}趋势图".format(value_cols))

    max_values = data[value_cols].max() * 1.1
    # 用阴影部分绘制周末
    for i in range(max(data.holiday_no2) + 1):
        if i % 2 == 1:
            date_pair = data.loc[data.holiday_no2 == i, "imp_date"]
            plt.fill_between(
                [date_pair.iloc[0], date_pair.iloc[-1]],
                0,
                max_values,
                facecolor=colors[1],
                alpha=0.1,
            )
    plt.ylim(0, max_values)



def draw_trend_stack(data, value_cols, time_col, holiday_col, color=None):
    """
    画时序的堆积图,
    data数据集,
    time_range:时间区间的列表或元组，需要有起止两个时间
    value_col:需要关注的序列列名，字符串或者列表、或者元组
    time_col:代表时间的列，默认“imp_date”

    holiday_col：代表是否节假日的表示，格式需为 0或1 的数值

    数据要求：
    最好是经过semi_auto_writer.time_handel包调整过的数据框。
    需要有值列（value_cols）,时间列（time_col）和节假日标识列（holiday_col）
    ...  | time_col  |  value_col1  | value_col2 | holiday_col |
     0   | 20210101  |    34523     |    43212   |     1       |
     1   | 20210102  |    34343     |    43512   |     1       |
     2   | 20210103  |    34343     |    43512   |     0       |
    """
    ## 用数据确定周末节假日的起止时间
    data["holiday_no"] = [1] + [
        (data[holiday_col].iloc[i - 1] * 1 + data[holiday_col].iloc[i] * 1)
        for i in range(1, data.shape[0])
    ]
    data["holiday_no"]
    data["holiday_no2"] = 0
    for i in range(1, data.shape[0]):
        if data.holiday_no.iloc[i] == 1:
            data.holiday_no2.iloc[i] = data.holiday_no2.iloc[i - 1] + 1
        else:
            data.holiday_no2.iloc[i] = data.holiday_no2.iloc[i - 1]

    fig = plt.figure(figsize=(14, 5))
    matplotlib.rcParams["font.family"] = ["Heiti TC"]

    # 颜色设置
    if color == None:
        colors = set_color(num=len(value_cols))
    else:
        colors = color[: len(value_cols)]

    print(data[value_cols].values.T)


    plt.stackplot(
        data[time_col], data[value_cols].values.T, labels=tuple(value_cols), alpha=0.5,
        colors=colors,
    )

    if len(value_cols) <= 4:
        plt.title("{}趋势堆叠图图".format("&".join(value_cols)))
        plt.legend()
    elif len(value_cols) <= 8:
        plt.title("趋势堆叠图图")
        plt.legend()
    else:
        plt.title("趋势堆叠图图")

    max_values = max(data[value_cols].sum(1)) * 1.1
    # 用阴影部分绘制周末
    if data["holiday_no"].iloc[0] == 0:
        k = 1
    else:
        k = 0
    for i in range(max(data.holiday_no2) + 1):
        if i % 2 == k:
            date_pair = data.loc[data.holiday_no2 == i, "imp_date"]
            plt.fill_between(
                [date_pair.iloc[0], date_pair.iloc[-1]],
                0,
                max_values,
                facecolor="b",
                alpha=0.1,
            )
    plt.ylim(0, max_values * 1)
    plt.yticks(fontsize="x-large")
    plt.xticks(fontsize="x-large")
    plt.legend(loc=2, bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0)



def time_compare_describe(
    data, time_col, value_col, focus_time=None, basic_time=None, p=0.05
):
    
    '''
    间比较函数：
    data:数据,
    time_col:data中代表时间的列名
    value_col:data中关注的数值的列名
    focus_time:报告期,需要列表
    basic_time:基期,需要列表
     p=0.05:阈值 

    数据要求：
    最好是经过semi_auto_writer.time_handel包调整过的数据框。
    需要有值列（value_cols）,时间列（time_col）和节假日标识列（holiday_col）
    ...  | time_col  |  value_col1  | value_col2 | holiday_col |
     0   | 20210101  |    34523     |    43212   |     1       |
     1   | 20210102  |    34343     |    43512   |     1       |
     2   | 20210103  |    34343     |    43512   |     0       |
    
    '''
    data = data.sort_values(time_col)

    if focus_time == None:
        focus_time = [data["imp_date"].iloc[-1], data["imp_date"].iloc[-1]]
    if basic_time == None:
        basic_time = [data["imp_date"].iloc[0], data["imp_date"].iloc[0]]

    start = data.loc[
        (data[time_col] >= basic_time[0]) & (data[time_col] <= basic_time[1]), value_col
    ].mean()
    end = data.loc[
        (data[time_col] >= focus_time[0]) & (data[time_col] <= focus_time[1]), value_col
    ].mean()
    delta = end - start
    delta_percent = delta / start

    def over_10t(num):
        if abs(num) > 10000:
            return str(np.round(abs(num) / 10000, 1)) + "w"
        else:
            return str(np.round(abs(num), 1))

    starts = over_10t(start)
    ends = over_10t(end)
    deltas = over_10t(delta)


    start_head = (
        "({}~{},{})".format(str(basic_time[0])[:10], str(basic_time[1])[:10], starts)
        if basic_time[0] != basic_time[1]
        else "({},{})".format(str(basic_time[0])[:10], starts)
    )
    end_head = (
        "({}~{},{})".format(str(focus_time[0])[:10], str(focus_time[1])[:10], ends)
        if focus_time[0] != focus_time[1]
        else "({},{})".format(str(focus_time[0])[:10], ends)
    )
    head = "在{}方面".format(value_col) + end_head + "相对于" + start_head

    if abs(delta_percent) <= 0.01:
        towords = "变化不明显。"
        detail = ""
    elif (delta_percent > 0.01) & (delta_percent <= p):
        towords = "有一定提升。"
        detial = "提升幅度为{}({:.2%})".format(deltas, delta_percent)
    elif (delta_percent < -0.01) & (delta_percent >= -1 * p):
        towords = "有一定下降。"
        detial = "下降幅度为{}({:.2%})".format(deltas, delta_percent)
    elif delta_percent <= -1 * p:
        towords = "有显著下降。"
        detial = "下降幅度为{}({:.2%})".format(deltas, delta_percent)
    elif delta_percent > p:
        towords = "有显著提升。"
        detial = "提升幅度为{}({:.2%})".format(deltas, delta_percent)

    return head + towords + detial
