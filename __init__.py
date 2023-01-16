

import os
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import datetime
import re
import pandas as pd

import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


import chinese_calendar as calendar # 导入日历包
from chinese_calendar import is_holiday ##是否节假日


# 画图
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


def set_color(method="mat", name="terrain", num=8, path=None):
    '''
    设置颜色的函数
    method:提供两种设置颜色的方式,"mat"代表matplotlib中的cmap获取配色方案，self代表从一张图片中获取配色方案
    name:在method为“mat”的时候生效,默认为“terrain”
    mun:需要颜色的个数
    path:当method="self"的时候，需要
    
    '''
    
    if path != None:
        method = "self"

    if method not in ("mat", "self"):
        print("方法只能在mat和self中选择!如果选择mat,则需要在matplotlib的cmap中选择一种，默认terrain")
    elif method == "mat":
        select = np.linspace(0.1, 0.95, num)
        colors = plt.get_cmap(name)(select)
    elif method == "self":
        from haishoku.haishoku import Haishoku

        a = Haishoku.loadHaishoku(path)
        colors = [[i[1][0] / 255, i[1][1] / 255, i[1][2] / 255] for i in a.palette]
        times = np.int(np.ceil(num / 8.0))
        colors = colors * times
        colors = colors[:num]
    return colors