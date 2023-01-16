
import pandas as pd
import numpy as np
import chinese_calendar as calendar      # 导入日历包
from chinese_calendar import is_holiday  # 是否节假日
import sys


def transefer_to_date(x):
    '''
    将字符串或数字转化为日期的函数,
    x必须为字符串或者数字,长度必须为8,格式为yyyyMMdd
    '''
    if len(str(x))!=8:
        print('输入日期格式必须长度为8，且为yyyyMMdd格式!')
        sys.exit()
       
    if type(x) != "str":
        x = str(x)
    return pd.to_datetime(x[:4]+'-'+x[4:6]+'-'+x[6:])


def time_handle(data, time_col):
    """
    该函数负责将数据集中的时间字符串处理成时间格式，并且判断是否中国的节假日
    data:数据集,time_col数据集中时间戳的列表名字
    返回新的数据框,包含是否节假日，月份，周几，一年第几周，年份

    需要数据集中有时间列（time_col）和数值列 （value_col）的格式如下：
    ...  | time_col  |  value_col1  | value_col2 |
     0   | 20210101  |    34523     |    43212   |
     1   | 20210102  |    34343     |    43512   |
     2   | 20210103  |    34343     |    43512   |
    ……
    
    """
    data[time_col] = data[time_col].map(lambda x: transefer_to_date(x))
    data["is_holiday"] = data[time_col].map(lambda x: is_holiday(x))
    data['imp_month'] =data[time_col].map(lambda x:x.to_period(freq="M"))
    data['weekday'] =data[time_col].map(lambda x:x.weekday() + 1)
    data['year_week']=data[time_col].map(lambda x:x.isocalendar()[1])
    data['year']=data[time_col].map(lambda x:x.isocalendar()[0])
      
    return data


