#arima函数，输出未来三年以及历史数据
import pandas as pd
import numpy as np
# 设置matplotlib字体为Times New Roman
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False 

def predict_arima(p=5,d=1,q=6,T=3,symbol="002594.SZ",yaer_num=11):
    # 读取数据
    df_sita=pd.read_excel('合并_关键词得分汇总.xlsx')
    df_total=pd.read_csv('EV2&MV&PB&TotalAsset.csv')

    df=df_total.loc[df_total["asharevalue_stat_symbol"] == symbol][-(yaer_num):]["asharevalue_ev2"]

    #23年再往前取10个
    sl=df_sita[df_sita['stock_id']==2594]['关键词得分(%)_总和'][-10:].values
    extended_sl = 1-np.concatenate([
        sl,                                          # 原始数组
        np.array([0.040869227149405])                # 末尾添加的1个值
    ])
    s=extended_sl*df.values

    model = ARIMA(s, order=(p,d,q))
    results = model.fit()

    # 预测未来 3 年(周期)
    forecast = results.forecast(steps=T)

    a = np.concatenate([s, forecast])

    return a
print(predict_arima(p=5,d=1,q=6,T=3))
