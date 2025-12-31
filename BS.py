import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
from arima import predict_arima

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
 
# 读取数据
df_sita=pd.read_excel('./data/input/合并_关键词得分汇总.xlsx')
df_rate=pd.read_excel('./data/input/三年期国债收益率.xlsx')
df_total=pd.read_csv('./data/input/EV2&MV&PB&TotalAsset.csv')

df_sig=pd.read_csv('./data/input/比亚迪年sig.csv')
#BS函数
def BS(v_0,sigma,T,K,r):
    
    #单利转复利
    R=(1+T*r)**(1/T)-1
    d1 = (log(v_0/K) + (R + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call_price = v_0 * norm.cdf(d1) - K * exp(-R * T) * norm.cdf(d2)
    print("-----------当前模型参数-----------")
    print(f"当前V_0:{v_0}\n当前sigma:{sigma}\n当前T:{T}\n当前NDAT:{K}\n当前利率r:{r}")
    print("-----------模型结果-----------")
    print('数据资产价值为{},\nd1为{},N(d1)为{},\nd2为{},N(d2)为{}'.format(call_price,d1,norm.cdf(d1),d2,norm.cdf(d2)))
    return call_price,K * exp(-R * T),norm.cdf(d1),norm.cdf(d2)

def caculate_sigma(symbol,yaer_num):
    #sigma = (df_total.loc[df_total["asharevalue_stat_symbol"] == symbol][-(yaer_num):]["asharevalue_ev2"]).pct_change().mean()
    #sigma = (df_total.loc[df_total["asharevalue_stat_symbol"] == symbol][-(yaer_num):]["balance_stat_total_assets"]).pct_change().std()
    #sigma1 = (df_total.loc[df_total["asharevalue_stat_symbol"] == symbol][-(yaer_num):]["balance_stat_total_assets"]).pct_change()[1:]
    sigma = df_sig['sig'].values[-(yaer_num):]
    #print("sigma:", sigma1)
    return sigma

#返回K的时间序列
def caculate_K(symbol,yaer_num):
    return predict_arima(p=5,d=1,q=6,T=3,symbol=symbol,yaer_num=11)

def save_bs_results(a,symbol,yaer_num):
    years = np.arange(2015,yaer_num+2015)

    out = pd.DataFrame({'year': years, 'bs_price': a})
    out.to_csv(f'./data/output/bs_prices_{symbol}_with_year.csv', index=False)
    print(f'BS价格已保存到bs_prices_{symbol}_with_year.csv文件中。')

def main_bs(symbol,yaer_num):
    v_0 = df_total.loc[df_total["asharevalue_stat_symbol"] == symbol][-(yaer_num):]["asharevalue_ev2"].values
    sigma = caculate_sigma(symbol,yaer_num)
    K_series = caculate_K(symbol,yaer_num)[-(yaer_num+3):]
    r = df_rate['rate_3'].values[-(yaer_num):]
    bs_results = []
    for i in range(yaer_num):
        print(f"--------------------------{2015+i}年--------------------------")
        bs_result = BS(v_0[i],sigma[i],3,K_series[i+3],r[i])
        bs_results.append(bs_result[0])
    save_bs_results(bs_results,symbol,yaer_num)
    return 0

a = main_bs(symbol="002594.SZ",yaer_num=10)
#print(main_bs(symbol="600028.SH",yaer_num=10))
