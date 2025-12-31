import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
from arima import predict_arima
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False 
# 读取数据
df_sita=pd.read_excel('./data/input/合并_关键词得分汇总.xlsx')
df_rate=pd.read_excel('./data/input/三年期国债收益率.xlsx')
df_total=pd.read_csv('./data/input/EV2&MV&PB&TotalAsset.csv')

company = input("请输入公司名称：").strip()
try:
    df_sig=pd.read_csv(f'./data/input/{company}年sig.csv')
    out_dir = f'./data/output/{company}'
    os.makedirs(out_dir, exist_ok=True)
except:
    print("没找到对应公司的sigma文件")

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
def caculate_K(symbol,yaer_num,company=company):
    return predict_arima(p=5,d=1,q=6,T=3,symbol=symbol,yaer_num=11,company=company)

def save_bs_results(a,symbol,yaer_num):
    years = np.arange(2015,yaer_num+2015)

    out = pd.DataFrame({'year': years, 'bs_price': a})
    out.to_csv(f'./data/output/{company}/bs_prices_{symbol}_with_year.csv', index=False)
    print(f'BS价格已保存到bs_prices_{symbol}_with_year.csv文件中。')

def compare_bs_ev2(bs_vals, symbol, yaer_num):
    ###############  绘制BS估值与EV2对比图  #################

    # 取对应的 11 期原始数据（用于年份和对比）
    mask = df_total["asharevalue_stat_symbol"] == symbol
    last11 = df_total.loc[mask].tail(yaer_num)

    x = np.arange(2015, 2015+yaer_num)

    # 可选：叠加原始 EV2 进行对比
    ev2_vals = last11["asharevalue_ev2"].values if "asharevalue_ev2" in last11.columns else None
    mkvt_vals = last11["asharevalue_stat_total_mv"].values if "asharevalue_stat_total_mv" in last11.columns else None
    ta = last11["balance_stat_total_assets"].values if "balance_stat_total_assets" in last11.columns else None
    plt.figure(figsize=(10, 6), dpi=100)

    plt.plot(x, bs_vals, 'b-o', label='BS_price', linewidth=2, markersize=5)
    if mkvt_vals is not None:
        plt.plot(x, mkvt_vals, 'orange', label='Market Value', linewidth=2)
    if ev2_vals is not None:
        plt.plot(x, ev2_vals, 'gray', label='EV', linewidth=2)
    if ta is not None:
        plt.plot(x, ta, 'g--', label='Total Assets', linewidth=2)

    plt.title('Comparison')
    plt.xlabel('Year' if isinstance(x[0], (np.integer, int)) else 'Index')
    plt.ylabel('Price')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'./data/output/{company}/Comparison_{symbol}.png', dpi=400)
    plt.show()



def main_bs(symbol,yaer_num,company=company):
    v_0 = df_total.loc[df_total["asharevalue_stat_symbol"] == symbol][-(yaer_num):]["asharevalue_ev2"].values
    sigma = caculate_sigma(symbol,yaer_num)
    K_series = caculate_K(symbol,yaer_num,company=company)[-(yaer_num+3):]
    r = df_rate['rate_3'].values[-(yaer_num):]
    bs_results = []
    for i in range(yaer_num):
        print(f"--------------------------{2015+i}年--------------------------")
        bs_result = BS(v_0[i],sigma[i],3,K_series[i+3],r[i])
        bs_results.append(bs_result[0])
    save_bs_results(bs_results,symbol,yaer_num)
    compare_bs_ev2(bs_results, symbol, yaer_num)
    return 0


#a = main_bs(symbol="002594.SZ",yaer_num=10)#比亚迪
a=(main_bs(symbol="600028.SH",yaer_num=10))#中国石化
#a=(main_bs(symbol="002230.SZ",yaer_num=10))#科大讯飞


