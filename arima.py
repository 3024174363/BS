#arima函数，输出未来三年以及历史数据
import pandas as pd
import numpy as np
# 设置matplotlib字体为Times New Roman
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False 

# 评估ARIMA模型的拟合效果和预测性能
def evaluate_arima_model(results, s, forecast, df, symbol):
    df.index=range(len(df))
    # 获取模型内样本预测值
    in_sample_predictions = results.predict(start=df.index[0], end=df.index[-1])

    # 计算各种拟合优度指标
    #因为差分了一期，所以从第二个开始比较
    r2 = r2_score(s[1:], in_sample_predictions[1:])
    mse = mean_squared_error(s[1:], in_sample_predictions[1:])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(s[1:], in_sample_predictions[1:])

    # 获取 AIC 和 BIC
    aic = results.aic
    bic = results.bic
    print("--------------------------预测指标--------------------------")
    print(f"R² (决定系数): {r2:.4f}")
    print(f"MSE (均方误差): {mse:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"AIC: {aic:.4f}")
    print(f"BIC: {bic:.4f}")

    # 可视化拟合效果 + 未来预测
    plt.figure(figsize=(12, 7), dpi=400)

    # 创建2013-2027年的年份标签
    years = np.arange(2014, 2029)
    historical_years = years[:len(df)]  # 历史数据年份
    forecast_years = years[len(df):len(df)+3]  # 预测年份

    # 1. 绘制历史实际值
    plt.plot(historical_years, df, 'b-', label='Actual values', linewidth=2)

    # 2. 绘制样本内预测值
    plt.plot(historical_years, in_sample_predictions, 'r--', label='In-sample predictions', linewidth=2)

    # 3. 绘制未来预测值
    plt.plot(forecast_years, forecast, 'g-o', label='Future forecasts', linewidth=2, markersize=6)

    # 添加预测区域的阴影
    plt.axvspan(historical_years[-1], forecast_years[-1], alpha=0.1, color='green', label='Forecast period')

    # 连接样本内预测的最后一点和未来预测的第一点
    plt.plot([historical_years[-1], forecast_years[0]], [in_sample_predictions[-1], forecast[0]], 
            'g-o', linewidth=2)

    # 图表装饰
    plt.legend(loc='best')
    #plt.title('ARIMA(5,1,6)模型拟合与预测')
    plt.xlabel('Year')
    #plt.ylabel('NDA (yuan)')
    plt.ylabel(r'NDA$_t$ (yuan)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 设置X轴的刻度和范围
    plt.xticks(np.arange(2014, 2029, step=1))
    plt.xlim(2013, 2028)

    plt.tight_layout()
    plt.savefig(f'./data/output/arima_model_fit_forecast_{symbol}.png',dpi=400)
    print("ARIMA模型拟合与预测图已保存到./data/output/路径中")
    #plt.show()
    



def predict_arima(p,d,q,T,symbol,yaer_num):
    # 读取数据
    df_sita=pd.read_excel('./data/input/合并_关键词得分汇总.xlsx')
    df_total=pd.read_csv('./data/input/EV2&MV&PB&TotalAsset.csv')

    df=df_total.loc[df_total["asharevalue_stat_symbol"] == symbol][-(yaer_num):]["asharevalue_ev2"]

    #23年再往前取10个
    sl=df_sita[df_sita['stock_id']==int(symbol[0:6])]['关键词得分(%)_总和'][(1-yaer_num):].values
    extended_sl = 1-np.concatenate([
        sl,                                          # 原始数组
        np.array([0.040869227])#比亚迪     
        #np.array([0.125580161])#中国石化
        #np.array([0.158201213])#科大讯飞
    ])
    s=extended_sl*df.values

    model = ARIMA(s, order=(p,d,q))
    results = model.fit()

    # 预测未来 3 年(周期)
    forecast = results.forecast(steps=T)

    a = np.concatenate([s, forecast])
    #print("s", s)
    evaluate_arima_model(results, s, forecast, df, symbol)
    return a
#print(predict_arima(p=5,d=1,q=6,T=3))

